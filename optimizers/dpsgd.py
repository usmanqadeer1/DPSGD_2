"""
Damped Preconditioned Stochastic Gradient Descent (DPSGD).

Implements Algorithm 1 (PSGD Kronecker preconditioner update) and Algorithm 2
(DPSGD training step) from DPSGD_SN paper, including:
  - Tikhonov damping (Eq. 36 additive / Eq. 46 factor-absorbed)
  - Levenberg-Marquardt trust-region tau adaptation (Eq. 29-33)
  - Exponential factor averaging (Eq. 47)
  - Step-size clipping (Eq. 48)
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

from optimizers.psgd_kron import precond_grad_kron, update_precond_kron

_TINY = 1.2e-38


def _trace_ratio_pi(p1: Tensor, p2: Tensor) -> float:
    return (torch.trace(p1) * p2.shape[0]) / (torch.trace(p2) * p1.shape[0] + _TINY)


def _damped_factors(
    ql: Tensor,
    qr: Tensor,
    tau: float,
    eta: float,
    factor_absorbed: bool,
) -> Tuple[Tensor, Tensor]:
    """Form damped P_L, P_R from Q factors (Eq. 36 or 46)."""
    p1 = ql.t().mm(ql)
    p2 = qr.t().mm(qr)
    pi = _trace_ratio_pi(p1, p2)
    il = torch.ones(p1.shape[0], device=p1.device, dtype=p1.dtype)
    ir = torch.ones(p2.shape[0], device=p2.device, dtype=p2.dtype)
    damp = math.sqrt(eta + tau)
    if factor_absorbed:
        # Absorb sqrt(tau) into Q before forming P (Eq. 46).
        scale_l = math.sqrt(pi * damp)
        scale_r = math.sqrt((1.0 / pi) * damp)
        ql_eff = ql + scale_l * torch.diag(il)
        qr_eff = qr + scale_r * torch.diag(ir)
        return ql_eff.t().mm(ql_eff), qr_eff.t().mm(qr_eff)
    # Additive correction on P (Eq. 36).
    p1 = p1 + torch.diag(torch.sqrt(pi * damp) * il)
    p2 = p2 + torch.diag(torch.sqrt((1.0 / pi) * damp) * ir)
    return p1, p2


def _precond_grad_additive(pl: Tensor, pr: Tensor, grad: Tensor) -> Tensor:
    return pl.mm(grad).mm(pr)


def _update_lambda(loss1: Tensor, loss2: Tensor, m: float, lambd: float, omega: float) -> float:
    """Levenberg-Marquardt damping update (Eq. 33)."""
    if m <= 0:
        return lambd
    r = abs(loss2.item() - loss1.item()) / m
    if r > 0.75:
        return lambd * omega
    if r < 0.25:
        return lambd / omega
    return lambd


class DPSGD(Optimizer):
    """
    DPSGD optimizer for 2D weight matrices (Linear / reshaped Conv weights).

    Expects gradients with create_graph=True when step() is called after backward
    with retain_graph, or uses internal autograd.grad on matrix parameters.
    """

    def __init__(
        self,
        matrix_params: Iterable[Tensor],
        lr: float = 0.1,
        T1: int = 1,
        T2: int = 5,
        beta: float = 0.7,
        tau0: float = 1.0,
        eta: float = 1e-5,
        nu_scale: float = 0.1,
        factor_absorbed: bool = True,
        precond_lr: float = 0.1,
        weight_decay: float = 0.0,
    ):
        matrix_params = list(matrix_params)
        if not matrix_params:
            raise ValueError("DPSGD requires at least one 2D parameter tensor.")
        n_params = sum(p.numel() for p in matrix_params)
        nu = nu_scale * math.sqrt(n_params)
        defaults = dict(
            lr=lr,
            T1=T1,
            T2=T2,
            beta=beta,
            tau=tau0,
            eta=eta,
            nu=nu,
            factor_absorbed=factor_absorbed,
            precond_lr=precond_lr,
            weight_decay=weight_decay,
        )
        super().__init__(matrix_params, defaults)
        self._step_count = 0
        self._omega = (19.0 / 20.0) ** T1
        self._init_state()

    def _init_state(self) -> None:
        for p in self.param_groups[0]["params"]:
            st = self.state[p]
            m, n = p.shape
            dev, dt = p.device, p.dtype
            st["Ql"] = torch.eye(m, device=dev, dtype=dt)
            st["Qr"] = torch.eye(n, device=dev, dtype=dt)
            st["Pl"] = torch.zeros(m, m, device=dev, dtype=dt)
            st["Pr"] = torch.zeros(n, n, device=dev, dtype=dt)

    @property
    def needs_closure(self) -> bool:
        return True

    def _clip_threshold(self) -> float:
        params = self.param_groups[0]["params"]
        return 0.1 * math.sqrt(sum(p.numel() for p in params))

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        if closure is None:
            raise ValueError("DPSGD requires a closure that returns the loss.")
        loss = closure()
        if loss is None:
            return None

        group = self.param_groups[0]
        lr = group["lr"]
        T1 = group["T1"]
        T2 = group["T2"]
        beta_max = group["beta"]
        tau = group["tau"]
        eta = group["eta"]
        nu = group["nu"]
        factor_absorbed = group["factor_absorbed"]
        precond_lr = group["precond_lr"]

        params = group["params"]
        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            allow_unused=True,
        )
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]

        self._step_count += 1
        n = self._step_count - 1
        beta = min(n / (n + 1), beta_max) if n > 0 else beta_max

        # Algorithm 1: preconditioner update every T1 steps.
        if self._step_count % T1 == 0:
            v_list = [torch.randn_like(p) for p in params]
            hv_list = torch.autograd.grad(
                grads,
                params,
                grad_outputs=v_list,
                retain_graph=True,
            )
            hv_list = [h if h is not None else torch.zeros_like(p) for h, p in zip(hv_list, params)]
            with torch.no_grad():
                for p, v, hv in zip(params, v_list, hv_list):
                    st = self.state[p]
                    st["Ql"], st["Qr"] = update_precond_kron(
                        st["Ql"],
                        st["Qr"],
                        v,
                        hv,
                        step=precond_lr,
                    )
                    # Double update as in reference notebooks for T1=1.
                    st["Ql"], st["Qr"] = update_precond_kron(
                        st["Ql"],
                        st["Qr"],
                        v,
                        hv,
                        step=precond_lr,
                    )

        pre_grads: List[Tensor] = []
        raw_grads = grads

        with torch.no_grad():
            for p, g in zip(params, raw_grads):
                st = self.state[p]
                ql, qr = st["Ql"], st["Qr"]
                if factor_absorbed:
                    # Eq. 47: EMA on undamped factors; Eq. 46: damped step.
                    p1, p2 = ql.t().mm(ql), qr.t().mm(qr)
                    st["Pl"] = beta * st["Pl"] + (1.0 - beta) * p1
                    st["Pr"] = beta * st["Pr"] + (1.0 - beta) * p2
                    p1d, p2d = _damped_factors(ql, qr, tau, eta, factor_absorbed=True)
                    pg = _precond_grad_additive(p1d, p2d, g.detach())
                else:
                    # Eq. 36: additive damping inside EMA factors and gradient.
                    p1, p2 = _damped_factors(ql, qr, tau, eta, factor_absorbed=False)
                    st["Pl"] = beta * st["Pl"] + (1.0 - beta) * p1
                    st["Pr"] = beta * st["Pr"] + (1.0 - beta) * p2
                    pg = _precond_grad_additive(st["Pl"], st["Pr"], g.detach())
                pre_grads.append(pg)

            grad_norm = torch.sqrt(sum(torch.sum(pg * pg) for pg in pre_grads))
            clip_thr = self._clip_threshold()
            step_adjust = min(clip_thr / (grad_norm + _TINY), 1.0)
            alpha_adj = min(nu / (grad_norm + _TINY), 1.0) * lr * step_adjust

            for p, pg in zip(params, pre_grads):
                p.add_(pg, alpha=-alpha_adj)

            # LM damping update every T2 steps (Eq. 29-33).
            if self._step_count % T2 == 0 and tau >= eta:
                m_vals = []
                for g, pg in zip(raw_grads, pre_grads):
                    m_vals.append(0.5 * torch.dot(g.detach().view(-1), (lr * pg).view(-1)).item())
                m_pred = min(m_vals) if m_vals else 0.0
                loss1 = loss.detach()
                loss2 = closure().detach()
                tau = _update_lambda(loss1, loss2, m_pred, tau, self._omega)
                group["tau"] = tau

        return loss
