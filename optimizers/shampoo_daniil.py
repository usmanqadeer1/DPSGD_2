"""
PyTorch Shampoo following Daniil-Selikhanovych/Shampoo_optimizer.

The upstream repo is TensorFlow-based (``shampoo_optimizer.py`` + ``matrix_square_root_power.py``).
This module ports the same update rule to PyTorch:

  - Accumulate Kronecker factors Gbar_i from gradient outer products (mat_gbar_decay=1).
  - Precondition with Gbar_i^{-alpha/n} via matrix SVD (``_compute_power_svd`` path).
  - For dim > max_matrix_size use diagonal Shampoo (``_apply_gradient`` large-tensor branch).

Defaults from ``ShampooOptimizer`` / paper Table 1:
  learning_rate (α), max_matrix_size (limit), alpha=0.5, epsilon=1e-4.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer

_TINY = 1.2e-38


def matrix_power_svd(matrix: Tensor, power: float, epsilon: float = 1e-4) -> Tensor:
    """Matrix power via SVD with damping (TF ``_compute_power_svd`` equivalent)."""
    device = matrix.device
    n = matrix.shape[0]
    if n == 1:
        return torch.pow(matrix + epsilon, power)
    mat = matrix
    damping = epsilon * torch.eye(n, device=device, dtype=mat.dtype)
    mat_cpu = (mat + damping).cpu()
    u, s, vh = torch.linalg.svd(mat_cpu)
    s = torch.clamp(s, min=epsilon)
    result = u @ torch.diag(s.pow(power)) @ vh
    return result.to(device)


class Shampoo(Optimizer):
    """
    Shampoo optimizer (PyTorch port of Daniil-Selikhanovych/Shampoo_optimizer).

    Only 2D parameter tensors receive full Kronecker Shampoo; others use diagonal
    accumulation or are skipped (1D biases updated with plain scaled gradient).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.5,
        max_matrix_size: int = 200,
        alpha: float = 0.5,
        epsilon: float = 1e-4,
        mat_gbar_decay: float = 1.0,
        weight_decay: float = 0.0,
    ):
        params = [p for p in params if p.requires_grad]
        defaults = dict(
            lr=lr,
            max_matrix_size=max_matrix_size,
            alpha=alpha,
            epsilon=epsilon,
            mat_gbar_decay=mat_gbar_decay,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)
        n_params = sum(p.numel() for p in params)
        self._nu = 0.1 * math.sqrt(n_params)

        for p in params:
            self._init_state(p)

    def _init_state(self, p: Tensor) -> None:
        st = self.state[p]
        if p.dim() == 2:
            m, n = p.shape
            dev, dt = p.device, p.dtype
            lim = self.defaults["max_matrix_size"]
            eps = self.defaults["epsilon"]
            if m <= lim:
                st["Gbar_L"] = eps * torch.eye(m, device=dev, dtype=dt)
                st["inv_L"] = torch.eye(m, device=dev, dtype=dt)
            else:
                st["Gbar_L"] = eps * torch.ones(m, device=dev, dtype=dt)
                st["inv_L"] = torch.ones(m, device=dev, dtype=dt)
            if n <= lim:
                st["Gbar_R"] = eps * torch.eye(n, device=dev, dtype=dt)
                st["inv_R"] = torch.eye(n, device=dev, dtype=dt)
            else:
                st["Gbar_R"] = eps * torch.ones(n, device=dev, dtype=dt)
                st["inv_R"] = torch.ones(n, device=dev, dtype=dt)
        elif p.dim() == 1:
            st["Gbar"] = self.defaults["epsilon"] * torch.ones_like(p)

    def _update_gbar(
        self, gbar: Tensor, grad: Tensor, axes: Tuple[int, ...], decay: float, weight: float
    ) -> Tensor:
        grad_outer = torch.tensordot(grad, grad, dims=(axes, axes))
        return decay * gbar + weight * grad_outer

    def _diagonal_shampoo(self, gbar: Tensor, grad: Tensor, axes: Tuple[int, ...], neg_alpha: float) -> Tensor:
        if axes:
            normalizer = float(grad.shape[axes[0]]) if axes else 1.0
            grad_outer = grad.pow(2).sum(dim=axes) / normalizer
        else:
            grad_outer = grad.pow(2)
        gbar.mul_(self.defaults["mat_gbar_decay"]).add_(grad_outer)
        return torch.where(gbar > 0, torch.pow(gbar, neg_alpha), torch.zeros_like(gbar))

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lim = group["max_matrix_size"]
            alpha = group["alpha"]
            eps = group["epsilon"]
            decay = group["mat_gbar_decay"]
            wd = group["weight_decay"]

            pre_grads: List[Tensor] = []
            params_with_grad: List[Tensor] = []

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                if wd > 0:
                    g = g + wd * p.data

                st = self.state[p]

                if p.dim() == 2:
                    m, n = g.shape
                    neg_alpha = -alpha / 2.0  # n=2 Kronecker dims

                    if m <= lim:
                        axes_l = (1,)
                        st["Gbar_L"] = self._update_gbar(st["Gbar_L"], g, axes_l, decay, 1.0)
                        st["inv_L"] = matrix_power_svd(st["Gbar_L"], neg_alpha, eps)
                        pre = torch.matmul(st["inv_L"], g)
                    else:
                        st["Gbar_L"] = self._diagonal_shampoo(st["Gbar_L"], g, (1,), neg_alpha)
                        pre = st["Gbar_L"].reshape(-1, 1) * g

                    if n <= lim:
                        axes_r = (0,)
                        st["Gbar_R"] = self._update_gbar(st["Gbar_R"], g, axes_r, decay, 1.0)
                        st["inv_R"] = matrix_power_svd(st["Gbar_R"], neg_alpha, eps)
                        pre = torch.matmul(pre, st["inv_R"])
                    else:
                        st["Gbar_R"] = self._diagonal_shampoo(st["Gbar_R"], g, (0,), neg_alpha)
                        pre = pre * st["Gbar_R"].reshape(1, -1)

                    pre_grads.append(pre)
                    params_with_grad.append(p)

                elif p.dim() == 1:
                    neg_alpha = -alpha
                    st["Gbar"] = self._diagonal_shampoo(st["Gbar"], g, (), neg_alpha)
                    pre_grads.append(st["Gbar"] * g)
                    params_with_grad.append(p)

            if not pre_grads:
                continue

            grad_norm = torch.sqrt(sum(torch.sum(pg * pg) for pg in pre_grads))
            clip_thr = 0.1 * math.sqrt(sum(p.numel() for p in params_with_grad))
            step_adjust = min(clip_thr / (grad_norm + _TINY), 1.0)
            alpha_adj = min(self._nu / (grad_norm + _TINY), 1.0) * lr * step_adjust

            for p, pg in zip(params_with_grad, pre_grads):
                p.add_(pg, alpha=-alpha_adj)

        return loss


def build_shampoo(
    params: Iterable[Tensor],
    lr: float = 0.5,
    limit: int = 200,
    alpha: float = 0.5,
    epsilon: float = 1e-4,
    weight_decay: float = 0.0,
) -> Shampoo:
    """Factory matching configs ``lr`` and ``limit`` (→ ``max_matrix_size``)."""
    return Shampoo(
        params,
        lr=lr,
        max_matrix_size=limit,
        alpha=alpha,
        epsilon=epsilon,
        mat_gbar_decay=1.0,
        weight_decay=weight_decay,
    )
