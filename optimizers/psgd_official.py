"""
PSGD baseline via official ``lixilinx/psgd_torch`` (``KronNewton``).

Matches the notebook / paper setup: Newton-type Kronecker preconditioner with
exact Hessian–vector products, ``lr_params`` = Table-1 α, ``lr_preconditioner`` = 0.1,
preconditioner updates every ``T1`` steps, gradient clipping ``0.1√N``.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, List, Optional

import torch
from torch import Tensor

from optimizers.third_party_loader import ensure_third_party_paths

ensure_third_party_paths()

from psgd import KronNewton  # noqa: E402  (third_party/psgd_torch)


class PSGD(torch.optim.Optimizer):
    """
    Thin ``torch.optim.Optimizer`` adapter around ``psgd.KronNewton``.

    Uses all trainable parameters (official PSGD handles tensor reshaping internally).
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 0.1,
        precond_lr: float = 0.1,
        T1: int = 1,
        weight_decay: float = 0.0,
        preconditioner_max_size: float = float("inf"),
        preconditioner_max_skew: float = 1.0,
    ):
        params = [p for p in params if p.requires_grad]
        if not params:
            raise ValueError("PSGD requires at least one trainable parameter.")
        n_params = sum(p.numel() for p in params)
        grad_clip = 0.1 * math.sqrt(n_params)
        update_prob = min(1.0, 1.0 / max(T1, 1))

        defaults = dict(lr=lr, precond_lr=precond_lr, T1=T1, weight_decay=weight_decay)
        # Register params so ``torch.optim.Optimizer`` state dict works; updates use KronNewton.
        super().__init__(params, defaults)

        self._kron = KronNewton(
            params,
            lr_params=lr,
            lr_preconditioner=precond_lr,
            preconditioner_update_probability=update_prob,
            exact_hessian_vector_product=True,
            grad_clip_max_norm=grad_clip,
            preconditioner_init_scale=1.0,
            preconditioner_max_size=preconditioner_max_size,
            preconditioner_max_skew=preconditioner_max_skew,
            damping=1e-9,
            betaL=0.9,
            momentum=0.0,
        )
        self._weight_decay = weight_decay

    @property
    def needs_closure(self) -> bool:
        return True

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self._kron._params_with_grad:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    def step(self, closure: Optional[Callable[[], Tensor]] = None) -> Optional[Tensor]:
        if closure is None:
            raise ValueError("PSGD (KronNewton) requires a closure returning the loss.")

        if self._weight_decay > 0:

            def wrapped_closure() -> Tensor:
                loss = closure()
                if self._weight_decay > 0:
                    wd = sum(
                        (p * p).sum() for p in self._kron._params_with_grad
                    )
                    return loss + 0.5 * self._weight_decay * wd
                return loss

            return self._kron.step(wrapped_closure)
        return self._kron.step(closure)
