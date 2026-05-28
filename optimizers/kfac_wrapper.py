"""
K-FAC via ``gpauloski/kfac-pytorch`` (``KFACPreconditioner`` + SGD).

Paper Table 1 mapping:
  α  -> SGD learning rate
  β  -> Polyak averaging on Kronecker factors → ``factor_decay = 1 - β``
  γ  -> Tikhonov damping on ``KFACPreconditioner``

Training loop (from upstream README):
  loss.backward() → preconditioner.step() → optimizer.step()
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class KFACBundle:
    """SGD + KFACPreconditioner with unified ``zero_grad`` / ``step``."""

    def __init__(self, optimizer: torch.optim.Optimizer, preconditioner: Any):
        self.optimizer = optimizer
        self.preconditioner = preconditioner

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.preconditioner.step()
        self.optimizer.step()


def build_kfac(
    model: nn.Module,
    lr: float = 0.5,
    damping: float = 0.001,
    momentum: float = 0.05,
    weight_decay: float = 0.0,
    factor_update_steps: int = 1,
    inv_update_steps: int = 1,
) -> KFACBundle:
    """
    Build K-FAC from gpauloski/kfac-pytorch (required).

    ``momentum`` is Table-1 β (Polyak coefficient); factor EMA uses ``1 - β``.
    """
    try:
        from kfac.preconditioner import KFACPreconditioner
    except ImportError as exc:
        raise ImportError(
            "Install official K-FAC: pip install git+https://github.com/gpauloski/kfac-pytorch.git"
        ) from exc

    factor_decay = 1.0 - momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.0,
        weight_decay=0.0,
    )
    preconditioner = KFACPreconditioner(
        model,
        damping=damping,
        factor_decay=factor_decay,
        factor_update_steps=factor_update_steps,
        inv_update_steps=inv_update_steps,
        lr=lr,
        kl_clip=0.001,
    )
    if weight_decay > 0:
        # Decoupled WD: add L2 to loss in trainer if needed; SGD WD off by default.
        pass
    return KFACBundle(optimizer, preconditioner)
