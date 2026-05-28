"""
SOAP via official ``nikhilvyas/SOAP`` (``soap.py``).

Repository defaults (see SOAP README):
  lr=3e-3, betas=(0.95, 0.95), weight_decay=0.01, precondition_frequency=10

For these benchmarks we set ``lr`` from DPSGD paper Table-1 α (tuned per task) and
``weight_decay=0`` to match the other baselines. Other SOAP defaults are unchanged.
"""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import Tensor

from optimizers.third_party_loader import ensure_third_party_paths

ensure_third_party_paths()

from soap import SOAP  # noqa: E402  (third_party/SOAP/soap.py)


def build_soap(
    params: Iterable[Tensor],
    lr: float = 0.003,
    betas: Tuple[float, float] = (0.95, 0.95),
    weight_decay: float = 0.0,
    precondition_frequency: int = 10,
    max_precond_dim: int = 10000,
) -> SOAP:
    return SOAP(
        params,
        lr=lr,
        betas=betas,
        shampoo_beta=-1,
        eps=1e-8,
        weight_decay=weight_decay,
        precondition_frequency=precondition_frequency,
        max_precond_dim=max_precond_dim,
        merge_dims=False,
        precondition_1d=False,
        normalize_grads=False,
        data_format="channels_first",
        correct_bias=True,
    )
