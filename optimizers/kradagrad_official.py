"""
KrADagrad via official ``jonathanmei/kradagrad`` (``KradagradMM``).

Uses ``KradagradMM`` (KrADagrad in the paper) which does not depend on the missing
``batched_matrix_functions`` module required by ``KradagradPP``.

Default hyperparameters follow ``experiments/deepobs_single.py``:
  matrix_eps=1e-4, preconditioning_compute_steps=20, lr tuned per task (Table-1 style).
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor

from optimizers.third_party_loader import ensure_third_party_paths

ensure_third_party_paths()

from kradagrad.kradagradmm import KradagradMM  # noqa: E402
from kradagrad.third_party.shampoo.shampoo import ShampooHyperParams  # noqa: E402


def build_kradagrad(
    params: Iterable[Tensor],
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    matrix_eps: float = 1e-4,
    preconditioning_compute_steps: int = 20,
    statistics_compute_steps: int = 1,
    block_size: int = 128,
) -> KradagradMM:
    hps = ShampooHyperParams(
        beta2=1.0,
        matrix_eps=matrix_eps,
        weight_decay=weight_decay,
        preconditioning_compute_steps=preconditioning_compute_steps,
        statistics_compute_steps=statistics_compute_steps,
        block_size=block_size,
        best_effort_shape_interpretation=True,
        nesterov=True,
    )
    return KradagradMM(
        params,
        lr=lr,
        momentum=momentum,
        hyperparams=hps,
    )
