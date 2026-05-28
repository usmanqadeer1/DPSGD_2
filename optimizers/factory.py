"""Factory for building optimizers with unified configuration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from configs import DPSGDConfig, ExperimentConfig
from models import matrix_params_for_kron
from optimizers.dpsgd import DPSGD
from optimizers.kfac_wrapper import KFACBundle, build_kfac
from optimizers.kradagrad_official import build_kradagrad
from optimizers.psgd_official import PSGD
from optimizers.shampoo_daniil import build_shampoo
from optimizers.soap_official import build_soap


@dataclass
class OptimizerBundle:
    """Optimizer plus metadata for trainer."""

    name: str
    optimizer: Any
    needs_closure: bool = False
    is_kfac: bool = False
    aux_optimizer: Optional[torch.optim.Optimizer] = None


def _split_matrix_bias(model: nn.Module) -> Tuple[List[torch.nn.Parameter], List[torch.nn.Parameter]]:
    """Split for DPSGD (matrix Kronecker + SGD on biases)."""
    matrix = matrix_params_for_kron(model)
    matrix_ids = {id(p) for p in matrix}
    other = [p for p in model.parameters() if id(p) not in matrix_ids]
    return matrix, other


def build_optimizer(
    name: str,
    model: nn.Module,
    experiment: ExperimentConfig,
) -> OptimizerBundle:
    name = name.lower()
    cfg = experiment.baseline_lrs.get(name, {})
    dpsgd_cfg: DPSGDConfig = experiment.dpsgd
    all_params = list(model.parameters())

    if name == "sgd":
        opt = torch.optim.SGD(all_params, lr=cfg.get("lr", 0.1))
        return OptimizerBundle("sgd", opt)

    if name == "adam":
        opt = torch.optim.Adam(
            all_params,
            lr=cfg.get("lr", 0.001),
            betas=cfg.get("betas", (0.9, 0.999)),
        )
        return OptimizerBundle("adam", opt)

    if name == "dpsgd":
        matrix_params, other_params = _split_matrix_bias(model)
        opt = DPSGD(
            matrix_params,
            lr=dpsgd_cfg.alpha,
            T1=dpsgd_cfg.T1,
            T2=dpsgd_cfg.T2,
            beta=dpsgd_cfg.beta,
            tau0=dpsgd_cfg.tau0,
            eta=dpsgd_cfg.eta,
            nu_scale=dpsgd_cfg.nu_scale,
            factor_absorbed=dpsgd_cfg.factor_absorbed,
            precond_lr=cfg.get("precond_lr", 0.1),
        )
        aux = (
            torch.optim.SGD(other_params, lr=dpsgd_cfg.alpha)
            if other_params
            else None
        )
        return OptimizerBundle("dpsgd", opt, needs_closure=True, aux_optimizer=aux)

    if name == "psgd":
        # Official lixilinx/psgd_torch KronNewton — all trainable params
        opt = PSGD(
            all_params,
            lr=cfg.get("lr", 0.1),
            precond_lr=cfg.get("precond_lr", 0.1),
            T1=cfg.get("T1", 1),
            preconditioner_max_size=cfg.get("preconditioner_max_size", float("inf")),
            preconditioner_max_skew=cfg.get("preconditioner_max_skew", 1.0),
        )
        return OptimizerBundle("psgd", opt, needs_closure=True)

    if name == "shampoo":
        opt = build_shampoo(
            all_params,
            lr=cfg.get("lr", 0.5),
            limit=cfg.get("limit", 200),
            alpha=cfg.get("alpha", 0.5),
            epsilon=cfg.get("epsilon", 1e-4),
            weight_decay=cfg.get("weight_decay", 0.0),
        )
        return OptimizerBundle("shampoo", opt)

    if name == "kfac":
        bundle = build_kfac(
            model,
            lr=cfg.get("lr", 0.5),
            damping=cfg.get("damping", 0.001),
            momentum=cfg.get("momentum", 0.05),
            factor_update_steps=cfg.get("factor_update_steps", 1),
            inv_update_steps=cfg.get("inv_update_steps", 1),
        )
        return OptimizerBundle("kfac", bundle, is_kfac=True)

    if name == "soap":
        opt = build_soap(
            all_params,
            lr=cfg.get("lr", 0.003),
            betas=tuple(cfg.get("betas", (0.95, 0.95))),
            weight_decay=cfg.get("weight_decay", 0.0),
            precondition_frequency=cfg.get("precondition_frequency", 10),
            max_precond_dim=cfg.get("max_precond_dim", 10000),
        )
        return OptimizerBundle("soap", opt)

    if name == "kradagrad":
        opt = build_kradagrad(
            all_params,
            lr=cfg.get("lr", 0.1),
            momentum=cfg.get("momentum", 0.9),
            weight_decay=cfg.get("weight_decay", 0.0),
            matrix_eps=cfg.get("matrix_eps", 1e-4),
            preconditioning_compute_steps=cfg.get("preconditioning_compute_steps", 20),
            statistics_compute_steps=cfg.get("statistics_compute_steps", 1),
            block_size=cfg.get("block_size", 128),
        )
        return OptimizerBundle("kradagrad", opt)

    raise ValueError(f"Unknown optimizer: {name}")
