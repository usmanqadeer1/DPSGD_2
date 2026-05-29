"""Experiment and optimizer hyperparameter configuration (DPSGD paper Table 1 + repo defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DPSGDConfig:
    alpha: float = 0.1
    T1: int = 1
    T2: int = 5
    beta: float = 0.7
    factor_absorbed: bool = True
    tau0: float = 1.0
    eta: float = 1e-5
    nu_scale: float = 0.1  # nu = nu_scale * sqrt(N)


@dataclass
class ExperimentConfig:
    name: str
    task: str
    epochs: int
    batch_size: int
    test_batch_size: int
    lr_metric: str  # "loss" or "error"
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    dpsgd: DPSGDConfig = field(default_factory=DPSGDConfig)
    baseline_lrs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    seed: int = 1


# Table 1 (DPSGD paper) + official repo defaults for SOAP / KrADagrad
EXPERIMENTS: Dict[str, ExperimentConfig] = {
    "curves": ExperimentConfig(
        name="curves",
        task="curves",
        epochs=20,
        batch_size=64,
        test_batch_size=1000,
        lr_metric="loss",
        model_kwargs={"dims": [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]},
        dpsgd=DPSGDConfig(
            alpha=0.1,
            T1=1,
            T2=5,
            beta=0.7,
            factor_absorbed=True,
        ),
        baseline_lrs={
            "sgd": {"lr": 0.1},
            "adam": {"lr": 0.001},
            "kfac": {
                "lr": 0.5,
                "momentum": 0.05,
                "damping": 0.001,
            },
            "shampoo": {
                "lr": 0.5,
                "limit": 200,
                "alpha": 0.5,
                "epsilon": 1e-4,
            },
            "psgd": {
                "lr": 0.1,
                "precond_lr": 0.1,
                "T1": 1,
            },
            "soap": {
                "lr": 0.1,
                "precondition_frequency": 10,
                "betas": (0.95, 0.95),
            },
            "kradagrad": {
                "lr": 0.1,
                "matrix_eps": 1e-4,
                "preconditioning_compute_steps": 20,
            },
        },
    ),
    "mnist": ExperimentConfig(
        name="mnist",
        task="mnist",
        epochs=20,
        batch_size=64,
        test_batch_size=1000,
        lr_metric="loss",
        model_kwargs={},
        dpsgd=DPSGDConfig(
            alpha=0.001,
            T1=1,
            T2=5,
            beta=0.9,
            factor_absorbed=True,
        ),
        baseline_lrs={
            "sgd": {"lr": 0.1},
            "adam": {"lr": 0.005},
            "kfac": {
                "lr": 0.01,
                "momentum": 0.05,
                "damping": 0.001,
            },
            "shampoo": {
                "lr": 0.5,
                "limit": 200,
                "alpha": 0.5,
                "epsilon": 1e-4,
            },
            "psgd": {
                "lr": 0.05,
                "precond_lr": 0.1,
                "T1": 1,
            },
            "soap": {
                "lr": 0.05,
                "precondition_frequency": 10,
                "betas": (0.95, 0.95),
            },
            "kradagrad": {
                "lr": 0.05,
                "matrix_eps": 1e-4,
                "preconditioning_compute_steps": 20,
            },
        },
    ),
    "fashionmnist": ExperimentConfig(
        name="fashionmnist",
        task="fashionmnist",
        epochs=20,
        batch_size=64,
        test_batch_size=1000,
        lr_metric="loss",
        model_kwargs={},
        dpsgd=DPSGDConfig(
            alpha=0.1,
            T1=1,
            T2=5,
            beta=0.7,
            factor_absorbed=False,
        ),
        baseline_lrs={
            "sgd": {"lr": 0.1},
            "adam": {"lr": 0.005},
            "kfac": {
                "lr": 0.01,
                "momentum": 0.05,
                "damping": 0.001,
            },
            "shampoo": {
                "lr": 0.5,
                "limit": 200,
                "alpha": 0.5,
                "epsilon": 1e-4,
            },
            "psgd": {
                "lr": 0.1,
                "precond_lr": 0.1,
                "T1": 1,
            },
            "soap": {
                "lr": 0.1,
                "precondition_frequency": 10,
                "betas": (0.95, 0.95),
            },
            "kradagrad": {
                "lr": 0.1,
                "matrix_eps": 1e-4,
                "preconditioning_compute_steps": 20,
            },
        },
    ),
    "addition": ExperimentConfig(
        name="addition",
        task="addition",
        epochs=10,
        batch_size=100,
        test_batch_size=100,
        lr_metric="loss",
        model_kwargs={"input_dim": 2, "hidden_dim": 20, "output_dim": 1},
        dpsgd=DPSGDConfig(
            alpha=0.1,
            T1=1,
            T2=5,
            beta=0.7,
            factor_absorbed=True,
        ),
        baseline_lrs={
            "sgd": {"lr": 0.1},
            "adam": {"lr": 0.001},
            "kfac": {
                "lr": 0.5,
                "momentum": 0.05,
                "damping": 0.001,
            },
            "shampoo": {
                "lr": 0.1,
                "limit": 10,
                "alpha": 0.5,
                "epsilon": 1e-4,
            },
            "psgd": {
                "lr": 0.1,
                "precond_lr": 0.1,
                "T1": 1,
            },
            "soap": {
                "lr": 0.1,
                "precondition_frequency": 10,
                "betas": (0.95, 0.95),
            },
            "kradagrad": {
                "lr": 0.1,
                "matrix_eps": 1e-4,
                "preconditioning_compute_steps": 20,
            },
        },
    ),
}

DEFAULT_OPTIMIZERS: List[str] = [
    "sgd",
    "adam",
    "psgd",
    "dpsgd",
    "kfac",
    "shampoo",
    "soap",
    "kradagrad",
]

TASK_OPTIMIZERS: Dict[str, List[str]] = {name: list(DEFAULT_OPTIMIZERS) for name in EXPERIMENTS}
