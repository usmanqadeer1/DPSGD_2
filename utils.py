"""Utilities: seeding, metrics tracking, and matplotlib plotting."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

PLOT_COLORS = plt.cm.tab10.colors  # type: ignore[attr-defined]


def set_seed(seed: int) -> None:
    """Fix seeds for reproducible initialization and batch order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def cuda_kernels_work() -> bool:
    """True if CUDA is available and a simple GPU op runs (catches sm_60 P100 vs new PyTorch)."""
    if not torch.cuda.is_available():
        return False
    try:
        x = torch.ones(1, device="cuda")
        (x + 1).item()
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """
    Pick training device. Falls back to CPU when CUDA is unavailable or kernels
    cannot run (e.g. Tesla P100 sm_60 with PyTorch builds that require sm_70+).
    """
    if requested is not None:
        req = requested.strip().lower()
        if req == "cpu":
            return torch.device("cpu")
        if req in ("cuda", "gpu"):
            if cuda_kernels_work():
                return torch.device("cuda")
            name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
            cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
            print(
                f"WARNING: --device cuda requested but GPU kernels failed on {name} "
                f"(capability {cap}). Falling back to CPU.\n"
                "  Kaggle P100 (sm_60): use Settings → Accelerator → GPU T4, or run with --device cpu.\n"
                "  Or install an older PyTorch build that supports your GPU: https://pytorch.org"
            )
            return torch.device("cpu")
        return torch.device(requested)

    if cuda_kernels_work():
        return torch.device("cuda")
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        print(
            f"WARNING: CUDA device {name} (capability {cap}) is visible but PyTorch "
            "cannot run kernels on it; using CPU.\n"
            "  On Kaggle: switch to a T4 GPU (sm_75+) or pass --device cpu."
        )
    return torch.device("cpu")


def get_device() -> torch.device:
    """Auto-select cuda if usable, else cpu."""
    return resolve_device(None)


@dataclass
class MetricsTracker:
    """
    Training metrics per iteration (batch step); test metrics per epoch.

    Wall-clock ``time`` is cumulative seconds from training start.
    """

    optimizer_name: str
    # Training (per optimizer step / batch)
    train_iteration: List[int] = field(default_factory=list)
    train_epoch: List[int] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    train_time: List[float] = field(default_factory=list)
    # Test (end of each epoch)
    test_epoch: List[int] = field(default_factory=list)
    test_loss: List[float] = field(default_factory=list)
    test_acc: List[float] = field(default_factory=list)
    test_time: List[float] = field(default_factory=list)

    def record_train_step(
        self,
        iteration: int,
        epoch: int,
        loss: float,
        elapsed_sec: float,
        acc: Optional[float] = None,
    ) -> None:
        self.train_iteration.append(iteration)
        self.train_epoch.append(epoch)
        self.train_loss.append(loss)
        self.train_time.append(elapsed_sec)
        if acc is not None:
            self.train_acc.append(acc)

    def record_test_epoch(
        self,
        epoch: int,
        loss: float,
        elapsed_sec: float,
        acc: Optional[float] = None,
    ) -> None:
        self.test_epoch.append(epoch)
        self.test_loss.append(loss)
        self.test_time.append(elapsed_sec)
        if acc is not None:
            self.test_acc.append(acc)

    def to_dict(self) -> Dict:
        d: Dict = {
            "optimizer": self.optimizer_name,
            "train_iteration": self.train_iteration,
            "train_epoch": self.train_epoch,
            "train_loss": self.train_loss,
            "train_time": self.train_time,
            "test_epoch": self.test_epoch,
            "test_loss": self.test_loss,
            "test_time": self.test_time,
        }
        if self.train_acc:
            d["train_acc"] = self.train_acc
        if self.test_acc:
            d["test_acc"] = self.test_acc
        return d

    def save_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _plot_series(
    series: Dict[str, Dict[str, Sequence[float]]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: str,
    log_y: bool = False,
    ylim: Optional[tuple] = None,
) -> str:
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, data) in enumerate(series.items()):
        x = data.get(x_key, [])
        y = data.get(y_key, [])
        if not x or not y:
            continue
        ax.plot(x, y, label=name, color=PLOT_COLORS[i % len(PLOT_COLORS)], linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_comparison_plots(
    all_metrics: Dict[str, MetricsTracker],
    results_dir: str,
    has_accuracy: bool = False,
) -> List[str]:
    """Generate matplotlib comparison plots (all optimizers on each figure)."""
    paths: List[str] = []
    opts = list(all_metrics.keys())

    train_loss = {
        k: {
            "x": v.train_iteration,
            "y": v.train_loss,
        }
        for k, v in all_metrics.items()
    }
    train_loss_time = {
        k: {"x": v.train_time, "y": v.train_loss} for k, v in all_metrics.items()
    }
    test_loss_epoch = {
        k: {"x": v.test_epoch, "y": v.test_loss} for k, v in all_metrics.items()
    }
    test_loss_time = {
        k: {"x": v.test_time, "y": v.test_loss} for k, v in all_metrics.items()
    }

    paths.append(
        _plot_series(
            train_loss,
            "x",
            "y",
            "Train Loss vs Iteration",
            "Iteration",
            "Train Loss",
            os.path.join(results_dir, "train_loss_vs_iteration.png"),
            log_y=True,
        )
    )
    paths.append(
        _plot_series(
            train_loss_time,
            "x",
            "y",
            "Train Loss vs Wall-Clock Time",
            "Time (s)",
            "Train Loss",
            os.path.join(results_dir, "train_loss_vs_time.png"),
            log_y=True,
        )
    )
    paths.append(
        _plot_series(
            test_loss_epoch,
            "x",
            "y",
            "Test Loss vs Epoch",
            "Epoch",
            "Test Loss",
            os.path.join(results_dir, "test_loss_vs_epoch.png"),
            log_y=True,
        )
    )
    paths.append(
        _plot_series(
            test_loss_time,
            "x",
            "y",
            "Test Loss vs Wall-Clock Time",
            "Time (s)",
            "Test Loss",
            os.path.join(results_dir, "test_loss_vs_time.png"),
            log_y=True,
        )
    )

    if has_accuracy:
        train_acc = {
            k: {"x": v.train_iteration, "y": v.train_acc}
            for k, v in all_metrics.items()
            if v.train_acc
        }
        train_acc_time = {
            k: {"x": v.train_time, "y": v.train_acc}
            for k, v in all_metrics.items()
            if v.train_acc
        }
        test_acc = {
            k: {"x": v.test_epoch, "y": v.test_acc}
            for k, v in all_metrics.items()
            if v.test_acc
        }
        test_acc_time = {
            k: {"x": v.test_time, "y": v.test_acc}
            for k, v in all_metrics.items()
            if v.test_acc
        }
        if train_acc:
            paths.append(
                _plot_series(
                    train_acc,
                    "x",
                    "y",
                    "Train Accuracy vs Iteration",
                    "Iteration",
                    "Train Accuracy",
                    os.path.join(results_dir, "train_accuracy_vs_iteration.png"),
                )
            )
            paths.append(
                _plot_series(
                    train_acc_time,
                    "x",
                    "y",
                    "Train Accuracy vs Wall-Clock Time",
                    "Time (s)",
                    "Train Accuracy",
                    os.path.join(results_dir, "train_accuracy_vs_time.png"),
                )
            )
        if test_acc:
            paths.append(
                _plot_series(
                    test_acc,
                    "x",
                    "y",
                    "Test Accuracy vs Epoch",
                    "Epoch",
                    "Test Accuracy",
                    os.path.join(results_dir, "test_accuracy_vs_epoch.png"),
                    ylim=(0.0, 1.0),
                )
            )
            paths.append(
                _plot_series(
                    test_acc_time,
                    "x",
                    "y",
                    "Test Accuracy vs Wall-Clock Time",
                    "Time (s)",
                    "Test Accuracy",
                    os.path.join(results_dir, "test_accuracy_vs_time.png"),
                    ylim=(0.97, 1.0),
                )
            )

    return paths
