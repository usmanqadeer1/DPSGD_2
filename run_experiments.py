#!/usr/bin/env python3
"""
Run DPSGD benchmark experiments.

Each task compares all configured optimizers and saves:
  - Per-iteration train loss/accuracy + wall-clock time
  - Per-epoch test loss/accuracy + wall-clock time
  - Matplotlib comparison plots (PNG)

Example:
  python run_experiments.py --task curves
  python run_experiments.py --task mnist --optimizers dpsgd psgd
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from configs import EXPERIMENTS, TASK_OPTIMIZERS
from datasets import get_data
from models import build_model
from optimizers.third_party_loader import ensure_third_party_paths

ensure_third_party_paths()  # auto-clone third_party/ on Kaggle / fresh clones

from optimizers.factory import build_optimizer
from trainer import train
from utils import get_device, save_comparison_plots, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DPSGD optimizer benchmarks")
    p.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment: curves | mnist | fashionmnist | addition",
    )
    p.add_argument(
        "--optimizers",
        nargs="+",
        default=None,
        help="Optimizers to run (default: all optimizers for this task)",
    )
    p.add_argument("--seed", type=int, default=1, help="Random seed (shared across optimizers)")
    p.add_argument("--data-dir", type=str, default="./data", help="Dataset cache directory")
    p.add_argument("--results-dir", type=str, default=None, help="Output directory for plots/metrics")
    p.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
    p.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if omitted)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    exp = EXPERIMENTS[args.task]
    optimizers = args.optimizers or TASK_OPTIMIZERS[args.task]
    results_dir = args.results_dir or os.path.join("results", args.task)
    os.makedirs(results_dir, exist_ok=True)

    device = get_device() if args.device is None else __import__("torch").device(args.device)
    print(f"Task: {args.task} | Device: {device} | Seed: {args.seed}")
    print(f"Optimizers ({len(optimizers)}): {optimizers}")

    data_bundle = get_data(
        exp.task,
        args.data_dir,
        exp.batch_size,
        exp.test_batch_size,
        num_workers=args.num_workers,
    )

    all_metrics = {}

    for opt_name in optimizers:
        print(f"\n{'=' * 60}\nTraining with {opt_name}\n{'=' * 60}")
        set_seed(args.seed)
        model = build_model(exp.task, exp.model_kwargs)
        opt_bundle = build_optimizer(opt_name, model, exp)
        metrics = train(model, data_bundle, opt_bundle, exp.epochs, device)
        all_metrics[opt_name] = metrics

        out_json = os.path.join(results_dir, f"metrics_{opt_name}.json")
        metrics.save_json(out_json)
        print(f"Saved metrics to {out_json}")

    has_acc = exp.task in ("mnist", "fashionmnist")
    plot_paths = save_comparison_plots(all_metrics, results_dir, has_accuracy=has_acc)
    print(f"\nPlots saved to {results_dir}:")
    for p in plot_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
