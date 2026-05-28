"""Generalized training loop with per-iteration train and per-epoch test metrics."""

from __future__ import annotations

import time
from typing import Optional, Tuple

import torch
import torch.nn as nn

from datasets import DataBundle, get_loss_fn
from optimizers.factory import OptimizerBundle
from utils import MetricsTracker, get_device


def _compute_batch_loss(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    task: str,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[float]]:
    data = data.to(device)
    target = target.to(device)

    if task == "curves":
        data = data.view(data.size(0), -1)
        target = target.view(target.size(0), -1)
        out = model(data)
        loss = criterion(out, target)
        return loss, None

    if task == "addition":
        out = model(data)
        loss = criterion(out, target)
        return loss, None

    out = model(data)
    loss = criterion(out, target)
    pred = out.argmax(dim=1)
    acc = (pred == target).float().mean().item()
    return loss, acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_bundle: DataBundle,
    task: str,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, Optional[float]]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    for data, target in data_bundle.test_loader:
        loss, acc = _compute_batch_loss(model, data, target, task, criterion, device)
        total_loss += loss.item()
        if acc is not None:
            total_acc += acc
        n_batches += 1
    avg_loss = total_loss / max(n_batches, 1)
    avg_acc = (total_acc / n_batches) if task in ("mnist", "fashionmnist") else None
    return avg_loss, avg_acc


def train(
    model: nn.Module,
    data_bundle: DataBundle,
    opt_bundle: OptimizerBundle,
    epochs: int,
    device: Optional[torch.device] = None,
) -> MetricsTracker:
    device = device or get_device()
    model = model.to(device)
    task = data_bundle.task
    criterion = get_loss_fn(task)
    tracker = MetricsTracker(opt_bundle.name)
    has_acc = task in ("mnist", "fashionmnist")

    t_start = time.time()
    global_iter = 0

    # Epoch 0 / iteration 0: baseline before any weight updates
    model.eval()
    test_loss, test_acc = evaluate(model, data_bundle, task, criterion, device)
    tracker.record_test_epoch(0, test_loss, time.time() - t_start, test_acc)

    for epoch in range(epochs):
        model.train()

        for data, target in data_bundle.train_loader:
            if opt_bundle.is_kfac:
                batch_loss, batch_acc = _train_step_kfac(
                    model, data, target, task, criterion, device, opt_bundle
                )
            elif opt_bundle.needs_closure:
                batch_loss, batch_acc = _train_step_closure(
                    model, data, target, task, criterion, device, opt_bundle
                )
            else:
                batch_loss, batch_acc = _train_step_first_order(
                    model, data, target, task, criterion, device, opt_bundle
                )

            tracker.record_train_step(
                global_iter,
                epoch + 1,
                batch_loss,
                time.time() - t_start,
                batch_acc if has_acc else None,
            )
            global_iter += 1

        model.eval()
        test_loss, test_acc = evaluate(model, data_bundle, task, criterion, device)
        tracker.record_test_epoch(epoch + 1, test_loss, time.time() - t_start, test_acc)

        last_train = tracker.train_loss[-1] if tracker.train_loss else float("nan")
        msg = (
            f"[{opt_bundle.name}] epoch {epoch + 1}/{epochs} "
            f"last_train_loss={last_train:.6f} test_loss={test_loss:.6f}"
        )
        if test_acc is not None:
            msg += f" test_acc={test_acc:.4f}"
        msg += f" time={tracker.test_time[-1]:.1f}s"
        print(msg)

    return tracker


def _train_step_first_order(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    task: str,
    criterion: nn.Module,
    device: torch.device,
    opt_bundle: OptimizerBundle,
) -> Tuple[float, Optional[float]]:
    opt = opt_bundle.optimizer
    opt.zero_grad()
    loss, acc = _compute_batch_loss(model, data, target, task, criterion, device)
    loss.backward()
    opt.step()
    return loss.item(), acc


def _train_step_kfac(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    task: str,
    criterion: nn.Module,
    device: torch.device,
    opt_bundle: OptimizerBundle,
) -> Tuple[float, Optional[float]]:
    bundle = opt_bundle.optimizer
    bundle.zero_grad()
    loss, acc = _compute_batch_loss(model, data, target, task, criterion, device)
    loss.backward()
    bundle.step()
    return loss.item(), acc


def _train_step_closure(
    model: nn.Module,
    data: torch.Tensor,
    target: torch.Tensor,
    task: str,
    criterion: nn.Module,
    device: torch.device,
    opt_bundle: OptimizerBundle,
) -> Tuple[float, Optional[float]]:
    opt = opt_bundle.optimizer
    aux = opt_bundle.aux_optimizer
    last_loss: float = 0.0
    last_acc: Optional[float] = None

    def closure() -> torch.Tensor:
        nonlocal last_loss, last_acc
        opt.zero_grad(set_to_none=True)
        if aux is not None:
            aux.zero_grad(set_to_none=True)
        loss, acc = _compute_batch_loss(model, data, target, task, criterion, device)
        last_loss = loss.item()
        last_acc = acc
        return loss

    opt.step(closure)

    if aux is not None:
        aux.zero_grad(set_to_none=True)
        loss_b, acc_b = _compute_batch_loss(model, data, target, task, criterion, device)
        loss_b.backward()
        aux.step()
        if acc_b is not None:
            last_acc = acc_b

    return last_loss, last_acc
