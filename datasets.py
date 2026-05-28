"""Dataset loaders for all four benchmarks."""

from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np
import scipy.io
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


CURVES_MAT_URL = "http://www.cs.toronto.edu/~jmartens/digs3pts_1.mat"


@dataclass
class DataBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    task: str
    n_train: int
    n_test: int


def _make_loader(
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    pin = torch.cuda.is_available()
    return DataLoader(
        TensorDataset(x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin,
    )


def download_curves_mat(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "digs3pts_1.mat")
    if not os.path.isfile(path):
        print(f"Downloading CURVES dataset to {path} ...")
        urllib.request.urlretrieve(CURVES_MAT_URL, path)
    return path


def load_curves(
    data_dir: str,
    batch_size: int = 64,
    test_batch_size: int = 1000,
    max_train: int = 30000,
    num_workers: int = 0,
) -> DataBundle:
    """Load CURVES autoencoder data from digs3pts_1.mat (ported from notebook)."""
    mat_path = download_curves_mat(data_dir)
    mat = scipy.io.loadmat(mat_path)
    x_train = torch.tensor(mat["bdata"], dtype=torch.float32)
    x_test = torch.tensor(mat["bdatatest"], dtype=torch.float32)

    if max_train and x_train.shape[0] > max_train:
        x_train = x_train[:max_train]

    train_loader = _make_loader(x_train, x_train, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = _make_loader(x_test, x_test, test_batch_size, shuffle=False, num_workers=num_workers)
    return DataBundle(train_loader, test_loader, "curves", len(x_train), len(x_test))


def load_mnist(
    data_dir: str,
    batch_size: int = 64,
    test_batch_size: int = 1000,
    fashion: bool = False,
    num_workers: int = 0,
) -> DataBundle:
    """MNIST / FashionMNIST with standard 60k/10k split."""
    os.makedirs(data_dir, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    cls = datasets.FashionMNIST if fashion else datasets.MNIST
    root = os.path.join(data_dir, "fashionmnist" if fashion else "mnist")
    train_ds = cls(root=root, train=True, download=True, transform=tfm)
    test_ds = cls(root=root, train=False, download=True, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    task = "fashionmnist" if fashion else "mnist"
    return DataBundle(train_loader, test_loader, task, len(train_ds), len(test_ds))


def get_synthetic_addition_dataset(
    batch_size: int = 50000,
    seq_len0: int = 30,
    dim_in: int = 2,
    dim_out: int = 1,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Synthetic addition task data generation (ported from LSTM_XOR.ipynb).
    Returns x: [N, seq_len, dim_in], y: [N, dim_out].
    """
    if seed is not None:
        rng = np.random.RandomState(seed)
        torch.manual_seed(seed)
    else:
        rng = np.random

    seq_len = round(seq_len0 + 0.1 * rng.rand() * seq_len0)
    x = torch.zeros(batch_size, seq_len, dim_in)
    y = torch.zeros(batch_size, dim_out)

    for i in range(batch_size):
        x[i, :, 0] = 2.0 * torch.rand(seq_len) - 1.0
        while True:
            i1, i2 = list(np.floor(rng.rand(2) * seq_len / 2).astype(int))
            if i1 != i2:
                break
        x[i, i1, 1] = 1.0
        x[i, i2, 1] = 1.0
        y[i] = 0.5 * (x[i, i1, 0] + x[i, i2, 0])
    return x, y


def load_synthetic_addition(
    data_dir: str,
    batch_size: int = 100,
    test_batch_size: int = 100,
    n_samples: int = 50000,
    test_fraction: float = 0.1,
    seed: int = 42,
    num_workers: int = 0,
) -> DataBundle:
    """50k synthetic addition sequences with train/test split."""
    os.makedirs(data_dir, exist_ok=True)
    x, y = get_synthetic_addition_dataset(batch_size=n_samples, seed=seed)
    x_train, x_test, y_train, y_test = train_test_split(
        x.numpy(),
        y.numpy(),
        test_size=test_fraction,
        random_state=seed,
    )
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    train_loader = _make_loader(x_train, y_train, batch_size, shuffle=True, num_workers=num_workers)
    test_loader = _make_loader(x_test, y_test, test_batch_size, shuffle=False, num_workers=num_workers)
    return DataBundle(train_loader, test_loader, "addition", len(x_train), len(x_test))


def get_data(
    task: str,
    data_dir: str,
    batch_size: int,
    test_batch_size: int,
    num_workers: int = 0,
) -> DataBundle:
    if task == "curves":
        return load_curves(data_dir, batch_size, test_batch_size, num_workers=num_workers)
    if task == "mnist":
        return load_mnist(data_dir, batch_size, test_batch_size, fashion=False, num_workers=num_workers)
    if task == "fashionmnist":
        return load_mnist(data_dir, batch_size, test_batch_size, fashion=True, num_workers=num_workers)
    if task == "addition":
        return load_synthetic_addition(data_dir, batch_size, test_batch_size, num_workers=num_workers)
    raise ValueError(f"Unknown task: {task}")


def get_loss_fn(task: str) -> Callable:
    if task in ("curves", "addition"):
        return torch.nn.MSELoss()
    return torch.nn.CrossEntropyLoss()
