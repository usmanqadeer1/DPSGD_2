"""Model architectures for DPSGD benchmarks."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class CurvesAutoencoder(nn.Module):
    """12-layer FCN autoencoder (784-400-...-784), ReLU hidden, linear output."""

    DEFAULT_DIMS = [784, 400, 200, 100, 50, 25, 6, 25, 50, 100, 200, 400, 784]

    def __init__(self, dims: Sequence[int] | None = None):
        super().__init__()
        dims = list(dims or self.DEFAULT_DIMS)
        layers: List[nn.Linear] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
        self.layers = nn.ModuleList(layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for layer in self.layers:
            w = 0.1 * torch.randn(layer.weight.shape)
            nn.init.xavier_uniform_(w)
            with torch.no_grad():
                layer.weight.copy_(w)
                layer.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
        return self.layers[-1](x)


class LeNet5(nn.Module):
    """LeNet-5 variant for 28x28 inputs with ReLU (ported from PSGD_MNIST.ipynb)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                w = 0.1 * torch.randn_like(m.weight)
                nn.init.xavier_uniform_(w)
                with torch.no_grad():
                    m.weight.copy_(w)
                    if m.bias is not None:
                        m.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SimpleRNN(nn.Module):
    """Elman RNN for synthetic addition (hidden dim 20, ported from LSTM_XOR setup)."""

    def __init__(self, input_dim: int = 2, hidden_dim: int = 20, output_dim: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.normal_(param, mean=0.0, std=0.1)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc(last)


def build_model(task: str, model_kwargs: dict | None = None) -> nn.Module:
    model_kwargs = model_kwargs or {}
    if task == "curves":
        return CurvesAutoencoder(**model_kwargs)
    if task in ("mnist", "fashionmnist"):
        return LeNet5(num_classes=10)
    if task == "addition":
        return SimpleRNN(**model_kwargs)
    raise ValueError(f"Unknown task: {task}")


def matrix_params_for_kron(model: nn.Module) -> List[nn.Parameter]:
    """Collect 2D weight matrices used for Kronecker preconditioning."""
    params: List[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.Linear):
            params.append(module.weight)
        elif isinstance(module, nn.Conv2d):
            w = module.weight
            params.append(w.view(w.size(0), -1))
    return params
