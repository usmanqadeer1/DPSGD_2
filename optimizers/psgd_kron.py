"""
Vendored Kronecker PSGD update rules from lixilinx/psgd_torch
(preconditioned_stochastic_gradient_descent.py), Apache-2.0 compatible.
"""

from __future__ import annotations

import torch


def norm_lower_bound(A: torch.Tensor) -> torch.Tensor:
    max_abs = torch.max(torch.abs(A))
    if max_abs > 0:
        A = A / max_abs
        aa = torch.real(A * A.conj())
        value0, i = torch.max(torch.sum(aa, dim=0), 0)
        value1, j = torch.max(torch.sum(aa, dim=1), 0)
        if value0 > value1:
            x = A[:, i].conj() @ A
            return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))
    return max_abs


def _update_precond_dense_dense(
    Ql: torch.Tensor,
    Qr: torch.Tensor,
    dX: torch.Tensor,
    dG: torch.Tensor,
    step: float = 0.01,
    _tiny: float = 1.2e-38,
):
    max_l = torch.max(torch.diag(Ql))
    max_r = torch.max(torch.diag(Qr))
    rho = torch.sqrt(max_l / max_r)
    Ql = Ql / rho
    Qr = Qr * rho

    A = torch.linalg.multi_dot([Ql, dG, Qr.t()])
    Bt = torch.linalg.solve_triangular(
        Ql.t(),
        torch.linalg.solve_triangular(Qr, dX, upper=True, left=False),
        upper=False,
    )
    grad1 = torch.triu(A.mm(A.t()) - Bt.mm(Bt.t()))
    grad2 = torch.triu(A.t().mm(A) - Bt.t().mm(Bt))
    step1 = step / (norm_lower_bound(grad1) + _tiny)
    step2 = step / (norm_lower_bound(grad2) + _tiny)
    return Ql - step1 * grad1.mm(Ql), Qr - step2 * grad2.mm(Qr)


def _precond_grad_dense_dense(Ql: torch.Tensor, Qr: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    return torch.linalg.multi_dot([Ql.t(), Ql, grad, Qr.t(), Qr])


def update_precond_kron(
    Ql: torch.Tensor,
    Qr: torch.Tensor,
    dX: torch.Tensor,
    dG: torch.Tensor,
    step: float = 0.01,
    _tiny: float = 1.2e-38,
):
    m, n = Ql.shape
    p, q = Qr.shape
    if m == n and p == q:
        return _update_precond_dense_dense(Ql, Qr, dX, dG, step, _tiny)
    raise ValueError(f"Unsupported Kronecker preconditioner shapes: Ql {Ql.shape}, Qr {Qr.shape}")


def precond_grad_kron(Ql: torch.Tensor, Qr: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    m, n = Ql.shape
    p, q = Qr.shape
    if m == n and p == q:
        return _precond_grad_dense_dense(Ql, Qr, grad)
    raise ValueError(f"Unsupported Kronecker preconditioner shapes: Ql {Ql.shape}, Qr {Qr.shape}")
