# Hyperparameters by experiment and optimizer

Values come from `configs.py` (DPSGD paper Table 1 where noted) and `optimizers/factory.py` defaults when a key is omitted.

**Notation (paper):** α = learning rate, β = EMA / Polyak factor, γ = damping (K-FAC).

---

## Training setup (all optimizers)

| Experiment | Model | Epochs | Train batch | Test batch | Loss |
|------------|-------|--------|-------------|------------|------|
| **curves** | 12-layer FCN autoencoder | 20 | 64 | 1000 | MSE |
| **mnist** | LeNet-5 | 20 | 64 | 1000 | Cross-entropy |
| **fashionmnist** | LeNet-5 | 20 | 64 | 1000 | Cross-entropy |
| **addition** | Simple RNN (hidden 20) | 10 | 100 | 100 | MSE |

**Optimizers run per task:** all eight (`sgd`, `adam`, `psgd`, `dpsgd`, `kfac`, `shampoo`, `soap`, `kradagrad`) for every experiment unless you override with `--optimizers`.

---

## DPSGD — global (all experiments)

| Symbol | Parameter | Value | Notes |
|--------|-----------|-------|--------|
| τ₀ | `tau0` | 1.0 | Initial trust-region scale |
| ω | schedule | (19/20)^T₁ | LM τ update (T₁ from experiment) |
| η | `eta` | 1e-5 | Damping constant in Eq. 36 / 46 |
| ν | `nu_scale` | 0.1 | Step clip: ν = 0.1√N |
| — | `precond_lr` | 0.1 | Kronecker preconditioner LR (factory default) |
| — | bias update | SGD @ α | Non-matrix params use same α as DPSGD |

---

## CURVES

| Optimizer | α / `lr` | Other hyperparameters |
|-----------|----------|------------------------|
| **SGD** | 0.1 | momentum 0, weight_decay 0 |
| **Adam** | 0.001 | betas (0.9, 0.999) |
| **PSGD** | 0.1 | `precond_lr` 0.1, `T1` 1, grad clip 0.1√N, `preconditioner_max_size` ∞, `preconditioner_max_skew` 1.0 |
| **DPSGD** | α 0.1 | β 0.7, `T1` 1, `T2` 5, damping **factor-absorbed** (Eq. 46), τ₀/η/ν global |
| **K-FAC** | 0.5 | β→`momentum` 0.05, γ→`damping` 0.001, `factor_decay` 0.95, `factor_update_steps` 1, `inv_update_steps` 1, `kl_clip` 0.001 |
| **Shampoo** | 0.5 | `limit` 200, `alpha` 0.5, `epsilon` 1e-4, `weight_decay` 0 |
| **SOAP** | 0.1 | betas (0.95, 0.95), `precondition_frequency` 10, `weight_decay` 0, `max_precond_dim` 10000 |
| **KrADagrad** | 0.1 | `momentum` 0.9, `matrix_eps` 1e-4, `preconditioning_compute_steps` 20, `statistics_compute_steps` 1, `block_size` 128, `weight_decay` 0, Nesterov on |

---

## MNIST

| Optimizer | α / `lr` | Other hyperparameters |
|-----------|----------|------------------------|
| **SGD** | 0.1 | momentum 0, weight_decay 0 |
| **Adam** | 0.005 | betas (0.9, 0.999) |
| **PSGD** | 0.05 | `precond_lr` 0.1, `T1` 1, grad clip 0.1√N |
| **DPSGD** | α 0.001 | β 0.9, `T1` 1, `T2` 5, damping **factor-absorbed** (Eq. 46) |
| **K-FAC** | 0.01 | β→`momentum` 0.05, γ→`damping` 0.001, `factor_decay` 0.95, updates every step |
| **Shampoo** | 0.5 | `limit` 200, `alpha` 0.5, `epsilon` 1e-4 |
| **SOAP** | 0.05 | betas (0.95, 0.95), `precondition_frequency` 10 |
| **KrADagrad** | 0.05 | same KrADagrad defaults as CURVES |

---

## FashionMNIST

| Optimizer | α / `lr` | Other hyperparameters |
|-----------|----------|------------------------|
| **SGD** | 0.1 | — |
| **Adam** | 0.005 | betas (0.9, 0.999) |
| **PSGD** | 0.1 | `precond_lr` 0.1, `T1` 1 |
| **DPSGD** | α 0.1 | β 0.7, `T1` 1, `T2` 5, damping **additive** (Eq. 36), `factor_absorbed` false |
| **K-FAC** | 0.01 | β→`momentum` 0.05, γ→`damping` 0.001 (same as MNIST) |
| **Shampoo** | 0.5 | `limit` 200, `alpha` 0.5, `epsilon` 1e-4 (same as MNIST) |
| **SOAP** | 0.1 | betas (0.95, 0.95), `precondition_frequency` 10 |
| **KrADagrad** | 0.1 | KrADagrad defaults as above |

---

## Synthetic addition (RNN)

| Optimizer | α / `lr` | Other hyperparameters |
|-----------|----------|------------------------|
| **SGD** | 0.1 | — |
| **Adam** | 0.001 | betas (0.9, 0.999) |
| **PSGD** | 0.1 | `precond_lr` 0.1, `T1` 1 |
| **DPSGD** | α 0.1 | β 0.7, `T1` 1, `T2` 5, factor-absorbed (Eq. 46) |
| **K-FAC** | 0.5 | β→`momentum` 0.05, γ→`damping` 0.001 (same as CURVES) |
| **Shampoo** | 0.1 | `limit` **10**, `alpha` 0.5, `epsilon` 1e-4 |
| **SOAP** | 0.1 | betas (0.95, 0.95), `precondition_frequency` 10 |
| **KrADagrad** | 0.1 | KrADagrad defaults as above |

---

## Fixed defaults (not varied per experiment in `configs.py`)

| Optimizer | Parameter | Value | Where set |
|-----------|-----------|-------|-----------|
| Adam | `betas` | (0.9, 0.999) | `factory.py` |
| PSGD | `precond_lr` | 0.1 | configs (all tasks) |
| PSGD | `T1` | 1 | configs |
| PSGD | grad clip | 0.1√N | `psgd_official.py` |
| SOAP | `betas` | (0.95, 0.95) | configs |
| SOAP | `precondition_frequency` | 10 | configs |
| SOAP | `weight_decay` | 0 | factory |
| KrADagrad | `momentum` | 0.9 | factory |
| KrADagrad | `beta2` (Shampoo HP) | 1.0 | `kradagrad_official.py` |
| KrADagrad | `nesterov` | True | `kradagrad_official.py` |
| Shampoo | `mat_gbar_decay` | 1.0 | `shampoo_daniil.py` |
| K-FAC | `kl_clip` | 0.001 | `kfac_wrapper.py` |

---

## Source files

| What | File |
|------|------|
| Per-task α, β, γ, epochs, batches | `configs.py` → `EXPERIMENTS` |
| Optimizer construction | `optimizers/factory.py` |
| DPSGD algorithm | `optimizers/dpsgd.py` |

To change hyperparameters, edit `configs.py` (and re-run `python run_experiments.py --task <name>`).
