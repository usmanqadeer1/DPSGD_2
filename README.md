# DPSGD Benchmark Suite

Research-grade PyTorch codebase to benchmark **Damped Preconditioned Stochastic Gradient Descent (DPSGD)** against Kronecker-factored and first-order optimizers on four tasks from the DPSGD paper:

| Task | Model | Loss |
|------|-------|------|
| CURVES autoencoder | 12-layer FCN | MSE |
| MNIST | LeNet-5 (ReLU) | Cross-entropy |
| FashionMNIST | LeNet-5 (ReLU) | Cross-entropy |
| Synthetic addition | Simple RNN (hidden 20) | MSE |

## Project layout

```
DPSGD/
├── configs.py           # Experiment hyperparameters
├── datasets.py          # Data loading (CURVES .mat, MNIST, synthetic addition)
├── models.py            # FCN, LeNet-5, SimpleRNN
├── trainer.py           # Unified training loop
├── utils.py             # set_seed, Plotly plots
├── run_experiments.py   # CLI entry point
├── optimizers/
│   ├── dpsgd.py         # DPSGD (Algorithms 1–2)
│   ├── psgd.py          # PSGD baseline
│   ├── psgd_kron.py     # Vendored Kronecker PSGD updates
│   ├── shampoo.py       # Shampoo
│   ├── kfac_wrapper.py  # K-FAC
│   ├── soap_wrapper.py  # SOAP
│   ├── kradagrad_wrapper.py
│   └── factory.py
├── requirements.txt
└── README.md
```

## Quick start (local CPU smoke test)

```bash
cd DPSGD
pip install -r requirements.txt
python run_experiments.py --task curves --optimizers sgd dpsgd --seed 1
```

Outputs under `results/<task>/`:
- `metrics_<optimizer>.json` — train loss/acc **per iteration**, test loss/acc **per epoch**, wall-clock time
- `*.png` — matplotlib overlays comparing **all** optimizers (loss vs iteration/epoch/time)

## CLI

```bash
python run_experiments.py --task curves          # all optimizers for task
python run_experiments.py --task mnist --optimizers dpsgd psgd adam sgd
python run_experiments.py --task fashionmnist --seed 1 --data-dir ./data
python run_experiments.py --task addition --optimizers dpsgd shampoo
```

## DPSGD implementation notes

- **Algorithm 1:** `update_precond_kron` (PSGD Kronecker factor update via HVP)
- **Algorithm 2:** damped preconditioned step, EMA (Eq. 47), LM `τ` schedule (Eq. 29–33), clipping (Eq. 48)
- **Formulations:** `--task fashionmnist` uses additive damping (Eq. 36); other tasks use factor-absorbed (Eq. 46) per `configs.py`
- **Globals:** `τ₀=1`, `ω=(19/20)^T₁`, `η=10⁻⁵`, `ν=0.1√N`

## Third-party optimizer sources (official implementations)

**First run** auto-clones missing repos into `third_party/` (needs `git` + internet).  
Or clone manually:

```powershell
.\scripts\setup_third_party.ps1
```

Then:

```powershell
pip install -r requirements.txt
pip install git+https://github.com/gpauloski/kfac-pytorch.git
```

**Kaggle / Colab:** after `git clone`, run experiments as usual — `third_party/` is created automatically. Install deps in a cell:

```python
!pip install -q opt_einsum
!pip install -q git+https://github.com/gpauloski/kfac-pytorch.git
!python run_experiments.py --task mnist --device cuda --optimizers kradagrad
```

| Optimizer | Implementation | Hyperparameters |
|-----------|----------------|-----------------|
| **DPSGD** | Custom (`optimizers/dpsgd.py` + vendored PSGD Kronecker updates) | Paper Table 1 + global τ₀, ω, η, ν |
| **PSGD** | [`lixilinx/psgd_torch`](https://github.com/lixilinx/psgd_torch) `KronNewton` | Table 1 α, `lr_precond=0.1`, `T1=1`, exact HVP |
| **Shampoo** | PyTorch port of [`Daniil-Selikhanovych/Shampoo_optimizer`](https://github.com/Daniil-Selikhanovych/Shampoo_optimizer) (`optimizers/shampoo_daniil.py`) | Table 1 α, `limit`→`max_matrix_size`, `alpha=0.5`, `epsilon=1e-4` |
| **K-FAC** | [`gpauloski/kfac-pytorch`](https://github.com/gpauloski/kfac-pytorch) | Table 1 α, β→`factor_decay=1-β`, γ→`damping` |
| **SOAP** | [`nikhilvyas/SOAP`](https://github.com/nikhilvyas/SOAP) `soap.py` | Table 1 α; repo defaults `betas=(0.95,0.95)`, `precond_freq=10` |
| **KrADagrad** | [`jonathanmei/kradagrad`](https://github.com/jonathanmei/kradagrad) `KradagradMM` | `deepobs_single.py` defaults: `matrix_eps=1e-4`, `precond_steps=20` |

Notebook reimplementations of Shampoo/K-FAC/PSGD are **not** used for baselines.

## Reproducibility

`set_seed(seed)` fixes Python, NumPy, and PyTorch (CPU/CUDA). The same `--seed` is used for every optimizer in a run so weight init and batch order match.

## RunPod (GPU cloud) guide

### 1. Create a GPU pod

1. Sign in at [runpod.io](https://www.runpod.io).
2. **Deploy** → **GPU Pod** (e.g. **RTX 4090** or **A100**).
3. Template: **PyTorch 2.x** (CUDA 12.x).
4. GPU count: **1** is enough for these benchmarks.

### 2. Upload or clone the project

**Option A – Git (recommended)**

```bash
cd /workspace
git clone <your-repo-url> DPSGD
cd DPSGD
```

**Option B – ZIP upload**

Upload the `DPSGD` folder via RunPod file browser or `scp` into `/workspace/DPSGD`.

### 3. Install dependencies

```bash
cd /workspace/DPSGD
pip install -r requirements.txt

# Optional baselines
pip install git+https://github.com/gpauloski/kfac-pytorch.git
pip install git+https://github.com/nikhilvyas/SOAP.git
pip install git+https://github.com/jonathanmei/kradagrad.git
```

### 4. Run experiments

```bash
# Full CURVES benchmark (~20 epochs, all optimizers)
python run_experiments.py --task curves --seed 1 --data-dir /workspace/data --results-dir /workspace/results/curves

# MNIST
python run_experiments.py --task mnist --optimizers sgd adam psgd dpsgd kfac shampoo --seed 1

# FashionMNIST (additive damping for DPSGD)
python run_experiments.py --task fashionmnist --seed 1

# Synthetic addition RNN
python run_experiments.py --task addition --seed 1
```

### 5. Retrieve results

- Open `results/<task>/*.html` in a browser (Plotly interactive loss curves).
- Download `metrics_*.json` from the pod **Connect** → file manager or:

```bash
scp -r root@<pod-ip>:/workspace/DPSGD/results ./local_results
```

### 6. Long runs / disconnect safety

Use `tmux` or `screen` so training continues if SSH drops:

```bash
tmux new -s dpsgd
python run_experiments.py --task curves --seed 1
# Detach: Ctrl+B then D
# Reattach: tmux attach -t dpsgd
```

### 7. Tips

- Set `--num-workers 4` on GPU pods for faster data loading (MNIST/FashionMNIST).
- CURVES downloads `digs3pts_1.mat` automatically on first run (~few MB).
- If CUDA OOM on CURVES+Shampoo, run optimizers one at a time with `--optimizers dpsgd`.
- Verify GPU: `python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`

## Citation

If you use this code, please cite the DPSGD paper (DPSGD_SN) and the respective baseline optimizer references (PSGD, K-FAC, Shampoo, SOAP, KrADagrad).
