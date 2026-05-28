#!/usr/bin/env bash
# Clone official optimizer repositories into third_party/
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="${ROOT}/third_party"
mkdir -p "${TP}"

clone_if_missing() {
  local url="$1"
  local dir="$2"
  local branch="${3:-}"
  local path="${TP}/${dir}"

  if [[ -d "${path}/.git" ]]; then
    echo "Already cloned: ${dir}"
    return
  fi

  if [[ -n "${branch}" ]]; then
    git clone --depth 1 -b "${branch}" "${url}" "${path}"
  else
    git clone --depth 1 "${url}" "${path}"
  fi
}

clone_if_missing "https://github.com/lixilinx/psgd_torch.git" "psgd_torch"
clone_if_missing "https://github.com/jonathanmei/kradagrad.git" "kradagrad" "release"
clone_if_missing "https://github.com/nikhilvyas/SOAP.git" "SOAP"
clone_if_missing "https://github.com/Daniil-Selikhanovych/Shampoo_optimizer.git" "Shampoo_optimizer"

echo "Done. Install K-FAC: pip install git+https://github.com/gpauloski/kfac-pytorch.git"
echo "Shampoo: PyTorch port in optimizers/shampoo_daniil.py (algorithm from Shampoo_optimizer repo)."
