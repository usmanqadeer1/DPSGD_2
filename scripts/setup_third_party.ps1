# Clone official optimizer repositories into third_party/
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$TP = Join-Path $Root "third_party"
New-Item -ItemType Directory -Force -Path $TP | Out-Null

function Clone-IfMissing($url, $dir, $branch = $null) {
    $path = Join-Path $TP $dir
    if (Test-Path (Join-Path $path ".git")) {
        Write-Host "Already cloned: $dir"
        return
    }
    if ($branch) {
        git clone --depth 1 -b $branch $url $path
    } else {
        git clone --depth 1 $url $path
    }
}

Clone-IfMissing "https://github.com/lixilinx/psgd_torch.git" "psgd_torch"
Clone-IfMissing "https://github.com/jonathanmei/kradagrad.git" "kradagrad" "release"
Clone-IfMissing "https://github.com/nikhilvyas/SOAP.git" "SOAP"
Clone-IfMissing "https://github.com/Daniil-Selikhanovych/Shampoo_optimizer.git" "Shampoo_optimizer"

Write-Host "Done. Install K-FAC: pip install git+https://github.com/gpauloski/kfac-pytorch.git"
Write-Host "Shampoo: PyTorch port in optimizers/shampoo_daniil.py (algorithm from Shampoo_optimizer repo)."
