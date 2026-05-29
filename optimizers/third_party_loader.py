"""Clone vendored optimizer repos if missing, then register them on ``sys.path``."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _ROOT / "third_party"

# (folder_name, git_url, branch or None, marker file inside repo)
_VENDORED_REPOS: Tuple[Tuple[str, str, Optional[str], str], ...] = (
    ("psgd_torch", "https://github.com/lixilinx/psgd_torch.git", None, "psgd.py"),
    ("kradagrad", "https://github.com/jonathanmei/kradagrad.git", "release", "kradagradmm.py"),
    ("SOAP", "https://github.com/nikhilvyas/SOAP.git", None, "soap.py"),
)

_PATHS = (
    _THIRD_PARTY,
    _THIRD_PARTY / "psgd_torch",
    _THIRD_PARTY / "SOAP",
)

_REPOS_READY = False

# Upstream __init__.py imports KradagradPP -> needs missing batched_matrix_functions.
_KRADAGRAD_INIT_PATCH = '''\
"""KrADagrad (vendored). Import submodules directly, e.g. ``kradagrad.kradagradmm.KradagradMM``."""

# Do not import KradagradPP here (release branch lacks batched_matrix_functions).
__all__: list[str] = []
'''


def third_party_root() -> Path:
    return _THIRD_PARTY


def _patch_kradagrad_init() -> None:
    """Replace upstream package __init__ that pulls in broken KradagradPP."""
    init_path = _THIRD_PARTY / "kradagrad" / "__init__.py"
    if not init_path.parent.is_dir():
        return
    current = init_path.read_text(encoding="utf-8") if init_path.is_file() else ""
    if current == _KRADAGRAD_INIT_PATCH:
        return
    if "KradagradPP" in current or "batched_matrix_functions" in current or current.strip() != _KRADAGRAD_INIT_PATCH.strip():
        init_path.write_text(_KRADAGRAD_INIT_PATCH, encoding="utf-8")
        print("[third_party] Patched kradagrad/__init__.py (skip KradagradPP import).")


def _clone_repo(name: str, url: str, branch: Optional[str], marker: str) -> None:
    dest = _THIRD_PARTY / name
    if (dest / marker).is_file():
        return

    _THIRD_PARTY.mkdir(parents=True, exist_ok=True)
    cmd: List[str] = ["git", "clone", "--depth", "1"]
    if branch:
        cmd.extend(["-b", branch])
    cmd.extend([url, str(dest)])
    print(f"[third_party] Cloning {name} from {url} ...")
    subprocess.run(cmd, check=True, cwd=_ROOT)


def ensure_third_party_repos() -> None:
    """Shallow-clone official optimizer repos into ``third_party/`` when absent."""
    global _REPOS_READY
    if _REPOS_READY:
        return
    for name, url, branch, marker in _VENDORED_REPOS:
        _clone_repo(name, url, branch, marker)
    _patch_kradagrad_init()
    _REPOS_READY = True


def ensure_third_party_paths() -> None:
    ensure_third_party_repos()
    for path in _PATHS:
        p = str(path)
        if path.is_dir() and p not in sys.path:
            sys.path.insert(0, p)
