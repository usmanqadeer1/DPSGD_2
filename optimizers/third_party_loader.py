"""Register vendored optimizer repos under ``third_party/`` on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_THIRD_PARTY = _ROOT / "third_party"

_PATHS = (
    _THIRD_PARTY,
    _THIRD_PARTY / "psgd_torch",
    _THIRD_PARTY / "SOAP",
)


def ensure_third_party_paths() -> None:
    for path in _PATHS:
        p = str(path)
        if path.is_dir() and p not in sys.path:
            sys.path.insert(0, p)


def third_party_root() -> Path:
    return _THIRD_PARTY
