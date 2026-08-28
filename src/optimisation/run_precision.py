#!/usr/bin/env python3
"""Deprecated alias — use run_quantize.py."""
from __future__ import annotations

from pathlib import Path as _Path
import sys


def _ensure_src_on_path() -> None:
    p = _Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent


_ensure_src_on_path()

from optimisation.run_quantize import main  # noqa: E402

if __name__ == "__main__":
    main()
