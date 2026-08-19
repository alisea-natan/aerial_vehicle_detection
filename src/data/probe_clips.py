#!/usr/bin/env python3
"""Backward-compatible entrypoint → preprocess_clips.py.

Old middle-only tile probe is replaced by start/middle/end car preprocess
(tiles, size, distance drift, frame_step). Prefer:

  python src/data/preprocess_clips.py
  python src/data/preprocess_clips.py --clip NAME
"""
from __future__ import annotations

import sys
from pathlib import Path as _Path

def _ensure_src_on_path() -> None:
    """Allow `python src/<pkg>/….py` without PYTHONPATH."""
    p = _Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "common").is_dir() and (p / "labeling").is_dir():
            s = str(p)
            if s not in sys.path:
                sys.path.insert(0, s)
            return
        p = p.parent

_ensure_src_on_path()

from data.preprocess_clips import main

if __name__ == "__main__":
    main()
