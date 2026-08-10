#!/usr/bin/env python3
"""Backward-compatible entrypoint → preprocess_clips.py.

Old middle-only tile probe is replaced by start/middle/end car preprocess
(tiles, size, distance drift, frame_step). Prefer:

  python src/preprocess_clips.py
  python src/preprocess_clips.py --clip NAME
"""

from __future__ import annotations

from preprocess_clips import main

if __name__ == "__main__":
    main()
