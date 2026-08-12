"""Load repo-root ``.env`` into ``os.environ`` (idempotent)."""

from __future__ import annotations

from pathlib import Path

_LOADED = False


def load_project_env(*, override: bool = False) -> Path | None:
    """Load ``.env`` from the repository root if present.

    Returns the path loaded, or None if missing / dotenv unavailable.
    Existing process env wins unless ``override=True``.
    """
    global _LOADED
    if _LOADED and not override:
        return None

    # src/common/env.py → parents[2] = repo root
    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"

    try:
        from dotenv import load_dotenv
    except ImportError:
        _LOADED = True
        return None

    if env_path.is_file():
        load_dotenv(env_path, override=override)
        _LOADED = True
        return env_path

    _LOADED = True
    return None
