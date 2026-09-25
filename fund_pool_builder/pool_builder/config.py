"""JSON config loader + metadata writer for the pool-builder pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent  # fund_pool_builder/
DEFAULT_SHARED_CONFIG = _PACKAGE_ROOT / "shared_filter_config.json"


def load_shared_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load ``shared_filter_config.json``.

    Resolution order:
      1. Explicit ``config_path`` argument.
      2. ``fund_pool_builder/shared_filter_config.json`` (new canonical location).
      3. Legacy ``2_filter/shared_filter_config.json`` for back-compat.
    """
    candidates = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(DEFAULT_SHARED_CONFIG)
    # Legacy fallback so existing 2_filter/ config still works during migration.
    legacy = _PACKAGE_ROOT.parent / "2_filter" / "shared_filter_config.json"
    candidates.append(legacy)

    for cand in candidates:
        if cand and cand.exists():
            with cand.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        f"shared_filter_config.json not found. Tried: {[str(c) for c in candidates]}"
    )


def write_json(path: str | os.PathLike, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
