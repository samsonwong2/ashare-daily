"""JSON config loader + metadata writer for the pool-builder pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ashare_daily.paths import repo_root


def _config_path() -> Path:
    return repo_root() / "configs" / "shared_filter_config.json"


def _example_path() -> Path:
    return repo_root() / "configs" / "shared_filter_config.json.example"


def load_shared_config(config_path: str | os.PathLike | None = None) -> dict:
    """Load ``configs/shared_filter_config.json`` from the checkout root."""
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))
    candidates.append(_config_path())
    for cand in candidates:
        if cand.is_file():
            with cand.open("r", encoding="utf-8") as handle:
                return json.load(handle)
    raise FileNotFoundError(
        "shared_filter_config.json not found at "
        f"{_config_path()}. Copy {_example_path().name} and fill it in."
    )


def write_json(path: str | os.PathLike, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
