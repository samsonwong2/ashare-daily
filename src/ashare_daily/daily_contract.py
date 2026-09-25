"""Pure helpers for the eight daily commands.

These functions do not import path loaders or Qlib. Tests call them directly.
"""
from __future__ import annotations

import os
import re
import sys
from datetime import date, timedelta
from pathlib import Path


def slice_after_command(argv: list[str]) -> tuple[str, list[str]]:
    """Split ``['hrp', '--dist-t', '0.8']`` into the name and the old parser's argv."""
    if not argv:
        raise SystemExit("missing subcommand")
    return argv[0], list(argv[1:])


def parse_as_of(value: str) -> str:
    """Return ``YYYY-MM-DD`` or exit 2."""
    text = str(value).strip()
    if len(text) != 10 or text[4] != "-" or text[7] != "-":
        print(f"[ERROR] --as-of must be YYYY-MM-DD, got: {text}", file=sys.stderr)
        raise SystemExit(2)
    try:
        date.fromisoformat(text)
    except ValueError:
        print(f"[ERROR] --as-of is not a valid date: {text}", file=sys.stderr)
        raise SystemExit(2)
    return text


def as_of_tag(value: str) -> str:
    return parse_as_of(value).replace("-", "")


def month_of(value: str) -> str:
    return parse_as_of(value)[:7]


def clear_regime_shards(shard_dir: Path, month: str, *, force: bool) -> None:
    """Match the stock regime shell.

    Default and ``--skip-rebuild`` delete ``{month}.done`` and keep the csv.
    ``--force-month-rebuild`` deletes both. Step 1 recreates the csv.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / f"{month}.done").unlink(missing_ok=True)
    if force:
        (shard_dir / f"{month}.signals.csv").unlink(missing_ok=True)


def require_validation_dir(path: Path) -> None:
    if not path.is_dir():
        print(f"missing validation dir: {path}", file=sys.stderr)
        raise SystemExit(1)


def require_signals_file(validation_dir: Path, month: str) -> None:
    csv = validation_dir / "shards" / f"{month}.signals.csv"
    if not csv.is_file():
        print(f"missing signals file: {csv}", file=sys.stderr)
        raise SystemExit(1)


def require_config_source(path: Path) -> None:
    if not path.is_dir():
        print(f"missing config source dir: {path}", file=sys.stderr)
        raise SystemExit(1)


def is_off(value: str | None) -> bool:
    text = str(value or "").strip().lower()
    return text in {"", "off", "none", "0", "false", "no"}


def find_prev_listing_dir(plotly_root: Path, as_of_tag_value: str) -> Path | None:
    """Newest ``YYYYMMDD_from_listing_stock`` strictly before ``as_of_tag_value``."""
    if not plotly_root.is_dir():
        return None
    best: Path | None = None
    best_tag = ""
    for directory in plotly_root.glob("*_from_listing_stock"):
        if not directory.is_dir() or not (directory / "listing_dates.csv").is_file():
            continue
        tag = directory.name[: -len("_from_listing_stock")]
        if len(tag) != 8 or not tag.isdigit() or tag >= as_of_tag_value:
            continue
        if tag > best_tag:
            best_tag = tag
            best = directory
    return best


def listing_worker_prefix(python: str) -> list[str]:
    return [python, "-m", "ashare_daily.plots.plot_adaptive_from_listing"]


_LISTING_DIR_NAME = re.compile(r"^(\d{8})_from_listing_stock$")


def as_of_from_listing_dir(path: Path) -> str:
    """``20260924_from_listing_stock`` -> ``2026-09-24``. Exit 2 if the name has no date."""
    match = _LISTING_DIR_NAME.match(path.name)
    if match is None:
        print(
            f"[ERROR] cannot read YYYYMMDD from {path.name}; pass --as-of and --next-day",
            file=sys.stderr,
        )
        raise SystemExit(2)
    raw = match.group(1)
    return parse_as_of(f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}")


def next_weekday(value: str) -> str:
    """Next Mon–Fri after ``value``. Does not skip exchange holidays."""
    current = date.fromisoformat(parse_as_of(value)) + timedelta(days=1)
    while current.weekday() >= 5:
        current += timedelta(days=1)
    return current.isoformat()


def resolve_listing_dir(raw: str, plotly_root: str | None = None) -> Path:
    """Turn a shell path into the listing directory under ``PLOTLY_ROOT``."""
    root = (plotly_root if plotly_root is not None else os.environ.get("PLOTLY_ROOT", "")).strip()
    text = os.path.expandvars(str(raw).strip())
    path = Path(text)
    if path.is_dir():
        return path
    if root and _LISTING_DIR_NAME.match(path.name):
        candidate = Path(root) / path.name
        if candidate.is_dir():
            return candidate
    return path
