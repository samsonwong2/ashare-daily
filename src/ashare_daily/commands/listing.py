from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from ashare_daily.daily_contract import (
    find_prev_listing_dir,
    is_off,
    month_of,
    parse_as_of,
    require_config_source,
    require_signals_file,
    require_validation_dir,
)
from ashare_daily.paths import VALIDATION_DIR, plotly_outputs_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily listing")
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--train-cutoff", default="2026-04-01")
    p.add_argument("--codes", default="")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--no-incremental", action="store_true")
    p.add_argument("--prev-listing-dir", type=Path, default=None)
    p.add_argument("--train-cache-dir", default=None)
    p.add_argument("--listing-cache", default=None)
    p.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    p.add_argument("--config-source-dir", type=Path, default=None)
    p.add_argument("--html-out-dir", type=Path, default=None)
    p.add_argument("--no-fig12", action="store_true")
    p.add_argument("--ru-diag", action="store_true")
    p.add_argument("--fair-path-extra-trail-years", default="1,0.5,0.25,1/12")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    as_of = parse_as_of(args.as_of)
    require_validation_dir(args.validation_dir)
    require_signals_file(args.validation_dir, month_of(as_of))
    root = plotly_outputs_dir()
    tag = as_of.replace("-", "")
    config_dir = args.config_source_dir or (root / f"{tag}all_adaptive_stock")
    require_config_source(config_dir)
    html_dir = args.html_out_dir or (root / f"{tag}_from_listing_stock")
    incremental = not args.no_incremental
    prev = args.prev_listing_dir
    if incremental and prev is None:
        prev = find_prev_listing_dir(root, tag)
    if incremental and (prev is None or not prev.is_dir()):
        print("[WARN] no previous *_from_listing_stock found; falling back to full rebuild", file=sys.stderr)
        incremental = False
        prev = None
    cache = args.train_cache_dir if args.train_cache_dir is not None else str(root / "_train_cache")
    listing_cache = args.listing_cache if args.listing_cache is not None else str(root / "_listing_dates_stock.csv")
    forwarded = [
        "--as-of",
        as_of,
        "--train-cutoff",
        args.train_cutoff,
        "--validation-dir",
        str(args.validation_dir),
        "--config-source-dir",
        str(config_dir),
        "--html-out-dir",
        str(html_dir),
        "--jobs",
        str(args.jobs),
        "--incremental-mode",
        "html" if incremental else "off",
    ]
    if is_off(cache):
        forwarded.append("--no-train-cache")
    else:
        forwarded += ["--train-cache-dir", str(cache)]
    if is_off(listing_cache):
        forwarded.append("--no-listing-cache")
    else:
        forwarded += ["--listing-cache", str(listing_cache)]
    if args.retrain:
        forwarded.append("--retrain")
    if not args.ru_diag:
        forwarded.append("--disable-ru-diag")
    if args.max_codes:
        forwarded += ["--max-codes", str(args.max_codes)]
    codes = str(args.codes).split()
    for code in codes:
        forwarded += ["--code", code]
    if incremental and prev is not None:
        forwarded += ["--incremental-from", str(prev)]
    if not is_off(args.fair_path_extra_trail_years):
        forwarded += ["--fair-path-extra-trail-years", args.fair_path_extra_trail_years]
    from ashare_daily.plots.run_from_listing_sharded import main as shard_main

    rc = shard_main(forwarded)
    if isinstance(rc, int) and rc != 0:
        return rc
    ewma = ["--html-dir", str(html_dir), "--jobs", str(args.jobs)]
    for code in codes:
        ewma += ["--code", code]
    from ashare_daily.plots.add_ewma_rails import main as ewma_main

    rc = ewma_main(ewma)
    if isinstance(rc, int) and rc != 0:
        return rc
    if not args.no_fig12:
        from ashare_daily.plots.add_fig12_six_states import main as fig12_main

        rc = fig12_main(ewma)
        if isinstance(rc, int) and rc != 0:
            return rc
    return 0
