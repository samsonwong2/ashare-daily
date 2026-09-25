from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ashare_daily.daily_contract import (
    is_off,
    month_of,
    parse_as_of,
    require_signals_file,
    require_validation_dir,
)
from ashare_daily.paths import VALIDATION_DIR, plotly_outputs_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily adaptive")
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--plot-start", default="2026-04-01")
    p.add_argument("--train-cutoff", default=None)
    p.add_argument("--codes", default="")
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--score-mode", default="expect_no_trade")
    p.add_argument("--regime-mode", default="legacy_score_method")
    p.add_argument("--trade-mode", default="hold_up")
    p.add_argument("--train-cache-dir", default=None)
    p.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    p.add_argument("--html-out-dir", type=Path, default=None)
    p.add_argument("--no-ticket-card", action="store_true")
    p.add_argument("--ru-diag", action="store_true")
    p.add_argument("--fair-path-extra-trail-years", default="1,0.5,0.25,1/12")
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    as_of = parse_as_of(args.as_of)
    require_validation_dir(args.validation_dir)
    require_signals_file(args.validation_dir, month_of(as_of))
    tag = as_of.replace("-", "")
    html_dir = args.html_out_dir or (plotly_outputs_dir() / f"{tag}all_adaptive_stock")
    train_cutoff = args.train_cutoff or args.plot_start
    cache = args.train_cache_dir
    if cache is None:
        cache = str(plotly_outputs_dir() / "_train_cache")
    forwarded = [
        "--start-date",
        args.plot_start,
        "--end-date",
        as_of,
        "--train-cutoff",
        train_cutoff,
        "--validation-dir",
        str(args.validation_dir),
        "--html-out-dir",
        str(html_dir),
        "--prior-k",
        "5",
        "--score-mode",
        args.score_mode,
        "--regime-mode",
        args.regime_mode,
        "--trade-mode",
        args.trade_mode,
        "--jobs",
        str(args.jobs),
        "--continue-on-error",
    ]
    if not is_off(cache):
        forwarded += ["--train-cache-dir", str(cache)]
    if args.retrain:
        forwarded.append("--retrain")
    if not args.ru_diag:
        forwarded.append("--disable-ru-diag")
    if args.max_codes:
        forwarded += ["--max-codes", str(args.max_codes)]
    for code in str(args.codes).split():
        forwarded += ["--code", code]
    if not is_off(args.fair_path_extra_trail_years):
        forwarded += ["--fair-path-extra-trail-years", args.fair_path_extra_trail_years]
    from ashare_daily.plots.plot_adaptive_stage_pool import main as plot_main

    rc = plot_main(forwarded)
    if isinstance(rc, int) and rc != 0:
        return rc
    if not args.no_ticket_card:
        from ashare_daily.plots.build_daily_ticket_card import main as ticket_main

        rc = ticket_main(
            ["--as-of", as_of, "--adaptive-dir", str(html_dir), "--out", str(html_dir / "ticket_card.csv")]
        )
        if isinstance(rc, int) and rc != 0:
            return rc
    return 0
