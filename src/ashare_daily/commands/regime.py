from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ashare_daily.daily_contract import (
    clear_regime_shards,
    month_of,
    parse_as_of,
    require_validation_dir,
)
from ashare_daily.paths import MODEL_CACHE_DIR, VALIDATION_DIR, plotly_outputs_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily regime")
    p.add_argument("--as-of", default=date.today().isoformat())
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--plot-start", default="2026-04-01")
    p.add_argument("--skip-rebuild", action="store_true")
    p.add_argument("--skip-html", action="store_true")
    p.add_argument("--force-month-rebuild", action="store_true")
    p.add_argument("--html-pool-dir", type=Path, default=None)
    p.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    p.add_argument("--cache-dir", type=Path, default=MODEL_CACHE_DIR)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    as_of = parse_as_of(args.as_of)
    require_validation_dir(args.validation_dir)
    month = month_of(as_of)
    clear_regime_shards(
        args.validation_dir / "shards",
        month,
        force=bool(args.force_month_rebuild),
    )
    tag = as_of.replace("-", "")
    html_dir = args.html_pool_dir or (plotly_outputs_dir() / f"{tag}all_stock")
    from ashare_daily.regime.backtest_regime_transition_signals import main as step1

    rc = step1(
        [
            "--output-dir",
            str(args.validation_dir),
            "--cache-dir",
            str(args.cache_dir),
            "--eval-start",
            "2024-06-01",
            "--eval-end",
            as_of,
            "--horizon",
            "10",
            "--jobs",
            str(args.jobs),
        ]
    )
    if isinstance(rc, int) and rc != 0:
        return rc
    if not args.skip_rebuild:
        from ashare_daily.regime.rebuild_reversal_event_metrics import main as step2

        rc = step2(["--validation-dir", str(args.validation_dir)])
        if isinstance(rc, int) and rc != 0:
            return rc
    if not args.skip_html:
        from ashare_daily.plots.plot_regime_transition_example import main as step3

        rc = step3(
            [
                "--validation-dir",
                str(args.validation_dir),
                "--start-date",
                args.plot_start,
                "--end-date",
                as_of,
                "--html-out-dir",
                str(html_dir),
                "--continue-on-error",
            ]
        )
        if isinstance(rc, int) and rc != 0:
            return rc
    return 0
