#!/usr/bin/env python3
"""Generate adaptive HTML charts from each ETF's listing date through as-of.

Reuses frozen ``hold_up`` configs from a source adaptive directory (no retrain
by default). Each symbol gets its own plot window starting at the first Qlib
bar (or ``--start-date`` if given). Production default includes multi-scale
fair-path panels fig8–11 (1y / 6m / 3m / 1m); pass
``--fair-path-extra-trail-years off`` to disable.

Examples::

    # All pool codes, listing → 2026-08-13
    PYTHONPATH=. python decision_pack/scripts/plot_adaptive_from_listing.py \\
      --as-of 2026-08-13

    # One symbol only
    PYTHONPATH=. python decision_pack/scripts/plot_adaptive_from_listing.py \\
      --as-of 2026-08-13 --code SH513090

    # Custom plot window (Y axes fit this window; HTML also Y-rescales on X zoom)
    PYTHONPATH=. python decision_pack/scripts/plot_adaptive_from_listing.py \\
      --as-of 2026-08-14 --start-date 2025-11-01 --end-date 2026-08-14 \\
      --code SZ159985 \\
      --config-source-dir workspace/plotly_outputs/20260814all_adaptive \\
      --html-out-dir workspace/plotly_outputs/20260814_from_listing

    # Strong incremental: reuse yesterday HTML, append one causal bar
    PYTHONPATH=. python decision_pack/scripts/plot_adaptive_from_listing.py \\
      --as-of 2026-08-28 \\
      --config-source-dir workspace/plotly_outputs/20260828all_adaptive \\
      --html-out-dir workspace/plotly_outputs/20260828_from_listing \\
      --incremental-from workspace/plotly_outputs/20260827_from_listing \\
      --incremental-mode html
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from ashare_daily.plots.adaptive_stage_common import (  # noqa: E402
    METHOD_IMPL_VERSION,
    PRIOR_K,
    REGIME_SELECT_LEGACY,
    TRADE_MODE_HOLD_UP,
    TRUTH_VERSION,
)
from ashare_daily.hrp.hrp_cluster_router import DEFAULT_MEMBERSHIP_CSV  # noqa: E402
from ashare_daily.lib.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from ashare_daily.paths import (  # noqa: E402
    CLUSTER_MAPPING_SELECTED_TXT,
    VALIDATION_DIR,
    plotly_outputs_dir,
    repo_root,
)

_PROJECT_ROOT = repo_root()

DEFAULT_VALIDATION_DIR = VALIDATION_DIR
DEFAULT_TRAIN_CUTOFF = "2026-04-01"
DEFAULT_SCORE_MODE = "expect_no_trade"
LISTING_PROBE_START = "2005-01-01"


def _load_pm():
    from ashare_daily.plots import plot_regime_transition_example as mod

    return mod


def _load_pool_main():
    from ashare_daily.plots import plot_adaptive_stage_pool as mod

    return mod


def cache_tag(
    *,
    train_cutoff: str,
    trade_mode: str,
    score_mode: str,
    regime_mode: str,
) -> str:
    return (
        f"{train_cutoff.replace('-', '')}_{trade_mode}_{score_mode}_"
        f"{regime_mode}_{TRUTH_VERSION.replace('+', '-')}_{METHOD_IMPL_VERSION}"
    )


def resolve_config_source(as_of: str, explicit: Path | None) -> Path:
    if explicit is not None:
        p = explicit if explicit.is_absolute() else _PROJECT_ROOT / explicit
        if not p.exists():
            raise FileNotFoundError(f"config source missing: {p}")
        return p
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    candidates = [
        plotly_outputs_dir() / f"{tag}all_adaptive_stock",
    ]
    for c in candidates:
        if (c / "configs").is_dir():
            return c
    raise FileNotFoundError(
        "no config source found; pass --config-source-dir "
        "(expected */all_adaptive/configs)"
    )


def seed_out_dir(
    out_dir: Path,
    source: Path,
    codes: list[str],
    *,
    tag: str,
    train_cutoff: str,
    retrain: bool,
) -> None:
    """Copy frozen configs (+ train cache marker) so pool script reuses them.

    Always refresh from ``source`` when the source file exists. Skipping an
    existing ``out_dir/configs/{code}.json`` left stale methods after
    ``all_adaptive`` was retrained into the same html-out-dir.
    """
    cfg_dst = out_dir / "configs"
    cfg_dst.mkdir(parents=True, exist_ok=True)
    src_cfg = source / "configs"
    missing: list[str] = []
    for code in codes:
        src = src_cfg / f"{code}.json"
        dst = cfg_dst / f"{code}.json"
        if src.exists():
            # Refresh even if dst exists (source is the frozen-method authority).
            shutil.copy2(src, dst)
        elif not dst.exists():
            missing.append(code)
    cache_name = f".train_cache_{tag}.json"
    src_cache = source / cache_name
    dst_cache = out_dir / cache_name
    if src_cache.exists():
        shutil.copy2(src_cache, dst_cache)
    elif not dst_cache.exists() and not retrain:
        # Minimal marker so pool reuse path can engage when configs exist.
        dst_cache.write_text(
            json.dumps(
                {
                    "train_cutoff": train_cutoff,
                    "n_codes": len(codes),
                    "seeded_from": str(source),
                    "trade_mode": TRADE_MODE_HOLD_UP,
                    "score_mode": DEFAULT_SCORE_MODE,
                    "regime_mode": REGIME_SELECT_LEGACY,
                    "truth_version": TRUTH_VERSION,
                    "method_impl_version": METHOD_IMPL_VERSION,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if missing and not retrain:
        raise FileNotFoundError(
            "missing frozen configs for: "
            + ", ".join(missing)
            + f" (looked in {src_cfg}; pass --retrain to build)"
        )


def first_bar_date(pm, code: str, end_date: str) -> str:
    ohlcv = pm.load_qlib_ohlcv(
        code,
        LISTING_PROBE_START,
        end_date,
        init_qlib=False,
        lookback_calendar_days=0,
        clip_to_window=False,
        apply_qfq=False,
    )
    if ohlcv is None or ohlcv.empty:
        raise ValueError(f"no Qlib bars for {code} through {end_date}")
    d = pd.to_datetime(ohlcv["datetime"]).min()
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def load_listing_lookup(path: Path | None) -> dict[str, str]:
    """Load code→listing YYYY-MM-DD from a listing_dates.csv-like file."""
    if path is None:
        return {}
    p = path if path.is_absolute() else _PROJECT_ROOT / path
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return {}
    if "code" not in df.columns or "listing" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, r in df.iterrows():
        code = str(r["code"]).upper().strip()
        listing = str(r["listing"])[:10]
        if code and listing and listing.lower() not in {"nan", "none", ""}:
            out[code] = listing
    return out


def merge_listing_lookups(*lookups: dict[str, str]) -> dict[str, str]:
    """Later dicts override earlier ones for the same code."""
    out: dict[str, str] = {}
    for lu in lookups:
        out.update(lu)
    return out


def write_listing_cache(path: Path, lookup: dict[str, str]) -> None:
    p = path if path.is_absolute() else _PROJECT_ROOT / path
    p.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"code": c, "listing": lookup[c]} for c in sorted(lookup)]
    pd.DataFrame(rows).to_csv(p, index=False, encoding="utf-8-sig")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--as-of",
        default=None,
        help="plot end date YYYY-MM-DD (used if --end-date omitted; also for "
        "default config-source / out-dir naming)",
    )
    p.add_argument(
        "--start-date",
        default=None,
        help="plot window start YYYY-MM-DD; default = each symbol listing date "
        "(clamped to listing if earlier)",
    )
    p.add_argument(
        "--end-date",
        default=None,
        help="plot window end YYYY-MM-DD; default = --as-of",
    )
    p.add_argument(
        "--code",
        action="append",
        default=None,
        help="limit to one or more codes (repeatable); default = cluster pool",
    )
    p.add_argument(
        "--train-cutoff",
        default=DEFAULT_TRAIN_CUTOFF,
        help=f"frozen train cutoff (default {DEFAULT_TRAIN_CUTOFF})",
    )
    p.add_argument(
        "--config-source-dir",
        type=Path,
        default=None,
        help="adaptive dir with configs/ to reuse (default: {as_of}all_adaptive)",
    )
    p.add_argument(
        "--html-out-dir",
        type=Path,
        default=None,
        help="output directory (default: {as_of}_from_listing, or "
        "{end}_from_listing_{start}_{end} when --start-date is set)",
    )
    p.add_argument("--validation-dir", type=Path, default=DEFAULT_VALIDATION_DIR)
    p.add_argument("--cluster-mapping", type=Path, default=None)
    p.add_argument("--live-snapshot", type=Path, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument(
        "--hrp-membership-csv",
        type=Path,
        default=DEFAULT_MEMBERSHIP_CSV,
    )
    p.add_argument("--disable-ru-diag", action="store_true")
    p.add_argument(
        "--fair-path-extra-trail-years",
        type=str,
        default="1,0.5,0.25,1/12",
        help=(
            "extra fair-path trails after fig7=2y (default fig8–11: 1y,6m,3m,1m). "
            "Use off|none|0 to disable."
        ),
    )
    p.add_argument(
        "--incremental-mode",
        choices=("off", "html"),
        default="off",
        help="off=full rebuild (default). html=reuse yesterday from_listing HTML "
        "and append one bar (qfq/gap mismatch falls back to full).",
    )
    p.add_argument(
        "--incremental-from",
        type=Path,
        default=None,
        help="previous *_from_listing directory (required when --incremental-mode=html)",
    )
    p.add_argument(
        "--incremental-close-tol",
        type=float,
        default=1e-6,
        help="max relative close error on overlap to treat as qfq change",
    )
    p.add_argument(
        "--train-cache-dir",
        type=Path,
        default=None,
        help=(
            "shared pass1 config cache across AS_OF dirs "
            "(default _train_cache under PLOTLY_ROOT)"
        ),
    )
    p.add_argument(
        "--listing-cache",
        type=Path,
        default=None,
        help=(
            "persistent code→listing CSV to skip Qlib first-bar probes "
            "(default _listing_dates_stock.csv under PLOTLY_ROOT)"
        ),
    )
    p.add_argument(
        "--no-listing-cache",
        action="store_true",
        help="disable persistent listing-date cache",
    )
    p.add_argument(
        "--no-train-cache",
        action="store_true",
        help="disable shared train-cache-dir reuse",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    end_raw = args.end_date or args.as_of
    if not end_raw:
        raise SystemExit("need --end-date or --as-of")
    end_date = pd.Timestamp(end_raw).strftime("%Y-%m-%d")
    # Prefer --as-of for naming/config discovery when present; else end_date.
    as_of = pd.Timestamp(args.as_of or end_date).strftime("%Y-%m-%d")
    as_of_tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    end_tag = pd.Timestamp(end_date).strftime("%Y%m%d")
    start_override = (
        pd.Timestamp(args.start_date).strftime("%Y-%m-%d") if args.start_date else None
    )
    custom_window = start_override is not None
    train_cutoff = pd.Timestamp(args.train_cutoff).strftime("%Y-%m-%d")
    trade_mode = TRADE_MODE_HOLD_UP
    score_mode = DEFAULT_SCORE_MODE
    regime_mode = REGIME_SELECT_LEGACY
    tag = cache_tag(
        train_cutoff=train_cutoff,
        trade_mode=trade_mode,
        score_mode=score_mode,
        regime_mode=regime_mode,
    )

    out_dir = args.html_out_dir
    if out_dir is None:
        if custom_window:
            start_tag = pd.Timestamp(start_override).strftime("%Y%m%d")
            out_dir = (
                plotly_outputs_dir()
                / f"{end_tag}_from_listing_{start_tag}_{end_tag}"
            )
        else:
            out_dir = plotly_outputs_dir() / f"{as_of_tag}_from_listing"
    out_dir = out_dir if out_dir.is_absolute() else _PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = (
        Path(args.cluster_mapping)
        if args.cluster_mapping
        else Path(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping
    codes = [c.upper() for c in (args.code or [])] or list(
        load_cluster_mapping_codes(mapping)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]
    if not codes:
        raise SystemExit("no codes to plot")

    source = resolve_config_source(as_of, args.config_source_dir)
    print(f"[INFO] as_of={as_of} end_date={end_date} train_cutoff={train_cutoff}")
    print(f"[INFO] start_date={'listing' if not start_override else start_override}")
    print(f"[INFO] config_source={source}")
    print(f"[INFO] html_out_dir={out_dir}")
    print(f"[INFO] codes={len(codes)}")
    print(f"[INFO] incremental_mode={args.incremental_mode}")
    prev_listing: Path | None = None
    if args.incremental_mode == "html":
        if args.incremental_from is None:
            raise SystemExit("html incremental requires --incremental-from")
        prev_listing = args.incremental_from
        if not prev_listing.is_absolute():
            prev_listing = _PROJECT_ROOT / prev_listing
        if not prev_listing.is_dir():
            raise SystemExit(f"incremental-from missing: {prev_listing}")
        print(f"[INFO] incremental_from={prev_listing}")

    seed_out_dir(
        out_dir,
        source,
        codes,
        tag=tag,
        train_cutoff=train_cutoff,
        retrain=bool(args.retrain),
    )

    pool = _load_pool_main()
    listing_cache_path: Path | None = None
    if not args.no_listing_cache:
        if args.listing_cache is not None:
            listing_cache_path = (
                args.listing_cache
                if args.listing_cache.is_absolute()
                else _PROJECT_ROOT / args.listing_cache
            )
        else:
            listing_cache_path = plotly_outputs_dir() / "_listing_dates_stock.csv"
    persistent_lookup = load_listing_lookup(listing_cache_path)
    prev_lookup: dict[str, str] = {}
    if prev_listing is not None:
        prev_lookup = load_listing_lookup(prev_listing / "listing_dates.csv")
    listing_lookup = merge_listing_lookups(persistent_lookup, prev_lookup)
    if listing_lookup:
        print(
            f"[INFO] listing_lookup codes={len(listing_lookup)} "
            f"(cache={len(persistent_lookup)} prev={len(prev_lookup)})"
        )

    train_cache_dir: Path | None = None
    if not args.no_train_cache:
        if args.train_cache_dir is not None:
            train_cache_dir = args.train_cache_dir
        else:
            train_cache_dir = plotly_outputs_dir() / "_train_cache"

    rt = pool.prepare_adaptive_runtime(
        end=end_date,
        start=end_date,
        train_cutoff=train_cutoff,
        out_dir=out_dir,
        codes=codes,
        validation_dir=args.validation_dir,
        cluster_mapping=args.cluster_mapping,
        live_snapshot=args.live_snapshot,
        prior_k=float(PRIOR_K),
        score_mode_name=score_mode,
        regime_mode=regime_mode,
        trade_mode=trade_mode,
        retrain=bool(args.retrain),
        fail_fast=bool(args.fail_fast),
        hrp_membership_csv=args.hrp_membership_csv,
        disable_ru_diag=bool(args.disable_ru_diag),
        enable_shallow_up_trade=False,
        train_cache_dir=train_cache_dir,
    )
    pm = rt["pm"]
    ev = rt["ev"]
    from ashare_daily.plots.incremental_from_listing import (
        try_incremental_html,
        write_book_csvs,
    )

    listing_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    summary_rows: list[dict[str, Any]] = []
    for i, code in enumerate(codes, 1):
        try:
            listing = listing_lookup.get(code)
            if not listing:
                listing = first_bar_date(pm, code, end_date)
            if start_override:
                plot_start = max(pd.Timestamp(start_override), pd.Timestamp(listing))
                plot_start = plot_start.strftime("%Y-%m-%d")
            else:
                plot_start = listing
            if pd.Timestamp(plot_start) > pd.Timestamp(end_date):
                raise ValueError(
                    f"empty window: start={plot_start} > end={end_date} "
                    f"(listing={listing})"
                )
            print(
                f"[{i}/{len(codes)}] {code} listing={listing} "
                f"window={plot_start}→{end_date}"
            )
            ohlcv = pool._load_ohlcv(pm, code, end_date, train_cutoff=train_cutoff)
            cfg = rt["configs"][code]
            name = rt["name_map"].get(code) or code
            used_incr = False
            if args.incremental_mode == "html" and prev_listing is not None:
                book = write_book_csvs(
                    pool=pool,
                    code=code,
                    cfg=cfg,
                    ohlcv=ohlcv,
                    start_date=plot_start,
                    end_date=end_date,
                    out_dir=out_dir,
                    ev=ev,
                    hrp_mode_by_code=rt["hrp_mode_by_code"],
                    enable_ru_diag=bool(rt["enable_ru_diag"]),
                    enable_shallow_up_trade=bool(rt["enable_shallow_up_trade"]),
                    name=name,
                )
                html_out, reason = try_incremental_html(
                    code=code,
                    prev_dir=prev_listing,
                    out_dir=out_dir,
                    ohlcv=ohlcv,
                    end_date=end_date,
                    close_tol=float(args.incremental_close_tol),
                    extra_trail_years=args.fair_path_extra_trail_years,
                    aev=book.get("aev") or [],
                    display_name=name,
                    rv20_anchor=rt.get("rv20_anchor"),
                    rv5_anchor=rt.get("rv5_anchor"),
                    nav_last=book.get("nav_last"),
                    nav_by_date=book.get("nav_by_date"),
                )
                if html_out is not None and reason in {"ok", "same"}:
                    print(f"[{i}/{len(codes)}] INCREMENT html {code} reason={reason}")
                    used_incr = True
                    row = dict(book)
                    row.pop("aev", None)
                    row.pop("nav_last", None)
                    row.pop("nav_by_date", None)
                    row["out"] = html_out.name
                    summary_rows.append(row)
                else:
                    print(f"[{i}/{len(codes)}] FALLBACK full {code} reason={reason}")
            if not used_incr:
                r = pool.oos_one(
                    code=code,
                    cfg=cfg,
                    ohlcv=ohlcv,
                    start_date=plot_start,
                    end_date=end_date,
                    ev=ev,
                    pm=pm,
                    oos=rt["oos"],
                    rev=rt["rev"],
                    evt=rt["evt"],
                    name_map=rt["name_map"],
                    out_dir=out_dir,
                    live_snapshot=rt["live"],
                    hrp_mode_by_code=rt["hrp_mode_by_code"],
                    enable_ru_diag=bool(rt["enable_ru_diag"]),
                    enable_shallow_up_trade=bool(rt["enable_shallow_up_trade"]),
                    rv20_anchor=rt["rv20_anchor"],
                    rv5_anchor=rt["rv5_anchor"],
                    fair_path_extra_trail_years=args.fair_path_extra_trail_years,
                )
                summary_rows.append(r)
            htmls = list(out_dir.glob(f"regime_transition_{code}_*_adaptive.html"))
            htmls = [
                p
                for p in htmls
                if "_fig9" not in p.stem
                and "_fig10" not in p.stem
                and "_oracle" not in p.stem
            ]
            if not htmls:
                raise RuntimeError(
                    f"no HTML for {code} under {out_dir}"
                )
            listing_rows.append(
                dict(
                    code=code,
                    listing=listing,
                    plot_start=plot_start,
                    plot_end=end_date,
                    as_of=as_of,
                )
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(dict(code=code, error=str(exc)))
            print(f"[{i}/{len(codes)}] FAIL {code}: {exc}")
            if args.fail_fast:
                traceback.print_exc()
                return 1

    pd.DataFrame(listing_rows).to_csv(
        out_dir / "listing_dates.csv", index=False, encoding="utf-8-sig"
    )
    if listing_cache_path is not None and listing_rows:
        for row in listing_rows:
            c = str(row.get("code") or "").upper()
            listing = str(row.get("listing") or "")[:10]
            if c and listing:
                listing_lookup[c] = listing
        write_listing_cache(listing_cache_path, listing_lookup)
        print(f"[INFO] wrote listing cache n={len(listing_lookup)} -> {listing_cache_path}")
    if summary_rows:
        pool.write_merged_batch_summary(out_dir, pd.DataFrame(summary_rows))
    if errors:
        pd.DataFrame(errors).to_csv(
            out_dir / "from_listing_errors.csv", index=False, encoding="utf-8-sig"
        )
        pool.write_merged_batch_errors(
            out_dir,
            [dict(code=e["code"], phase="from_listing", error=e["error"]) for e in errors],
            clear_codes=[str(r.get("code")) for r in summary_rows],
        )
    elif summary_rows:
        pool.write_merged_batch_errors(
            out_dir,
            [],
            clear_codes=[str(r.get("code")) for r in summary_rows],
        )
    summary_path = out_dir / "batch_summary.csv"
    n_ok = len(codes) - len(errors)
    print(f"[OK] from_listing done ok={n_ok} fail={len(errors)} out={out_dir}")
    if summary_path.exists():
        try:
            n_sum = len(pd.read_csv(summary_path))
        except Exception:  # noqa: BLE001
            n_sum = -1
        print(f"[OK] batch_summary={summary_path} rows={n_sum}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
