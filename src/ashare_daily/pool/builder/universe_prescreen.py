"""Stratified liquidity prescreen before clustering (MLFAM Ch.4 coverage guard).

Filters by median volume (not return), with per-theme bucket floors.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as C
from .theme_labels import refined_industry_label

# Theme buckets that should keep more names at prescreen (liquidity-only, not return).
BUCKET_MIN_OVERRIDES: dict[str, int] = {
    "ai_compute": 3,
    "semiconductor": 3,
}


def _min_for_bucket(bucket: str, default: int) -> int:
    return int(BUCKET_MIN_OVERRIDES.get(bucket, default))


def _median_volume(vol: pd.DataFrame, code: str, lookback: int) -> float:
    if code not in vol.columns:
        return float("nan")
    series = pd.to_numeric(vol[code], errors="coerce").dropna()
    if series.empty:
        return float("nan")
    window = series.tail(lookback)
    if window.empty:
        return float("nan")
    return float(window.median())


def prescreen_universe(
    codes: Iterable[str],
    vol: pd.DataFrame,
    code_name_map: dict[str, str],
    *,
    vol_pctl: float = C.PRESCREEN_VOL_PCTL,
    min_per_bucket: int = C.PRESCREEN_MIN_PER_BUCKET,
    max_universe: int = C.PRESCREEN_MAX_UNIVERSE,
    lookback_days: int = C.CLUSTER_LOOKBACK_DAYS,
) -> tuple[list[str], pd.DataFrame]:
    """Return kept codes and diagnostics (code, bucket, median_vol, kept_reason)."""
    code_list = list(dict.fromkeys(str(c).strip() for c in codes if str(c).strip()))
    if not code_list:
        return [], pd.DataFrame(columns=["code", "bucket", "median_vol", "kept_reason"])

    rows: list[dict[str, object]] = []
    for code in code_list:
        name = code_name_map.get(code) or code_name_map.get(code[2:]) if len(code) > 2 else code
        bucket = refined_industry_label(name or code)
        med_vol = _median_volume(vol, code, lookback_days)
        rows.append({"code": code, "bucket": bucket, "median_vol": med_vol})

    frame = pd.DataFrame(rows)
    finite_vol = frame["median_vol"].replace([np.inf, -np.inf], np.nan).dropna()
    if finite_vol.empty:
        frame["kept_reason"] = "no_volume_data"
        return code_list, frame

    vol_threshold = float(np.percentile(finite_vol.values, vol_pctl))
    frame["passes_vol_floor"] = frame["median_vol"].fillna(-1.0) >= vol_threshold

    kept: set[str] = set()
    kept_reason: dict[str, str] = {}

    # Stage A: per-bucket minimum by liquidity (coverage floor).
    for bucket, group in frame.groupby("bucket", sort=False):
        ordered = group.sort_values("median_vol", ascending=False, na_position="last")
        floor_n = min(_min_for_bucket(str(bucket), min_per_bucket), len(ordered))
        for _, row in ordered.head(floor_n).iterrows():
            code = str(row["code"])
            kept.add(code)
            kept_reason[code] = f"bucket_floor:{bucket}"

    # Stage B: fill remaining seats up to max_universe by global liquidity among vol-passers.
    remaining_cap = max(0, int(max_universe) - len(kept))
    if remaining_cap > 0:
        candidates = frame.loc[frame["passes_vol_floor"] & ~frame["code"].isin(kept)].copy()
        candidates = candidates.sort_values("median_vol", ascending=False, na_position="last")
        for _, row in candidates.head(remaining_cap).iterrows():
            code = str(row["code"])
            kept.add(code)
            kept_reason[code] = "vol_rank"

    # If still under cap (strict vol floor removed too many), backfill by liquidity without vol floor.
    remaining_cap = max(0, int(max_universe) - len(kept))
    if remaining_cap > 0:
        backfill = frame.loc[~frame["code"].isin(kept)].sort_values(
            "median_vol", ascending=False, na_position="last"
        )
        for _, row in backfill.head(remaining_cap).iterrows():
            code = str(row["code"])
            kept.add(code)
            kept_reason[code] = "backfill_liquidity"

    frame["kept_reason"] = frame["code"].map(kept_reason).fillna("dropped")
    kept_ordered = [c for c in code_list if c in kept]
    print(
        f"[INFO] prescreen: {len(code_list)} -> {len(kept_ordered)} codes "
        f"(vol_pctl={vol_pctl}, min_per_bucket={min_per_bucket}, max={max_universe})"
    )
    return kept_ordered, frame
