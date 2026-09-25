"""Tests for mapping trend-board metrics."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from ashare_daily.pool.builder.trend_board import (
    build_mapping_trend_frame,
    build_risk_flags,
    compute_trend_metrics,
    DEFAULT_TREND_THRESHOLDS,
)


def test_compute_trend_metrics_bull_stack():
    dates = pd.bdate_range("2025-01-01", periods=80)
    # Steady uptrend
    prices = np.linspace(10.0, 20.0, len(dates))
    close = pd.Series(prices, index=dates)
    end = dates[-1]
    metrics = compute_trend_metrics(close, end, vol_window=20, vol_lookback=60)
    assert metrics["ma_stack"] == "bull"
    assert metrics["dist_ma20_pct"] is not None
    assert metrics["dist_ma20_pct"] > 0


def test_build_risk_flags_below_ma20():
    flags = build_risk_flags(
        dist_ma20_pct=-2.0,
        vol_pct_120d=0.5,
        weight=0.0,
        pnl_pct=None,
        ma_stack="bear",
        rs_20d_vs_bench=-0.1,
        days_below_ma20=5,
        thresholds=DEFAULT_TREND_THRESHOLDS,
    )
    assert "below_ma20" in flags
    assert "bear_stack" in flags
    assert "heavy_wt" not in flags


def test_build_mapping_trend_frame_columns():
    dates = pd.bdate_range("2025-01-01", periods=200)
    prices = np.linspace(10.0, 15.0, len(dates))
    close = pd.DataFrame({"AAA": prices}, index=dates)
    as_of = dates[-1].strftime("%Y-%m-%d")
    frame = build_mapping_trend_frame(close, ["AAA"], as_of, provider_uri=None)
    assert list(frame.columns) == [
        "code",
        "weight",
        "close",
        "ma5",
        "ma10",
        "ma20",
        "dist_ma20_pct",
        "ret_5d",
        "ret_20d",
        "ma_stack",
        "vol_pct_120d",
        "vol_ann_20d",
        "vol_ann_5d",
        "vol_ratio_5_20",
        "vol_level",
        "dd_20d_high",
        "pnl_pct",
        "days_below_ma20",
        "rs_20d_vs_bench",
        "risk_flags",
    ]
    assert float(frame.loc[0, "weight"]) == 0.0
