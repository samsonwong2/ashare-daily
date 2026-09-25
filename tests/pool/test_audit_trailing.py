"""Trailing return metrics ending on review_end."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


from ashare_daily.pool.builder.audit import compute_trailing_metrics


def test_compute_trailing_metrics_20d_ending_on_review_end():
    dates = pd.bdate_range("2026-03-01", periods=40)
    prices = np.linspace(80.0, 100.0, len(dates))
    close = pd.DataFrame({"AAA": prices}, index=dates)
    out = compute_trailing_metrics(close, "2026-04-30", [20])
    ret = float(out.loc[out["code"] == "AAA", "trail_ret_20d"].iloc[0])
    anchor_pos = int(np.where(close.index <= pd.Timestamp("2026-04-30"))[0][-1])
    expected = close["AAA"].iloc[anchor_pos] / close["AAA"].iloc[anchor_pos - 20] - 1.0
    assert abs(ret - expected) < 1e-9
