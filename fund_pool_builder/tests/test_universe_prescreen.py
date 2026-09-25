"""Tests for stratified liquidity universe prescreen."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pool_builder.universe_prescreen import prescreen_universe


def _toy_volume(codes: list[str], medians: dict[str, float]) -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    data = {}
    for code in codes:
        base = medians.get(code, 1.0)
        data[code] = np.full(len(idx), base)
    return pd.DataFrame(data, index=idx)


def test_prescreen_keeps_bucket_floor_not_return_based():
    codes = ["A", "B", "C", "D"]
    vol = _toy_volume(codes, {"A": 100.0, "B": 90.0, "C": 5.0, "D": 4.0})
    name_map = {
        "A": "人工智能ETF",
        "B": "AI芯片",
        "C": "冷门消费",
        "D": "冷门other",
    }
    kept, diag = prescreen_universe(
        codes,
        vol,
        name_map,
        vol_pctl=50.0,
        min_per_bucket=1,
        max_universe=3,
    )
    assert "C" in kept or "D" in kept
    assert len(kept) <= 3
    assert set(diag["code"]) == set(codes)
    assert "kept_reason" in diag.columns


def test_prescreen_respects_max_universe():
    codes = [f"C{i}" for i in range(10)]
    vol = _toy_volume(codes, {c: float(i + 1) for i, c in enumerate(codes)})
    name_map = {c: f"name{c}" for c in codes}
    kept, _ = prescreen_universe(
        codes,
        vol,
        name_map,
        vol_pctl=0.0,
        min_per_bucket=1,
        max_universe=5,
    )
    assert len(kept) == 5


def test_prescreen_diagnostics_marks_dropped():
    codes = ["HIGH", "LOW"]
    vol = _toy_volume(codes, {"HIGH": 1000.0, "LOW": 1.0})
    name_map = {"HIGH": "high vol", "LOW": "low vol"}
    kept, diag = prescreen_universe(
        codes,
        vol,
        name_map,
        vol_pctl=50.0,
        min_per_bucket=1,
        max_universe=1,
    )
    assert len(kept) == 1
    dropped = diag.loc[diag["kept_reason"] == "dropped", "code"].tolist()
    assert len(dropped) == 1
