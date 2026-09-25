"""Tests for selection-stage liquidity floor."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


import ashare_daily.pool.builder.constants as C
from ashare_daily.pool.builder.liquidity_gate import (
    compute_median_dollar_volume,
    effective_liquidity_floor,
    passes_liquidity_floor,
)


def _panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = pd.date_range("2020-01-01", periods=260, freq="B")
    close = pd.DataFrame({"LOW": np.full(len(idx), 10.0), "HIGH": np.full(len(idx), 10.0)}, index=idx)
    volume = pd.DataFrame({"LOW": np.full(len(idx), 1.0e6), "HIGH": np.full(len(idx), 1.0e7)}, index=idx)
    return close, volume


def test_compute_median_dollar_volume():
    close, volume = _panels()
    med = compute_median_dollar_volume(close, volume, lookback_days=252)
    assert med["LOW"] == 1.0e7
    assert med["HIGH"] == 1.0e8


def test_passes_liquidity_floor_singleton_multiplier():
    floor = effective_liquidity_floor(
        1,
        min_dollar=0,
        singleton_min_dollar=2.45e7,
        singleton_mult=1.5,
        apply_mode="singleton",
    )
    assert floor == pytest.approx(2.45e7 * 1.5)
    assert passes_liquidity_floor(
        9.0e7,
        n_members=1,
        min_dollar=0,
        singleton_min_dollar=2.45e7,
        singleton_mult=1.5,
        apply_mode="singleton",
        gate_enabled=True,
    )
    assert not passes_liquidity_floor(
        3.0e7,
        n_members=1,
        min_dollar=0,
        singleton_min_dollar=2.45e7,
        singleton_mult=1.5,
        apply_mode="singleton",
        gate_enabled=True,
    )


def test_liquidity_gate_disabled_by_default():
    assert not C.SELECTION_LIQUIDITY_GATE_ENABLED
    assert passes_liquidity_floor(
        1.0e6,
        n_members=1,
        min_dollar=0,
        singleton_min_dollar=5.5e7,
        singleton_mult=1.0,
        apply_mode="singleton",
    )


def test_passes_liquidity_floor_multi_member_skips_singleton_mode():
    assert passes_liquidity_floor(
        1.0e6,
        n_members=3,
        min_dollar=5.5e7,
        singleton_min_dollar=2.45e7,
        singleton_mult=1.0,
        apply_mode="singleton",
    )


def test_hot_rank_cutoff_skips_gate_for_cold_singleton():
    assert passes_liquidity_floor(
        1.0e6,
        n_members=1,
        min_dollar=0,
        singleton_min_dollar=5.5e7,
        singleton_mult=1.0,
        apply_mode="singleton",
        cross_cluster_rank=100,
        hot_rank_cutoff=50,
        gate_enabled=True,
    )
    assert not passes_liquidity_floor(
        1.0e6,
        n_members=1,
        min_dollar=0,
        singleton_min_dollar=5.5e7,
        singleton_mult=1.0,
        apply_mode="singleton",
        cross_cluster_rank=10,
        hot_rank_cutoff=50,
        gate_enabled=True,
    )
