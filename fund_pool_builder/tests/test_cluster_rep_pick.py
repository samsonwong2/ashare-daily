"""Tests for POOL_REP_PICK representative ranking."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pool_builder.constants as C
from pool_builder.clustering import (
    _build_multi_rep_cluster_pool,
    _corr_to_cluster_center,
    _global_cross_cluster_scores,
    _rank_members_for_rep_pick,
    _return_scores,
)


def _toy_returns() -> pd.DataFrame:
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    return pd.DataFrame(
        {
            "HIGH": np.linspace(0.001, 0.003, len(idx)),
            "MID": np.linspace(0.0005, 0.0015, len(idx)),
            "LOW": np.linspace(-0.001, 0.0005, len(idx)),
        },
        index=idx,
    )


def _toy_corr(members: list[str]) -> pd.DataFrame:
    n = len(members)
    base = np.full((n, n), 0.5)
    np.fill_diagonal(base, 1.0)
    return pd.DataFrame(base, index=members, columns=members)


def test_corr_to_cluster_center_single_member():
    assert _corr_to_cluster_center("A", ["A"], pd.DataFrame()) == 1.0


def test_rank_members_center_corr_prefers_central_name():
    members = ["HIGH", "MID", "LOW"]
    returns = _toy_returns()
    corr = _toy_corr(members)
    corr.loc["HIGH", "MID"] = corr.loc["MID", "HIGH"] = 0.95
    corr.loc["HIGH", "LOW"] = corr.loc["LOW", "HIGH"] = 0.2
    corr.loc["MID", "LOW"] = corr.loc["LOW", "MID"] = 0.3
    ranked = _rank_members_for_rep_pick(members, returns, corr, rep_pick="center_corr")
    assert ranked.index[0] == "MID"


def test_rank_members_blend_penalizes_low_center_corr():
    members = ["HIGH", "LOW"]
    returns = _toy_returns()[members]
    corr = pd.DataFrame([[1.0, 0.1], [0.1, 1.0]], index=members, columns=members)
    ranked = _rank_members_for_rep_pick(
        members,
        returns,
        corr,
        rep_pick="blend",
        min_center_corr=0.55,
    )
    assert len(ranked) == 2


def test_return_scores_cumulative_return_picks_highest_total(monkeypatch):
    monkeypatch.setattr(C, "SCORING_MODE", "cumulative_return")
    monkeypatch.setattr("pool_builder.clustering.SCORING_MODE", "cumulative_return")
    returns = _toy_returns()
    members = ["HIGH", "MID", "LOW"]
    ranked = _rank_members_for_rep_pick(members, returns, None, rep_pick="return")
    assert ranked.index[0] == "HIGH"
    expected_high = float((returns["HIGH"] + 1.0).prod() - 1.0)
    assert ranked["HIGH"] == pytest.approx(expected_high)


def test_return_scores_cumulative_return_respects_scoring_mode_env(monkeypatch):
    returns = _toy_returns()
    members = ["HIGH", "MID", "LOW"]

    monkeypatch.setattr(C, "SCORING_MODE", "cumulative_return")
    monkeypatch.setattr("pool_builder.clustering.SCORING_MODE", "cumulative_return")
    cum_ranked = _return_scores(members, returns)

    monkeypatch.setattr(C, "SCORING_MODE", "sharpe_mom")
    monkeypatch.setattr("pool_builder.clustering.SCORING_MODE", "sharpe_mom")
    sharpe_ranked = _return_scores(members, returns)

    assert cum_ranked.index[0] == "HIGH"
    assert sharpe_ranked.index[0] == "HIGH"
    assert cum_ranked["HIGH"] > 0.0
    assert 0.0 <= sharpe_ranked["HIGH"] <= 1.0
    assert cum_ranked["HIGH"] != sharpe_ranked["HIGH"]


def test_global_cross_cluster_scores_cumulative_return():
    returns = _toy_returns()
    scores = _global_cross_cluster_scores(returns, mode="cumulative_return")
    assert scores.idxmax() == "HIGH"
    assert scores["HIGH"] > scores["LOW"]


def test_global_cross_cluster_scores_low_corr_prefers_diversifier():
    idx = pd.date_range("2020-01-01", periods=80, freq="B")
    rng = np.random.default_rng(0)
    market = rng.normal(0.0, 0.01, size=len(idx))
    # A/B hug the market; DIV is nearly orthogonal noise.
    returns = pd.DataFrame(
        {
            "A": market + rng.normal(0.0, 0.001, size=len(idx)),
            "B": market + rng.normal(0.0, 0.001, size=len(idx)),
            "DIV": rng.normal(0.0, 0.01, size=len(idx)),
        },
        index=idx,
    )
    scores = _global_cross_cluster_scores(returns, mode="low_corr")
    assert scores.idxmax() == "DIV"
    assert scores["DIV"] > scores["A"]
    assert scores["DIV"] > scores["B"]


def test_global_cross_cluster_scores_low_corr_return_balances_return(monkeypatch):
    monkeypatch.setattr(C, "CROSS_LOW_CORR_WEIGHT", 0.5)
    monkeypatch.setattr("pool_builder.clustering.CROSS_LOW_CORR_WEIGHT", 0.5)
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    n = len(idx)
    # CORR_A/CORR_B move together (high corr, modest return).
    # DIV_HOT is orthogonal and drifts up strongly.
    base = np.linspace(0.001, 0.0015, n)
    noise = np.sin(np.linspace(0, 6 * np.pi, n)) * 0.002
    returns = pd.DataFrame(
        {
            "CORR_A": base + noise,
            "CORR_B": base + noise * 0.95,
            "DIV_HOT": np.linspace(0.003, 0.006, n) + np.cos(np.linspace(0, 5 * np.pi, n)) * 0.001,
        },
        index=idx,
    )
    scores = _global_cross_cluster_scores(returns, mode="low_corr_return")
    assert scores.idxmax() == "DIV_HOT"
    assert scores["DIV_HOT"] > scores["CORR_A"]
    assert scores["DIV_HOT"] > scores["CORR_B"]


def test_plan_b_orders_by_global_cumulative_return(monkeypatch):
    monkeypatch.setattr(C, "CROSS_CLUSTER_RANK", "cumulative_return")
    monkeypatch.setattr("pool_builder.clustering.CROSS_CLUSTER_RANK", "cumulative_return")
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    returns = pd.DataFrame(
        {
            "HOT": np.linspace(0.01, 0.02, len(idx)),
            "COLD_A": np.linspace(0.0001, 0.0002, len(idx)),
            "COLD_B": np.linspace(0.0001, 0.0002, len(idx)),
        },
        index=idx,
    )
    cluster_df = pd.DataFrame(
        [
            {"cluster": 1, "members": ["HOT"]},
            {"cluster": 2, "members": ["COLD_A", "COLD_B"]},
        ]
    )
    history_days = pd.Series({code: 200.0 for code in returns.columns})
    selected, _, _ = _build_multi_rep_cluster_pool(
        cluster_df,
        returns,
        history_days,
        target_count=1,
        cross_cluster_rank_mode="cumulative_return",
    )
    assert selected == ["HOT"]


def test_plan_b_skips_low_liquidity_rank1(monkeypatch):
    monkeypatch.setattr(C, "CROSS_CLUSTER_RANK", "cumulative_return")
    monkeypatch.setattr("pool_builder.clustering.CROSS_CLUSTER_RANK", "cumulative_return")
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    returns = pd.DataFrame(
        {
            "ILLIQ": np.linspace(0.01, 0.02, len(idx)),
            "LIQUID": np.linspace(0.0001, 0.0002, len(idx)),
        },
        index=idx,
    )
    cluster_df = pd.DataFrame(
        [
            {"cluster": 1, "members": ["ILLIQ"], "n": 1},
            {"cluster": 2, "members": ["LIQUID"], "n": 1},
        ]
    )
    history_days = pd.Series({code: 200.0 for code in returns.columns})
    median_dollar = pd.Series({"ILLIQ": 1.0e7, "LIQUID": 1.0e8})
    selected, reasons, drop_count = _build_multi_rep_cluster_pool(
        cluster_df,
        returns,
        history_days,
        target_count=1,
        cross_cluster_rank_mode="cumulative_return",
        median_dollar_volume=median_dollar,
        liquidity_gate_enabled=True,
        min_dollar_volume=0,
        singleton_min_dollar_volume=5.5e7,
        liquidity_apply_mode="singleton",
        liquidity_hot_rank_cutoff=50,
    )
    assert selected == ["LIQUID"]
    assert drop_count >= 1
    assert "ILLIQ" in reasons and "liquidity_floor_miss" in reasons["ILLIQ"]

