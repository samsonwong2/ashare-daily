"""Tests for hand-pool daily review helpers (ETF-style missed / rep-lag)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pool_builder import audit
from pool_builder.daily_review import (
    build_missed_clusters,
    build_rep_lag,
    overlay_hand_selected,
)


def test_assign_review_priority_cluster_cut_is_low():
    row = pd.Series(
        {
            "miss_type": "cluster_cut_miss",
            "is_strong_post": True,
            "regret_20d": 0.5,
        }
    )
    assert audit.assign_review_priority(row, "regret_20d", 0.10) == "low"


def test_assign_review_priority_wrong_cluster_strong_is_high():
    row = pd.Series(
        {
            "miss_type": "wrong_cluster_miss",
            "is_strong_post": True,
            "regret_20d": 0.05,
        }
    )
    assert audit.assign_review_priority(row, "regret_20d", 0.10) == "high"


def test_build_cluster_quality_summary_columns():
    mapping = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "name": ["a", "b", "c", "d"],
            "cluster": [1, 1, 2, 2],
            "selected": [True, False, False, False],
            "mean_corr": [0.6, 0.6, 0.4, 0.4],
            "n": [2, 2, 2, 2],
        }
    )
    audit_df = pd.DataFrame(
        {
            "code": ["B", "D"],
            "cluster": [1, 2],
            "miss_type": ["wrong_cluster_miss", "cluster_cut_miss"],
            "cluster_has_selected_rep": [True, False],
            "corr_to_rep_in_build_window": [0.3, None],
            "is_strong_post": [True, True],
        }
    )
    summary = audit.build_cluster_quality_summary(mapping, audit_df)
    assert not summary.empty
    assert "cohesion_flag" in summary.columns
    row1 = summary[summary["cluster"] == 1].iloc[0]
    assert int(row1["wrong_cluster_n"]) == 1
    assert bool(row1["has_selected_rep"]) is True
    row2 = summary[summary["cluster"] == 2].iloc[0]
    assert int(row2["cluster_cut_flag"]) == 1


def test_overlay_hand_selected_marks_selection_source():
    mapping = pd.DataFrame(
        {
            "code": ["SH600000", "SZ000001", "SH601398"],
            "cluster": [1, 1, 2],
            "selected": [True, False, True],
            "n": [2, 2, 1],
            "name": ["a", "b", "c"],
            "reason": ["", "", ""],
        }
    )
    out = overlay_hand_selected(mapping, ["SZ000001"])
    assert out.loc[out["code"] == "SZ000001", "selected"].iloc[0]
    assert not out.loc[out["code"] == "SH600000", "selected"].iloc[0]
    assert out.loc[out["code"] == "SZ000001", "selection_source"].iloc[0] == "hand_curated"


def test_build_missed_clusters_finds_empty_hand_clusters():
    mapping = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "name": ["a", "b", "c", "d"],
            "cluster": [1, 1, 2, 2],
            "selected": [True, False, False, False],
            "n": [2, 2, 2, 2],
            "reason": ["", "", "", ""],
        }
    )
    future = pd.DataFrame(
        {
            "code": ["A", "B", "C", "D"],
            "fwd_ret_20d": [0.01, 0.02, 0.30, 0.10],
        }
    )
    missed = build_missed_clusters(mapping, ["A"], future, min_cluster_n=2, focus_window=20)
    assert len(missed) == 1
    assert int(missed.iloc[0]["cluster"]) == 2
    assert missed.iloc[0]["best_code"] == "C"
    assert missed.iloc[0]["reason_kind"] == "target_count_not_selected"


def test_build_rep_lag_flags_non_top_selected():
    mapping = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "name": ["a", "b", "c"],
            "cluster": [1, 1, 1],
            "selected": [True, False, False],
            "n": [3, 3, 3],
            "reason": ["", "", ""],
        }
    )
    future = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "fwd_ret_20d": [0.05, 0.20, 0.10],
        }
    )
    build_metrics = pd.DataFrame(
        {
            "code": ["A", "B", "C"],
            "volume_rank_in_cluster": [1.0, 2.0, 3.0],
        }
    )
    lag = build_rep_lag(
        mapping, ["A"], future, build_metrics, min_regret=0.01, focus_window=20
    )
    assert len(lag) == 1
    assert lag.iloc[0]["selected_code"] == "A"
    assert lag.iloc[0]["best_code"] == "B"
    assert abs(float(lag.iloc[0]["regret_20d"]) - 0.15) < 1e-9


def test_compute_trailing_return_metrics_columns():
    idx = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame(
        {
            "SH600000": np.linspace(10, 13, len(idx)),
            "SZ000001": np.linspace(20, 18, len(idx)),
        },
        index=idx,
    )
    out = audit.compute_trailing_return_metrics(close, idx[-1].strftime("%Y-%m-%d"), [5, 20])
    assert set(out["code"]) == {"SH600000", "SZ000001"}
    assert "fwd_ret_5d" in out.columns
    assert "fwd_ret_20d" in out.columns
    assert out.loc[out["code"] == "SH600000", "fwd_ret_5d"].iloc[0] > 0
    assert out.loc[out["code"] == "SZ000001", "fwd_ret_5d"].iloc[0] < 0


def test_assign_unstable_reason():
    row_wrong = pd.Series({"miss_type": "wrong_cluster_miss", "cluster_has_selected_rep": True})
    assert audit.assign_unstable_reason(row_wrong, 0.55) == "wrong_cluster_assignment"
    row_weak = pd.Series(
        {
            "miss_type": "same_cluster_miss",
            "cluster_has_selected_rep": True,
            "corr_to_rep_in_build_window": 0.4,
        }
    )
    assert audit.assign_unstable_reason(row_weak, 0.55) == "weak_similarity_to_rep"
