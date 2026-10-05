"""Synthetic tests for the cluster-map recall auditor."""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ashare_daily.pool.builder.data_loading import load_data, shape_loaded_panels
from ashare_daily.pool.builder.recall_audit import (
    AUDIT_TARGET,
    CLONE_DIAGNOSTIC,
    CLONE_THRESHOLD,
    HORIZON,
    LOOKBACK,
    TAIL_ROWS,
    WINNER_N,
    anchor_positions,
    assign_band,
    bounded_jobs,
    clone_hit,
    decision_positions,
    jaccard,
    msg_anchor_skipped,
    msg_provider_missing,
    msg_snapshot_missing,
    pick_winners,
    recall_at,
    run_payloads,
    score_window,
    select_at,
    summarize,
    view_asof,
)


def _prices(rows: int, cols: int, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=rows)
    shock = rng.normal(0, 0.01, size=(rows, cols))
    close = 100 * np.exp(np.cumsum(shock, axis=0))
    vol = rng.integers(1_000, 5_000, size=(rows, cols)).astype(float)
    columns = [f"S{i:02d}" for i in range(cols)]
    return (
        pd.DataFrame(close, index=dates, columns=columns),
        pd.DataFrame(vol, index=dates, columns=columns),
    )


def test_load_data_ffill_defaults_true_and_false_keeps_nan():
    assert inspect.signature(load_data).parameters["ffill"].default is True
    dates = pd.bdate_range("2024-01-01", periods=4)
    close = pd.DataFrame({"AAA": [1.0, np.nan, np.nan, 4.0]}, index=dates)
    vol = pd.DataFrame({"AAA": [10.0, 10.0, 10.0, 10.0]}, index=dates)
    filled, _ = shape_loaded_panels(close, vol)
    raw, _ = shape_loaded_panels(close, vol, ffill=False)
    assert filled["AAA"].iloc[1] == 1.0
    assert np.isnan(raw["AAA"].iloc[1])


def test_view_asof_length_and_volume_index():
    close, vol = _prices(400, 3)
    asof = close.index[300]
    window, window_vol = view_asof(close, vol, str(asof.date()))
    assert len(window) == TAIL_ROWS
    assert window.index[-1] == asof
    assert window_vol.index.equals(window.index)


def test_view_asof_ignores_price_after_asof():
    close, vol = _prices(400, 2)
    asof = close.index[300]
    before, _ = view_asof(close, vol, str(asof.date()))
    close = close.copy()
    close.iloc[301, 0] = close.iloc[301, 0] * 10
    after, _ = view_asof(close, vol, str(asof.date()))
    assert before.equals(after)


def test_stale_price_before_tail_stays_ineligible():
    close, vol = _prices(400, 1)
    asof = close.index[300]
    close = close.copy()
    close.iloc[:, 0] = np.nan
    close.iloc[300 - TAIL_ROWS, 0] = 10.0
    window, _ = view_asof(close, vol, str(asof.date()))
    assert window["S00"].isna().all()


def test_anchor_grid_step_and_decision_subset():
    positions = anchor_positions(400, lookback=LOOKBACK, forward=HORIZON, step=5)
    decision = decision_positions(positions, origin=LOOKBACK, decision_step=10)
    assert positions[1] - positions[0] == 5
    assert decision[1] - decision[0] == 10
    assert set(decision).issubset(positions)


def test_short_history_name_never_enters_winners():
    dates = pd.bdate_range("2024-01-01", periods=30)
    close = pd.DataFrame({"LONG": np.linspace(10, 11, len(dates)), "SHORT": np.linspace(10, 40, len(dates))}, index=dates)
    from ashare_daily.pool.builder.audit import compute_forward_metrics

    forward = compute_forward_metrics(close, str(dates[10].date()), [HORIZON])
    winners = pick_winners(forward, {"LONG"}, HORIZON)
    assert winners is None or "SHORT" not in set(winners["code"])


def test_fewer_than_50_winners_skips():
    frame = pd.DataFrame({"code": [f"C{i}" for i in range(10)], "fwd_ret_10d": np.linspace(0.1, 0.2, 10)})
    assert pick_winners(frame, set(frame["code"]), HORIZON) is None


def test_clone_085_counts_and_075_does_not():
    codes = ["W", "P"]
    corr = pd.DataFrame([[1.0, 0.85], [0.85, 1.0]], index=codes, columns=codes)
    winners = pd.DataFrame({"code": ["W"], "fwd_ret_10d": [0.2]})
    assert clone_hit("W", {"P"}, corr, CLONE_THRESHOLD)
    assert recall_at(winners, {"P"}, corr, CLONE_THRESHOLD) == 1.0
    loose = corr.copy()
    loose.loc["W", "P"] = 0.75
    loose.loc["P", "W"] = 0.75
    assert not clone_hit("W", {"P"}, loose, CLONE_THRESHOLD)
    assert clone_hit("W", {"P"}, loose, CLONE_DIAGNOSTIC)


def test_empty_winner_mean_is_not_a_band():
    summary = summarize(
        [{"asof": "2024-01-02", "skip": "insufficient_winners"}],
        decision_asofs={"2024-01-02"},
        snapshot_codes=None,
        trend_as_of=None,
        band_scope="full",
    )
    assert summary["status"] == "failed"
    assert summary["band"] is None


def test_dominance_forces_raise_target():
    band, dominated = assign_band(0.90, [0.88])
    assert dominated
    assert band == "raise_target"


def test_select_at_target_is_literal_100(monkeypatch):
    captured = {}

    def fake(returns, volume_df, **kwargs):
        captured.update(kwargs)
        codes = list(returns.columns)
        clusters = pd.Series(1, index=codes)
        return None, clusters, pd.DataFrame(), codes[:1], {}

    import ashare_daily.pool.builder.clustering as clustering

    monkeypatch.setattr(clustering, "cluster_and_select", fake)
    close, vol = _prices(TAIL_ROWS + 5, 4)
    asof = str(close.index[LOOKBACK].date())
    os.environ["POOL_TARGET_COUNT"] = "130"
    try:
        selected = select_at(close, vol, asof)
    finally:
        os.environ.pop("POOL_TARGET_COUNT", None)
    assert captured["target_count"] == AUDIT_TARGET == 100
    assert captured["detone_corr"] is False
    assert captured["liquidity_gate_enabled"] is False
    assert selected["selected"]


def test_select_at_matches_direct_call():
    from ashare_daily.pool.builder import constants as C
    from ashare_daily.pool.builder.clustering import cluster_and_select
    from ashare_daily.pool.builder.selector import _build_diagnostic_view

    close, vol = _prices(TAIL_ROWS + 15, 6, seed=1)
    asof = str(close.index[LOOKBACK + 5].date())
    audited = select_at(close.loc[:asof], vol.loc[:asof], asof)
    window, window_vol = view_asof(close.loc[:asof], vol.loc[:asof], asof)
    view = _build_diagnostic_view(window, window_vol)
    eligible = view["eligible_codes"]
    _z, _clusters, _df, selected, _reasons = cluster_and_select(
        view["returns"],
        view["cluster_vol"],
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        force_keep=list(C.ALWAYS_KEEP_CODES),
        target_count=100,
        history_days=view["recent_valid_days"].reindex(eligible).fillna(0),
        corr=view["corr"],
        overlap=view["overlap"],
        coverage_rescue=C.COVERAGE_RESCUE_ENABLED,
        coverage_rescue_max=C.COVERAGE_RESCUE_MAX if C.COVERAGE_RESCUE_ENABLED else 0,
        detone_corr=False,
        merge_low_cohesion=C.MERGE_LOW_COHESION_ENABLED,
        median_dollar_volume=None,
        liquidity_gate_enabled=False,
        min_dollar_volume=C.SELECTION_MIN_DOLLAR_VOLUME,
        singleton_min_dollar_volume=C.SINGLETON_MIN_DOLLAR_VOLUME,
        singleton_liquidity_mult=C.SINGLETON_LIQUIDITY_MULT,
        liquidity_apply_mode=C.SELECTION_LIQUIDITY_APPLY,
        liquidity_hot_rank_cutoff=C.LIQUIDITY_HOT_RANK_CUTOFF,
    )
    assert audited["selected"] == list(selected)


def test_score_window_skips_short_forward_names_and_keeps_clone_column():
    rows = TAIL_ROWS + HORIZON
    close, vol = _prices(rows, 3, seed=2)
    asof = str(close.index[LOOKBACK].date())
    # Three names cannot fill a 50-name winner set.
    result = score_window(close, vol, asof)
    assert result["skip"] == "insufficient_winners"
    assert "finite forward return" in result["message"]
    assert "omitted from the decision mean" in result["message"]


def test_jobs_cap_and_worker_error_propagates():
    cpu = os.cpu_count() or 1
    assert bounded_jobs(cpu + 10, cpu=cpu) == cpu
    with pytest.raises(ValueError, match="--jobs"):
        bounded_jobs(0, cpu=cpu)
    close, vol = _prices(10, 2)
    with pytest.raises(ValueError, match="fewer than"):
        run_payloads([{"close": close, "volume": vol, "asof": str(close.index[-1].date()), "seed": 0}], 1)


def test_pool_worker_module_is_the_package():
    from ashare_daily.pool.builder.recall_audit import _pool_worker

    worker = _pool_worker()
    assert worker.__module__ == "ashare_daily.pool.builder.recall_audit"


def test_spawn_jobs_match_sequential():
    rows = TAIL_ROWS + HORIZON
    close, vol = _prices(rows, 4, seed=3)
    asof = str(close.index[LOOKBACK].date())
    payload = {"close": close, "volume": vol, "asof": asof, "seed": 0}
    one = run_payloads([payload], 1)
    two = run_payloads([payload, payload], 2)
    assert one[0]["skip"] == two[0]["skip"] == two[1]["skip"]


def test_env_leak_subprocess_keeps_target_100():
    script = (
        "import os\n"
        "os.environ['POOL_TARGET_COUNT'] = '130'\n"
        "os.environ['POOL_DIST_T'] = '0.99'\n"
        "import ashare_daily.pool.builder.constants as constants\n"
        "assert constants.TARGET_SELECTED_COUNT == 130\n"
        "assert constants.DIST_T == 0.99\n"
        "from ashare_daily.pool.builder.recall_audit import (\n"
        "    AUDIT_TARGET, drop_builder_modules, scrub_pool_env, select_at,\n"
        ")\n"
        "assert AUDIT_TARGET == 100\n"
        "scrub_pool_env()\n"
        "drop_builder_modules()\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import ashare_daily.pool.builder.clustering as clustering\n"
        "captured = {}\n"
        "def fake(returns, volume_df, **kwargs):\n"
        "    captured.update(kwargs)\n"
        "    codes = list(returns.columns)\n"
        "    clusters = pd.Series(1, index=codes)\n"
        "    return None, clusters, pd.DataFrame(), codes[:1], {}\n"
        "clustering.cluster_and_select = fake\n"
        "rows = 258\n"
        "dates = pd.bdate_range('2020-01-01', periods=rows)\n"
        "rng = np.random.default_rng(0)\n"
        "shock = rng.normal(0, 0.01, size=(rows, 4))\n"
        "close = pd.DataFrame(100 * np.exp(np.cumsum(shock, axis=0)), index=dates, columns=list('ABCD'))\n"
        "vol = pd.DataFrame(np.full((rows, 4), 1000.0), index=dates, columns=close.columns)\n"
        "select_at(close, vol, str(dates[252].date()))\n"
        "assert captured['target_count'] == 100\n"
        "assert captured['t'] == 0.60\n"
        "assert captured['detone_corr'] is False\n"
        "assert captured['liquidity_gate_enabled'] is False\n"
        "print('ok')\n"
    )
    env = os.environ.copy()
    env["POOL_TARGET_COUNT"] = "130"
    env["PYTHONPATH"] = "src" + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_error_text_names_the_fix():
    assert "provider_uri" in msg_provider_missing("/missing")
    assert "Fix:" in msg_provider_missing("/missing")
    assert "omitted from the decision mean" in msg_anchor_skipped("2024-01-02")
    text = msg_snapshot_missing(Path("/tmp/stock_cluster_mapping_selected_snapshot.csv"))
    assert "stock_cluster_mapping_selected_snapshot.csv" in text
    assert "Fix:" in text


def test_jaccard_is_intersection_over_union():
    assert jaccard({"A", "B"}, {"B", "C"}) == pytest.approx(1 / 3)


@pytest.mark.recall_qlib
def test_recall_qlib_snapshot_or_skip():
    from ashare_daily.pool.builder.recall_audit import (
        default_snapshot_metadata_path,
        default_snapshot_path,
        read_snapshot_codes,
        read_trend_as_of,
    )

    path = default_snapshot_path()
    if not path.is_file():
        pytest.skip(msg_snapshot_missing(path))
    codes = read_snapshot_codes(path)
    trend = read_trend_as_of(default_snapshot_metadata_path())
    assert codes
    assert trend == "2026-06-30"
