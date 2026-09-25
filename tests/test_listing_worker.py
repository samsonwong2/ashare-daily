"""Listing workers are a module invocation, and a missing config dir stops first."""
from __future__ import annotations

from pathlib import Path

import pytest

from ashare_daily.daily_contract import listing_worker_prefix, require_config_source
from ashare_daily.plots.run_from_listing_sharded import build_worker_argv


def test_worker_command_is_module() -> None:
    prefix = listing_worker_prefix("python")
    assert "-m" in prefix
    assert "ashare_daily.plots.plot_adaptive_from_listing" in prefix
    assert not any("decision_pack/scripts" in part for part in prefix)


def test_build_worker_argv_matches_prefix() -> None:
    class Args:
        python = "python"
        as_of = "2026-09-25"
        start_date = None
        end_date = None
        train_cutoff = None
        config_source_dir = None
        validation_dir = None
        cluster_mapping = None
        live_snapshot = None
        retrain = False
        fail_fast = False
        hrp_membership_csv = None
        disable_ru_diag = True
        fair_path_extra_trail_years = None
        incremental_mode = "off"
        incremental_from = None
        incremental_close_tol = None
        train_cache_dir = None
        listing_cache = None
        no_train_cache = False
        no_listing_cache = False

    cmd = build_worker_argv(Args(), shard_codes_list=["SH600000"], shard_out=Path("/tmp/out"))
    assert "-m" in cmd
    assert "ashare_daily.plots.plot_adaptive_from_listing" in cmd
    assert not any("decision_pack/scripts" in part for part in cmd)


def test_missing_config_source_exits_before_workers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "20260925all_adaptive_stock"
    with pytest.raises(SystemExit) as exc:
        require_config_source(missing)
    assert exc.value.code == 1
    assert "20260925all_adaptive_stock" in capsys.readouterr().err
