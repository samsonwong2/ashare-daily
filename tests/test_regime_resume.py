"""Regime .done cleanup does not touch a real shard."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_daily.daily_contract import clear_regime_shards
from ashare_daily.regime.backtest_regime_transition_signals import (
    day_already_scored,
    month_backfill_plan,
)


def _seed(shard: Path, month: str) -> None:
    shard.mkdir()
    (shard / f"{month}.done").write_text("done", encoding="utf-8")
    (shard / f"{month}.signals.csv").write_text("signals", encoding="utf-8")


def test_default_keeps_signals(tmp_path: Path) -> None:
    shard = tmp_path / "shards"
    _seed(shard, "2026-09")
    clear_regime_shards(shard, "2026-09", force=False)
    assert not (shard / "2026-09.done").exists()
    assert (shard / "2026-09.signals.csv").is_file()


def test_skip_rebuild_same_as_default(tmp_path: Path) -> None:
    shard = tmp_path / "shards"
    _seed(shard, "2026-09")
    clear_regime_shards(shard, "2026-09", force=False)
    assert not (shard / "2026-09.done").exists()
    assert (shard / "2026-09.signals.csv").is_file()


def test_force_deletes_both(tmp_path: Path) -> None:
    shard = tmp_path / "shards"
    _seed(shard, "2026-09")
    clear_regime_shards(shard, "2026-09", force=True)
    assert not (shard / "2026-09.done").exists()
    assert not (shard / "2026-09.signals.csv").exists()


def test_backfill_plan_scores_missing_codes_across_the_month() -> None:
    days = pd.bdate_range("2026-09-01", "2026-09-04")
    score_days = days[:0]
    plan = month_backfill_plan(["AAA", "BBB"], days, score_days, {"AAA"})
    assert list(plan) == ["BBB"]
    assert list(plan["BBB"]) == list(days)


def test_day_already_scored_requires_every_code(tmp_path: Path) -> None:
    shard = tmp_path / "shard.csv"
    signals = tmp_path / "signals.csv"
    rows = "as_of,code\n2026-09-30,AAA\n"
    shard.write_text(rows, encoding="utf-8")
    signals.write_text(rows, encoding="utf-8")
    assert day_already_scored(shard, signals, "2026-09-30", ["AAA"])
    assert not day_already_scored(shard, signals, "2026-09-30", ["AAA", "BBB"])
