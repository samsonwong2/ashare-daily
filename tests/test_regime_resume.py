"""Regime .done cleanup does not touch a real shard."""
from __future__ import annotations

from pathlib import Path

from ashare_daily.daily_contract import clear_regime_shards


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
