"""Validation precheck is split: regime checks the directory, the others check the csv."""
from __future__ import annotations

from pathlib import Path

import pytest

from ashare_daily.daily_contract import require_signals_file, require_validation_dir


def test_regime_missing_dir_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    missing = tmp_path / "no-validation"
    with pytest.raises(SystemExit) as exc:
        require_validation_dir(missing)
    assert exc.value.code == 1
    assert "no-validation" in capsys.readouterr().err


def test_regime_does_not_require_signals_csv(tmp_path: Path) -> None:
    require_validation_dir(tmp_path)


def test_adaptive_and_listing_missing_csv_exit_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        require_signals_file(tmp_path, "2026-09")
    assert exc.value.code == 1
    assert "2026-09.signals.csv" in capsys.readouterr().err
