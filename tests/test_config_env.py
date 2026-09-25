"""PLOTLY_ROOT comes from the environment or config.env."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from ashare_daily.cli import load_plotly_root, main


def test_missing_config_exits_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.delenv("PLOTLY_ROOT", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["pool-merge"], root=tmp_path)
    assert exc.value.code == 1
    assert "config.env.example" in capsys.readouterr().err


def test_temp_config_sets_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLOTLY_ROOT", raising=False)
    out = tmp_path / "plots"
    (tmp_path / "config.env").write_text(f"PLOTLY_ROOT={out}\n", encoding="utf-8")
    assert load_plotly_root(tmp_path) == str(out)
    assert os.environ["PLOTLY_ROOT"] == str(out)


def test_preset_env_wins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    chosen = tmp_path / "from-env"
    monkeypatch.setenv("PLOTLY_ROOT", str(chosen))
    (tmp_path / "config.env").write_text("PLOTLY_ROOT=/somewhere/else\n", encoding="utf-8")
    assert load_plotly_root(tmp_path) == str(chosen)
