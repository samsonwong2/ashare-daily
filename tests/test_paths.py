"""paths.repo_root, empty paths, and fail-loud json."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ashare_daily.paths import EXAMPLE_CONFIG_PATH, _as_path, load_paths, repo_root


def test_missing_json_names_example(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.json"
    with pytest.raises(FileNotFoundError) as exc:
        load_paths(missing)
    message = str(exc.value)
    assert "production_regime_switch_ewma_shrink.json.example" in message
    assert EXAMPLE_CONFIG_PATH.name in message


def test_present_json_wins_over_public_defaults(tmp_path: Path) -> None:
    config = tmp_path / "local.json"
    config.write_text(
        json.dumps({"paths": {"provider_uri": str(tmp_path / "qlib"), "temp_dir": str(tmp_path / "out")}}),
        encoding="utf-8",
    )
    paths = load_paths(config)
    assert paths["provider_uri"] == str(tmp_path / "qlib")
    assert paths["temp_dir"] == str(tmp_path / "out")
    assert "etf_risk_report_script" in paths


def test_missing_risk_script_path_raises() -> None:
    from ashare_daily.pool.builder.cli import _require_risk_path

    with pytest.raises(FileNotFoundError) as exc:
        _require_risk_path(Path("/no/such/generate_fat_tail_risk_report.py"), "etf_risk_report_script")
    assert "etf_risk_report_script" in str(exc.value)


def test_repo_root_is_pyproject_dir(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    package = tmp_path / "src" / "ashare_daily"
    package.mkdir(parents=True)
    start = package / "paths.py"
    start.write_text("", encoding="utf-8")
    assert repo_root(start) == tmp_path


def test_empty_path_is_not_cwd() -> None:
    assert _as_path("") is None
    assert _as_path("   ") is None
