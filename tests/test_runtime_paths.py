"""runtime_paths fails closed when the local json is missing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from runtime_paths import EXAMPLE_CONFIG_PATH, load_paths


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
    from fund_pool_builder.pool_builder.cli import _require_risk_path

    with pytest.raises(FileNotFoundError) as exc:
        _require_risk_path(Path("/no/such/generate_fat_tail_risk_report.py"), "etf_risk_report_script")
    assert "etf_risk_report_script" in str(exc.value)
