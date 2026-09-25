"""Pool filter config is read from configs/ at the checkout root."""
from __future__ import annotations

from pathlib import Path

from ashare_daily.pool.builder.config import _config_path, load_shared_config
from ashare_daily.paths import repo_root


def test_loader_path_is_configs() -> None:
    assert _config_path() == repo_root() / "configs" / "shared_filter_config.json"


def test_explicit_path_loads(tmp_path: Path) -> None:
    path = tmp_path / "shared_filter_config.json"
    path.write_text('{"ok": true}\n', encoding="utf-8")
    assert load_shared_config(path)["ok"] is True
