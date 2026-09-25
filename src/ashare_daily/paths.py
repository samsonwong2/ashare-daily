"""Runtime paths for the stock daily handbook.

Machine-specific paths live in ``configs/production_regime_switch_ewma_shrink.json``
under the top-level ``paths`` object. That file is gitignored. Copy
``configs/production_regime_switch_ewma_shrink.json.example`` and fill it in.
Import fails if the local file is missing.

Empty path strings stay empty. They do not resolve to the current directory.
HTML output uses ``PLOTLY_ROOT`` from the environment, not a directory inside
this checkout.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def repo_root(start: Path | None = None) -> Path:
    """Directory that contains ``pyproject.toml``.

    Walks parents of this file, or of ``start`` when a test passes a temp tree.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    seeds = [here] if here.is_file() else []
    for parent in [*seeds, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise FileNotFoundError(
        f"pyproject.toml not found walking from {here}. "
        "Install from the checkout that contains pyproject.toml."
    )


PROJECT_ROOT = repo_root()
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json"
EXAMPLE_CONFIG_PATH = PROJECT_ROOT / "configs" / "production_regime_switch_ewma_shrink.json.example"
VALIDATION_DIR = (
    PROJECT_ROOT
    / "runtime"
    / "decision_packs"
    / "20260901_stock"
    / "regime_transition_validation_q90_stock"
)
MODEL_CACHE_DIR = PROJECT_ROOT / "runtime" / "decision_packs" / "regime_transition_model_cache_stock"


def plotly_outputs_dir() -> Path:
    raw = os.environ.get("PLOTLY_ROOT", "").strip()
    if not raw:
        raise RuntimeError(
            "PLOTLY_ROOT is not set. Copy config.env.example and set PLOTLY_ROOT."
        )
    return Path(raw).expanduser().resolve()


def notes_dir() -> Path:
    return plotly_outputs_dir() / "notes"


def _public_defaults() -> dict[str, str]:
    """Used only to fill keys omitted from a local json that does exist."""
    provider_uri = Path("~/.qlib/qlib_data/cn_data")
    temp_dir = Path("~/ashare-daily-output")
    workspace_dir = temp_dir / "workspace"
    log_dir = workspace_dir / "logs"
    return {
        "provider_uri": str(provider_uri),
        "qlib_scripts_dir": "",
        "temp_dir": str(temp_dir),
        "workspace_dir": str(workspace_dir),
        "log_dir": str(log_dir),
        "local_library_dir": "",
        "fund_list_csv": str(temp_dir / "stock_list.csv"),
        "qlib_change_csv_dir": str(provider_uri / "change_csv"),
        "all_instruments_txt": str(provider_uri / "instruments" / "all.txt"),
        "cluster_mapping_path": str(provider_uri / "instruments" / "cluster_mapping_selected.txt"),
        "etf_risk_report_script": "",
        "etf_risk_pack_dir": "",
        "etf_risk_source_pack_dir": "",
    }


def load_paths(config_path: Path | None = None) -> dict[str, str]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    example = EXAMPLE_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(
            f"missing local config {path}. Copy {example.name} and fill in paths."
        )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    paths = payload.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError(f"Config paths must be a JSON object: {path}")
    expanded: dict[str, str] = {
        **_public_defaults(),
        **{str(key): str(value) for key, value in paths.items() if value is not None},
    }
    expanded["project_root"] = str(PROJECT_ROOT)
    for _ in range(8):
        changed = False
        context = dict(expanded)
        for key, value in list(expanded.items()):
            formatted = value.format(**context)
            if formatted != value:
                expanded[key] = formatted
                changed = True
        if not changed:
            break
    return expanded


def _as_path(value: str) -> Path | None:
    if not str(value).strip():
        return None
    return Path(value).expanduser().resolve()


def _under(parent: Path | None, *parts: str) -> Path | None:
    if parent is None:
        return None
    return parent.joinpath(*parts)


_PATHS = load_paths()

QLIB_PROVIDER_URI = _as_path(_PATHS["provider_uri"])
TEMP_DIR = _as_path(_PATHS["temp_dir"])
LOG_DIR = _as_path(_PATHS["log_dir"])
WORKSPACE_DIR = _as_path(_PATHS["workspace_dir"])
FUND_CRAWLER_LOG_DIR = _under(LOG_DIR, "fund_crawler")
PIPELINE_RUN_LOG_DIR = _under(LOG_DIR, "pipeline_runs")
HISTORY_DIR = _under(WORKSPACE_DIR, "history")
DOCS_DIR = HISTORY_DIR
NOTES_DIR = notes_dir
WORKSPACE_SCRIPTS_DIR = _under(WORKSPACE_DIR, "scripts")
PLOTLY_OUTPUTS_DIR = plotly_outputs_dir
FUND_LIST_CSV = _as_path(_PATHS["fund_list_csv"])
QLIB_CHANGE_CSV_DIR = _as_path(_PATHS["qlib_change_csv_dir"])
QLIB_INSTRUMENTS_DIR = _under(QLIB_PROVIDER_URI, "instruments")
ALL_INSTRUMENTS_TXT = _as_path(_PATHS["all_instruments_txt"])
CLUSTER_MAPPING_SELECTED_TXT = _as_path(_PATHS["cluster_mapping_path"])
QLIB_SCRIPTS_DIR = _as_path(_PATHS["qlib_scripts_dir"])
LOCAL_LIBRARY_DIR = _as_path(_PATHS["local_library_dir"])
ETF_RISK_REPORT_SCRIPT = _as_path(_PATHS["etf_risk_report_script"])
ETF_RISK_PACK_DIR = _as_path(_PATHS["etf_risk_pack_dir"])
ETF_RISK_SOURCE_PACK_DIR = _as_path(_PATHS["etf_risk_source_pack_dir"])


def as_str(path: str | Path) -> str:
    return str(Path(path).expanduser())
