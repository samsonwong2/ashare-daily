"""Tests for cluster mapping CSV export schema."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pool_builder.export import MAPPING_EXPORT_COLUMNS, export_mapping
from pool_builder.trend_board import TREND_BOARD_COLUMNS


def test_export_mapping_columns_and_no_diagnostics_on_main_csv(tmp_path):
    trend_df = pd.DataFrame(
        {
            "code": ["AAA"],
            "weight": [0.0],
            **{col: [1.0] for col in TREND_BOARD_COLUMNS if col != "weight"},
        }
    )
    trend_df["ma_stack"] = "bull"
    trend_df["vol_level"] = "mid"
    trend_df["risk_flags"] = ""
    trend_df["pnl_pct"] = None

    out_csv = tmp_path / "mapping.csv"
    diag_csv = tmp_path / "diag.csv"
    cluster_df = pd.DataFrame({"cluster": [1], "n": [3]})
    export_mapping(
        ["AAA"],
        pd.Series({"AAA": 1}, name="cluster"),
        ["AAA"],
        {"AAA": "test_reason"},
        {"AAA": "测试"},
        out_csv=str(out_csv),
        selected_csv=str(tmp_path / "selected.csv"),
        cluster_df=cluster_df,
        trend_df=trend_df,
        diagnostics_df=pd.DataFrame({"code": ["AAA"], "eligible_for_cluster": [True]}),
        diagnostics_csv=str(diag_csv),
    )

    main = pd.read_csv(out_csv)
    assert list(main.columns) == MAPPING_EXPORT_COLUMNS
    assert "mean_corr" not in main.columns
    assert "eligible_for_cluster" not in main.columns
    assert main.loc[0, "cluster"] == 1
    assert int(main.loc[0, "n"]) == 3
    assert main.loc[0, "ma_stack"] == "bull"

    diag = pd.read_csv(diag_csv)
    assert "eligible_for_cluster" in diag.columns
