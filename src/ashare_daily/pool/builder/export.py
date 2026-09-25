"""CSV mapping exporter and metadata writer."""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd

from . import constants as C

MAPPING_BASE_COLUMNS = ["name", "code", "cluster", "selected", "reason", "n"]
MAPPING_TREND_COLUMNS = ["weight", "close"]
MAPPING_EXPORT_COLUMNS = MAPPING_BASE_COLUMNS + MAPPING_TREND_COLUMNS


def export_mapping(
    all_codes: Iterable[str],
    clusters: pd.Series | None,
    selected: Iterable[str],
    reasons: dict[str, str],
    code_name_map: dict[str, str],
    out_csv: str,
    selected_csv: str | None = None,
    cluster_df: pd.DataFrame | None = None,
    trend_df: pd.DataFrame | None = None,
    diagnostics_df: pd.DataFrame | None = None,
    diagnostics_csv: str | None = None,
) -> str:
    all_codes = list(dict.fromkeys(list(all_codes)))
    selected_set = set(selected)
    df = pd.DataFrame({"code": all_codes})
    if clusters is not None:
        df["cluster"] = df["code"].map(clusters)
    else:
        df["cluster"] = np.nan
    df["selected"] = df["code"].isin(selected_set)
    df["reason"] = df["code"].map(lambda c: reasons.get(c, ""))
    df["name"] = df["code"].map(
        lambda c: code_name_map.get(str(c).strip())
        or code_name_map.get(str(c).strip().upper())
        or (code_name_map.get(str(c)[2:]) if str(c).upper().startswith(("SH", "SZ")) else "")
    )

    if cluster_df is not None and not cluster_df.empty and "n" in cluster_df.columns:
        cluster_sizes = cluster_df.set_index("cluster")["n"]
        df["n"] = df["cluster"].map(cluster_sizes)
    else:
        df["n"] = np.nan

    if trend_df is not None and not trend_df.empty:
        trend_part = trend_df.drop_duplicates(subset=["code"], keep="first")
        df = df.merge(trend_part, on="code", how="left")
    else:
        for col in MAPPING_TREND_COLUMNS:
            df[col] = np.nan if col != "weight" else 0.0

    for col in MAPPING_EXPORT_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan if col != "weight" else 0.0

    df = df[MAPPING_EXPORT_COLUMNS]
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Wrote mapping to", out_csv)

    if selected_csv:
        df[df["selected"]].to_csv(selected_csv, index=False, encoding="utf-8-sig")
        print("Wrote selected-only mapping to", selected_csv)

    diag_path = diagnostics_csv or C.DIAGNOSTICS_CSV
    if diagnostics_df is not None and not diagnostics_df.empty:
        diagnostics_df.to_csv(diag_path, index=False, encoding="utf-8-sig")
        print("Wrote diagnostics to", diag_path)

    return out_csv


def export_pool_risk_csv(source_csv: str | None, out_csv: str | None = None) -> str | None:
    if not source_csv:
        return None
    if not os.path.exists(source_csv):
        print("[INFO] pool risk source missing, skipped:", source_csv)
        return None

    target_path = out_csv or os.path.join(C.OUT_DIR, "stock_cluster_mapping_selected_pool_risk.csv")
    for enc in ("gb18030", "utf-8-sig", "utf-8"):
        try:
            risk_df = pd.read_csv(source_csv, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("utf-8", b"", 0, 1, "unable to decode pool risk CSV")

    risk_df.to_csv(target_path, index=False, encoding="utf-8-sig")
    print("Wrote pool risk to", target_path)
    return target_path
