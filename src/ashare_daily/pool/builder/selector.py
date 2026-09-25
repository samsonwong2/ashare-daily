"""High-level :func:`select_pool` pipeline.

Mirrors ``2_filter/0.2_cluster_select_from_dendrogram.py::main``. Writes:

- ``cluster_mapping.csv`` (full diagnostic mapping, mainline rule)
- ``cluster_mapping_selected.csv`` (selected rows only, mainline rule)
- ``cluster_mapping_exp.csv`` / ``cluster_mapping_selected_exp.csv``
  (benchmark rule mapping, when different from mainline)
- ``cluster_mapping_selected_metadata.json``
- ``dendrogram_selected_reps.png`` / ``.svg``
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import constants as C
from .clustering import (
    cluster_and_select,
    pairwise_overlap_matrices,
    prepare_recent_window,
    summarize_overlap_counts,
)
from .config import load_shared_config, write_json
from .data_loading import build_code_name_map, ensure_stock_list_csv, load_data, load_fund_list_universe
from .dendrogram import plot_selected_dendrogram
from .export import export_mapping
from .trend_board import DEFAULT_BENCHMARK_CODE, build_mapping_trend_frame
from .liquidity_gate import build_liquidity_diagnostics_with_clusters, compute_median_dollar_volume
from .universe_prescreen import prescreen_universe


def _build_diagnostic_view(
    close: pd.DataFrame,
    vol: pd.DataFrame | None,
) -> dict[str, object]:
    """Build row-level diagnostics and clustering inputs for a given universe."""
    _, recent_vol, returns_raw = prepare_recent_window(close, vol, C.CLUSTER_LOOKBACK_DAYS)
    recent_valid_days = returns_raw.notna().sum().astype(int)
    available_history_days = close.notna().sum().astype(int)
    eligible_for_cluster = recent_valid_days >= C.MIN_CLUSTER_HISTORY_DAYS
    eligible_for_selection = recent_valid_days >= C.MIN_SELECTION_HISTORY_DAYS
    diagnostics_df = pd.DataFrame(
        {
            "code": close.columns,
            "available_history_days": close.columns.map(lambda c: int(available_history_days.get(c, 0))),
            "recent_valid_days": close.columns.map(lambda c: int(recent_valid_days.get(c, 0))),
            "eligible_for_cluster": close.columns.map(lambda c: bool(eligible_for_cluster.get(c, False))),
            "eligible_for_selection": close.columns.map(lambda c: bool(eligible_for_selection.get(c, False))),
        }
    )
    eligible_codes = diagnostics_df.loc[diagnostics_df["eligible_for_cluster"], "code"].tolist()
    overlap_columns = ["overlap_min_days", "overlap_median_days", "overlap_max_days"]
    if not eligible_codes:
        for col in overlap_columns:
            diagnostics_df[col] = 0
        empty_returns = returns_raw.iloc[:, 0:0]
        empty_vol = recent_vol.iloc[:, 0:0] if recent_vol is not None else None
        return {
            "diagnostics_df": diagnostics_df,
            "eligible_codes": eligible_codes,
            "returns": empty_returns,
            "cluster_vol": empty_vol,
            "recent_valid_days": recent_valid_days,
            "corr": None,
            "overlap": None,
        }

    returns = returns_raw[eligible_codes]
    cluster_vol = recent_vol[eligible_codes] if recent_vol is not None else None
    corr, overlap, _ = pairwise_overlap_matrices(returns, C.MIN_PAIR_OVERLAP_DAYS, C.LOW_OVERLAP_DISTANCE)
    overlap_summary = summarize_overlap_counts(overlap)
    diagnostics_df = diagnostics_df.merge(overlap_summary, on="code", how="left")
    diagnostics_df[overlap_columns] = diagnostics_df[overlap_columns].fillna(0)
    return {
        "diagnostics_df": diagnostics_df,
        "eligible_codes": eligible_codes,
        "returns": returns,
        "cluster_vol": cluster_vol,
        "recent_valid_days": recent_valid_days,
        "corr": corr,
        "overlap": overlap,
    }


def select_pool(
    *,
    shared_config_path: str | None = None,
    inception_cutoff: str | None = None,
    auto_selected_suffix: str | None = None,
    refresh_stock_list: bool = False,
) -> dict:
    """Run the full selection pipeline and return a metadata dict."""
    cfg = load_shared_config(shared_config_path)
    provider_uri = cfg["provider_uri"]
    test_period = tuple(cfg["test_period"])
    universe_type = str(cfg.get("universe_type", "stock")).strip().lower()
    fund_list_csv = cfg.get("fund_list_csv")
    metadata_csv = cfg.get("stock_list_csv") or fund_list_csv
    universe_file = cfg.get("universe_file")
    max_universe_size = int(cfg.get("max_universe_size", C.MAX_UNIVERSE_SIZE) or 0)
    if not universe_file:
        universe_file = str(Path(provider_uri).expanduser() / "instruments" / "all.txt")
    if metadata_csv:
        ensure_stock_list_csv(
            metadata_csv,
            universe_file=universe_file,
            refresh=refresh_stock_list,
        )

    if inception_cutoff is None:
        inception_cutoff = test_period[1]  # match legacy default

    effective_manual_excludes: list[str] = list(C.ALWAYS_DROP_CODES)
    exclude_types: set[str] = set()
    cluster_excludes: list[str] = list(C.ALWAYS_DROP_CLUSTER_CODES)

    code_map, excluded_codes, selected_codes = build_code_name_map(
        universe_file,
        exclude_types,
        None,
        manual_excludes=effective_manual_excludes,
        cluster_excludes=cluster_excludes,
        cluster_reference_map_csv=C.CLUSTER_REFERENCE_MAP_CSV,
        name_csv=metadata_csv,
    )
    universe_df, full_code_map = load_fund_list_universe(
        universe_file,
        name_csv=metadata_csv,
    )
    code_map.update(full_code_map)
    excluded_prefix_count = 0
    if C.STOCK_EXCLUDE_CODE_PREFIXES:
        exclude_prefixes = tuple(prefix.upper() for prefix in C.STOCK_EXCLUDE_CODE_PREFIXES)
        before_count = len(selected_codes)
        selected_codes = [code for code in selected_codes if not str(code).upper().startswith(exclude_prefixes)]
        universe_df = universe_df[
            ~universe_df["code"].astype(str).str.upper().str.startswith(exclude_prefixes)
        ].copy()
        excluded_prefix_count = before_count - len(selected_codes)
        if excluded_prefix_count:
            print(
                f"Excluded {excluded_prefix_count} stock instruments with prefixes "
                f"{','.join(exclude_prefixes)}"
            )
    if max_universe_size > 0 and len(selected_codes) > max_universe_size:
        raise RuntimeError(
            f"Instrument universe has {len(selected_codes)} codes from {universe_file}, "
            f"exceeding max_universe_size={max_universe_size}. Use a narrower qlib "
            "instrument file such as csi300/csi500/csi800, or raise POOL_MAX_UNIVERSE_SIZE knowingly."
        )
    print(f"Loaded {len(selected_codes)} {universe_type} instruments from {universe_file}")
    initial_universe_count = len(selected_codes)

    force_keep_codes: list[str] = list(C.ALWAYS_KEEP_CODES)

    close, vol = load_data(
        selected_codes,
        provider_uri=provider_uri,
        test_period=test_period,
        excluded_codes=excluded_codes.union(effective_manual_excludes),
    )
    prescreen_diagnostics_df: pd.DataFrame | None = None
    prescreen_universe_count: int | None = None
    if C.PRESCREEN_ENABLED and vol is not None and not vol.empty:
        prescreen_codes, prescreen_diagnostics_df = prescreen_universe(
            close.columns.tolist(),
            vol,
            code_map,
            vol_pctl=C.PRESCREEN_VOL_PCTL,
            min_per_bucket=C.PRESCREEN_MIN_PER_BUCKET,
            max_universe=C.PRESCREEN_MAX_UNIVERSE,
        )
        prescreen_universe_count = len(prescreen_codes)
        prescreen_diagnostics_df.to_csv(C.PRESCREEN_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
        print(f"Wrote prescreen diagnostics to {C.PRESCREEN_DIAGNOSTICS_CSV}")
        keep_set = set(prescreen_codes)
        close = close[[c for c in close.columns if c in keep_set]]
        vol = vol[[c for c in vol.columns if c in keep_set]]
        selected_codes = [c for c in selected_codes if c in keep_set]
        universe_df = universe_df[universe_df["code"].isin(keep_set)].copy()
    missing_provider_codes = sorted(set(selected_codes) - set(close.columns.tolist()))
    diag_view = _build_diagnostic_view(close, vol)
    diagnostics_df = diag_view["diagnostics_df"]
    eligible_codes = diag_view["eligible_codes"]
    if not eligible_codes:
        raise RuntimeError("No instrument meets the minimum recent-history requirement for clustering")
    returns = diag_view["returns"]
    cluster_vol = diag_view["cluster_vol"]
    recent_valid_days = diag_view["recent_valid_days"]
    corr = diag_view["corr"]
    overlap = diag_view["overlap"]
    reasons = {
        code: "insufficient_history_for_clustering"
        for code in diagnostics_df.loc[~diagnostics_df["eligible_for_cluster"], "code"].tolist()
    }
    for code in missing_provider_codes:
        reasons.setdefault(code, "missing_from_qlib_data")
    print("Returns shape", returns.shape)

    median_dollar_volume = (
        compute_median_dollar_volume(close, vol, C.CLUSTER_LOOKBACK_DAYS)
        if C.SELECTION_LIQUIDITY_GATE_ENABLED
        else pd.Series(dtype=float)
    )
    liquidity_diagnostics_df: pd.DataFrame | None = None

    Z, clusters, cluster_df, selected, cluster_reasons = cluster_and_select(
        returns,
        cluster_vol,
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        force_keep=force_keep_codes,
        target_count=C.TARGET_SELECTED_COUNT,
        history_days=recent_valid_days.reindex(eligible_codes).fillna(0),
        corr=corr,
        overlap=overlap,
        coverage_rescue=C.COVERAGE_RESCUE_ENABLED,
        coverage_rescue_max=C.COVERAGE_RESCUE_MAX if C.COVERAGE_RESCUE_ENABLED else 0,
        detone_corr=C.DETONE_CORR_ENABLED,
        merge_low_cohesion=C.MERGE_LOW_COHESION_ENABLED,
        median_dollar_volume=median_dollar_volume if C.SELECTION_LIQUIDITY_GATE_ENABLED else None,
        liquidity_gate_enabled=C.SELECTION_LIQUIDITY_GATE_ENABLED,
        min_dollar_volume=C.SELECTION_MIN_DOLLAR_VOLUME,
        singleton_min_dollar_volume=C.SINGLETON_MIN_DOLLAR_VOLUME,
        singleton_liquidity_mult=C.SINGLETON_LIQUIDITY_MULT,
        liquidity_apply_mode=C.SELECTION_LIQUIDITY_APPLY,
        liquidity_hot_rank_cutoff=C.LIQUIDITY_HOT_RANK_CUTOFF,
    )
    reasons.update(cluster_reasons)
    if C.SELECTION_LIQUIDITY_GATE_ENABLED and not clusters.empty and not median_dollar_volume.empty:
        code_to_cluster = clusters.to_dict()
        cluster_member_counts = dict(zip(cluster_df["cluster"], cluster_df["n"])) if not cluster_df.empty else {}
        liquidity_diagnostics_df = build_liquidity_diagnostics_with_clusters(
            median_dollar_volume.reindex(list(returns.columns)).dropna(),
            cluster_member_counts,
            code_to_cluster,
            min_dollar=C.SELECTION_MIN_DOLLAR_VOLUME,
            singleton_min_dollar=C.SINGLETON_MIN_DOLLAR_VOLUME,
            singleton_mult=C.SINGLETON_LIQUIDITY_MULT,
            apply_mode=C.SELECTION_LIQUIDITY_APPLY,
            gate_enabled=C.SELECTION_LIQUIDITY_GATE_ENABLED,
        )
        liquidity_diagnostics_df.to_csv(C.LIQUIDITY_DIAGNOSTICS_CSV, index=False, encoding="utf-8-sig")
        print(f"Wrote liquidity gate diagnostics to {C.LIQUIDITY_DIAGNOSTICS_CSV}")
    full_diagnostics_df = universe_df[["code"]].copy().merge(diagnostics_df, on="code", how="left")
    int_fill_columns = ["available_history_days", "recent_valid_days", "overlap_min_days", "overlap_max_days"]
    float_fill_columns = ["overlap_median_days"]
    bool_fill_columns = ["eligible_for_cluster", "eligible_for_selection"]
    for col in int_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = pd.to_numeric(full_diagnostics_df[col], errors="coerce").fillna(0).astype(int)
    for col in float_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = pd.to_numeric(full_diagnostics_df[col], errors="coerce").fillna(0.0)
    for col in bool_fill_columns:
        if col in full_diagnostics_df.columns:
            full_diagnostics_df[col] = full_diagnostics_df[col].eq(True)
    experiment_suffix = str(auto_selected_suffix).strip() if auto_selected_suffix else ""
    if experiment_suffix:
        mainline_selected_csv = C.dated_selected_csv_path(experiment_suffix)
        mapping_csv = C.dated_mapping_csv_path(experiment_suffix)
        metadata_out = C.dated_metadata_path(experiment_suffix)
        print(
            f"[INFO] A/B suffix={experiment_suffix!r}: mapping -> {mapping_csv}, "
            f"selected -> {mainline_selected_csv} (production CSVs not overwritten)."
        )
    else:
        mainline_selected_csv = C.CSV_SELECTED_OUT
        mapping_csv = C.CSV_OUT
        metadata_out = C.METADATA_OUT

    trend_df = build_mapping_trend_frame(
        close,
        universe_df["code"].tolist(),
        test_period[1],
        provider_uri=provider_uri,
        benchmark_code=DEFAULT_BENCHMARK_CODE,
    )
    export_mapping(
        universe_df["code"].tolist(),
        clusters,
        selected,
        reasons,
        code_map,
        out_csv=mapping_csv,
        selected_csv=mainline_selected_csv,
        cluster_df=cluster_df,
        trend_df=trend_df,
        diagnostics_df=full_diagnostics_df,
    )

    benchmark_selected: list[str] = []
    if C.BENCHMARK_RULE and str(C.BENCHMARK_RULE).strip().lower() != str(C.RULE).strip().lower():
        _, benchmark_clusters, benchmark_cluster_df, benchmark_selected, benchmark_reasons = cluster_and_select(
            returns,
            cluster_vol,
            t=C.DIST_T,
            per_cluster=C.PER_CLUSTER,
            rule=C.BENCHMARK_RULE,
            force_keep=force_keep_codes,
            target_count=C.TARGET_SELECTED_COUNT,
            history_days=recent_valid_days.reindex(eligible_codes).fillna(0),
            corr=corr,
            overlap=overlap,
            coverage_rescue=C.COVERAGE_RESCUE_ENABLED,
            coverage_rescue_max=C.COVERAGE_RESCUE_MAX if C.COVERAGE_RESCUE_ENABLED else 0,
            detone_corr=C.DETONE_CORR_ENABLED,
            merge_low_cohesion=C.MERGE_LOW_COHESION_ENABLED,
            median_dollar_volume=median_dollar_volume if C.SELECTION_LIQUIDITY_GATE_ENABLED else None,
            liquidity_gate_enabled=C.SELECTION_LIQUIDITY_GATE_ENABLED,
            min_dollar_volume=C.SELECTION_MIN_DOLLAR_VOLUME,
            singleton_min_dollar_volume=C.SINGLETON_MIN_DOLLAR_VOLUME,
            singleton_liquidity_mult=C.SINGLETON_LIQUIDITY_MULT,
            liquidity_apply_mode=C.SELECTION_LIQUIDITY_APPLY,
            liquidity_hot_rank_cutoff=C.LIQUIDITY_HOT_RANK_CUTOFF,
        )
        export_mapping(
            universe_df["code"].tolist(),
            benchmark_clusters,
            benchmark_selected,
            benchmark_reasons,
            code_map,
            out_csv=C.CSV_BENCHMARK_OUT,
            selected_csv=C.CSV_SELECTED_BENCHMARK_OUT,
            cluster_df=benchmark_cluster_df,
            trend_df=trend_df,
            diagnostics_df=full_diagnostics_df,
            diagnostics_csv=os.path.join(C.OUT_DIR, "stock_cluster_mapping_exp_diagnostics.csv"),
        )

    cluster_count = int(clusters.nunique()) if clusters is not None and not clusters.empty else 0
    selected_cluster_ids = set(clusters.loc[clusters.index.isin(selected)].tolist()) if not clusters.empty else set()
    selected_cluster_coverage_pct = (
        float(len(selected_cluster_ids) / cluster_count * 100.0) if cluster_count > 0 else 0.0
    )
    ai_compute_codes = {"SH603019", "SH688041", "SH688256"}
    ai_compute_selected = [code for code in selected if code in ai_compute_codes]

    metadata = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "universe_type": universe_type,
        "universe_file": str(universe_file),
        "metadata_csv": str(metadata_csv) if metadata_csv else None,
        "initial_universe_count": int(initial_universe_count),
        "prescreen_enabled": bool(C.PRESCREEN_ENABLED),
        "prescreen_universe_count": prescreen_universe_count,
        "prescreen_diagnostics_csv": C.PRESCREEN_DIAGNOSTICS_CSV if prescreen_diagnostics_df is not None else None,
        "prescreen_max_universe": int(C.PRESCREEN_MAX_UNIVERSE),
        "prescreen_vol_pctl": float(C.PRESCREEN_VOL_PCTL),
        "prescreen_min_per_bucket": int(C.PRESCREEN_MIN_PER_BUCKET),
        "detone_corr_enabled": bool(C.DETONE_CORR_ENABLED),
        "merge_low_cohesion_enabled": bool(C.MERGE_LOW_COHESION_ENABLED),
        "merge_corr_floor": float(C.MERGE_CORR_FLOOR),
        "merge_count": int(getattr(cluster_df, "attrs", {}).get("merge_count", 0)),
        "singleton_merge_count": int(getattr(cluster_df, "attrs", {}).get("singleton_merge_count", 0)),
        "merge_singletons_enabled": bool(C.MERGE_SINGLETONS_ENABLED),
        "merge_singleton_corr": float(C.MERGE_SINGLETON_CORR),
        "cross_cluster_rank_mode": str(C.CROSS_CLUSTER_RANK),
        "cross_low_corr_weight": float(C.CROSS_LOW_CORR_WEIGHT),
        "cross_rep_by_score": bool(C.CROSS_REP_BY_SCORE),
        "selection_liquidity_gate_enabled": bool(C.SELECTION_LIQUIDITY_GATE_ENABLED),
        "selection_liquidity_apply": str(C.SELECTION_LIQUIDITY_APPLY),
        "selection_min_dollar_volume": float(C.SELECTION_MIN_DOLLAR_VOLUME),
        "singleton_min_dollar_volume": float(C.SINGLETON_MIN_DOLLAR_VOLUME),
        "singleton_liquidity_mult": float(C.SINGLETON_LIQUIDITY_MULT),
        "liquidity_hot_rank_cutoff": int(C.LIQUIDITY_HOT_RANK_CUTOFF),
        "liquidity_diagnostics_csv": C.LIQUIDITY_DIAGNOSTICS_CSV if liquidity_diagnostics_df is not None else None,
        "liquidity_floor_drop_count": int(getattr(cluster_df, "attrs", {}).get("liquidity_floor_drop_count", 0)),
        "sz300757_selected": bool("SZ300757" in set(selected)),
        "ai_compute_selected_count": int(len(ai_compute_selected)),
        "ai_compute_selected_codes": ai_compute_selected,
        "coverage_rescue_enabled": bool(C.COVERAGE_RESCUE_ENABLED),
        "coverage_rescue_max": int(C.COVERAGE_RESCUE_MAX),
        "cluster_count": cluster_count,
        "selected_cluster_coverage_pct": round(selected_cluster_coverage_pct, 2),
        "excluded_code_prefixes": list(C.STOCK_EXCLUDE_CODE_PREFIXES),
        "excluded_prefix_count": int(excluded_prefix_count),
        "always_keep_codes": list(C.ALWAYS_KEEP_CODES),
        "always_drop_codes": list(C.ALWAYS_DROP_CODES),
        "always_drop_cluster_codes": list(C.ALWAYS_DROP_CLUSTER_CODES),
        "production_rule": C.RULE,
        "production_scheme": C.PRODUCTION_MULTI_REP_MAINLINE_SCHEME,
        "benchmark_rule": C.BENCHMARK_RULE,
        "benchmark_scheme": C.PRODUCTION_MULTI_REP_BENCHMARK_SCHEME,
        "deprecated_schemes": list(C.DEPRECATED_MULTI_REP_WEIGHT_SCHEMES),
        "dist_t": float(C.DIST_T),
        "linkage": str(C.CLUSTER_LINKAGE),
        "rep_pick_mode": str(C.REP_PICK_MODE),
        "rep_min_center_corr": float(C.REP_MIN_CENTER_CORR),
        "scoring_mode": str(C.SCORING_MODE),
        "per_cluster": int(C.PER_CLUSTER),
        "target_selected_count": int(C.TARGET_SELECTED_COUNT) if C.TARGET_SELECTED_COUNT is not None else None,
        "experiment_suffix": experiment_suffix or None,
        "mapping_csv": mapping_csv,
        "cluster_lookback_days": int(C.CLUSTER_LOOKBACK_DAYS),
        "min_cluster_history_days": int(C.MIN_CLUSTER_HISTORY_DAYS),
        "min_selection_history_days": int(C.MIN_SELECTION_HISTORY_DAYS),
        "min_pair_overlap_days": int(C.MIN_PAIR_OVERLAP_DAYS),
        "mainline_selected_csv": mainline_selected_csv,
        "benchmark_selected_csv": C.CSV_SELECTED_BENCHMARK_OUT if benchmark_selected else None,
        "mainline_selected_count": int(len(selected)),
        "benchmark_selected_count": int(len(benchmark_selected)) if benchmark_selected else 0,
        "trend_as_of": str(test_period[1]),
        "trend_board_columns": list(trend_df.columns),
        "benchmark_code": DEFAULT_BENCHMARK_CODE,
        "diagnostics_csv": C.DIAGNOSTICS_CSV,
    }
    write_json(metadata_out, metadata)
    print("Wrote generation metadata to", metadata_out)

    # 谱系图叶子顺序与聚类使用的标的集一致。
    codes = list(clusters.index)
    try:
        plot_selected_dendrogram(Z, codes, set(selected), code_map, C.IMG_OUT, C.SVG_OUT)
    except RecursionError:
        print(
            f"[WARN] Skipped dendrogram plot: universe too large for scipy dendrogram "
            f"({len(codes)} leaves). Mapping CSVs were written successfully."
        )
    return metadata


if __name__ == "__main__":
    select_pool()


