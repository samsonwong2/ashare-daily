"""Windowing, pair overlap, multi-rep pool, trimming and clustering.

Extracted verbatim from ``2_filter/0.2_cluster_select_from_dendrogram.py``.
Public entry point is :func:`cluster_and_select`.
"""
from __future__ import annotations

import os
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from .code_utils import build_normalized_code_set, code_in_normalized_set
from .constants import (
    CLUSTER_LINKAGE,
    CLUSTER_LOOKBACK_DAYS,
    COVERAGE_RESCUE_MAX,
    COVERAGE_RESCUE_ENABLED,
    CROSS_CLUSTER_RANK,
    CROSS_LOW_CORR_WEIGHT,
    CROSS_REP_BY_SCORE,
    DETONE_CORR_ENABLED,
    DIST_T,
    FINAL_MAX_ABS_CORR_THRESHOLDS,
    INTRA_CLUSTER_MAX_ABS_CORR,
    LOW_OVERLAP_DISTANCE,
    MERGE_CORR_FLOOR,
    MERGE_LOW_COHESION_ENABLED,
    MERGE_MAX_CLUSTER_SIZE,
    MERGE_SINGLETON_CORR,
    MERGE_SINGLETONS_ENABLED,
    MIN_PAIR_OVERLAP_DAYS,
    MIN_SELECTION_HISTORY_DAYS,
    MULTI_REP_EXP_BASE,
    PER_CLUSTER,
    REP_MIN_CENTER_CORR,
    REP_PICK_MODE,
    RULE,
    SCORING_MODE,
    LIQUIDITY_HOT_RANK_CUTOFF,
    SELECTION_LIQUIDITY_GATE_ENABLED,
    SELECTION_LIQUIDITY_APPLY,
    SELECTION_MIN_DOLLAR_VOLUME,
    SINGLETON_LIQUIDITY_MULT,
    SINGLETON_MIN_DOLLAR_VOLUME,
)
from .liquidity_gate import passes_liquidity_floor


def apply_detone_to_corr(corr: pd.DataFrame, t_obs: int) -> pd.DataFrame:
    """MP denoise + detone correlation matrix (MLFAM §4.4.1)."""
    try:
        from mlfam.denoise import denoise_corr_matrix
    except ImportError:
        import sys
        from pathlib import Path

        mlfam_root = Path(__file__).resolve().parents[3] / "etf_strategy_MLFAM"
        if mlfam_root.exists() and str(mlfam_root) not in sys.path:
            sys.path.insert(0, str(mlfam_root))
        from mlfam.denoise import denoise_corr_matrix

    n_assets = len(corr.columns)
    q = float(t_obs) / float(max(n_assets, 1))
    corr_out, n_facts = denoise_corr_matrix(corr, q=q, detone=True)
    print(f"[INFO] detone corr: q={q:.3f}, n_facts={n_facts}, N={n_assets}")
    return corr_out.clip(lower=-1.0, upper=1.0)


def merge_low_cohesion_clusters(
    clusters: pd.Series,
    corr: pd.DataFrame,
    *,
    max_size: int = MERGE_MAX_CLUSTER_SIZE,
    corr_floor: float = MERGE_CORR_FLOOR,
) -> tuple[pd.Series, int]:
    """Merge small / low-cohesion clusters into nearest high-cohesion neighbor."""
    if clusters.empty or corr.empty:
        return clusters, 0

    work = clusters.copy()
    merge_count = 0
    changed = True
    while changed:
        changed = False
        cluster_stats: list[tuple[object, list[str], float, int]] = []
        for cid, members in work.groupby(work).groups.items():
            members = list(members)
            n = len(members)
            if n <= 1:
                mean_corr = 1.0
            else:
                sub = corr.reindex(index=members, columns=members).fillna(0.0).values
                mean_corr = float((sub.sum() - n) / (n * (n - 1)))
            cluster_stats.append((cid, members, mean_corr, n))

        weak = [
            (cid, members, mean_corr, n)
            for cid, members, mean_corr, n in cluster_stats
            if n <= max_size and mean_corr < corr_floor
        ]
        if not weak:
            break

        strong = [
            (cid, members, mean_corr, n)
            for cid, members, mean_corr, n in cluster_stats
            if not (n <= max_size and mean_corr < corr_floor)
        ]
        if not strong:
            break

        for weak_cid, weak_members, _, _ in sorted(weak, key=lambda x: (len(x[1]), x[0])):
            if work.loc[weak_members[0]] != weak_cid:
                continue
            best_target = None
            best_sim = -np.inf
            for strong_cid, strong_members, strong_corr, _ in strong:
                if strong_cid == weak_cid:
                    continue
                cross = corr.reindex(index=weak_members, columns=strong_members).fillna(0.0)
                if cross.empty:
                    continue
                sim = float(cross.values.mean())
                if sim > best_sim:
                    best_sim = sim
                    best_target = strong_cid
            if best_target is None:
                continue
            for code in weak_members:
                work.loc[code] = best_target
            merge_count += 1
            changed = True

    if merge_count:
        print(f"[INFO] merge low-cohesion clusters: {merge_count} merges, K={work.nunique()}")
    return work, merge_count


def merge_singleton_clusters(
    clusters: pd.Series,
    corr: pd.DataFrame,
    *,
    min_corr: float = MERGE_SINGLETON_CORR,
) -> tuple[pd.Series, int]:
    """Merge n=1 clusters into the best-correlated multi-member neighbor."""
    if clusters.empty or corr.empty:
        return clusters, 0

    work = clusters.copy()
    merge_count = 0
    changed = True
    while changed:
        changed = False
        groups = {cid: list(members) for cid, members in work.groupby(work).groups.items()}
        for cid, members in sorted(groups.items(), key=lambda item: len(item[1])):
            if len(members) != 1:
                continue
            code = members[0]
            if work.loc[code] != cid:
                continue
            best_target = None
            best_sim = -np.inf
            best_size = 0
            for other_cid, other_members in groups.items():
                if other_cid == cid or len(other_members) < 2:
                    continue
                cross = corr.reindex(index=[code], columns=other_members).fillna(0.0)
                if cross.empty:
                    continue
                sim = float(cross.values.mean())
                other_size = len(other_members)
                if sim > best_sim or (sim == best_sim and other_size > best_size):
                    best_sim = sim
                    best_target = other_cid
                    best_size = other_size
            if best_target is None or best_sim < float(min_corr):
                continue
            work.loc[code] = best_target
            merge_count += 1
            changed = True

    if merge_count:
        print(f"[INFO] merge singleton clusters: {merge_count} merges, K={work.nunique()}")
    return work, merge_count


def _rank_score(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    series = series.astype(float)
    if series.empty:
        return series
    return series.rank(method="average", pct=True)


def _corr_to_cluster_center(code: str, members: list[str], corr: pd.DataFrame) -> float:
    if not members or code not in members:
        return float("nan")
    if len(members) == 1:
        return 1.0
    if code not in corr.index:
        return float("nan")
    peers = [m for m in members if m != code and m in corr.columns]
    if not peers:
        return 1.0
    return float(corr.loc[code, peers].mean())


def _rank_members_for_rep_pick(
    members: list[str],
    returns: pd.DataFrame,
    corr: pd.DataFrame | None,
    rep_pick: str | None = None,
    min_center_corr: float | None = None,
) -> pd.Series:
    """Higher score = better representative candidate within ``members``."""
    mode = (rep_pick or REP_PICK_MODE).strip().lower()
    floor = float(min_center_corr if min_center_corr is not None else REP_MIN_CENTER_CORR)
    return_scores = _return_scores(members, returns)
    if mode == "return" or corr is None or corr.empty:
        return return_scores.sort_values(ascending=False)

    center_corrs = pd.Series({m: _corr_to_cluster_center(m, members, corr) for m in members})
    if mode == "center_corr":
        return center_corrs.sort_values(ascending=False)

    ret_rank = _rank_score(return_scores.reindex(members).fillna(0.0))
    center_rank = _rank_score(center_corrs.reindex(members).fillna(0.0))
    blend = 0.5 * ret_rank + 0.5 * center_rank
    below = center_corrs.reindex(members) < floor
    blend = blend.where(~below, blend * 0.5)
    return blend.sort_values(ascending=False)


def _return_scores(codes: Iterable[str], returns: pd.DataFrame) -> pd.Series:
    """Score candidate instruments for within-cluster representative ranking.

    Default ``cumulative_return`` (``POOL_SCORING_MODE``): full lookback window
    cumulative return ``(1+r).prod()-1`` over the clustering panel (252d typically).

    Alternatives:
      - ``sharpe_mom``: 50/50 blend of percentile-ranked Sharpe and 12-1 momentum
      - ``legacy_mean``: percentile rank of simple mean daily return
    """
    codes = list(codes)
    if not codes:
        return pd.Series(dtype=float)
    code_list = list(dict.fromkeys(codes))
    panel = returns[code_list]

    mode = SCORING_MODE
    if mode == "cumulative_return":
        total_return = (panel + 1.0).prod() - 1.0
        return total_return.reindex(code_list).fillna(-np.inf).sort_values(ascending=False)

    if mode == "legacy_mean":
        mean_return = panel.mean().reindex(code_list).fillna(0.0)
        return _rank_score(mean_return).sort_values(ascending=False)

    # sharpe_mom: annualized Sharpe over the full panel window.
    mean_daily = panel.mean()
    std_daily = panel.std(ddof=0).replace(0.0, np.nan)
    sharpe = (mean_daily / std_daily * np.sqrt(252.0)).reindex(code_list).fillna(0.0)

    # 12-1 momentum: cumulative return over t-252 ... t-21 (skip last month).
    n_rows = len(panel)
    skip_recent = min(21, max(0, n_rows - 1))
    momentum_window = panel.iloc[: n_rows - skip_recent] if skip_recent > 0 else panel
    momentum = (momentum_window + 1.0).prod() - 1.0
    momentum = momentum.reindex(code_list).fillna(0.0)

    score = 0.5 * _rank_score(sharpe) + 0.5 * _rank_score(momentum)
    return score.sort_values(ascending=False)


def _mean_abs_corr_to_peers(
    codes: list[str],
    returns: pd.DataFrame,
    corr: pd.DataFrame | None = None,
) -> pd.Series:
    """Mean pairwise |corr| of each code to all other codes (diag ignored)."""
    if not codes:
        return pd.Series(dtype=float)
    if corr is not None and not corr.empty:
        mat = corr.reindex(index=codes, columns=codes)
    else:
        mat = returns[codes].corr()
    mat = mat.astype(float)
    abs_mat = mat.abs()
    vals = abs_mat.to_numpy(dtype=float, copy=True)
    np.fill_diagonal(vals, np.nan)
    return pd.Series(np.nanmean(vals, axis=1), index=codes, dtype=float)


def _global_cross_cluster_scores(
    returns: pd.DataFrame,
    mode: str | None = None,
    corr: pd.DataFrame | None = None,
) -> pd.Series:
    """Global score for Plan B cross-cluster rank-1 ordering.

    Modes:
      - ``low_corr`` (production default): ``-mean_j≠i(|corr_ij|)`` — diversifiers only.
      - ``low_corr_return``: percentile blend of low-corr and cumulative return
        (weight ``CROSS_LOW_CORR_WEIGHT``, default 0.5/0.5).
      - ``cumulative_return`` / ``sharpe_mom``: return/momentum-tilted modes.
      - other / ``within_blend``: zeros here; caller uses within-cluster blend score.
    """
    rank_mode = (mode or CROSS_CLUSTER_RANK).strip().lower()
    codes = list(returns.columns)
    if not codes:
        return pd.Series(dtype=float)
    panel = returns[codes]
    if rank_mode in ("low_corr", "low_corr_return"):
        mean_abs = _mean_abs_corr_to_peers(codes, panel, corr=corr)
        low_score = (-mean_abs).reindex(codes).fillna(-np.inf)
        if rank_mode == "low_corr":
            return low_score
        cum = ((panel + 1.0).prod() - 1.0).reindex(codes).fillna(-np.inf)
        w = float(np.clip(CROSS_LOW_CORR_WEIGHT, 0.0, 1.0))
        return (w * _rank_score(low_score) + (1.0 - w) * _rank_score(cum)).reindex(codes).fillna(0.0)
    if rank_mode == "cumulative_return":
        return ((panel + 1.0).prod() - 1.0).reindex(codes).fillna(-np.inf)
    if rank_mode == "sharpe_mom":
        mean_daily = panel.mean()
        std_daily = panel.std(ddof=0).replace(0.0, np.nan)
        sharpe = (mean_daily / std_daily * np.sqrt(252.0)).reindex(codes).fillna(0.0)
        n_rows = len(panel)
        skip_recent = min(21, max(0, n_rows - 1))
        momentum_window = panel.iloc[: n_rows - skip_recent] if skip_recent > 0 else panel
        momentum = (momentum_window + 1.0).prod() - 1.0
        momentum = momentum.reindex(codes).fillna(0.0)
        return (0.5 * _rank_score(sharpe) + 0.5 * _rank_score(momentum)).reindex(codes).fillna(0.0)
    return pd.Series({code: 0.0 for code in codes}, dtype=float)


def _multi_rep_rank_weight(rank_index: int, scheme: str, exp_base: float = MULTI_REP_EXP_BASE) -> float:
    rank_number = int(rank_index) + 1
    if scheme == "equal":
        return 1.0
    if scheme == "harmonic":
        return 1.0 / float(rank_number)
    if scheme == "exp":
        return float(exp_base) ** float(rank_index)
    raise ValueError(f"Unsupported multi-rep weight scheme: {scheme}")


def _build_multi_rep_candidate_caps(cluster_df: pd.DataFrame, target_count: int) -> pd.DataFrame:
    quota_rows = []
    for _, row in cluster_df.iterrows():
        quota_rows.append(
            {
                "cluster": row["cluster"],
                "member_count": len(row["candidate_members"]),
                "top_return": float(row["candidate_top_return"]),
            }
        )
    quota_df = pd.DataFrame(quota_rows)
    total_members = int(quota_df["member_count"].sum()) if not quota_df.empty else 0
    if total_members <= 0:
        return pd.DataFrame(columns=["cluster", "member_count", "top_return", "raw_quota", "candidate_cap"])

    quota_df["raw_quota"] = quota_df["member_count"] / total_members * target_count
    quota_df["candidate_cap"] = np.ceil(quota_df["raw_quota"]).astype(int)
    quota_df["candidate_cap"] = quota_df[["candidate_cap", "member_count"]].min(axis=1)
    quota_df["candidate_cap"] = quota_df["candidate_cap"].clip(lower=1)

    target_candidate_count = int(target_count) + min(
        max(5, int(np.ceil(target_count * 0.15))),
        max(0, total_members - int(target_count)),
    )
    current_candidates = int(quota_df["candidate_cap"].sum())
    quota_df["remainder"] = quota_df["raw_quota"] - np.floor(quota_df["raw_quota"])
    while current_candidates < target_candidate_count:
        available = quota_df.loc[quota_df["candidate_cap"] < quota_df["member_count"]].copy()
        if available.empty:
            break
        available = available.sort_values(
            ["remainder", "member_count", "top_return", "cluster"],
            ascending=[False, False, False, True],
        )
        pick_index = available.index[0]
        quota_df.loc[pick_index, "candidate_cap"] += 1
        current_candidates += 1
    return quota_df


def _build_multi_rep_cluster_pool(
    cluster_df: pd.DataFrame,
    returns: pd.DataFrame,
    history_days: pd.Series,
    target_count: int | None,
    force_keep_codes: Iterable[str] | None = None,
    weight_scheme: str = "equal",
    exp_base: float = MULTI_REP_EXP_BASE,
    corr: pd.DataFrame | None = None,
    near_clone_corr_limit: float | None = None,
    intra_cluster_corr_limit: float | None = None,
    coverage_rescue: bool = True,
    coverage_rescue_max: int | None = None,
    rep_pick: str | None = None,
    rep_min_center_corr: float | None = None,
    cross_cluster_rank_mode: str | None = None,
    median_dollar_volume: pd.Series | None = None,
    liquidity_gate_enabled: bool | None = None,
    min_dollar_volume: float | None = None,
    singleton_min_dollar_volume: float | None = None,
    singleton_liquidity_mult: float | None = None,
    liquidity_apply_mode: str | None = None,
    liquidity_hot_rank_cutoff: int | None = None,
) -> tuple[list[str], dict[str, str], int]:
    if cluster_df is None or cluster_df.empty:
        return [], {}, 0

    target_count = int(target_count or len(returns.columns))
    history_days = history_days if history_days is not None else pd.Series(dtype=float)
    use_liq_gate = SELECTION_LIQUIDITY_GATE_ENABLED if liquidity_gate_enabled is None else bool(liquidity_gate_enabled)
    liq_min_dollar = float(min_dollar_volume if min_dollar_volume is not None else SELECTION_MIN_DOLLAR_VOLUME)
    liq_singleton_min = float(
        singleton_min_dollar_volume if singleton_min_dollar_volume is not None else SINGLETON_MIN_DOLLAR_VOLUME
    )
    liq_singleton_mult = float(singleton_liquidity_mult if singleton_liquidity_mult is not None else SINGLETON_LIQUIDITY_MULT)
    liq_apply_mode = (liquidity_apply_mode or SELECTION_LIQUIDITY_APPLY).strip().lower()
    liq_hot_rank_cutoff = int(
        liquidity_hot_rank_cutoff if liquidity_hot_rank_cutoff is not None else LIQUIDITY_HOT_RANK_CUTOFF
    )
    liquidity_floor_drop_count = 0
    liquidity_miss_reasons: dict[str, str] = {}
    work_df = cluster_df.copy()
    work_df["candidate_members"] = work_df["members"].apply(
        lambda members: [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
        or list(members)
    )
    work_df["candidate_top_return"] = work_df["candidate_members"].apply(
        lambda members: float(_return_scores(members, returns).iloc[0]) if members else -np.inf
    )

    cap_df = _build_multi_rep_candidate_caps(work_df, target_count)
    cap_map = dict(zip(cap_df["cluster"], cap_df["candidate_cap"]))
    raw_quota_map = dict(zip(cap_df["cluster"], cap_df["raw_quota"]))
    if "n" in work_df.columns:
        cluster_n_map = dict(zip(work_df["cluster"], work_df["n"]))
    else:
        cluster_n_map = {
            row["cluster"]: len(row["members"]) if isinstance(row["members"], list) else 1
            for _, row in work_df.iterrows()
        }
    rank_mode = (cross_cluster_rank_mode or CROSS_CLUSTER_RANK).strip().lower()
    global_cross_scores = _global_cross_cluster_scores(returns, mode=rank_mode, corr=corr)
    # For low_corr / low_corr_return: nominate the in-cluster member that best
    # matches the cross-cluster objective (low corr ∧ high return), instead of
    # the blend center-rep. Blend remains available via POOL_CROSS_REP_BY_SCORE=0.
    use_cross_rep = bool(CROSS_REP_BY_SCORE) and rank_mode in ("low_corr", "low_corr_return")

    # Pre-compute force_keep set so that a force_keep instrument which is already a
    # cluster member "consumes" that cluster's quota instead of letting the
    # cluster emit an extra (near-clone) representative on top of it.
    normalized_keep_for_cluster = build_normalized_code_set(force_keep_codes or [])

    selected_rows = []
    for _, row in work_df.iterrows():
        members = row["candidate_members"]
        candidate_cap = int(cap_map.get(row["cluster"], 0))
        if candidate_cap <= 0 or not members:
            continue
        if use_cross_rep:
            member_scores = (
                global_cross_scores.reindex(list(members)).fillna(-np.inf).sort_values(ascending=False)
            )
        else:
            member_scores = _rank_members_for_rep_pick(
                members,
                returns,
                corr,
                rep_pick=rep_pick,
                min_center_corr=rep_min_center_corr,
            )
        # If a force_keep instrument is a member of this cluster, treat the cluster as
        # already "covered" by that force_keep and skip emitting any additional
        # auto representatives — this prevents near-clone pairs like
        # SH518880 (force_keep gold) + SH518600 (auto-picked gold) ending up
        # together in the final pool.
        forced_in_cluster = [
            code for code in member_scores.index
            if code_in_normalized_set(code, normalized_keep_for_cluster)
        ]
        if forced_in_cluster:
            picks = forced_in_cluster
        else:
            picks = member_scores.head(candidate_cap).index.tolist()
        for rank_index, code in enumerate(picks):
            within_score = float(member_scores.get(code, 0.0))
            if rank_mode == "within_blend":
                cross_score = within_score
            else:
                cross_score = float(global_cross_scores.get(code, -np.inf))
            selected_rows.append(
                {
                    "code": code,
                    "cluster": row["cluster"],
                    "cluster_quota": float(raw_quota_map.get(row["cluster"], 0.0)),
                    "within_cluster_rank": rank_index + 1,
                    "return_score": within_score,
                    "cross_cluster_score": cross_score,
                    "rep_weight": _multi_rep_rank_weight(rank_index, weight_scheme, exp_base),
                    "short_history": float(history_days.get(code, 0.0)) < MIN_SELECTION_HISTORY_DAYS,
                }
            )

    selected_df = pd.DataFrame(selected_rows)
    if selected_df.empty:
        selected: list[str] = []
        selected_reason_map: dict[str, str] = {}
    else:
        selected_df["priority_score"] = selected_df["return_score"] * selected_df["rep_weight"]
        selected_df = selected_df.sort_values(
            [
                "priority_score",
                "return_score",
                "cluster_quota",
                "rep_weight",
                "cluster",
                "within_cluster_rank",
                "code",
            ],
            ascending=[False, False, False, False, True, True, True],
        )
        unique_selected_df = selected_df.drop_duplicates(subset=["code"], keep="first").copy()
        # Plan B: every cluster contributes rank-1 first, ordered by cross_cluster_score
        # (global low_corr_return by default; low_corr / cumulative_return for rollback).
        _plan_b_coverage = os.environ.get("POOL_PLAN_B_COVERAGE", "1") == "1"
        if _plan_b_coverage:
            rank1 = unique_selected_df[unique_selected_df["within_cluster_rank"] == 1].copy()
            rank1 = rank1.sort_values(
                ["cross_cluster_score", "cluster_quota", "cluster", "code"],
                ascending=[False, False, True, True],
            )
            rankN = unique_selected_df[unique_selected_df["within_cluster_rank"] > 1]
            if use_liq_gate and median_dollar_volume is not None and not median_dollar_volume.empty:
                kept_rank1 = []
                for hot_rank, (_, row) in enumerate(rank1.iterrows(), start=1):
                    code = str(row["code"])
                    if code_in_normalized_set(code, normalized_keep_for_cluster):
                        kept_rank1.append(row)
                        continue
                    n_mem = int(cluster_n_map.get(row["cluster"], 1))
                    med = float(median_dollar_volume.get(code, np.nan))
                    if passes_liquidity_floor(
                        med,
                        n_members=n_mem,
                        min_dollar=liq_min_dollar,
                        singleton_min_dollar=liq_singleton_min,
                        singleton_mult=liq_singleton_mult,
                        apply_mode=liq_apply_mode,
                        gate_enabled=use_liq_gate,
                        cross_cluster_rank=hot_rank,
                        hot_rank_cutoff=liq_hot_rank_cutoff,
                    ):
                        kept_rank1.append(row)
                    else:
                        liquidity_floor_drop_count += 1
                        liquidity_miss_reasons[code] = (
                            f"multi_rep_{weight_scheme}_cluster_{row['cluster']}_rank_1_liquidity_floor_miss"
                        )
                rank1 = pd.DataFrame(kept_rank1) if kept_rank1 else rank1.iloc[0:0]
            ordered_df = pd.concat([rank1, rankN], ignore_index=False)
        else:
            ordered_df = unique_selected_df

        selected = ordered_df["code"].tolist()
        selected_reason_map = {}
        for _, picked in ordered_df.iterrows():
            short_history_suffix = "_short_history" if bool(picked["short_history"]) else ""
            coverage_tag = "_coverage" if (_plan_b_coverage and int(picked["within_cluster_rank"]) == 1) else ""
            selected_reason_map[picked["code"]] = (
                f"multi_rep_{weight_scheme}_cluster_{picked['cluster']}_rank_{int(picked['within_cluster_rank'])}{coverage_tag}{short_history_suffix}"
            )

    normalized_keep = build_normalized_code_set(force_keep_codes or [])
    forced_codes: list[str] = []
    if force_keep_codes:
        for code in returns.columns:
            if code_in_normalized_set(code, normalized_keep):
                forced_codes.append(code)
    forced_codes = list(dict.fromkeys(forced_codes))

    non_forced_selected = [code for code in selected if code not in set(forced_codes)]
    seats_left = max(0, int(target_count) - len(forced_codes))
    final_selected = (forced_codes + non_forced_selected[:seats_left])[:target_count]

    dedup_reason_overrides: dict[str, str] = {}

    # Fix #4: final near-clone dedup pass. Drops any non-force instrument whose
    # |corr| with an already-kept instrument exceeds ``near_clone_corr_limit``
    # (lowest rank in the priority order wins). Force_keep instruments are anchors
    # and are never dropped. Runs AFTER quota filling so that eliminated
    # slots don't get back-filled with further near-clones.
    if (
        near_clone_corr_limit is not None
        and corr is not None
        and not corr.empty
    ):
        forced_set = set(forced_codes)
        # Map cluster_id -> ordered candidate members (highest return first), so
        # that if a cluster's rank-1 rep is trimmed we can try its rank-2/3/...
        # instead of leaving the cluster with zero representatives (regression
        # observed e.g. 煤炭/能源 cluster 7 getting zeroed out when its rank-1
        # correlated >0.75 with an earlier-picked 油气 rep).
        cluster_to_candidates: dict[object, list[str]] = {}
        cluster_of_code: dict[str, object] = {}
        for _, row in work_df.iterrows():
            members = list(row["candidate_members"] or [])
            if not members:
                continue
            if use_cross_rep:
                ordered = (
                    global_cross_scores.reindex(members).fillna(-np.inf).sort_values(ascending=False).index.tolist()
                )
            else:
                ordered = _rank_members_for_rep_pick(
                    members,
                    returns,
                    corr,
                    rep_pick=rep_pick,
                    min_center_corr=rep_min_center_corr,
                ).index.tolist()
            cluster_to_candidates[row["cluster"]] = ordered
            for m in ordered:
                cluster_of_code[m] = row["cluster"]

        def _passes_near_clone(code: str, kept: list[str]) -> bool:
            if code not in corr.index:
                return True
            already = [c for c in kept if c in corr.columns]
            if not already:
                return True
            # Cross-cluster threshold (looser, e.g. 0.75).
            cross_limit = float(near_clone_corr_limit)
            # Intra-cluster threshold (stricter, e.g. 0.60) — same cluster means
            # same macro theme, so require stronger divergence to keep multiples.
            intra_limit = (
                float(intra_cluster_corr_limit)
                if intra_cluster_corr_limit is not None
                else cross_limit
            )
            code_cluster = cluster_of_code.get(code)
            for peer in already:
                peer_cluster = cluster_of_code.get(peer)
                limit = intra_limit if (
                    code_cluster is not None and peer_cluster == code_cluster
                ) else cross_limit
                val = float(abs(corr.loc[code, peer]))
                if val > limit:
                    return False
            return True

        dedup: list[str] = []
        clusters_with_rep: set[object] = set()
        dedup_reason_overrides: dict[str, str] = {}
        rescue_limit = int(coverage_rescue_max) if coverage_rescue_max is not None else 0
        rescue_added = 0
        liquidity_blocked = set(liquidity_miss_reasons.keys())

        def _score_for_swap(code: str) -> float:
            if not selected_df.empty:
                row_match = selected_df.loc[selected_df["code"] == code]
                if not row_match.empty:
                    return float(row_match.iloc[0]["cross_cluster_score"])
            return float(global_cross_scores.get(code, -np.inf))

        def _enforce_target_cap(pool: list[str]) -> list[str]:
            forced_set_local = set(forced_codes)
            capped = list(pool)
            while len(capped) > int(target_count):
                droppable = [
                    c for c in capped
                    if c not in forced_set_local and c not in dedup_reason_overrides
                ]
                if not droppable:
                    droppable = [c for c in capped if c not in forced_set_local]
                if not droppable:
                    break
                drop_code = min(droppable, key=_score_for_swap)
                capped.remove(drop_code)
            return capped

        for code in final_selected:
            if code in forced_set:
                dedup.append(code)
                clusters_with_rep.add(cluster_of_code.get(code))
                continue
            if _passes_near_clone(code, dedup):
                dedup.append(code)
                clusters_with_rep.add(cluster_of_code.get(code))

        if coverage_rescue:
            # Capped coverage rescue: try lower-ranked members for zero-rep clusters.
            for cluster_id, ordered_members in cluster_to_candidates.items():
                if cluster_id in clusters_with_rep:
                    continue
                if rescue_limit >= 0 and rescue_added >= rescue_limit:
                    break
                chosen_from_cluster = [c for c in final_selected if cluster_of_code.get(c) == cluster_id]
                chosen_set = set(chosen_from_cluster)
                for rank_index, candidate in enumerate(ordered_members):
                    if candidate in chosen_set:
                        continue
                    if candidate in dedup:
                        continue
                    if candidate not in returns.columns:
                        continue
                    if candidate in liquidity_blocked:
                        continue
                    if _passes_near_clone(candidate, dedup):
                        dedup.append(candidate)
                        clusters_with_rep.add(cluster_id)
                        rescue_added += 1
                        short_history = float(history_days.get(candidate, 0.0)) < MIN_SELECTION_HISTORY_DAYS
                        short_suffix = "_short_history" if short_history else ""
                        dedup_reason_overrides[candidate] = (
                            f"multi_rep_{weight_scheme}_cluster_{cluster_id}_rank_{rank_index + 1}_coverage_rescue{short_suffix}"
                        )
                        dedup = _enforce_target_cap(dedup)
                        break
        else:
            tried_codes = set(final_selected)
            for candidate in selected:
                if len(dedup) >= int(target_count):
                    break
                if candidate in tried_codes or candidate in dedup:
                    continue
                tried_codes.add(candidate)
                if candidate not in returns.columns:
                    continue
                if candidate in liquidity_blocked:
                    continue
                if _passes_near_clone(candidate, dedup):
                    dedup.append(candidate)

        # Preserve original priority order for already-kept items, then append rescue picks.
        final_selected = _enforce_target_cap(dedup)

    reasons: dict[str, str] = {}
    for code in final_selected:
        if code in forced_codes and code not in selected_reason_map:
            reasons[code] = "forced_keep"
        elif code in dedup_reason_overrides:
            reasons[code] = dedup_reason_overrides[code]
        else:
            reasons[code] = selected_reason_map.get(code, f"multi_rep_{weight_scheme}")
    reasons.update(liquidity_miss_reasons)
    return final_selected, reasons, liquidity_floor_drop_count


def prepare_recent_window(
    close: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    lookback_days: int = CLUSTER_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    recent_rows = max(int(lookback_days) + 1, 2)
    recent_close = close.sort_index().tail(recent_rows)
    recent_volume = volume_df.sort_index().reindex(recent_close.index) if volume_df is not None else None
    returns = recent_close.pct_change().replace([np.inf, -np.inf], np.nan)
    return recent_close, recent_volume, returns


def pairwise_overlap_matrices(
    returns: pd.DataFrame,
    min_overlap_days: int = MIN_PAIR_OVERLAP_DAYS,
    low_overlap_distance: float = LOW_OVERLAP_DISTANCE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fallback_corr = 1.0 - float(low_overlap_distance)
    overlap = returns.notna().astype(int).T.dot(returns.notna().astype(int))
    corr = returns.corr(min_periods=min_overlap_days).fillna(fallback_corr)
    corr = corr.clip(lower=-1.0, upper=1.0)
    np.fill_diagonal(corr.values, 1.0)

    dist = 1.0 - corr
    np.fill_diagonal(dist.values, 0.0)
    return corr, overlap, dist


def summarize_overlap_counts(overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for code in overlap.index:
        values = overlap.loc[code].drop(index=code, errors="ignore")
        values = values[values > 0]
        rows.append(
            {
                "code": code,
                "overlap_min_days": int(values.min()) if not values.empty else 0,
                "overlap_median_days": float(values.median()) if not values.empty else 0.0,
                "overlap_max_days": int(values.max()) if not values.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def _pick_cluster_representatives(
    members: list[str],
    returns: pd.DataFrame,
    per_cluster: int,
    history_days: pd.Series | None = None,
) -> list[str]:
    if not members:
        return []
    if history_days is None:
        history_days = pd.Series(dtype=float)
    eligible_members = [code for code in members if float(history_days.get(code, 0.0)) >= MIN_SELECTION_HISTORY_DAYS]
    candidate_members = eligible_members or members
    return _return_scores(candidate_members, returns).head(per_cluster).index.tolist()


def _trim_selected(
    selected: list[str],
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    target_count: int | None,
    force_keep: Iterable[str] | None = None,
) -> list[str]:
    if target_count is None or target_count <= 0:
        return selected
    selected = list(dict.fromkeys(selected))
    if len(selected) <= target_count:
        return selected

    force_keep = list(force_keep or [])
    normalized_keep = build_normalized_code_set(force_keep)
    forced = [code for code in selected if code_in_normalized_set(code, normalized_keep)]
    forced = list(dict.fromkeys(forced))
    if len(forced) >= target_count:
        return forced

    candidates = [code for code in selected if code not in forced]
    mean_returns = returns[candidates].mean().reindex(candidates).fillna(0.0)
    final_selected = list(forced)

    for max_abs_corr_limit in FINAL_MAX_ABS_CORR_THRESHOLDS:
        if len(final_selected) >= target_count:
            break
        while len(final_selected) < target_count:
            available_selected = [picked for picked in final_selected if picked in returns.columns]
            ranked_candidates = []
            for code in candidates:
                if code in final_selected:
                    continue
                if available_selected and code in corr.index:
                    corr_slice = corr.loc[code, available_selected]
                    if isinstance(corr_slice, pd.Series):
                        max_abs_corr = float(corr_slice.abs().max()) if not corr_slice.empty else 0.0
                    else:
                        max_abs_corr = float(abs(corr_slice))
                else:
                    max_abs_corr = 0.0

                if max_abs_corr <= max_abs_corr_limit:
                    ranked_candidates.append((code, max_abs_corr, float(mean_returns.get(code, 0.0))))

            if not ranked_candidates:
                break

            ranked_candidates.sort(key=lambda item: (item[1], -item[2], item[0]))
            final_selected.append(ranked_candidates[0][0])

    return final_selected


def _build_cluster_df(
    clusters: pd.Series,
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    history_days: pd.Series | None,
) -> pd.DataFrame:
    rows = []
    for cid, members in clusters.groupby(clusters).groups.items():
        members = list(members)
        n = len(members)
        if n <= 1:
            mean_corr = 1.0
        else:
            subcorr = corr.reindex(index=members, columns=members).fillna(0.0).values
            mean_corr = float((subcorr.sum() - n) / (n * (n - 1)))
        mean_volatility = returns[members].std().mean()
        mean_return = returns[members].mean().mean()
        try:
            avg_volume = volume_df[members].mean().mean() if volume_df is not None else None
        except Exception:
            avg_volume = None
        member_history = pd.Series(dtype=float) if history_days is None else history_days.reindex(members).fillna(0.0)
        rows.append(
            {
                "cluster": cid,
                "members": members,
                "n": n,
                "mean_corr": mean_corr,
                "mean_volatility": mean_volatility,
                "mean_return": mean_return,
                "avg_volume": avg_volume,
                "cluster_min_history_days": float(member_history.min()) if not member_history.empty else 0.0,
                "cluster_median_history_days": float(member_history.median()) if not member_history.empty else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_corr", "n"], ascending=[False, False])


def cluster_and_select(
    returns: pd.DataFrame,
    volume_df: pd.DataFrame | None,
    t: float = DIST_T,
    per_cluster: int = PER_CLUSTER,
    rule: str = RULE,
    force_keep: Iterable[str] | None = None,
    target_count: int | None = None,
    history_days: pd.Series | None = None,
    corr: pd.DataFrame | None = None,
    overlap: pd.DataFrame | None = None,
    coverage_rescue: bool = True,
    coverage_rescue_max: int | None = None,
    detone_corr: bool | None = None,
    merge_low_cohesion: bool | None = None,
    median_dollar_volume: pd.Series | None = None,
    liquidity_gate_enabled: bool | None = None,
    min_dollar_volume: float | None = None,
    singleton_min_dollar_volume: float | None = None,
    singleton_liquidity_mult: float | None = None,
    liquidity_apply_mode: str | None = None,
    liquidity_hot_rank_cutoff: int | None = None,
):
    """Run ward-linkage clustering and selection.

    Pipeline (2026-04-24 简化版)：
      1. 对 ``returns`` 所有列做 Ward 层级聚类
      2. 在每个簇内按 (Sharpe + 12-1 动量) 打分，挑 top-K 代表
      3. 最后一道 ``FINAL_MAX_ABS_CORR_THRESHOLDS`` 近克隆去冗余

    相关性由聚类保证，收益由簇内打分保证，两层互不干扰。

    Returns ``(Z, clusters, cluster_df, selected, reasons)``.
    """
    if corr is None or overlap is None:
        corr, overlap, _ = pairwise_overlap_matrices(returns, MIN_PAIR_OVERLAP_DAYS, LOW_OVERLAP_DISTANCE)

    use_detone = DETONE_CORR_ENABLED if detone_corr is None else bool(detone_corr)
    if use_detone:
        corr = apply_detone_to_corr(corr, t_obs=len(returns))

    codes = list(returns.columns)
    dist = 1.0 - corr
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method=CLUSTER_LINKAGE)
    labels = fcluster(Z, t=t, criterion="distance")
    clusters = pd.Series(labels, index=codes, name="cluster")

    use_merge = MERGE_LOW_COHESION_ENABLED if merge_low_cohesion is None else bool(merge_low_cohesion)
    merge_count = 0
    singleton_merge_count = 0
    if use_merge:
        clusters, merge_count = merge_low_cohesion_clusters(clusters, corr)
    if MERGE_SINGLETONS_ENABLED:
        clusters, singleton_merge_count = merge_singleton_clusters(clusters, corr)
        merge_count += singleton_merge_count

    cluster_df = _build_cluster_df(
        clusters,
        returns,
        corr,
        volume_df,
        history_days.reindex(codes).fillna(0.0) if history_days is not None else None,
    )

    selection_rule = str(rule or RULE).strip().lower()
    multi_rep_rule_to_scheme = {
        "multi_rep_equal_mainline": "equal",
        "multi_rep_equal": "equal",
        "equal": "equal",
        "multi_rep_exp_benchmark": "exp",
        "multi_rep_exp": "exp",
        "exp": "exp",
        "multi_rep_harmonic_deprecated": "harmonic",
        "multi_rep_harmonic": "harmonic",
        "harmonic": "harmonic",
    }

    liquidity_floor_drop_count = 0
    if selection_rule in multi_rep_rule_to_scheme:
        selected, reasons, liquidity_floor_drop_count = _build_multi_rep_cluster_pool(
            cluster_df=cluster_df,
            returns=returns,
            history_days=history_days.reindex(codes).fillna(0.0) if history_days is not None else pd.Series(dtype=float),
            target_count=target_count,
            force_keep_codes=force_keep,
            weight_scheme=multi_rep_rule_to_scheme[selection_rule],
            exp_base=MULTI_REP_EXP_BASE,
            corr=corr,
            # Fix #4 + Plan B: near-clone dedup cap. User goal is a pool with
            # LOW internal correlation (portfolio holds only 5-6 names/day, so
            # high corr = concentrated systemic risk). Uses the *strictest*
            # (first) entry of FINAL_MAX_ABS_CORR_THRESHOLDS so any pair above
            # that threshold keeps only the higher-priority code.
            # A/B 实验支持：POOL_NEAR_CLONE_IDX 环境变量可指定索引；0=最严(Plan B)，
            # -1=最宽(Fix#4 前)。
            near_clone_corr_limit=(
                float(FINAL_MAX_ABS_CORR_THRESHOLDS[int(os.environ.get("POOL_NEAR_CLONE_IDX", 0))])
                if FINAL_MAX_ABS_CORR_THRESHOLDS
                else None
            ),
            # 方案 A: 簇内（同 cluster）成员之间使用更严的 0.60 阈值，避免
            # 同一宽基族里出现高度重叠的多只代表（如上证中盘 vs 上证超大盘）。
            intra_cluster_corr_limit=float(INTRA_CLUSTER_MAX_ABS_CORR),
            coverage_rescue=coverage_rescue,
            coverage_rescue_max=(
                coverage_rescue_max
                if coverage_rescue_max is not None
                else (COVERAGE_RESCUE_MAX if COVERAGE_RESCUE_ENABLED else 0)
            ),
            rep_pick=REP_PICK_MODE,
            rep_min_center_corr=REP_MIN_CENTER_CORR,
            cross_cluster_rank_mode=CROSS_CLUSTER_RANK,
            median_dollar_volume=median_dollar_volume,
            liquidity_gate_enabled=liquidity_gate_enabled,
            min_dollar_volume=min_dollar_volume,
            singleton_min_dollar_volume=singleton_min_dollar_volume,
            singleton_liquidity_mult=singleton_liquidity_mult,
            liquidity_apply_mode=liquidity_apply_mode,
            liquidity_hot_rank_cutoff=liquidity_hot_rank_cutoff,
        )
    else:
        selected = []
        reasons = {}
        for _, row in cluster_df.iterrows():
            members = row["members"]
            cid = row["cluster"]
            if len(members) == 1:
                pick = members[0]
                selected.append(pick)
                if history_days is not None and float(history_days.get(pick, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[pick] = f"only_member_cluster_{cid}_short_history"
                else:
                    reasons[pick] = f"only_member_cluster_{cid}"
                continue
            picks = _pick_cluster_representatives(members, returns, per_cluster, history_days=history_days)
            for p in picks:
                selected.append(p)
                if history_days is not None and float(history_days.get(p, 0.0)) < MIN_SELECTION_HISTORY_DAYS:
                    reasons[p] = f"return_top_cluster_{cid}_short_history"
                else:
                    reasons[p] = f"return_top_cluster_{cid}"

        selected = list(dict.fromkeys(selected))
        if force_keep:
            normalized_keep = build_normalized_code_set(force_keep)
            for code in codes:
                if code_in_normalized_set(code, normalized_keep):
                    if code not in selected:
                        selected.append(code)
                    reasons.setdefault(code, "forced_keep")
        selected = _trim_selected(selected, returns, corr, target_count, force_keep)

    cluster_df.attrs["merge_count"] = merge_count
    cluster_df.attrs["singleton_merge_count"] = singleton_merge_count
    cluster_df.attrs["liquidity_floor_drop_count"] = liquidity_floor_drop_count
    cluster_df.attrs["detone_applied"] = use_detone
    return Z, clusters, cluster_df, selected, reasons
