"""Selection-stage liquidity floor (median dollar volume, not return-based)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import constants as C


def compute_median_dollar_volume(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback_days: int = C.CLUSTER_LOOKBACK_DAYS,
) -> pd.Series:
    """Median daily turnover ``volume * close`` over the clustering lookback window."""
    if close is None or volume is None or close.empty or volume.empty:
        return pd.Series(dtype=float)

    recent_rows = max(int(lookback_days) + 1, 2)
    recent_close = close.sort_index().tail(recent_rows)
    recent_volume = volume.sort_index().reindex(recent_close.index)
    common = [c for c in recent_close.columns if c in recent_volume.columns]
    if not common:
        return pd.Series(dtype=float)

    dollar = recent_volume[common] * recent_close[common]
    dollar = dollar.replace([np.inf, -np.inf], np.nan)
    return dollar.median(axis=0).astype(float)


def effective_liquidity_floor(
    n_members: int,
    *,
    min_dollar: float,
    singleton_min_dollar: float,
    singleton_mult: float,
    apply_mode: str = C.SELECTION_LIQUIDITY_APPLY,
) -> float:
    if apply_mode == "all":
        mult = float(singleton_mult) if int(n_members) <= 1 else 1.0
        base = float(min_dollar) if float(min_dollar) > 0 else float(singleton_min_dollar)
        return base * mult
    if int(n_members) <= 1:
        return float(singleton_min_dollar) * float(singleton_mult)
    return 0.0


def should_apply_liquidity_gate(
    n_members: int,
    *,
    apply_mode: str = C.SELECTION_LIQUIDITY_APPLY,
    gate_enabled: bool = C.SELECTION_LIQUIDITY_GATE_ENABLED,
    cross_cluster_rank: int | None = None,
    hot_rank_cutoff: int | None = None,
) -> bool:
    if not gate_enabled:
        return False
    cutoff = int(hot_rank_cutoff if hot_rank_cutoff is not None else C.LIQUIDITY_HOT_RANK_CUTOFF)
    if cross_cluster_rank is not None and int(cross_cluster_rank) > cutoff:
        return False
    if apply_mode == "all":
        return True
    return int(n_members) <= 1


def passes_liquidity_floor(
    median_dollar_volume: float,
    *,
    n_members: int,
    min_dollar: float,
    singleton_min_dollar: float | None = None,
    singleton_mult: float,
    apply_mode: str = C.SELECTION_LIQUIDITY_APPLY,
    gate_enabled: bool = C.SELECTION_LIQUIDITY_GATE_ENABLED,
    cross_cluster_rank: int | None = None,
    hot_rank_cutoff: int | None = None,
) -> bool:
    if not should_apply_liquidity_gate(
        n_members,
        apply_mode=apply_mode,
        gate_enabled=gate_enabled,
        cross_cluster_rank=cross_cluster_rank,
        hot_rank_cutoff=hot_rank_cutoff,
    ):
        return True
    singleton_floor = float(
        singleton_min_dollar if singleton_min_dollar is not None else C.SINGLETON_MIN_DOLLAR_VOLUME
    )
    floor = effective_liquidity_floor(
        n_members,
        min_dollar=min_dollar,
        singleton_min_dollar=singleton_floor,
        singleton_mult=singleton_mult,
        apply_mode=apply_mode,
    )
    if floor <= 0:
        return True
    if not np.isfinite(median_dollar_volume):
        return False
    return float(median_dollar_volume) >= floor


def build_liquidity_diagnostics(
    codes: list[str],
    median_dollar_volume: pd.Series,
    *,
    min_dollar: float = C.SELECTION_MIN_DOLLAR_VOLUME,
    singleton_min_dollar: float = C.SINGLETON_MIN_DOLLAR_VOLUME,
    singleton_mult: float = C.SINGLETON_LIQUIDITY_MULT,
    apply_mode: str = C.SELECTION_LIQUIDITY_APPLY,
    gate_enabled: bool = C.SELECTION_LIQUIDITY_GATE_ENABLED,
) -> pd.DataFrame:
    return build_liquidity_diagnostics_with_clusters(
        median_dollar_volume.reindex(codes).dropna(),
        cluster_member_counts={},
        code_to_cluster={},
        min_dollar=min_dollar,
        singleton_min_dollar=singleton_min_dollar,
        singleton_mult=singleton_mult,
        apply_mode=apply_mode,
        gate_enabled=gate_enabled,
    )

def build_liquidity_diagnostics_with_clusters(
    median_dollar_volume: pd.Series,
    cluster_member_counts: dict[object, int],
    code_to_cluster: dict[str, object],
    *,
    min_dollar: float = C.SELECTION_MIN_DOLLAR_VOLUME,
    singleton_min_dollar: float = C.SINGLETON_MIN_DOLLAR_VOLUME,
    singleton_mult: float = C.SINGLETON_LIQUIDITY_MULT,
    apply_mode: str = C.SELECTION_LIQUIDITY_APPLY,
    gate_enabled: bool = C.SELECTION_LIQUIDITY_GATE_ENABLED,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for code in median_dollar_volume.index:
        cid = code_to_cluster.get(code)
        n_members = int(cluster_member_counts.get(cid, 1)) if cid is not None else 1
        med = float(median_dollar_volume.get(code, np.nan))
        floor = effective_liquidity_floor(
            n_members,
            min_dollar=min_dollar,
            singleton_min_dollar=singleton_min_dollar,
            singleton_mult=singleton_mult,
            apply_mode=apply_mode,
        )
        applies = should_apply_liquidity_gate(n_members, apply_mode=apply_mode, gate_enabled=gate_enabled)
        passes = (
            passes_liquidity_floor(
                med,
                n_members=n_members,
                min_dollar=min_dollar,
                singleton_min_dollar=singleton_min_dollar,
                singleton_mult=singleton_mult,
                apply_mode=apply_mode,
                gate_enabled=gate_enabled,
            )
            if gate_enabled
            else True
        )
        rows.append(
            {
                "code": code,
                "cluster": cid,
                "cluster_n": n_members,
                "median_dollar_volume": med,
                "liquidity_floor": floor,
                "liquidity_gate_applies": bool(applies),
                "passes_liquidity_floor": bool(passes),
                "drop_reason": "" if passes or not gate_enabled or not applies else "liquidity_floor_miss",
            }
        )
    return pd.DataFrame(rows)
