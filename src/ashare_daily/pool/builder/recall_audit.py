"""Walk-forward recall auditor for the production cluster pool.

Not an ``ashare-daily`` subcommand. ``POOL_*`` is ignored. The decision band
uses locked defaults: target 100, distance 0.60, horizon 10, clone 0.8.

Smoke (overwrites the CSV and JSON)::

    python -m ashare_daily.pool.builder.recall_audit --max-anchors 5

Outputs land under ``TEMP_DIR`` as ``pool_recall_audit.csv`` and
``pool_recall_audit_summary.json``. A second run replaces both files.
Copy them if you need the previous result.

The real qlib marker test is::

    pytest -m recall_qlib -o addopts=

``load_data`` still forward-fills when the caller omits ``ffill``. This
module passes ``ffill=False`` and fills only the 253-row cluster window.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from datetime import date, timedelta
from pathlib import Path

AUDIT_TARGET = 100
LOOKBACK = 252
TAIL_ROWS = LOOKBACK + 1
HORIZON = 10
ANCHOR_STEP = 5
DECISION_STEP = 10
WINNER_N = 50
MIN_DECISION_DAYS = 180
CLONE_THRESHOLD = 0.8
CLONE_DIAGNOSTIC = 0.70
RANDOM_DRAWS = 50
RANDOM_SEED = 0
KEEP_BAND = 0.85
MID_BAND = 0.60
BASELINE_MARGIN = 0.05
LOAD_LOOKBACK_DAYS = int(3.5 * 365)

SURVIVORSHIP_NOTE = (
    "Universe is today's csi1800.txt. Index reconstitution and delisted names "
    "are not reconstructed. This does not block the band."
)


def pin_threads() -> None:
    """Cap BLAS threads before NumPy is imported in this process."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"


def scrub_pool_env() -> None:
    for key in list(os.environ):
        if key.startswith("POOL_"):
            del os.environ[key]


def drop_builder_modules() -> None:
    """Drop cached builder imports so a prior ``POOL_*`` binding cannot stick."""
    keep = {"ashare_daily.pool.builder.recall_audit"}
    for name in list(sys.modules):
        if name in keep:
            continue
        if name == "ashare_daily.pool.builder" or name.startswith("ashare_daily.pool.builder."):
            del sys.modules[name]


def bounded_jobs(requested: int, cpu: int | None = None) -> int:
    limit = cpu if cpu is not None else (os.cpu_count() or 1)
    limit = max(int(limit), 1)
    if requested < 1:
        raise ValueError(
            f"--jobs {requested} is below 1. "
            "Cause: the pool size must be a positive integer. "
            "Fix: pass --jobs 1, or omit the flag."
        )
    return min(int(requested), limit)


def msg_provider_missing(provider_uri: str) -> str:
    return (
        f"Cannot read prices from {provider_uri}. "
        "Cause: path missing or empty. "
        "Fix: set provider_uri in configs/shared_filter_config.json to the cn_data directory."
    )


def msg_anchor_skipped(asof: str) -> str:
    return (
        f"Anchor {asof} skipped: fewer than {WINNER_N} names with {MIN_DECISION_DAYS} "
        "valid days and a finite forward return. "
        "Cause: thin history. "
        "Fix: none for that date. It is omitted from the decision mean."
    )


def msg_snapshot_missing(path: Path) -> str:
    return (
        f"Production snapshot not found at {path}. "
        "Cause: cluster-map overwrites the live CSV. "
        f"Fix: copy stock_cluster_mapping_selected.csv and the metadata JSON to {path}, "
        "or pass --snapshot."
    )


def msg_short_tail(asof: str) -> str:
    return (
        f"Anchor {asof} has fewer than {TAIL_ROWS} price rows. "
        "Cause: the panel does not contain a 252-day tail at that date. "
        "Fix: start the load earlier, or drop that anchor."
    )


def view_asof(close, vol, asof: str):
    """Slice through ``asof``, keep 253 rows, and forward-fill that frame only."""
    import pandas as pd

    stamp = pd.Timestamp(asof)
    history = close.sort_index().loc[:stamp]
    if history.index.has_duplicates:
        history = history[~history.index.duplicated(keep="last")]
    if len(history) < TAIL_ROWS:
        raise ValueError(msg_short_tail(str(stamp.date())))
    window = history.tail(TAIL_ROWS).ffill()
    if vol is None:
        return window, None
    vol_hist = vol.sort_index().loc[:stamp]
    if vol_hist.index.has_duplicates:
        vol_hist = vol_hist[~vol_hist.index.duplicated(keep="last")]
    window_vol = vol_hist.reindex(window.index)
    return window, window_vol


def anchor_positions(
    n_rows: int,
    *,
    lookback: int = LOOKBACK,
    forward: int = HORIZON,
    step: int = ANCHOR_STEP,
) -> list[int]:
    """Trading-day positions that have a full tail and a full forward window."""
    start = lookback
    last = n_rows - 1 - forward
    if last < start:
        return []
    return list(range(start, last + 1, step))


def decision_positions(
    positions: list[int],
    *,
    origin: int = LOOKBACK,
    decision_step: int = DECISION_STEP,
) -> list[int]:
    return [pos for pos in positions if (pos - origin) % decision_step == 0]


def pick_winners(forward_df, decision_codes: set[str], horizon: int = HORIZON):
    """Top ``WINNER_N`` finite forward returns inside the 180-day set."""
    import numpy as np

    column = f"fwd_ret_{horizon}d"
    if forward_df is None or forward_df.empty or column not in forward_df.columns:
        return None
    subset = forward_df[forward_df["code"].isin(decision_codes)].copy()
    values = subset[column]
    subset = subset[np.isfinite(values.to_numpy(dtype=float))]
    if len(subset) < WINNER_N:
        return None
    return subset.nlargest(WINNER_N, column)


def clone_hit(code: str, selected: set[str], corr, threshold: float) -> bool:
    if code in selected:
        return True
    if corr is None or code not in corr.index:
        return False
    mates = [name for name in selected if name in corr.columns and name != code]
    if not mates:
        return False
    values = corr.loc[code, mates].abs()
    if values.empty:
        return False
    return float(values.max()) >= threshold


def recall_at(winners, selected: set[str], corr, threshold: float | None) -> float:
    if winners is None or len(winners) == 0:
        raise ValueError("recall requires a non-empty winner set")
    hits = 0
    for code in winners["code"].tolist():
        if threshold is None:
            hits += int(code in selected)
        elif clone_hit(code, selected, corr, threshold):
            hits += 1
    return hits / len(winners)


def missed_winner_mean(winners, selected: set[str], horizon: int = HORIZON) -> float | None:
    column = f"fwd_ret_{horizon}d"
    missed = winners[~winners["code"].isin(selected)]
    if missed.empty:
        return None
    return float(missed[column].mean())


def _sample_recall(eligible: list[str], chosen: list[str], winners, corr, threshold: float | None) -> float:
    return recall_at(winners, set(chosen), corr, threshold)


def random_recalls(eligible: list[str], winners, corr, *, draws: int = RANDOM_DRAWS, seed: int = RANDOM_SEED):
    """Mean bare and clone recall of ``draws`` seeded samples of size ``AUDIT_TARGET``."""
    import numpy as np

    if len(eligible) < AUDIT_TARGET:
        return None, None
    rng = np.random.default_rng(seed)
    bare = []
    clone = []
    pool = np.array(eligible)
    for _ in range(draws):
        pick = rng.choice(pool, size=AUDIT_TARGET, replace=False).tolist()
        bare.append(_sample_recall(eligible, pick, winners, corr, None))
        clone.append(_sample_recall(eligible, pick, winners, corr, CLONE_THRESHOLD))
    return float(np.mean(bare)), float(np.mean(clone))


def liquidity_top(close_window, vol_window, eligible: list[str]) -> list[str] | None:
    if vol_window is None or len(eligible) < AUDIT_TARGET:
        return None
    dollar = (close_window[eligible] * vol_window.reindex(columns=eligible)).median(axis=0)
    dollar = dollar.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(dollar) < AUDIT_TARGET:
        return None
    return dollar.nlargest(AUDIT_TARGET).index.tolist()


def momentum_top(close_window, eligible: list[str]) -> list[str] | None:
    """12-1 momentum: price at t-21 over price at t-252, skip the last 21 sessions."""
    if len(close_window) < TAIL_ROWS or len(eligible) < AUDIT_TARGET:
        return None
    start = close_window.iloc[0]
    end = close_window.iloc[-1 - 21]
    momentum = (end / start - 1.0).reindex(eligible)
    momentum = momentum.replace([float("inf"), float("-inf")], float("nan")).dropna()
    if len(momentum) < AUDIT_TARGET:
        return None
    return momentum.nlargest(AUDIT_TARGET).index.tolist()


def assign_band(recall: float, baseline_clone: list[float]) -> tuple[str, bool]:
    dominated = bool(baseline_clone) and recall <= max(baseline_clone) + BASELINE_MARGIN
    if dominated or recall < MID_BAND:
        return "raise_target", dominated
    if recall >= KEEP_BAND:
        return "keep_100", dominated
    return "reentry", dominated


def jaccard(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _json_ready(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def select_at(close, vol, asof: str) -> dict:
    """Run the production selector on the 253-row window ending at ``asof``."""
    from ashare_daily.pool.builder import constants as C
    from ashare_daily.pool.builder.clustering import cluster_and_select
    from ashare_daily.pool.builder.selector import _build_diagnostic_view

    window_close, window_vol = view_asof(close, vol, asof)
    view = _build_diagnostic_view(window_close, window_vol)
    eligible = list(view["eligible_codes"])
    diagnostics = view["diagnostics_df"]
    history = view["recent_valid_days"]
    if not eligible:
        return {
            "skip": "empty_cluster_eligible",
            "window_close": window_close,
            "window_vol": window_vol,
            "diagnostics": diagnostics,
            "history": history,
            "corr": None,
            "selected": [],
            "clusters": None,
        }
    selected_history = history.reindex(eligible).fillna(0)
    _z, clusters, _cluster_df, selected, _reasons = cluster_and_select(
        view["returns"],
        view["cluster_vol"],
        t=C.DIST_T,
        per_cluster=C.PER_CLUSTER,
        rule=C.RULE,
        force_keep=list(C.ALWAYS_KEEP_CODES),
        target_count=AUDIT_TARGET,
        history_days=selected_history,
        corr=view["corr"],
        overlap=view["overlap"],
        coverage_rescue=C.COVERAGE_RESCUE_ENABLED,
        coverage_rescue_max=C.COVERAGE_RESCUE_MAX if C.COVERAGE_RESCUE_ENABLED else 0,
        detone_corr=False,
        merge_low_cohesion=C.MERGE_LOW_COHESION_ENABLED,
        median_dollar_volume=None,
        liquidity_gate_enabled=False,
        min_dollar_volume=C.SELECTION_MIN_DOLLAR_VOLUME,
        singleton_min_dollar_volume=C.SINGLETON_MIN_DOLLAR_VOLUME,
        singleton_liquidity_mult=C.SINGLETON_LIQUIDITY_MULT,
        liquidity_apply_mode=C.SELECTION_LIQUIDITY_APPLY,
        liquidity_hot_rank_cutoff=C.LIQUIDITY_HOT_RANK_CUTOFF,
    )
    return {
        "skip": None,
        "window_close": window_close,
        "window_vol": window_vol,
        "diagnostics": diagnostics,
        "history": history,
        "corr": view["corr"],
        "selected": list(selected),
        "clusters": clusters,
    }


def score_window(close, vol, asof: str, *, seed: int = RANDOM_SEED) -> dict:
    """Score one anchor. ``close`` must include the forward horizon past ``asof``."""
    from ashare_daily.pool.builder.audit import compute_forward_metrics

    started = time.perf_counter()
    picked = select_at(close, vol, asof)
    history = picked["history"]
    decision_codes = set(history[history >= MIN_DECISION_DAYS].index.astype(str))
    mid_count = int(((history >= 120) & (history < MIN_DECISION_DAYS)).sum())
    if picked["skip"]:
        return {
            "asof": asof,
            "skip": picked["skip"],
            "seconds": time.perf_counter() - started,
            "names_120_179": mid_count,
            "message": msg_anchor_skipped(asof),
        }
    if not decision_codes:
        return {
            "asof": asof,
            "skip": "empty_decision_denom",
            "seconds": time.perf_counter() - started,
            "names_120_179": mid_count,
            "message": msg_anchor_skipped(asof),
        }
    forward = compute_forward_metrics(close.loc[:, list(decision_codes)], asof, [HORIZON])
    winners = pick_winners(forward, decision_codes, HORIZON)
    if winners is None:
        return {
            "asof": asof,
            "skip": "insufficient_winners",
            "seconds": time.perf_counter() - started,
            "names_120_179": mid_count,
            "message": msg_anchor_skipped(asof),
        }
    selected = set(picked["selected"])
    corr = picked["corr"]
    eligible = sorted(decision_codes)
    random_bare, random_clone = random_recalls(eligible, winners, corr, seed=seed)
    liquid = liquidity_top(picked["window_close"], picked["window_vol"], eligible)
    momentum = momentum_top(picked["window_close"], eligible)
    clusters = picked["clusters"]
    cluster_count = int(clusters.nunique()) if clusters is not None and not clusters.empty else 0
    covered_clusters = 0
    if clusters is not None and selected:
        covered_clusters = int(clusters.reindex(list(selected)).nunique())
    row = {
        "asof": asof,
        "skip": None,
        "seconds": round(time.perf_counter() - started, 3),
        "n_cluster_eligible": int((history >= 120).sum()),
        "n_decision": len(decision_codes),
        "names_120_179": mid_count,
        "n_selected": len(selected),
        "n_clusters": cluster_count,
        "cluster_coverage": (covered_clusters / cluster_count) if cluster_count else None,
        "n_winners": int(len(winners)),
        "recall_pool": recall_at(winners, selected, corr, None),
        "recall_pool_clone": recall_at(winners, selected, corr, CLONE_THRESHOLD),
        "recall_pool_clone_070": recall_at(winners, selected, corr, CLONE_DIAGNOSTIC),
        "missed_winner_mean_forward_return": missed_winner_mean(winners, selected),
        "recall_random_bare": random_bare,
        "recall_random_clone": random_clone,
        "recall_liquidity_bare": None if liquid is None else recall_at(winners, set(liquid), corr, None),
        "recall_liquidity_clone": None if liquid is None else recall_at(winners, set(liquid), corr, CLONE_THRESHOLD),
        "recall_momentum_bare": None if momentum is None else recall_at(winners, set(momentum), corr, None),
        "recall_momentum_clone": None if momentum is None else recall_at(winners, set(momentum), corr, CLONE_THRESHOLD),
        "selected": sorted(selected),
    }
    return row


def score_payload(payload: dict) -> dict:
    """Process-pool entry. Pins threads before numeric imports in a spawned worker."""
    pin_threads()
    return score_window(payload["close"], payload["volume"], payload["asof"], seed=payload["seed"])


def _slice_payload(close, vol, position: int, asof: str, seed: int) -> dict:
    start = position - LOOKBACK
    stop = position + HORIZON + 1
    return {
        "asof": asof,
        "seed": seed,
        "close": close.iloc[start:stop],
        "volume": None if vol is None else vol.iloc[start:stop],
    }


def _pool_worker():
    """Return the package function so a spawned child does not re-enter ``__main__``."""
    from ashare_daily.pool.builder.recall_audit import score_payload as worker

    return worker


def run_payloads(payloads: list[dict], jobs: int) -> list[dict]:
    if jobs == 1:
        return [score_payload(item) for item in payloads]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        rows = pool.map(_pool_worker(), payloads)
    return list(rows)


def output_paths() -> tuple[Path, Path]:
    from ashare_daily.paths import TEMP_DIR

    directory = Path(TEMP_DIR)
    return directory / "pool_recall_audit.csv", directory / "pool_recall_audit_summary.json"


def default_snapshot_path() -> Path:
    from ashare_daily.paths import TEMP_DIR

    return Path(TEMP_DIR) / "stock_cluster_mapping_selected_snapshot.csv"


def default_snapshot_metadata_path() -> Path:
    from ashare_daily.paths import TEMP_DIR

    return Path(TEMP_DIR) / "stock_cluster_mapping_selected_metadata_snapshot.json"


def read_snapshot_codes(path: Path) -> set[str]:
    import pandas as pd

    frame = pd.read_csv(path, dtype=str)
    if "code" not in frame.columns:
        raise ValueError(
            f"Snapshot {path} has no code column. "
            "Cause: the file is not stock_cluster_mapping_selected.csv. "
            "Fix: pass --snapshot pointing at that CSV."
        )
    if "selected" in frame.columns:
        flag = frame["selected"].astype(str).str.lower()
        frame = frame[flag.isin(["true", "1", "yes"])]
    return set(frame["code"].astype(str))


def read_trend_as_of(path: Path) -> str | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload.get("trend_as_of")
    return str(value) if value else None


def summarize(rows: list[dict], *, decision_asofs: set[str], snapshot_codes: set[str] | None, trend_as_of: str | None, band_scope: str) -> dict:
    import pandas as pd

    scored = [row for row in rows if not row.get("skip")]
    skipped = [row for row in rows if row.get("skip")]
    decision_rows = [row for row in scored if row["asof"] in decision_asofs]
    summary = {
        "status": "ok" if decision_rows else "failed",
        "band_scope": band_scope,
        "survivorship_note": SURVIVORSHIP_NOTE,
        "trend_as_of": trend_as_of,
        "locked_defaults": {
            "target": AUDIT_TARGET,
            "dist_t": 0.60,
            "horizon": HORIZON,
            "clone": CLONE_THRESHOLD,
            "pool_env": "ignored",
        },
        "n_scored": len(scored),
        "n_decision": len(decision_rows),
        "n_skipped": len(skipped),
        "skipped_insufficient_winners": sum(1 for row in skipped if row.get("skip") == "insufficient_winners"),
        "skipped_empty_decision_denom": sum(1 for row in skipped if row.get("skip") == "empty_decision_denom"),
        "skipped_empty_cluster_eligible": sum(1 for row in skipped if row.get("skip") == "empty_cluster_eligible"),
    }
    if not decision_rows:
        summary["band"] = None
        summary["message"] = "No decision anchor produced a winner set. No band was emitted."
        return summary
    frame = pd.DataFrame(decision_rows)
    recall = float(frame["recall_pool_clone"].mean())
    baselines = [
        float(value)
        for value in (
            frame["recall_random_clone"].mean(),
            frame["recall_liquidity_clone"].mean(),
            frame["recall_momentum_clone"].mean(),
        )
        if value == value
    ]
    band, dominated = assign_band(recall, baselines)
    summary.update(
        {
            "band": band,
            "baseline_dominated": dominated,
            "decision_recall_pool": float(frame["recall_pool"].mean()),
            "decision_recall_pool_clone": recall,
            "decision_recall_pool_clone_070": float(frame["recall_pool_clone_070"].mean()),
            "decision_recall_random_clone": None if frame["recall_random_clone"].isna().all() else float(frame["recall_random_clone"].mean()),
            "decision_recall_liquidity_clone": None if frame["recall_liquidity_clone"].isna().all() else float(frame["recall_liquidity_clone"].mean()),
            "decision_recall_momentum_clone": None if frame["recall_momentum_clone"].isna().all() else float(frame["recall_momentum_clone"].mean()),
            "decision_missed_winner_mean_forward_return": None
            if frame["missed_winner_mean_forward_return"].isna().all()
            else float(frame["missed_winner_mean_forward_return"].mean()),
        }
    )
    summary["jaccard_vs_prod_snapshot"] = _jaccard_at_trend(scored, snapshot_codes, trend_as_of)
    return summary


def _jaccard_at_trend(scored: list[dict], snapshot_codes: set[str] | None, trend_as_of: str | None) -> float | None:
    if not snapshot_codes or not trend_as_of:
        return None
    import pandas as pd

    target = pd.Timestamp(trend_as_of)
    match = None
    for row in scored:
        if pd.Timestamp(row["asof"]) == target:
            match = row
            break
    if match is None:
        return None
    return jaccard(set(match["selected"]), snapshot_codes)


def write_csv(path: Path, rows: list[dict]) -> None:
    import pandas as pd

    scored = [row for row in rows if not row.get("skip")]
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "asof",
        "seconds",
        "n_cluster_eligible",
        "n_decision",
        "names_120_179",
        "n_selected",
        "n_clusters",
        "cluster_coverage",
        "n_winners",
        "recall_pool",
        "recall_pool_clone",
        "recall_pool_clone_070",
        "missed_winner_mean_forward_return",
        "recall_random_bare",
        "recall_random_clone",
        "recall_liquidity_bare",
        "recall_liquidity_clone",
        "recall_momentum_bare",
        "recall_momentum_clone",
    ]
    frame = pd.DataFrame(scored, columns=columns) if scored else pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False)


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m ashare_daily.pool.builder.recall_audit",
        description=(
            "Score whether future winners landed in the 100-name cluster pool. "
            "POOL_* is ignored. The decision band uses target 100, dist 0.60, h=10, clone 0.8."
        ),
        epilog=(
            "The qlib marker is excluded by pytest addopts. "
            "Run it with: pytest -m recall_qlib -o addopts="
        ),
    )
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes. Capped at the CPU count. Default 1.")
    parser.add_argument(
        "--max-anchors",
        type=int,
        default=0,
        help="Score only the last N anchors. 0 scores every 5th session. A positive value marks the band as smoke.",
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Production selected CSV for Jaccard. Default: TEMP_DIR/stock_cluster_mapping_selected_snapshot.csv.",
    )
    return parser


def _load_universe():
    from ashare_daily.pool.builder import constants as C
    from ashare_daily.pool.builder.config import load_shared_config
    from ashare_daily.pool.builder.data_loading import build_code_name_map, load_data

    config = load_shared_config()
    provider = str(config.get("provider_uri") or "")
    if not provider or not Path(provider).expanduser().exists():
        raise FileNotFoundError(msg_provider_missing(provider or "<empty>"))
    universe_file = config.get("universe_file")
    if not universe_file:
        universe_file = str(Path(provider).expanduser() / "instruments" / "all.txt")
    metadata_csv = config.get("stock_list_csv") or config.get("fund_list_csv")
    _code_map, excluded, selected_codes = build_code_name_map(
        universe_file,
        set(),
        None,
        manual_excludes=list(C.ALWAYS_DROP_CODES),
        cluster_excludes=list(C.ALWAYS_DROP_CLUSTER_CODES),
        cluster_reference_map_csv=C.CLUSTER_REFERENCE_MAP_CSV,
        name_csv=metadata_csv,
    )
    prefixes = tuple(prefix.upper() for prefix in C.STOCK_EXCLUDE_CODE_PREFIXES)
    selected_codes = [code for code in selected_codes if not str(code).upper().startswith(prefixes)]
    max_universe = int(config.get("max_universe_size") or 0)
    if max_universe > 0 and len(selected_codes) > max_universe:
        raise RuntimeError(
            f"Instrument universe has {len(selected_codes)} codes, above max_universe_size={max_universe}. "
            "Cause: the configured universe file is wider than the cap. "
            "Fix: point universe_file at csi1800.txt, or raise max_universe_size in the local json."
        )
    end = date.today()
    start = end - timedelta(days=LOAD_LOOKBACK_DAYS)
    print(f"Config: {config.get('provider_uri')} universe={universe_file}")
    print(
        "POOL_* ignored. Decision band uses these defaults: "
        f"target={AUDIT_TARGET} dist_t=0.60 h={HORIZON} clone={CLONE_THRESHOLD}"
    )
    close, vol = load_data(
        selected_codes,
        provider_uri=provider,
        test_period=(start.isoformat(), end.isoformat()),
        excluded_codes=excluded.union(C.ALWAYS_DROP_CODES),
        ffill=False,
    )
    if C.PRESCREEN_ENABLED and vol is not None and not vol.empty:
        from ashare_daily.pool.builder.universe_prescreen import prescreen_universe

        kept, _diag = prescreen_universe(close.columns.tolist(), vol, _code_map)
        keep = set(kept)
        close = close[[column for column in close.columns if column in keep]]
        vol = vol[[column for column in vol.columns if column in keep]]
    if close.empty:
        raise RuntimeError(msg_provider_missing(provider))
    return close, vol


def audit(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    jobs = bounded_jobs(args.jobs)
    if jobs != args.jobs:
        print(f"--jobs {args.jobs} capped at {jobs}. Cause: that is the CPU count. Fix: none required.")
    csv_path, json_path = output_paths()
    snapshot = Path(args.snapshot) if args.snapshot else default_snapshot_path()
    metadata = snapshot.with_name(snapshot.name.replace("_snapshot.csv", "_metadata_snapshot.json"))
    if metadata == snapshot:
        metadata = default_snapshot_metadata_path()
    snapshot_codes = None
    trend = None
    if snapshot.is_file():
        snapshot_codes = read_snapshot_codes(snapshot)
        trend = read_trend_as_of(metadata if metadata.is_file() else default_snapshot_metadata_path())
    else:
        print(msg_snapshot_missing(snapshot), file=sys.stderr)
    try:
        close, vol = _load_universe()
        positions = anchor_positions(len(close))
        if args.max_anchors and args.max_anchors > 0:
            positions = positions[-args.max_anchors :]
            band_scope = "smoke"
        else:
            band_scope = "full"
        decision = set(decision_positions(positions))
        payloads = []
        for offset, position in enumerate(positions):
            asof = close.index[position].strftime("%Y-%m-%d")
            payloads.append(_slice_payload(close, vol, position, asof, RANDOM_SEED + offset))
        rows = run_payloads(payloads, jobs)
        rows.sort(key=lambda row: row["asof"])
        for row in rows:
            if row.get("message"):
                print(row["message"])
        decision_asofs = {close.index[pos].strftime("%Y-%m-%d") for pos in decision}
        summary = summarize(
            rows,
            decision_asofs=decision_asofs,
            snapshot_codes=snapshot_codes,
            trend_as_of=trend,
            band_scope=band_scope,
        )
        write_csv(csv_path, rows)
        write_summary(json_path, summary)
    except Exception as exc:
        failed = {
            "status": "failed",
            "band": None,
            "message": str(exc),
            "survivorship_note": SURVIVORSHIP_NOTE,
        }
        write_summary(json_path, failed)
        print(str(exc), file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1
    if summary["status"] != "ok":
        print(summary.get("message") or "status=failed", file=sys.stderr)
        print(f"json: {json_path}")
        return 1
    print(
        f"status=ok band={summary['band']} band_scope={summary['band_scope']} "
        f"decision_recall_pool_clone={summary['decision_recall_pool_clone']:.4f}"
    )
    print(f"csv: {csv_path}")
    print(f"json: {json_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    pin_threads()
    scrub_pool_env()
    drop_builder_modules()
    return audit(argv)


if __name__ == "__main__":
    pin_threads()
    raise SystemExit(main())
