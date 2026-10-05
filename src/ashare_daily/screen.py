"""Daily low-correlation / high-return screen over the csi1800 universe.

Writes ``stock_screen_scores.csv`` and ``stock_screen_selected.txt`` under
``TEMP_DIR``. The txt is a qlib instruments file filtered from ``all.txt``.
The same instruments file is also written to
``instruments/cluster_mapping_selected.txt``, which ``regime`` and ``hrp``
still read. ``adaptive`` and ``listing`` read the ``TEMP_DIR`` copy.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ashare_daily.paths import (
    ALL_INSTRUMENTS_TXT,
    CLUSTER_MAPPING_SELECTED_TXT,
    QLIB_PROVIDER_URI,
    TEMP_DIR,
)
from ashare_daily.pool.builder import constants as C
from ashare_daily.pool.builder.clustering import (
    _global_cross_cluster_scores,
    _mean_abs_corr_to_peers,
    pairwise_overlap_matrices,
    prepare_recent_window,
)
from ashare_daily.pool.builder.data_loading import (
    _load_metadata_name_map,
    _read_instrument_txt,
    load_data,
)
from ashare_daily.pool.builder.filter_txt import filter_all_txt

DEFAULT_SCORE_MIN = 0.85
SCORES_NAME = "stock_screen_scores.csv"
SELECTED_NAME = "stock_screen_selected.txt"


def default_paths() -> tuple[Path, Path]:
    root = Path(TEMP_DIR)
    return root / SCORES_NAME, root / SELECTED_NAME


def default_universe_txt() -> Path:
    """Screen list when it exists, otherwise the cluster-map instruments file."""
    if TEMP_DIR is not None:
        selected = Path(TEMP_DIR) / SELECTED_NAME
        if selected.is_file() and selected.stat().st_size > 0:
            return selected
    return Path(CLUSTER_MAPPING_SELECTED_TXT)


def score_table(
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    *,
    score_min: float,
    valid_days: pd.Series,
    name_map: dict[str, str],
) -> pd.DataFrame:
    scores = _global_cross_cluster_scores(
        returns, mode="low_corr_return", corr=corr
    ).sort_values(ascending=False)
    cum = ((returns + 1.0).prod() - 1.0)
    mean_abs = _mean_abs_corr_to_peers(list(returns.columns), returns, corr=corr)
    frame = pd.DataFrame(
        {
            "code": scores.index,
            "name": [name_map.get(c) or name_map.get(c[2:]) or c for c in scores.index],
            "score": scores.to_numpy(float),
            "cum_return": cum.reindex(scores.index).to_numpy(float),
            "mean_abs_corr": mean_abs.reindex(scores.index).to_numpy(float),
            "valid_days": valid_days.reindex(scores.index).fillna(0).astype(int).to_numpy(),
        }
    )
    frame["selected"] = frame["score"] >= float(score_min)
    return frame.sort_values("score", ascending=False).reset_index(drop=True)


def screen_universe(
    *,
    provider_uri: str | Path,
    universe_file: str | Path,
    all_txt: str | Path,
    score_min: float = DEFAULT_SCORE_MIN,
    test_period: tuple[str, str] | None = None,
    lookback_days: int = C.CLUSTER_LOOKBACK_DAYS,
    min_history_days: int = C.MIN_SELECTION_HISTORY_DAYS,
    stock_list_csv: str | Path | None = None,
    scores_out: Path | None = None,
    selected_out: Path | None = None,
) -> dict[str, object]:
    codes = _read_instrument_txt(universe_file)
    if not codes:
        raise RuntimeError(f"no codes in {universe_file}")
    if test_period is None:
        from ashare_daily.pool.builder.config import load_shared_config

        test_period = tuple(load_shared_config()["test_period"])
    close, _vol = load_data(
        codes,
        provider_uri=str(provider_uri),
        test_period=test_period,
    )
    _, _, returns_raw = prepare_recent_window(close, None, lookback_days)
    valid_days = returns_raw.notna().sum()
    eligible = valid_days[valid_days >= int(min_history_days)].index.tolist()
    if not eligible:
        raise RuntimeError("no instrument meets the screen history requirement")
    returns = returns_raw[eligible]
    corr, _, _ = pairwise_overlap_matrices(
        returns, C.MIN_PAIR_OVERLAP_DAYS, C.LOW_OVERLAP_DISTANCE
    )
    name_map = _load_metadata_name_map(stock_list_csv)
    frame = score_table(
        returns,
        corr,
        score_min=score_min,
        valid_days=valid_days,
        name_map=name_map,
    )
    scores_path, selected_path = (
        (scores_out, selected_out)
        if scores_out is not None and selected_out is not None
        else default_paths()
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(scores_path, index=False, encoding="utf-8-sig")
    selected_csv = scores_path.with_name(scores_path.stem + "_selected.csv")
    frame.loc[frame["selected"]].to_csv(selected_csv, index=False, encoding="utf-8-sig")
    filter_all_txt(
        csv_path=selected_csv,
        all_path=all_txt,
        output_path=selected_path,
        code_column="code",
    )
    return {
        "scores_csv": scores_path,
        "selected_txt": selected_path,
        "n_universe": len(codes),
        "n_eligible": len(eligible),
        "n_selected": int(frame["selected"].sum()),
        "score_min": float(score_min),
        "test_period": test_period,
    }


def publish_selected_txt(selected_txt: Path, canonical_txt: Path | None = None) -> Path:
    """Copy the screen instruments file to the path regime and hrp read."""
    dest = Path(canonical_txt) if canonical_txt is not None else Path(CLUSTER_MAPPING_SELECTED_TXT)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(Path(selected_txt).read_bytes())
    return dest


def run_screen(*, score_min: float = DEFAULT_SCORE_MIN) -> dict[str, object]:
    from ashare_daily.pool.builder.config import load_shared_config

    cfg = load_shared_config()
    result = screen_universe(
        provider_uri=cfg.get("provider_uri") or QLIB_PROVIDER_URI,
        universe_file=cfg["universe_file"],
        all_txt=cfg.get("all_txt") or ALL_INSTRUMENTS_TXT,
        score_min=score_min,
        test_period=tuple(cfg["test_period"]),
        stock_list_csv=cfg.get("stock_list_csv"),
    )
    result["canonical_txt"] = publish_selected_txt(Path(result["selected_txt"]))
    return result
