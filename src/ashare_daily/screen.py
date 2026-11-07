"""Daily low-correlation / high-return screen over the csi1800 universe.

Writes ``stock_screen_scores.csv`` and ``stock_screen_selected.txt`` under
``TEMP_DIR``, then one scores CSV each for the latest csi300, csi500, and
csi800 constituents. The csi300 file also carries ``权重日期`` and ``权重``.
The txt is a qlib instruments file filtered from ``all.txt``.
The same instruments file is also written to
``instruments/cluster_mapping_selected.txt``, which ``regime`` and ``hrp``
still read, unless ``--canonical csi300`` replaces that file with the current
300 CSI 300 constituents. ``adaptive`` and ``listing`` read the ``TEMP_DIR`` copy.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ashare_daily.paths import (
    ALL_INSTRUMENTS_TXT,
    CLUSTER_MAPPING_SELECTED_TXT,
    QLIB_INSTRUMENTS_DIR,
    QLIB_PROVIDER_URI,
    TEMP_DIR,
)
from ashare_daily.csi300_l2_hedge import L2_SCORES_NAME, industry_weight_representatives
from ashare_daily.pool.merge_csi1800 import latest_end, load_rows
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
INDEX_SCORE_NAMES = ("csi300", "csi500", "csi800")
SHENWAN_COLUMNS = ("申万三级代码", "申万三级名称", "申万二级代码", "申万二级名称")
SHENWAN_FILE_NAME = "sw_third_constituents.csv"
SHENWAN_L2_FILE_NAME = "sw_second_constituents.csv"
SHENWAN_L2_COLUMNS = ("申万二级代码", "申万二级名称")


def rolling_test_period(
    as_of: str | date | None = None,
    lookback_days: int = C.CLUSTER_LOOKBACK_DAYS,
) -> tuple[str, str]:
    """Calendar window that ends on ``as_of`` and holds enough sessions.

    The screen scores the last ``lookback_days`` trading rows. The window
    must follow the run date. ``test_period`` in ``shared_filter_config.json``
    is fixed text, so reading it froze every score on the day it was written.
    Qlib clips an end date past its calendar, so ``as_of`` may be today.
    """
    end = pd.Timestamp(as_of if as_of is not None else date.today()).normalize()
    span = int(int(lookback_days) * 1.6) + 30
    start = end - pd.Timedelta(days=span)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def default_paths() -> tuple[Path, Path]:
    root = Path(TEMP_DIR)
    return root / SCORES_NAME, root / SELECTED_NAME


def latest_constituent_rows(path: Path) -> list[tuple[str, str, str]]:
    """One row per code whose membership ends on this file's newest end date."""
    rows = load_rows(path)
    try:
        end = latest_end(rows)
    except ValueError as exc:
        raise RuntimeError(f"{path}: {exc}") from exc
    by_code: dict[str, tuple[str, str, str]] = {}
    for code, start, row_end in rows:
        if row_end == end:
            by_code[code.upper()] = (code.upper(), start, row_end)
    return [by_code[code] for code in sorted(by_code)]


def latest_constituent_codes(path: Path) -> set[str]:
    """Codes whose membership interval ends on this file's newest end date."""
    return {code for code, _, _ in latest_constituent_rows(path)}


def write_latest_constituent_txt(
    index_name: str,
    dest: Path,
    instruments_dir: Path,
    *,
    expect: int | None = None,
) -> Path:
    """Write the newest membership rows of an index instruments file."""
    source = Path(instruments_dir) / f"{index_name}.txt"
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = latest_constituent_rows(source)
    if expect is not None and len(rows) != expect:
        raise RuntimeError(f"{source} latest constituents are {len(rows)}, expected {expect}")
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("".join(f"{code}\t{start}\t{end}\n" for code, start, end in rows), encoding="utf-8")
    return dest


def _symbol_key(code: object) -> str:
    """Last six digits of a qlib code or a CSIndex constituent code."""
    digits = "".join(ch for ch in str(code).strip() if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else ""


def _normalize_csi300_weights(weights: pd.DataFrame) -> pd.DataFrame:
    required = {"日期", "成分券代码", "权重"}
    missing = required.difference(weights.columns)
    if missing:
        raise RuntimeError(
            "CSI 300 weight table missing columns: " + ", ".join(sorted(missing))
        )
    table = pd.DataFrame(
        {
            "_weight_key": weights["成分券代码"].map(_symbol_key),
            "权重日期": pd.to_datetime(weights["日期"], errors="coerce").dt.strftime("%Y-%m-%d"),
            "权重": pd.to_numeric(weights["权重"], errors="coerce"),
        }
    )
    table = table.loc[table["_weight_key"].str.len() == 6]
    return (
        table.sort_values("权重日期", kind="stable", na_position="first")
        .drop_duplicates("_weight_key", keep="last")
        .reset_index(drop=True)
    )


def fetch_csi300_weights() -> pd.DataFrame:
    """Latest CSI 300 close weights from the CSIndex file akshare downloads."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("akshare is required to load CSI 300 weights") from exc
    try:
        raw = ak.index_stock_cons_weight_csindex(symbol="000300")
    except Exception as exc:
        raise RuntimeError("failed to load CSI 300 weights from csindex") from exc
    if not isinstance(raw, pd.DataFrame):
        raise RuntimeError("CSI 300 weight table was not a dataframe")
    return raw


def attach_csi300_weights(frame: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Append 权重日期 and 权重. Unmatched constituents stay blank."""
    table = _normalize_csi300_weights(weights)
    out = frame.copy()
    out["_weight_key"] = out["code"].map(_symbol_key)
    merged = out.merge(table, on="_weight_key", how="left")
    merged = merged.drop(columns="_weight_key")
    merged["权重日期"] = merged["权重日期"].fillna("")
    return merged


def write_index_score_slices(
    frame: pd.DataFrame,
    scores_path: Path,
    instruments_dir: Path,
    *,
    csi300_weights: pd.DataFrame | None = None,
) -> list[dict[str, object]]:
    """Filter the score table to each index's latest constituents.

    All three instrument files are read before any slice is written.
    Only the csi300 slice gains 权重日期 and 权重. That slice also writes
    stock_screen_scores_csi300_l2.csv, one highest-score name per 申万二级.
    """
    prepared: list[tuple[str, set[str]]] = []
    for name in INDEX_SCORE_NAMES:
        path = Path(instruments_dir) / f"{name}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        prepared.append((name, latest_constituent_codes(path)))

    code_key = frame["code"].astype(str).str.upper()
    written: list[dict[str, object]] = []
    for name, codes in prepared:
        subset = frame.loc[code_key.isin(codes)].copy()
        if name == "csi300":
            weights = csi300_weights if csi300_weights is not None else fetch_csi300_weights()
            subset = attach_csi300_weights(subset, weights)
        out = Path(scores_path).with_name(f"stock_screen_scores_{name}.csv")
        subset.to_csv(out, index=False, encoding="utf-8-sig")
        item: dict[str, object] = {"name": name, "path": out, "rows": int(len(subset))}
        if name == "csi300":
            representatives = industry_weight_representatives(subset)
            l2_path = out.with_name(L2_SCORES_NAME)
            representatives.to_csv(l2_path, index=False, encoding="utf-8-sig")
            item["l2_path"] = l2_path
            item["l2_rows"] = int(len(representatives))
        written.append(item)
    return written


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


def append_unscored(
    frame: pd.DataFrame,
    universe: list[str],
    *,
    valid_days: pd.Series,
    returns_raw: pd.DataFrame,
    name_map: dict[str, str],
) -> pd.DataFrame:
    """Keep every universe code. Names that were not scored get score 0."""
    scored = set(frame["code"].astype(str).str.upper())
    missing = [code for code in universe if code.upper() not in scored]
    if not missing:
        return frame
    rows: list[dict[str, object]] = []
    for code in missing:
        if code in returns_raw.columns:
            series = returns_raw[code].dropna()
            cum = float((series + 1.0).prod() - 1.0) if len(series) else 0.0
            days = int(valid_days.get(code, 0))
        else:
            cum = 0.0
            days = 0
        rows.append(
            {
                "code": code,
                "name": name_map.get(code) or name_map.get(code[2:]) or code,
                "score": 0.0,
                "cum_return": cum,
                "mean_abs_corr": 0.0,
                "valid_days": days,
                "selected": False,
            }
        )
    extra = pd.DataFrame(rows).sort_values("code").reset_index(drop=True)
    return pd.concat([frame, extra], ignore_index=True)


def _merge_industry(frame: pd.DataFrame, industry_csv: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    if not industry_csv.is_file():
        return frame
    industry = pd.read_csv(industry_csv, dtype=str).fillna("")
    if "code" not in industry.columns:
        return frame
    keep = [column for column in ("code", *columns) if column in industry.columns]
    industry = industry[keep].drop_duplicates(subset="code", keep="first")
    industry["code"] = industry["code"].astype(str).str.upper()
    keyed = frame.copy()
    keyed["_code_key"] = keyed["code"].astype(str).str.upper()
    merged = keyed.merge(industry, left_on="_code_key", right_on="code", how="left", suffixes=("", "_sw"))
    return merged.drop(columns=["_code_key", "code_sw"], errors="ignore")


def attach_shenwan(frame: pd.DataFrame, industry_csv: Path, second_csv: Path | None = None) -> pd.DataFrame:
    """Insert Shenwan columns after ``name``.

    Level-3 rows fill all four columns. Where level-2 code is still blank,
    fill it from the level-2 constituent file. Level-3 stays blank in that case.
    """
    out = _merge_industry(frame, industry_csv, SHENWAN_COLUMNS)
    for column in SHENWAN_COLUMNS:
        if column not in out.columns:
            out[column] = ""
        out[column] = out[column].fillna("").astype(str)
    level2_path = second_csv if second_csv is not None else industry_csv.with_name(SHENWAN_L2_FILE_NAME)
    level2 = _merge_industry(out.drop(columns=list(SHENWAN_L2_COLUMNS), errors="ignore"), level2_path, SHENWAN_L2_COLUMNS)
    for column in SHENWAN_L2_COLUMNS:
        if column not in level2.columns:
            level2[column] = ""
        incoming = level2[column].fillna("").astype(str)
        current = out[column].fillna("").astype(str)
        out[column] = current.where(current.str.len() > 0, incoming)
    front = ["code", "name", *SHENWAN_COLUMNS]
    rest = [column for column in out.columns if column not in front]
    return out[front + rest]


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
    instruments_dir: str | Path | None = None,
) -> dict[str, object]:
    codes = _read_instrument_txt(universe_file)
    if not codes:
        raise RuntimeError(f"no codes in {universe_file}")
    if test_period is None:
        test_period = rolling_test_period(lookback_days=lookback_days)
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
    frame = append_unscored(
        score_table(
            returns,
            corr,
            score_min=score_min,
            valid_days=valid_days,
            name_map=name_map,
        ),
        codes,
        valid_days=valid_days,
        returns_raw=returns_raw,
        name_map=name_map,
    )
    scores_path, selected_path = (
        (scores_out, selected_out)
        if scores_out is not None and selected_out is not None
        else default_paths()
    )
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    frame = attach_shenwan(frame, scores_path.parent / SHENWAN_FILE_NAME)
    frame.to_csv(scores_path, index=False, encoding="utf-8-sig")
    selected_csv = scores_path.with_name(scores_path.stem + "_selected.csv")
    frame.loc[frame["selected"]].to_csv(selected_csv, index=False, encoding="utf-8-sig")
    filter_all_txt(
        csv_path=selected_csv,
        all_path=all_txt,
        output_path=selected_path,
        code_column="code",
    )
    index_dir = Path(instruments_dir) if instruments_dir is not None else Path(QLIB_INSTRUMENTS_DIR)
    return {
        "scores_csv": scores_path,
        "selected_txt": selected_path,
        "index_scores": write_index_score_slices(frame, scores_path, index_dir),
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


def run_screen(
    *,
    score_min: float = DEFAULT_SCORE_MIN,
    canonical: str = "score",
    as_of: str | None = None,
) -> dict[str, object]:
    from ashare_daily.pool.builder.config import load_shared_config

    cfg = load_shared_config()
    result = screen_universe(
        provider_uri=cfg.get("provider_uri") or QLIB_PROVIDER_URI,
        universe_file=cfg["universe_file"],
        all_txt=cfg.get("all_txt") or ALL_INSTRUMENTS_TXT,
        score_min=score_min,
        test_period=rolling_test_period(as_of),
        stock_list_csv=cfg.get("stock_list_csv"),
    )
    if canonical == "csi300":
        dest = write_latest_constituent_txt(
            "csi300",
            Path(CLUSTER_MAPPING_SELECTED_TXT),
            Path(QLIB_INSTRUMENTS_DIR),
            expect=300,
        )
        result["canonical_txt"] = dest
        result["canonical_n"] = 300
    elif canonical == "score":
        result["canonical_txt"] = publish_selected_txt(Path(result["selected_txt"]))
    else:
        raise RuntimeError(f"unknown canonical universe: {canonical}")
    return result
