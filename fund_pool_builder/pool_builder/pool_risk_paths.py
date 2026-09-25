"""Resolve fat-tail pool_risk CSV paths (canonical vs lookback-suffixed)."""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import pandas as pd

CANONICAL_POOL_RISK_NAME = "cluster_mapping_selected_pool_risk.csv"
RET_5D_COL = "近5日涨跌幅"
RET_20D_COL = "近20日涨跌幅"


def pack_pool_risk_canonical_path(pack_dir: Path) -> Path:
    return pack_dir / CANONICAL_POOL_RISK_NAME


def find_generated_pool_risk_csv(pack_dir: Path) -> Path | None:
    """Return canonical or newest ``cluster_mapping_selected_pool_risk_*.csv``."""
    canonical = pack_pool_risk_canonical_path(pack_dir)
    if canonical.exists():
        return canonical
    candidates = sorted(
        pack_dir.glob("cluster_mapping_selected_pool_risk_*.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def materialize_canonical_pool_risk(pack_dir: Path) -> Path:
    """Ensure ``cluster_mapping_selected_pool_risk.csv`` exists in pack_dir."""
    canonical = pack_pool_risk_canonical_path(pack_dir)
    found = find_generated_pool_risk_csv(pack_dir)
    if found is None:
        raise FileNotFoundError(
            f"No pool risk CSV under {pack_dir} "
            f"(expected {CANONICAL_POOL_RISK_NAME} or cluster_mapping_selected_pool_risk_*.csv)"
        )
    if found.resolve() != canonical.resolve():
        shutil.copy2(found, canonical)
        print(f"[INFO] pool risk {found.name} -> {canonical.name}")
    return canonical


def _resolve_as_of(provider_uri: str | Path, as_of: str | None = None) -> str:
    if as_of:
        return str(as_of).strip()
    calendar_path = Path(provider_uri).expanduser() / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise RuntimeError(f"Missing qlib calendar: {calendar_path}")
    lines = [ln.strip() for ln in calendar_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Empty qlib calendar: {calendar_path}")
    return lines[-1]


def attach_recent_returns_to_risk_csv(
    risk_csv_path: Path,
    *,
    provider_uri: str,
    as_of: str | None = None,
) -> None:
    """Add ``近5日涨跌幅`` / ``近20日涨跌幅`` columns (unit: percent, e.g. 12.34 = +12.34%)."""
    risk_csv_path = Path(risk_csv_path).expanduser()
    if not risk_csv_path.exists():
        raise FileNotFoundError(f"Missing pool risk CSV: {risk_csv_path}")

    frame = pd.read_csv(risk_csv_path, encoding="utf-8-sig")
    if "code" not in frame.columns or frame.empty:
        return

    codes = [str(c).strip() for c in frame["code"].tolist() if str(c).strip()]
    if not codes:
        return

    resolved_as_of = _resolve_as_of(provider_uri, as_of)
    start = (pd.Timestamp(resolved_as_of) - pd.tseries.offsets.BDay(40)).strftime("%Y-%m-%d")

    from .data_loading import load_data
    from .trend_board import window_return

    close, _ = load_data(codes, provider_uri=provider_uri, test_period=(start, resolved_as_of))
    end = pd.Timestamp(resolved_as_of)

    ret5_map: dict[str, float] = {}
    ret20_map: dict[str, float] = {}
    for code in dict.fromkeys(codes):
        series = close[code] if code in close.columns else None
        if series is None or series.dropna().empty:
            ret5_map[code] = float("nan")
            ret20_map[code] = float("nan")
            continue
        r5 = window_return(series, end, 5)
        r20 = window_return(series, end, 20)
        ret5_map[code] = round(r5 * 100.0, 4) if math.isfinite(r5) else float("nan")
        ret20_map[code] = round(r20 * 100.0, 4) if math.isfinite(r20) else float("nan")

    code_key = frame["code"].astype(str).str.strip()
    frame[RET_5D_COL] = code_key.map(ret5_map)
    frame[RET_20D_COL] = code_key.map(ret20_map)

    # Place the two return columns after cluster (or code) for readability.
    cols = [c for c in frame.columns if c not in (RET_5D_COL, RET_20D_COL)]
    if "cluster" in cols:
        insert_at = cols.index("cluster") + 1
    elif "code" in cols:
        insert_at = cols.index("code") + 1
    else:
        insert_at = len(cols)
    cols = cols[:insert_at] + [RET_5D_COL, RET_20D_COL] + cols[insert_at:]
    frame = frame.loc[:, cols]
    frame.to_csv(risk_csv_path, index=False, encoding="utf-8-sig")
    print(
        f"[INFO] attached {RET_5D_COL}/{RET_20D_COL} (as_of={resolved_as_of}, pct) -> {risk_csv_path}"
    )
