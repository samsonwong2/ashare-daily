"""One CSI300 name per Shenwan level-2 industry, hedged with SH000300.

Picks come from the existing screen scores. Returns start on 2026-07-01.
Run once with ``python -m ashare_daily.csi300_l2_hedge``.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from ashare_daily.paths import QLIB_PROVIDER_URI, TEMP_DIR

START = "2026-07-01"
INDEX_CODE = "SH000300"
INDUSTRY_COL = "申万二级名称"
WEIGHT_SUM_COL = "所有此申万二级权重"
SCORES_NAME = "stock_screen_scores_csi300.csv"
L2_SCORES_NAME = "stock_screen_scores_csi300_l2.csv"
PICKS_NAME = "csi300_l2_picks.csv"
DAILY_NAME = "csi300_l2_hedge_daily.csv"


def pick_one_per_industry(frame: pd.DataFrame) -> pd.DataFrame:
    """Highest score in each level-2 industry. Ties break by code."""
    work = frame.copy()
    work[INDUSTRY_COL] = work[INDUSTRY_COL].fillna("").astype(str).str.strip()
    work = work[work[INDUSTRY_COL] != ""]
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["code"] = work["code"].astype(str).str.upper()
    work = work.sort_values([INDUSTRY_COL, "score", "code"], ascending=[True, False, True])
    picked = work.groupby(INDUSTRY_COL, as_index=False, sort=False).head(1)
    columns = [name for name in ("code", "name", "申万二级代码", INDUSTRY_COL, "score") if name in picked.columns]
    return picked[columns].sort_values(INDUSTRY_COL).reset_index(drop=True)


def industry_weight_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """One full row per level-2 industry, plus that industry's total CSI 300 weight.

    Blank industry names are left out of both the rows and the weight sum.
    The kept row is the highest score; equal scores keep the smaller code.
    Missing weights are skipped. An industry whose weights are all missing
    gets an empty total.
    """
    if INDUSTRY_COL not in frame.columns or "score" not in frame.columns or "code" not in frame.columns:
        raise RuntimeError("CSI 300 score table needs 申万二级名称, code, and score")
    if "权重" not in frame.columns:
        raise RuntimeError("CSI 300 score table has no 权重 column")
    work = frame.copy()
    work[INDUSTRY_COL] = work[INDUSTRY_COL].fillna("").astype(str).str.strip()
    work = work[work[INDUSTRY_COL] != ""]
    totals = (
        pd.to_numeric(work["权重"], errors="coerce")
        .groupby(work[INDUSTRY_COL], sort=False)
        .sum(min_count=1)
    )
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["code"] = work["code"].astype(str).str.upper()
    work = work.sort_values([INDUSTRY_COL, "score", "code"], ascending=[True, False, True])
    picked = work.groupby(INDUSTRY_COL, as_index=False, sort=False).head(1).copy()
    picked[WEIGHT_SUM_COL] = picked[INDUSTRY_COL].map(totals)
    return picked.sort_values(INDUSTRY_COL).reset_index(drop=True)


def _calendar(provider: Path) -> list[str]:
    path = provider / "calendars" / "day.txt"
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _close_on_calendar(provider: Path, code: str, calendar: list[str]) -> dict[str, float]:
    path = provider / "features" / code.lower() / "close.day.bin"
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        return {}
    start = int(raw[0])
    values = raw[1:]
    out: dict[str, float] = {}
    for offset, value in enumerate(values):
        pos = start + offset
        if pos < 0 or pos >= len(calendar):
            continue
        number = float(value)
        if math.isfinite(number):
            out[calendar[pos]] = number
    return out


def hedge_window(
    codes: list[str],
    *,
    provider: Path,
    start: str = START,
) -> pd.DataFrame:
    """Equal-weight the codes from the session before ``start`` and short the index."""
    calendar = _calendar(provider)
    sessions = [day for day in calendar if day >= start]
    if not sessions:
        raise RuntimeError(f"no sessions on or after {start}")
    anchor = calendar[calendar.index(sessions[0]) - 1]
    prices = {code: _close_on_calendar(provider, code, calendar) for code in codes}
    index = _close_on_calendar(provider, INDEX_CODE, calendar)
    held = [code for code in codes if anchor in prices[code]]
    if not held:
        raise RuntimeError(f"no picked name has a close on {anchor}")
    carried = {code: prices[code][anchor] for code in held}
    market_value = {code: 1.0 / len(held) for code in held}
    rows: list[dict[str, object]] = []
    cum_basket = 0.0
    cum_index = 0.0
    cum_excess = 0.0
    previous_index = index.get(anchor)
    for day in sessions:
        if previous_index is None or day not in index:
            rows.append(
                {
                    "date": day,
                    "basket_ret": math.nan,
                    "index_ret": math.nan,
                    "excess": math.nan,
                    "cum_basket": cum_basket,
                    "cum_index": cum_index,
                    "cum_excess": cum_excess,
                }
            )
            if day in index:
                previous_index = index[day]
            continue
        index_ret = index[day] / previous_index - 1.0
        previous_index = index[day]
        total = sum(market_value.values())
        weighted = 0.0
        for code in held:
            close = prices[code].get(day)
            if close is None:
                ret = 0.0
            else:
                ret = close / carried[code] - 1.0
                carried[code] = close
            weighted += market_value[code] * ret
            market_value[code] *= 1.0 + ret
        basket_ret = weighted / total
        excess = basket_ret - index_ret
        cum_basket += basket_ret
        cum_index += index_ret
        cum_excess += excess
        rows.append(
            {
                "date": day,
                "basket_ret": basket_ret,
                "index_ret": index_ret,
                "excess": excess,
                "cum_basket": cum_basket,
                "cum_index": cum_index,
                "cum_excess": cum_excess,
            }
        )
    return pd.DataFrame(rows)


def run(scores_csv: Path | None = None, out_dir: Path | None = None) -> dict[str, object]:
    directory = Path(out_dir) if out_dir is not None else Path(TEMP_DIR)
    scores_path = Path(scores_csv) if scores_csv is not None else directory / SCORES_NAME
    frame = pd.read_csv(scores_path)
    picks = pick_one_per_industry(frame)
    daily = hedge_window(list(picks["code"]), provider=Path(QLIB_PROVIDER_URI))
    directory.mkdir(parents=True, exist_ok=True)
    picks_path = directory / PICKS_NAME
    daily_path = directory / DAILY_NAME
    picks.to_csv(picks_path, index=False, encoding="utf-8-sig")
    daily.to_csv(daily_path, index=False, encoding="utf-8-sig")
    last = daily.dropna(subset=["excess"]).iloc[-1]
    return {
        "picks": picks_path,
        "daily": daily_path,
        "n_picks": int(len(picks)),
        "start": str(daily["date"].iloc[0]),
        "end": str(last["date"]),
        "cum_basket": float(last["cum_basket"]),
        "cum_index": float(last["cum_index"]),
        "cum_excess": float(last["cum_excess"]),
    }


def main() -> None:
    result = run()
    print(f"[OK] picks={result['n_picks']} {result['start']} -> {result['end']}")
    print(f"[OK] basket={result['cum_basket']:.6f} SH000300={result['cum_index']:.6f} excess={result['cum_excess']:.6f}")
    print(f"[OK] {result['picks']}")
    print(f"[OK] {result['daily']}")


if __name__ == "__main__":
    main()
