"""Trend-board metrics for cluster mapping export (aligned with decision_pack holdings_trend_board)."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from . import data_loading as _dl

# Same column order as etf_strategy_clean/decision_pack TREND_BOARD_COLUMNS (after code/name).
TREND_BOARD_COLUMNS = [
    "weight",
    "close",
    "ma5",
    "ma10",
    "ma20",
    "dist_ma20_pct",
    "ret_5d",
    "ret_20d",
    "ma_stack",
    "vol_pct_120d",
    "vol_ann_20d",
    "vol_ann_5d",
    "vol_ratio_5_20",
    "vol_level",
    "dd_20d_high",
    "pnl_pct",
    "days_below_ma20",
    "rs_20d_vs_bench",
    "risk_flags",
]

DEFAULT_BENCHMARK_CODE = "SH510300"
DEFAULT_MA_WINDOW = 20
DEFAULT_VOL_WINDOW = 20
DEFAULT_VOL_LOOKBACK = 120

DEFAULT_TREND_THRESHOLDS = {
    "heavy_weight": 0.10,
    "high_vol_pct": 0.80,
    "vol_level_low_max": 0.33,
    "vol_level_mid_max": 0.80,
    "deep_below_ma20_pct": -5.0,
    "weak_rs_20d": -0.05,
    "stale_below_ma20_days": 3,
}

_METRIC_COLUMNS = TREND_BOARD_COLUMNS[1:]  # exclude weight


def _slice_through(end: pd.Timestamp, close: pd.Series) -> pd.Series:
    return close.loc[close.index <= end].dropna()


def window_return(close: pd.Series, end: pd.Timestamp, days: int) -> float:
    available = _slice_through(end, close)
    if len(available) < days + 1:
        return float("nan")
    window = available.tail(days + 1)
    start = float(window.iloc[0])
    finish = float(window.iloc[-1])
    if start <= 0 or not math.isfinite(start) or not math.isfinite(finish):
        return float("nan")
    return finish / start - 1.0


def moving_average(close: pd.Series, end: pd.Timestamp, window: int) -> float:
    available = _slice_through(end, close)
    if len(available) < window:
        return float("nan")
    return float(available.tail(window).mean())


def days_below_ma(close: pd.Series, end: pd.Timestamp, window: int = 20) -> int:
    available = _slice_through(end, close)
    if len(available) < window:
        return 0
    count = 0
    for end_idx in range(len(available) - 1, window - 2, -1):
        window_slice = available.iloc[: end_idx + 1]
        ma = float(window_slice.tail(window).mean())
        current = float(window_slice.iloc[-1])
        if not math.isfinite(ma) or not math.isfinite(current):
            break
        if current < ma:
            count += 1
        else:
            break
    return count


def vol_percentile(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int,
    lookback: int,
) -> float:
    available = _slice_through(end, close)
    if len(available) < vol_window + 5:
        return float("nan")
    returns = available.pct_change().dropna()
    realized = returns.rolling(vol_window).std() * math.sqrt(252.0)
    realized = realized.dropna()
    if realized.empty:
        return float("nan")
    current = float(realized.iloc[-1])
    history = realized.tail(lookback)
    if len(history) < 5:
        return float("nan")
    return float((history <= current).sum()) / float(len(history))


def realized_vol_annualized(close: pd.Series, end: pd.Timestamp, *, vol_window: int) -> float:
    available = _slice_through(end, close)
    if len(available) < vol_window + 1:
        return float("nan")
    returns = available.pct_change().dropna()
    window = returns.tail(vol_window)
    if len(window) < vol_window:
        return float("nan")
    std = float(window.std())
    if not math.isfinite(std):
        return float("nan")
    return std * math.sqrt(252.0)


def vol_level_label(
    vol_pct: float | None,
    *,
    low_max: float = 0.33,
    mid_max: float = 0.80,
) -> str | None:
    if vol_pct is None or not math.isfinite(vol_pct):
        return None
    if vol_pct <= low_max:
        return "low"
    if vol_pct <= mid_max:
        return "mid"
    return "high"


def drawdown_from_high(close: pd.Series, end: pd.Timestamp, *, lookback: int) -> float:
    available = _slice_through(end, close)
    if len(available) < 2:
        return float("nan")
    window = available.tail(lookback)
    peak = float(window.max())
    current = float(window.iloc[-1])
    if peak <= 0 or not math.isfinite(peak) or not math.isfinite(current):
        return float("nan")
    return current / peak - 1.0


def ma_stack_label(ma5: float, ma10: float, ma20: float) -> str:
    if not all(math.isfinite(x) for x in (ma5, ma10, ma20)):
        return "mixed"
    if ma5 > ma10 > ma20:
        return "bull"
    if ma5 < ma10 < ma20:
        return "bear"
    return "mixed"


def compute_trend_metrics(
    close: pd.Series,
    end: pd.Timestamp,
    *,
    vol_window: int,
    vol_lookback: int,
    dd_lookback: int = 20,
) -> dict[str, float | str | None]:
    available = _slice_through(end, close)
    last_close = float(available.iloc[-1]) if not available.empty else float("nan")

    ma5 = moving_average(close, end, 5)
    ma10 = moving_average(close, end, 10)
    ma20 = moving_average(close, end, 20)

    dist_ma20_pct = float("nan")
    if math.isfinite(last_close) and math.isfinite(ma20) and ma20 > 0:
        dist_ma20_pct = (last_close / ma20 - 1.0) * 100.0

    vol_pct = vol_percentile(close, end, vol_window=vol_window, lookback=vol_lookback)
    vol_ann_20 = realized_vol_annualized(close, end, vol_window=vol_window)
    vol_ann_5 = realized_vol_annualized(close, end, vol_window=5)
    vol_ratio_5_20 = float("nan")
    if math.isfinite(vol_ann_5) and math.isfinite(vol_ann_20) and vol_ann_20 > 0:
        vol_ratio_5_20 = vol_ann_5 / vol_ann_20
    ret_5d = window_return(close, end, 5)
    ret_20d = window_return(close, end, 20)
    dd_20d = drawdown_from_high(close, end, lookback=dd_lookback)

    def _opt(value: float) -> float | None:
        return value if math.isfinite(value) else None

    return {
        "close": _opt(last_close),
        "ma5": _opt(ma5),
        "ma10": _opt(ma10),
        "ma20": _opt(ma20),
        "vol_pct_120d": _opt(vol_pct),
        "vol_ann_20d": _opt(vol_ann_20),
        "vol_ann_5d": _opt(vol_ann_5),
        "vol_ratio_5_20": _opt(vol_ratio_5_20),
        "dist_ma20_pct": _opt(dist_ma20_pct),
        "ret_5d": _opt(ret_5d),
        "ret_20d": _opt(ret_20d),
        "ma_stack": ma_stack_label(ma5, ma10, ma20),
        "dd_20d_high": _opt(dd_20d),
    }


def build_risk_flags(
    *,
    dist_ma20_pct: float | None,
    vol_pct_120d: float | None,
    weight: float,
    pnl_pct: float | None,
    ma_stack: str | None,
    rs_20d_vs_bench: float | None,
    days_below_ma20: int | None,
    thresholds: dict[str, float] | None = None,
) -> str:
    t = thresholds or DEFAULT_TREND_THRESHOLDS
    flags: list[str] = []
    if dist_ma20_pct is not None and dist_ma20_pct < 0:
        flags.append("below_ma20")
    if dist_ma20_pct is not None and dist_ma20_pct < t["deep_below_ma20_pct"]:
        flags.append("deep_below_ma20")
    stale_days = int(t["stale_below_ma20_days"])
    if days_below_ma20 is not None and days_below_ma20 >= stale_days:
        flags.append(f"below_ma20_{stale_days}d+")
    if vol_pct_120d is not None and vol_pct_120d >= t["high_vol_pct"]:
        flags.append("high_vol")
    if weight >= t["heavy_weight"]:
        flags.append("heavy_wt")
    if pnl_pct is not None and pnl_pct < 0:
        flags.append("loss")
    if ma_stack == "bear":
        flags.append("bear_stack")
    if rs_20d_vs_bench is not None and rs_20d_vs_bench < t["weak_rs_20d"]:
        flags.append("weak_rs")
    return "|".join(flags)


def warmup_start(as_of: str, *, vol_lookback: int, vol_window: int, ma_window: int) -> str:
    end = pd.Timestamp(as_of)
    days = vol_lookback + vol_window + ma_window + 40
    return (end - pd.tseries.offsets.BDay(days)).strftime("%Y-%m-%d")


def load_benchmark_close(
    provider_uri: str,
    benchmark_code: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    if _dl.D is None:
        raise RuntimeError("qlib is not initialized; call load_data first")
    raw = _dl.D.features(
        [benchmark_code],
        ["$close"],
        start_time=start_date,
        end_time=end_date,
    )
    if raw is None or raw.empty:
        return pd.Series(dtype=float)
    series = raw["$close"]
    if isinstance(series.index, pd.MultiIndex):
        series = series.droplevel(0)
    series.index = pd.to_datetime(series.index)
    return series.sort_index().dropna()


def build_mapping_trend_frame(
    close: pd.DataFrame,
    codes: list[str],
    as_of: str,
    *,
    provider_uri: str | None = None,
    benchmark_code: str = DEFAULT_BENCHMARK_CODE,
    vol_window: int = DEFAULT_VOL_WINDOW,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    ma_window: int = DEFAULT_MA_WINDOW,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Per-code trend board rows for mapping export (weight=0, no pnl)."""
    t = thresholds or DEFAULT_TREND_THRESHOLDS
    end = pd.Timestamp(as_of)
    close = close.sort_index()
    start = warmup_start(as_of, vol_lookback=vol_lookback, vol_window=vol_window, ma_window=ma_window)
    panel = close.loc[close.index >= pd.Timestamp(start)]

    bench_ret_20d: float | None = None
    if provider_uri:
        bench = load_benchmark_close(provider_uri, benchmark_code, start, as_of)
        if not bench.empty:
            raw_bench_ret = window_return(bench, end, 20)
            bench_ret_20d = raw_bench_ret if pd.notna(raw_bench_ret) else None

    rows: list[dict[str, Any]] = []
    for code in codes:
        metrics: dict[str, Any] = {key: None for key in _METRIC_COLUMNS}
        if code in panel.columns:
            series = pd.to_numeric(panel[code], errors="coerce").dropna()
            if not series.empty:
                metrics = compute_trend_metrics(
                    series,
                    end,
                    vol_window=vol_window,
                    vol_lookback=vol_lookback,
                    dd_lookback=20,
                )
                metrics["days_below_ma20"] = days_below_ma(series, end, ma_window)
        metrics["vol_level"] = vol_level_label(
            metrics.get("vol_pct_120d"),
            low_max=t["vol_level_low_max"],
            mid_max=t["vol_level_mid_max"],
        )
        metrics["pnl_pct"] = None
        ret_20d = metrics.get("ret_20d")
        if ret_20d is not None and bench_ret_20d is not None:
            metrics["rs_20d_vs_bench"] = ret_20d - bench_ret_20d
        metrics["risk_flags"] = build_risk_flags(
            dist_ma20_pct=metrics.get("dist_ma20_pct"),
            vol_pct_120d=metrics.get("vol_pct_120d"),
            weight=0.0,
            pnl_pct=None,
            ma_stack=metrics.get("ma_stack"),
            rs_20d_vs_bench=metrics.get("rs_20d_vs_bench"),
            days_below_ma20=metrics.get("days_below_ma20"),
            thresholds=t,
        )
        rows.append(
            {
                "code": code,
                "weight": 0.0,
                **{key: metrics.get(key) for key in _METRIC_COLUMNS},
            }
        )

    return pd.DataFrame(rows, columns=["code", "weight"] + list(_METRIC_COLUMNS))
