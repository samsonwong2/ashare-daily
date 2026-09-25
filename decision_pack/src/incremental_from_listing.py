"""Reuse the previous from_listing HTML and append each missing causal bar.

Fair-path OLS and expanding P95 rails are causal in ``t``, so overlapping
history can be kept when qfq closes match. A one-day gap appends one bar
(same as the ETF daily run). A longer gap appends each missing trading day
in order instead of rebuilding the figure. Falls back to a full rebuild
when the previous HTML is missing or inconsistent.

Does not change production hold_up method selection.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from decision_pack.src.fair_path_band_signals import (
    FIG_PANELS,
    _decode_plotly_bdata,
    _extract_plotly_traces,
    _plotly_x_to_index,
    load_ohlcv_from_regime_html,
)
from decision_pack.src.plotly_html_autoscale import write_adaptive_html

_OVERLAY_HTML_MARK = re.compile(
    r"_(fig\d|oracle|aux_edge|hard_touch|near_edge)", re.IGNORECASE
)

TRAIL_YAXIS: dict[float, str] = {
    2.0: "y10",
    1.0: "y11",
    0.5: "y12",
    0.25: "y13",
    1.0 / 12.0: "y14",
}

CANDLE_YAXES = ("y", "y1", "y10", "y11", "y12", "y13", "y14")


def find_listing_html(listing_dir: Path, code: str) -> Path | None:
    """Primary adaptive HTML (not research overlays)."""
    code = str(code).upper()
    hits = []
    for p in listing_dir.glob(f"regime_transition_{code}_*_adaptive.html"):
        if _OVERLAY_HTML_MARK.search(p.stem):
            continue
        hits.append(p)
    if not hits:
        return None
    return sorted(hits)[0]


def _as_1d_list(arr: Any) -> list[Any]:
    if arr is None:
        return []
    if isinstance(arr, dict) and "bdata" in arr and "dtype" in arr:
        arr = _decode_plotly_bdata(arr)
    a = np.asarray(arr)
    if a.ndim == 0:
        return [a.item()]
    return a.ravel().tolist()


def _x_values(tr: dict[str, Any]) -> np.ndarray:
    return np.asarray(_as_1d_list(tr.get("x")))


def _trace_dates(tr: dict[str, Any]) -> pd.DatetimeIndex:
    x = tr.get("x")
    if x is None:
        return pd.DatetimeIndex([])
    if isinstance(x, dict) and "bdata" in x:
        x = _decode_plotly_bdata(x)
    return _plotly_x_to_index(x).normalize()


def _fig7_candle(traces: list[dict[str, Any]]) -> dict[str, Any] | None:
    for tr in traces:
        if tr.get("type") != "candlestick":
            continue
        if str(tr.get("yaxis") or "") == "y10":
            return tr
    for tr in traces:
        if tr.get("type") == "candlestick" and str(tr.get("name", "")).startswith("K线"):
            yax = str(tr.get("yaxis") or "y")
            if yax in {"y10", "y11", "y12", "y13", "y14"}:
                return tr
    return None


def overlap_close_ok(
    prev_close: pd.Series,
    new_close: pd.Series,
    *,
    tol: float,
) -> tuple[bool, str]:
    prev_close = pd.to_numeric(prev_close, errors="coerce")
    new_close = pd.to_numeric(new_close, errors="coerce")
    prev_close.index = pd.DatetimeIndex(pd.to_datetime(prev_close.index)).normalize()
    new_close.index = pd.DatetimeIndex(pd.to_datetime(new_close.index)).normalize()
    common = prev_close.index.intersection(new_close.index)
    if len(common) < 5:
        return False, "overlap_short"
    a = prev_close.reindex(common).to_numpy(float)
    b = new_close.reindex(common).to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 1e-12)
    if int(ok.sum()) < 5:
        return False, "overlap_nan"
    rel = np.abs(a[ok] / b[ok] - 1.0)
    if float(np.nanmax(rel)) > float(tol):
        return False, "qfq"
    return True, "ok"


def classify_date_gap(
    prev_last: pd.Timestamp,
    new_index: pd.DatetimeIndex,
) -> str:
    """Return plus_one | same | gap | empty."""
    if new_index.empty:
        return "empty"
    new_last = pd.Timestamp(new_index[-1]).normalize()
    prev_last = pd.Timestamp(prev_last).normalize()
    if new_last == prev_last:
        return "same"
    after = new_index[new_index > prev_last]
    if len(after) == 1:
        return "plus_one"
    if len(after) == 0:
        return "same"
    return "gap"


def _ols_last_path_g(
    close: pd.Series,
    *,
    trail_years: float,
    min_bars: int,
    trading_days_per_year: float = 252.0,
) -> tuple[float, float]:
    """Causal trailing log OLS at the last bar only."""
    y = pd.to_numeric(close, errors="coerce").to_numpy(float)
    n = len(y)
    tdy = float(trading_days_per_year)
    trail_bars = int(round(float(trail_years) * tdy)) if trail_years > 0 else 0
    t = n - 1
    start = 0
    if trail_bars > 0 and (t + 1) >= trail_bars:
        start = t + 1 - trail_bars
    sl = y[start : t + 1]
    ok = np.isfinite(sl) & (sl > 0)
    n_ok = int(ok.sum())
    if n_ok < max(2, int(min_bars)):
        if n_ok < 2:
            return float("nan"), float("nan")
        idx = np.flatnonzero(ok)
        p0 = float(sl[int(idx[0])])
        p1 = float(sl[int(idx[-1])])
        n_steps = int(idx[-1] - idx[0])
        g = (p1 / p0) ** (tdy / n_steps) - 1.0 if n_steps > 0 and p0 > 0 else float("nan")
        path = p0 * ((1.0 + float(g)) ** (n_steps / tdy)) if n_steps > 0 and np.isfinite(g) else p1
        return float(path), float(g)
    jj = np.flatnonzero(ok).astype(float)
    log_y = np.log(sl[ok])
    b, a = np.polyfit(jj, log_y, 1)
    if not np.isfinite(a) or not np.isfinite(b):
        return float("nan"), float("nan")
    path = float(np.exp(a + b * float(jj[-1])))
    g = float(np.exp(b * tdy) - 1.0)
    return path, g


def _new_envelope(
    path_hist: np.ndarray,
    close_hist: np.ndarray,
    *,
    path_new: float,
    close_new: float,
    q: float = 0.95,
    min_bars: int = 63,
) -> tuple[float, float, float]:
    abs_hist: list[float] = []
    for pt, ct in zip(path_hist, close_hist, strict=False):
        pt = float(pt)
        ct = float(ct)
        if np.isfinite(pt) and pt > 0 and np.isfinite(ct) and ct > 0:
            rt = float(np.log(ct / pt))
            if np.isfinite(rt):
                abs_hist.append(abs(rt))
    if np.isfinite(path_new) and path_new > 0 and np.isfinite(close_new) and close_new > 0:
        rt = float(np.log(close_new / path_new))
        if np.isfinite(rt):
            abs_hist.append(abs(rt))
    if len(abs_hist) < max(2, int(min_bars)):
        return float("nan"), float("nan"), float("nan")
    delta = float(np.quantile(np.asarray(abs_hist, dtype=float), float(q)))
    if not np.isfinite(delta) or delta < 0:
        return float("nan"), float("nan"), float("nan")
    upper = float(path_new * np.exp(delta))
    lower = float(path_new * np.exp(-delta))
    return upper, lower, delta


def _horizon_label(years: float) -> str:
    y = float(years)
    months = y * 12.0
    if abs(months - round(months)) < 1e-3 and 0.5 <= months < 12:
        return f"{int(round(months))}个月"
    if abs(y - round(y)) < 1e-6:
        return f"{int(round(y))}年"
    return f"{y:g}年"


def _find_scatter(
    traces: list[dict[str, Any]],
    *,
    yaxis: str,
    name_contains: str,
) -> dict[str, Any] | None:
    for tr in traces:
        if str(tr.get("yaxis") or "") != yaxis:
            continue
        if tr.get("type") not in (None, "scatter"):
            continue
        name = str(tr.get("name") or "")
        if name_contains in name:
            return tr
    return None


def _find_named(
    traces: list[dict[str, Any]],
    *,
    name: str,
    yaxis: str,
) -> dict[str, Any] | None:
    for tr in traces:
        if str(tr.get("name") or "") != name:
            continue
        if str(tr.get("yaxis") or "") != yaxis:
            continue
        return tr
    return None


def _append_scalar(tr: dict[str, Any], field: str, value: Any) -> None:
    vals = _as_1d_list(tr.get(field))
    vals.append(value)
    tr[field] = vals


def _encode_new_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> Any:
    xs = _as_1d_list(tr.get("x"))
    new_ts = pd.Timestamp(new_ts).normalize()
    if xs and isinstance(xs[0], (int, float, np.floating)) and not isinstance(xs[0], bool):
        v0 = float(xs[0])
        if v0 > 1e11:
            return int(new_ts.value // 1_000_000)
        return int(new_ts.timestamp() * 1000)
    sample = str(xs[0]) if xs else ""
    if "T" in sample:
        return f"{new_ts.strftime('%Y-%m-%d')}T{sample.split('T', 1)[1]}"
    return new_ts.strftime("%Y-%m-%d")


def _append_new_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> None:
    xs = _as_1d_list(tr.get("x"))
    xs.append(_encode_new_x(tr, new_ts))
    tr["x"] = xs


def _replace_last_x(tr: dict[str, Any], new_ts: pd.Timestamp) -> None:
    xs = _as_1d_list(tr.get("x"))
    if not xs:
        return
    xs[-1] = _encode_new_x(tr, new_ts)
    tr["x"] = xs


def _expected_trails(extra_trail_years: str | None) -> list[float]:
    from decision_pack.scripts.plot_regime_transition_example import (
        FAIR_PATH_LOG_TREND_TRAIL_YEARS,
        _normalize_extra_trail_years,
    )

    extra = _normalize_extra_trail_years(extra_trail_years)
    return [float(FAIR_PATH_LOG_TREND_TRAIL_YEARS)] + list(extra)


def traces_have_fair_panels(
    traces: list[dict[str, Any]],
    trails: list[float],
) -> bool:
    for y in trails:
        yax = TRAIL_YAXIS.get(float(y))
        if yax is None:
            continue
        hz = _horizon_label(float(y))
        path = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        up = _find_scatter(traces, yaxis=yax, name_contains=f"上轨{hz}")
        lo = _find_scatter(traces, yaxis=yax, name_contains=f"下轨{hz}")
        if path is None or up is None or lo is None:
            return False
    return True


def bump_html_end_tag(src_name: str, new_end: str) -> str:
    end_tag = pd.Timestamp(new_end).strftime("%Y%m%d")
    return re.sub(r"_(\d{8})_adaptive\.html$", f"_{end_tag}_adaptive.html", src_name)


def extract_layout(html: str) -> dict[str, Any]:
    from decision_pack.scripts.overlay_oracle_pivots_on_regime_html import (
        extract_layout as _el,
    )

    return _el(html)


def _update_title_g(layout: dict[str, Any], g_by_years: dict[float, float], deltas: dict[float, float]) -> None:
    title = layout.get("title")
    text = ""
    if isinstance(title, dict):
        text = str(title.get("text") or "")
    elif isinstance(title, str):
        text = title
    for years, g in g_by_years.items():
        if not np.isfinite(g):
            continue
        hz = _horizon_label(years)
        pat = re.compile(rf"({re.escape(hz)}\s*g≈)-?[\d.]+%")
        text = pat.sub(rf"\g<1>{g:.1%}", text)
        text = text.replace(f"{hz} g≈{g:.1%}".replace("%", "％"), f"{hz} g≈{g:.1%}")
    for years, d in deltas.items():
        if not np.isfinite(d):
            continue
        hz = _horizon_label(years)
        pct = f"{d:.1%}"
        text = re.sub(rf"(上轨{re.escape(hz)}\s*)\+[\d.]+%", rf"\g<1>+{pct}", text)
        text = re.sub(rf"(下轨{re.escape(hz)}\s*)-[\d.]+%", rf"\g<1>-{pct}", text)
    if isinstance(title, dict):
        layout["title"] = {**title, "text": text}
    else:
        layout["title"] = text


def _customdata_text(cell: Any) -> str:
    if cell is None:
        return ""
    if isinstance(cell, (list, tuple)):
        return "" if not cell else _customdata_text(cell[0])
    return str(cell)


def _append_customdata(tr: dict[str, Any], text: str) -> None:
    raw = tr.get("customdata")
    if raw is None:
        tr["customdata"] = [text]
        return
    vals = _as_1d_list(raw)
    if vals and isinstance(vals[0], (list, tuple)):
        vals.append([text])
    else:
        vals.append(text)
    tr["customdata"] = vals


def _align_customdata_after_y(tr: dict[str, Any], text: str) -> None:
    raw = tr.get("customdata")
    if raw is None:
        return
    vals = _as_1d_list(raw)
    ys = _as_1d_list(tr.get("y"))
    nested = bool(vals) and isinstance(vals[0], (list, tuple))
    cell: Any = [text] if nested else text
    if len(vals) == len(ys) - 1:
        vals.append(cell)
    elif len(vals) == len(ys) and vals:
        vals[-1] = cell
    else:
        return
    tr["customdata"] = vals


def _align_bar_color_after_y(tr: dict[str, Any], color: str) -> None:
    marker = tr.get("marker")
    if not isinstance(marker, dict):
        return
    colors = marker.get("color")
    if not isinstance(colors, list):
        return
    ys = _as_1d_list(tr.get("y"))
    if len(colors) == len(ys) - 1:
        colors.append(color)
    elif len(colors) == len(ys) and colors:
        colors[-1] = color
    else:
        return
    marker["color"] = colors
    tr["marker"] = marker


def _last_finite(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    v = v[np.isfinite(v.to_numpy(dtype=float))]
    if v.empty:
        return float("nan")
    return float(v.iloc[-1])


def _fig1_5_last_values(
    close: pd.Series,
    ohlcv_last: pd.Series,
    *,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    need_ann_fallback: float | None = None,
    need_ann_5_fallback: float | None = None,
) -> dict[str, Any]:
    """Causal last-bar values for fig1–5 overlays (MA / volume / vol / need-gap)."""
    from decision_pack.src.regime_transition_plot import compute_volatility_regime_frame
    from decision_pack.src.vol_need_return_board import (
        ANCHOR_ERP_ANN,
        HORIZON_5,
        HORIZON_20,
        fair_need_ann_series,
        gap_realized_minus_need,
    )

    out: dict[str, Any] = {}
    c = pd.to_numeric(close, errors="coerce")
    c.index = pd.DatetimeIndex(pd.to_datetime(c.index)).normalize()
    c = c[~c.index.duplicated(keep="last")].sort_index()
    for w in (5, 10, 20):
        out[f"MA{w}"] = float(c.rolling(int(w), min_periods=int(w)).mean().iloc[-1]) if len(c) else float("nan")
    vol_raw = ohlcv_last.get("$volume") if ohlcv_last is not None else None
    out["成交量"] = float(pd.to_numeric(pd.Series([vol_raw]), errors="coerce").iloc[0])
    vf = compute_volatility_regime_frame(c)
    if vf is None or vf.empty:
        return out
    vf = vf.copy()
    vf["as_of"] = pd.to_datetime(vf["as_of"]).dt.normalize()
    last = vf.iloc[-1]
    out["rv5"] = float(last.get("rv5", float("nan")))
    out["rv20"] = float(last.get("rv20", float("nan")))
    out["vol5_pct_120d"] = float(last.get("vol5_pct_120d", float("nan")))
    out["vol_regime"] = str(last.get("vol_regime") or "")
    indexed = vf.set_index("as_of")
    need = None
    if rv20_anchor is not None and not getattr(rv20_anchor, "empty", True):
        need = fair_need_ann_series(indexed["rv20"], rv20_anchor, erp_ann=ANCHOR_ERP_ANN)
    if need is None or need.empty or not np.isfinite(float(need.iloc[-1]) if len(need) else float("nan")):
        fb = need_ann_fallback if need_ann_fallback is not None else ANCHOR_ERP_ANN
        need = pd.Series(float(fb), index=c.index, dtype=float)
    need5 = need
    if (
        rv5_anchor is not None
        and not getattr(rv5_anchor, "empty", True)
        and "rv5" in indexed.columns
    ):
        need5_try = fair_need_ann_series(indexed["rv5"], rv5_anchor, erp_ann=ANCHOR_ERP_ANN)
        if need5_try is not None and not need5_try.empty:
            need5 = need5_try.reindex(need.index)
    elif need_ann_5_fallback is not None:
        need5 = pd.Series(float(need_ann_5_fallback), index=need.index, dtype=float)
    gap20 = gap_realized_minus_need(c, need, bars=HORIZON_20)
    gap5 = gap_realized_minus_need(c, need5, bars=HORIZON_5)
    out["公平年化收益"] = _last_finite(need)
    out["公平年化收益(rv5)"] = _last_finite(need5)
    out["20日差额(实现−公平)"] = _last_finite(gap20["gap"])
    out["5日差额(实现−公平)"] = _last_finite(gap5["gap"])
    out["r_need_20d"] = _last_finite(gap20["r_need_period"])
    out["r_20d"] = _last_finite(gap20["r_realized"])
    out["r_need_5d"] = _last_finite(gap5["r_need_period"])
    out["r_5d"] = _last_finite(gap5["r_realized"])
    return out


def _gap_bar_color(gap: float) -> str:
    if np.isfinite(gap) and float(gap) < 0:
        return "rgba(214,39,40,0.55)"
    return "rgba(44,160,44,0.50)"


def _append_named_overlay(
    traces: list[dict[str, Any]],
    *,
    name: str,
    yaxis: str,
    new_ts: pd.Timestamp,
    value: float,
    expected_len: int,
    customdata: str | None = None,
    bar_color: str | None = None,
) -> None:
    tr = _find_named(traces, name=name, yaxis=yaxis)
    if tr is None:
        return
    xs = _as_1d_list(tr.get("x"))
    ys = _as_1d_list(tr.get("y"))
    if len(xs) != int(expected_len) or len(ys) != int(expected_len):
        return
    _append_new_x(tr, new_ts)
    _append_scalar(tr, "y", value)
    if customdata is not None:
        _align_customdata_after_y(tr, customdata)
    if bar_color is not None:
        _align_bar_color_after_y(tr, bar_color)


def _extend_fig15_ref_lines(traces: list[dict[str, Any]], new_ts: pd.Timestamp) -> None:
    from decision_pack.src.vol_need_return_board import EQUITY_ANCHOR_LABEL

    for tr in traces:
        name = str(tr.get("name") or "")
        xs = _as_1d_list(tr.get("x"))
        if len(xs) != 2:
            continue
        if name in {"分位30%", "分位70%", "分位90%", "差额=0"} or (
            name.startswith(f"{EQUITY_ANCHOR_LABEL}公平")
        ):
            _replace_last_x(tr, new_ts)


def _append_fig1_5_overlays(
    traces: list[dict[str, Any]],
    *,
    close: pd.Series,
    ohlcv_last: pd.Series,
    new_ts: pd.Timestamp,
    expected_len: int,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
) -> None:
    """Recompute fig1–5 last-bar overlays. Do not copy yesterday's y."""
    from decision_pack.scripts.plot_regime_transition_example import (
        _fmt_pct_rank,
        _fmt_vol,
    )
    from decision_pack.src.vol_need_return_board import ANCHOR_ERP_ANN, EQUITY_ANCHOR_LABEL

    need_tr = _find_named(traces, name="公平年化收益", yaxis="y5")
    need5_tr = _find_named(traces, name="公平年化收益(rv5)", yaxis="y7")
    need_fb = None
    need5_fb = None
    if need_tr is not None:
        ys = pd.to_numeric(pd.Series(_as_1d_list(need_tr.get("y")), dtype=object), errors="coerce")
        if len(ys) and np.isfinite(float(ys.iloc[-1])):
            need_fb = float(ys.iloc[-1])
    if need5_tr is not None:
        ys = pd.to_numeric(pd.Series(_as_1d_list(need5_tr.get("y")), dtype=object), errors="coerce")
        if len(ys) and np.isfinite(float(ys.iloc[-1])):
            need5_fb = float(ys.iloc[-1])
    vals = _fig1_5_last_values(
        close,
        ohlcv_last,
        rv20_anchor=rv20_anchor,
        rv5_anchor=rv5_anchor,
        need_ann_fallback=need_fb,
        need_ann_5_fallback=need5_fb,
    )
    erp_txt = f"{float(ANCHOR_ERP_ANN):.1%}"
    date_s = pd.Timestamp(new_ts).strftime("%Y-%m-%d")
    rv5_hover = (
        f"日期={date_s}<br>"
        f"vol_regime={vals.get('vol_regime', '')}<br>"
        f"rv5={_fmt_vol(vals.get('rv5'))}<br>"
        f"rv20={_fmt_vol(vals.get('rv20'))}<br>"
        f"vol5_pct_120d={_fmt_pct_rank(vals.get('vol5_pct_120d'))}<br>"
        f"说明=分位用近120日rv5历史（含当日），无未来数据"
    )

    def _need_hover(bars: int, *, rv5: bool) -> str:
        key_need = "r_need_5d" if bars == 5 else "r_need_20d"
        key_real = "r_5d" if bars == 5 else "r_20d"
        key_gap = "5日差额(实现−公平)" if bars == 5 else "20日差额(实现−公平)"
        key_ann = "公平年化收益(rv5)" if rv5 else "公平年化收益"
        vol_tag = "rv5" if rv5 else "rv20"
        return (
            f"日期={date_s}<br>"
            f"公平年化={_fmt_vol(vals.get(key_ann))}<br>"
            f"公平{bars}日={_fmt_vol(vals.get(key_need))}<br>"
            f"已实现{bars}日={_fmt_vol(vals.get(key_real))}<br>"
            f"差额=已实现−公平{bars}日={_fmt_vol(vals.get(key_gap))}<br>"
            f"公式={erp_txt}×{vol_tag}/{vol_tag}_{EQUITY_ANCHOR_LABEL}（实现波动）"
        )

    def _gap_hover(bars: int) -> str:
        key_need = "r_need_5d" if bars == 5 else "r_need_20d"
        key_real = "r_5d" if bars == 5 else "r_20d"
        key_gap = "5日差额(实现−公平)" if bars == 5 else "20日差额(实现−公平)"
        g = vals.get(key_gap)
        if np.isfinite(float(g)) if g is not None else False:
            gf = float(g)
            note = (
                f"近{bars}日没赚够波动补偿"
                if gf < 0
                else f"近{bars}日覆盖了波动补偿"
            )
        else:
            note = "差额=—"
        return (
            f"日期={date_s}<br>"
            f"已实现{bars}日={_fmt_vol(vals.get(key_real))}<br>"
            f"公平{bars}日={_fmt_vol(vals.get(key_need))}<br>"
            f"差额={_fmt_vol(vals.get(key_gap))}<br>"
            f"{note}"
        )

    for name, yax in (("MA5", "y"), ("MA10", "y"), ("MA20", "y")):
        _append_named_overlay(
            traces, name=name, yaxis=yax, new_ts=new_ts,
            value=float(vals.get(name, float("nan"))), expected_len=expected_len,
        )
    _append_named_overlay(
        traces, name="成交量", yaxis="y2", new_ts=new_ts,
        value=float(vals.get("成交量", float("nan"))), expected_len=expected_len,
    )
    _append_named_overlay(
        traces, name="rv5", yaxis="y3", new_ts=new_ts,
        value=float(vals.get("rv5", float("nan"))), expected_len=expected_len,
        customdata=rv5_hover,
    )
    _append_named_overlay(
        traces, name="rv20", yaxis="y3", new_ts=new_ts,
        value=float(vals.get("rv20", float("nan"))), expected_len=expected_len,
    )
    _append_named_overlay(
        traces, name="vol5_pct_120d", yaxis="y4", new_ts=new_ts,
        value=float(vals.get("vol5_pct_120d", float("nan"))), expected_len=expected_len,
    )
    g20 = float(vals.get("20日差额(实现−公平)", float("nan")))
    g5 = float(vals.get("5日差额(实现−公平)", float("nan")))
    _append_named_overlay(
        traces, name="公平年化收益", yaxis="y5", new_ts=new_ts,
        value=float(vals.get("公平年化收益", float("nan"))), expected_len=expected_len,
        customdata=_need_hover(20, rv5=False),
    )
    _append_named_overlay(
        traces, name="20日差额(实现−公平)", yaxis="y6", new_ts=new_ts,
        value=g20, expected_len=expected_len,
        customdata=_gap_hover(20), bar_color=_gap_bar_color(g20),
    )
    _append_named_overlay(
        traces, name="公平年化收益(rv5)", yaxis="y7", new_ts=new_ts,
        value=float(vals.get("公平年化收益(rv5)", float("nan"))), expected_len=expected_len,
        customdata=_need_hover(5, rv5=True),
    )
    _append_named_overlay(
        traces, name="5日差额(实现−公平)", yaxis="y8", new_ts=new_ts,
        value=g5, expected_len=expected_len,
        customdata=_gap_hover(5), bar_color=_gap_bar_color(g5),
    )
    _extend_fig15_ref_lines(traces, new_ts)


def _append_fig6_nav(
    traces: list[dict[str, Any]],
    *,
    new_ts: pd.Timestamp,
    expected_len: int,
    nav_last: dict[str, float] | None,
) -> None:
    if not nav_last:
        return
    for name, key in (
        ("hold_up净值", "hold_up净值"),
        ("BH净值", "BH净值"),
        ("cash净值", "cash净值"),
    ):
        if key not in nav_last:
            continue
        _append_named_overlay(
            traces,
            name=name,
            yaxis="y9",
            new_ts=new_ts,
            value=float(nav_last[key]),
            expected_len=expected_len,
        )
    policy_v = nav_last.get("nav_policy")
    if policy_v is None:
        return
    for tr in traces:
        if str(tr.get("yaxis") or "") != "y9":
            continue
        if not str(tr.get("name") or "").startswith("本票自选"):
            continue
        xs = _as_1d_list(tr.get("x"))
        ys = _as_1d_list(tr.get("y"))
        if len(xs) != int(expected_len) or len(ys) != int(expected_len):
            continue
        _append_new_x(tr, new_ts)
        _append_scalar(tr, "y", float(policy_v))
        break


def _patch_trail_path_hover(
    prev: str,
    *,
    new_ts: pd.Timestamp,
    path_new: float,
    close_new: float,
    g_new: float,
    g_by: dict[float, float],
    signed_pct: float,
    asset_name: str | None,
) -> str:
    from decision_pack.scripts.plot_regime_transition_example import (
        _fmt_signed_path_percentile,
        _fmt_vol,
        fmt_path_gap_hover_line,
    )

    s = prev or ""
    s = re.sub(r"日期=\d{4}-\d{2}-\d{2}", f"日期={new_ts.date()}", s, count=1)
    s = re.sub(r"(诊断\))=[-+\d.]+", rf"\g<1>={path_new:.4f}", s, count=1)
    s = re.sub(
        r"(当日路径年化\([^)]+\)=)-?[\d.]+%",
        lambda m: f"{m.group(1)}{_fmt_vol(g_new)}",
        s,
        count=1,
    )
    pct_txt = f"相对路径因果分位={_fmt_signed_path_percentile(signed_pct)}"
    s = re.sub(r"相对路径因果分位=[^<]+", pct_txt, s, count=1)
    gap_line = fmt_path_gap_hover_line(asset_name, new_ts, close_new, path_new)
    if re.search(r"<br>[^<]* 约 (?:[+-]\d+%|—)", s):
        s = re.sub(r"<br>[^<]* 约 (?:[+-]\d+%|—)", f"<br>{gap_line}", s, count=1)
    else:
        s = s.replace(pct_txt, f"{pct_txt}<br>{gap_line}", 1)
    for years, g in g_by.items():
        hz = _horizon_label(years)
        s = re.sub(
            rf"(双尺度·{re.escape(hz)}路径年化=)-?[\d.]+%",
            lambda m, gg=g: f"{m.group(1)}{_fmt_vol(gg)}",
            s,
        )
    return s


def append_fair_path_bar(
    traces: list[dict[str, Any]],
    *,
    close: pd.Series,
    ohlcv_last: pd.Series,
    new_ts: pd.Timestamp,
    trails: list[float],
    asset_name: str | None = None,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    nav_last: dict[str, float] | None = None,
) -> tuple[dict[float, float], dict[float, float]]:
    """Append last-bar path/rails/candles. Returns g and delta by trail years."""
    from decision_pack.scripts.plot_regime_transition_example import (
        _trail_fit_min_bars,
        causal_path_residual_signed_percentile,
    )

    px_new = float(ohlcv_last["$close"])
    g_by: dict[float, float] = {}
    d_by: dict[float, float] = {}
    candle_payload = dict(
        open=float(ohlcv_last["$open"]),
        high=float(ohlcv_last["$high"]),
        low=float(ohlcv_last["$low"]),
        close=px_new,
    )
    n_prev = None
    fig7 = _fig7_candle(traces)
    if fig7 is not None:
        n_prev = len(_as_1d_list(fig7.get("close")))

    for tr in traces:
        if tr.get("type") != "candlestick":
            continue
        yax = str(tr.get("yaxis") or "y")
        if yax not in CANDLE_YAXES:
            continue
        _append_new_x(tr, new_ts)
        for k, v in candle_payload.items():
            _append_scalar(tr, k, v)

    for years in trails:
        yax = TRAIL_YAXIS.get(float(years))
        if yax is None:
            continue
        min_b = _trail_fit_min_bars(float(years))
        path_new, g_new = _ols_last_path_g(close, trail_years=float(years), min_bars=min_b)
        g_by[float(years)] = g_new
        hz = _horizon_label(float(years))
        path_tr = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        up_tr = _find_scatter(traces, yaxis=yax, name_contains=f"上轨{hz}")
        lo_tr = _find_scatter(traces, yaxis=yax, name_contains=f"下轨{hz}")
        if path_tr is None:
            continue
        path_hist = np.asarray(_as_1d_list(path_tr.get("y")), dtype=float)
        c_tr = None
        for tr in traces:
            if tr.get("type") == "candlestick" and str(tr.get("yaxis") or "") == yax:
                c_tr = tr
                break
        if c_tr is not None:
            close_hist = np.asarray(_as_1d_list(c_tr.get("close"))[:-1], dtype=float)
        else:
            close_hist = np.full_like(path_hist, np.nan)
        n = min(len(path_hist), len(close_hist))
        upper, lower, delta = _new_envelope(
            path_hist[:n],
            close_hist[:n],
            path_new=path_new,
            close_new=px_new,
            min_bars=min_b,
        )
        d_by[float(years)] = delta
        _append_new_x(path_tr, new_ts)
        _append_scalar(path_tr, "y", path_new)
        if up_tr is not None:
            _append_new_x(up_tr, new_ts)
            _append_scalar(up_tr, "y", upper)
        if lo_tr is not None:
            _append_new_x(lo_tr, new_ts)
            _append_scalar(lo_tr, "y", lower)

    for years in trails:
        yax = TRAIL_YAXIS.get(float(years))
        if yax is None:
            continue
        path_tr = _find_scatter(traces, yaxis=yax, name_contains="公平路径")
        if path_tr is None or path_tr.get("customdata") is None:
            continue
        cd = _as_1d_list(path_tr.get("customdata"))
        ys = _as_1d_list(path_tr.get("y"))
        if len(cd) != len(ys) - 1 or not cd:
            continue
        c_tr = None
        for tr in traces:
            if tr.get("type") == "candlestick" and str(tr.get("yaxis") or "") == yax:
                c_tr = tr
                break
        close_arr = (
            np.asarray(_as_1d_list(c_tr.get("close")), dtype=float)
            if c_tr is not None
            else np.full(len(ys), np.nan)
        )
        path_arr = np.asarray(ys, dtype=float)
        n = min(len(path_arr), len(close_arr))
        signed = causal_path_residual_signed_percentile(
            pd.Series(path_arr[:n]),
            pd.Series(close_arr[:n]),
            min_bars=_trail_fit_min_bars(float(years)),
        )
        last_pct = float(signed.iloc[-1]) if len(signed) else float("nan")
        g_this = float(g_by.get(float(years), float("nan")))
        new_hover = _patch_trail_path_hover(
            _customdata_text(cd[-1]),
            new_ts=new_ts,
            path_new=float(path_arr[-1]) if len(path_arr) else float("nan"),
            close_new=px_new,
            g_new=g_this,
            g_by=g_by,
            signed_pct=last_pct,
            asset_name=asset_name,
        )
        _append_customdata(path_tr, new_hover)

    overlay_len = n_prev
    if overlay_len is None:
        for tr in traces:
            if tr.get("type") != "candlestick":
                continue
            if str(tr.get("yaxis") or "y") == "y":
                n_c = len(_as_1d_list(tr.get("close")))
                overlay_len = max(0, n_c - 1)
                break
    if overlay_len is not None:
        _append_fig1_5_overlays(
            traces,
            close=close,
            ohlcv_last=ohlcv_last,
            new_ts=new_ts,
            expected_len=int(overlay_len),
            rv20_anchor=rv20_anchor,
            rv5_anchor=rv5_anchor,
        )
        _append_fig6_nav(
            traces,
            new_ts=new_ts,
            expected_len=int(overlay_len),
            nav_last=nav_last,
        )
    return g_by, d_by


def _add_trade_markers(
    traces: list[dict[str, Any]],
    *,
    aev: list[dict[str, Any]],
    new_date: str,
    ohlcv_last: pd.Series,
    layout: dict[str, Any],
) -> None:
    from decision_pack.scripts.overlay_oracle_pivots_on_regime_html import (
        FIG_PANEL_AXES,
        _panel_rail_ys,
        _rail_ys,
    )

    new_ev = [e for e in aev if str(e.get("date")) == new_date]
    if not new_ev:
        return
    dummy = pd.DataFrame(
        {
            "$high": [float(ohlcv_last["$high"])],
            "$low": [float(ohlcv_last["$low"])],
        }
    )
    ra, rb = _rail_ys(dummy)
    buys = [e for e in new_ev if str(e.get("side")) == "BUY"]
    sells = [e for e in new_ev if str(e.get("side")) == "SELL"]
    ts = pd.Timestamp(new_date)

    def _mk(side: str, y: float, yaxis: str, xaxis: str, showlegend: bool) -> dict[str, Any]:
        is_buy = side == "BUY"
        return dict(
            type="scatter",
            x=[ts],
            y=[y],
            mode="markers",
            name=("上涨买+" if is_buy else "上涨卖+") if showlegend else None,
            marker=dict(
                symbol="triangle-up" if is_buy else "triangle-down",
                size=12,
                color="#12a150" if is_buy else "#d62728",
            ),
            yaxis=yaxis,
            xaxis=xaxis,
            showlegend=showlegend,
        )

    if buys:
        traces.append(_mk("BUY", ra, "y", "x", True))
    if sells:
        traces.append(_mk("SELL", rb, "y", "x", True))
    for _panel, yaxis, xaxis in FIG_PANEL_AXES:
        pra, prb = _panel_rail_ys(traces, yaxis, layout)
        if buys:
            traces.append(_mk("BUY", pra, yaxis, xaxis, False))
        if sells:
            traces.append(_mk("SELL", prb, yaxis, xaxis, False))


def try_incremental_html(
    *,
    code: str,
    prev_dir: Path,
    out_dir: Path,
    ohlcv: pd.DataFrame,
    end_date: str,
    close_tol: float,
    extra_trail_years: str | None,
    aev: list[dict[str, Any]] | None = None,
    display_name: str | None = None,
    rv20_anchor: pd.Series | None = None,
    rv5_anchor: pd.Series | None = None,
    nav_last: dict[str, float] | None = None,
    nav_by_date: dict[str, dict[str, float]] | None = None,
) -> tuple[Path | None, str]:
    """Patch yesterday HTML. Returns (out_path, reason). reason ok|same|*fallback*."""
    prev = find_listing_html(prev_dir, code)
    if prev is None or not prev.exists():
        return None, "no_prev_html"
    try:
        prev_close, _ = load_ohlcv_from_regime_html(prev)
    except Exception:
        return None, "prev_html_ohlcv"
    work = ohlcv.copy()
    if "datetime" not in work.columns:
        work = work.reset_index()
        if "datetime" not in work.columns and "index" in work.columns:
            work = work.rename(columns={"index": "datetime"})
        if "datetime" not in work.columns:
            work = work.rename(columns={work.columns[0]: "datetime"})
    new_close = work.set_index(pd.to_datetime(work["datetime"]).dt.normalize())["$close"]
    new_close = pd.to_numeric(new_close, errors="coerce")
    new_close = new_close[~new_close.index.duplicated(keep="last")].sort_index()
    ok, why = overlap_close_ok(prev_close, new_close, tol=close_tol)
    if not ok:
        return None, why

    html_txt = prev.read_text(encoding="utf-8")
    try:
        traces = _extract_plotly_traces(html_txt)
        layout = extract_layout(html_txt)
    except Exception:
        return None, "parse_html"

    trails = _expected_trails(extra_trail_years)
    if not traces_have_fair_panels(traces, trails):
        return None, "missing_panels"

    prev_last = pd.Timestamp(prev_close.index.max()).normalize()
    after = new_close.index[new_close.index > prev_last]
    safe = (display_name or code).replace("/", "")
    start_in_name = prev.name.split("_")
    # regime_transition_CODE_NAME_START_END_adaptive.html — keep start from prev
    out_name = bump_html_end_tag(prev.name, end_date)
    out_path = out_dir / out_name

    if len(after) == 0:
        shutil.copy2(prev, out_path)
        return out_path, "same"

    ohlcv = work.copy()
    ohlcv["_d"] = pd.to_datetime(ohlcv["datetime"]).dt.normalize()
    g_by: dict[float, float] = {}
    d_by: dict[float, float] = {}
    print(
        f"[incr] {code} append_bars={len(after)} "
        f"{prev_last.date()} -> {pd.Timestamp(after[-1]).date()}"
    )
    for raw_ts in after:
        new_ts = pd.Timestamp(raw_ts).normalize()
        day_rows = ohlcv.loc[ohlcv["_d"].eq(new_ts)]
        if day_rows.empty:
            return None, "missing_bar"
        last_row = day_rows.iloc[-1]
        # OLS must be evaluated at this bar, not at the series end.
        close_upto = new_close.loc[:new_ts]
        day_key = new_ts.strftime("%Y-%m-%d")
        if nav_by_date is not None:
            day_nav = nav_by_date.get(day_key) or {
                "hold_up净值": float("nan"),
                "BH净值": float("nan"),
                "cash净值": float("nan"),
                "nav_policy": float("nan"),
            }
        elif new_ts == pd.Timestamp(after[-1]).normalize():
            day_nav = nav_last
        else:
            day_nav = {
                "hold_up净值": float("nan"),
                "BH净值": float("nan"),
                "cash净值": float("nan"),
                "nav_policy": float("nan"),
            }
        g_by, d_by = append_fair_path_bar(
            traces,
            close=close_upto,
            ohlcv_last=last_row,
            new_ts=new_ts,
            trails=trails,
            asset_name=display_name,
            rv20_anchor=rv20_anchor,
            rv5_anchor=rv5_anchor,
            nav_last=day_nav,
        )
        _update_title_g(layout, g_by, d_by)
        if aev:
            _add_trade_markers(
                traces,
                aev=aev,
                new_date=new_ts.strftime("%Y-%m-%d"),
                ohlcv_last=last_row,
                layout=layout,
            )
    from decision_pack.src.ewma_fair_rails import attach_ewma_rails

    traces = attach_ewma_rails(traces)
    fig = go.Figure(data=traces, layout=layout)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_adaptive_html(fig, out_path)
    _ = safe
    _ = start_in_name
    return out_path, "ok"


def write_book_csvs(
    *,
    pool: Any,
    code: str,
    cfg: dict,
    ohlcv: pd.DataFrame,
    start_date: str,
    end_date: str,
    out_dir: Path,
    ev: Any,
    hrp_mode_by_code: dict[str, str] | None,
    enable_ru_diag: bool,
    enable_shallow_up_trade: bool,
    name: str,
) -> dict[str, Any]:
    """Regime labels + hold_up book without rebuilding the figure."""
    import numpy as np
    from decision_pack.src.adaptive_stage_common import (
        FIXED_HYBRID_RULES,
        detect_early_regime_marks,
        prepare_features,
        segments,
        simulate_adaptive_book,
    )
    from decision_pack.src.early_down_alert import detect_early_down_marks
    from decision_pack.src.early_down_alert import early_down_flags_frame
    from decision_pack.src.hrp_cluster_router import mode_allows_trade_overlay
    from decision_pack.src.recovery_up_overlay import (
        DIAGNOSTIC_N,
        SHALLOW_N,
        SHALLOW_P20_MIN_DEFAULT,
        apply_recovery_up,
        detect_recovery_up_marks,
        ed_alert_bool_from_frame,
        latch_recovery_up_mask,
        recovery_up_mask,
    )
    from decision_pack.src.self_train_policy import (
        config_block as self_train_config_block,
        evaluate_self_train_policy,
        policy_frame_from_eval,
    )

    method = cfg["method"]
    method_params = cfg.get("method_params") or {}
    px = prepare_features(ev, ohlcv, method=method, method_params=method_params)
    native_labs = np.asarray(px.regime.to_numpy(), dtype=object)
    mode = (hrp_mode_by_code or {}).get(str(code).upper(), "default")
    if enable_shallow_up_trade and mode_allows_trade_overlay(mode):
        ed_full = early_down_flags_frame(px, regimes=native_labs)
        ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
        su_mask = recovery_up_mask(
            px, native_labs, ed_bool if ed_ok else None, N=SHALLOW_N, p20_min=SHALLOW_P20_MIN_DEFAULT
        )
        if ed_ok:
            su_mask = latch_recovery_up_mask(px, native_labs, su_mask, ed_bool)
        px = px.copy()
        px["regime"] = apply_recovery_up(native_labs, su_mask)
    px_w = px[(px.as_of >= start_date) & (px.as_of <= end_date)].reset_index(drop=True)
    if len(px_w) < 5:
        raise ValueError(f"too few OOS bars {len(px_w)}")
    close_w = px_w["$close"].to_numpy(float)
    dates = px_w.as_of.dt.strftime("%Y-%m-%d").to_numpy()
    seg_df = segments(px_w.regime.to_numpy(), dates, close_w)
    baseline_method = str((cfg.get("method_select_meta") or {}).get("baseline") or method)
    if baseline_method in ev.METHODS:
        baseline_full = ev.run_method(baseline_method, px)
        window_mask = ((px.as_of >= start_date) & (px.as_of <= end_date)).to_numpy()
        baseline_w = np.asarray(baseline_full, dtype=object)[window_mask]
    else:
        baseline_w = px_w.regime.to_numpy(object)
    baseline_seg_df = segments(baseline_w, dates, close_w)
    baseline_diff = float(np.mean(baseline_w != px_w.regime.to_numpy(object)))
    aev, stats, _open_mtm = simulate_adaptive_book(px_w, rules=pool._rules_for_sim(cfg))
    px_fix = prepare_features(ev, ohlcv, method=FIXED_HYBRID_RULES["method"])
    px_fx_w = px_fix[(px_fix.as_of >= start_date) & (px_fix.as_of <= end_date)].reset_index(drop=True)
    _, stats_fx, _ = simulate_adaptive_book(
        px_fx_w, rules=pool._rules_for_sim({"rules": FIXED_HYBRID_RULES})
    )
    train_cutoff = str(cfg.get("train_cutoff") or start_date)
    stp_eval = evaluate_self_train_policy(
        px["as_of"],
        px["$close"].to_numpy(dtype=float),
        px["regime"].to_numpy(dtype=object),
        train_cutoff=train_cutoff,
    )
    cfg["self_train_policy"] = self_train_config_block(stp_eval)
    policy_frame = policy_frame_from_eval(stp_eval)
    lo = pd.Timestamp(start_date).normalize()
    hi = pd.Timestamp(end_date).normalize()
    policy_frame = policy_frame.copy()
    policy_frame["as_of"] = pd.to_datetime(policy_frame["as_of"]).dt.normalize()
    policy_frame = policy_frame[
        (policy_frame["as_of"] >= lo) & (policy_frame["as_of"] <= hi)
    ]
    def _nav_row(row: pd.Series) -> dict[str, float]:
        return {
            "hold_up净值": float(row.get("nav_hold_up", float("nan"))),
            "BH净值": float(row.get("nav_bh", float("nan"))),
            "cash净值": float(row.get("nav_cash", float("nan"))),
            "nav_policy": float(row.get("nav_policy", float("nan"))),
        }

    nav_last: dict[str, float] = {}
    nav_by_date: dict[str, dict[str, float]] = {}
    if not policy_frame.empty:
        nav_last = _nav_row(policy_frame.iloc[-1])
        for _, row in policy_frame.iterrows():
            day = pd.Timestamp(row["as_of"]).strftime("%Y-%m-%d")
            nav_by_date[day] = _nav_row(row)
    agg_marks = detect_early_regime_marks(px_w, regimes=px_w.regime.to_numpy())
    ed_marks = detect_early_down_marks(px_w, regimes=px_w.regime.to_numpy())
    ru_marks: list[dict] = []
    if enable_ru_diag:
        try:
            ed_full = early_down_flags_frame(px, regimes=native_labs)
            ed_bool, ed_ok = ed_alert_bool_from_frame(ed_full, len(px))
            ru_all = detect_recovery_up_marks(
                px, native_labs, ed_bool if ed_ok else None, N=DIAGNOSTIC_N
            )
            lo_ts, hi_ts = pd.Timestamp(start_date), pd.Timestamp(end_date)
            for m in ru_all:
                d0 = pd.Timestamp(str(m.get("date") or ""))
                d1 = pd.Timestamp(str(m.get("end") or m.get("date") or ""))
                if d1 < lo_ts or d0 > hi_ts:
                    continue
                m2 = dict(m)
                m2["hrp_mode"] = mode
                ru_marks.append(m2)
        except Exception:
            ru_marks = []

    trades_dir = out_dir / "trades"
    trades_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(aev).to_csv(trades_dir / f"{code}_trades.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(trades_dir / f"{code}_segments.csv", index=False, encoding="utf-8-sig")
    baseline_seg_df.to_csv(
        trades_dir / f"{code}_baseline_segments.csv", index=False, encoding="utf-8-sig"
    )
    if agg_marks:
        pd.DataFrame(agg_marks).to_csv(
            trades_dir / f"{code}_early_regime_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ed_marks:
        pd.DataFrame(ed_marks).to_csv(
            trades_dir / f"{code}_early_down_marks.csv", index=False, encoding="utf-8-sig"
        )
    if ru_marks:
        pd.DataFrame(ru_marks).to_csv(
            trades_dir / f"{code}_recovery_up_marks.csv", index=False, encoding="utf-8-sig"
        )
    ru = cfg["rules"]
    cfg_out = dict(cfg)
    cfg_out["oos"] = dict(
        start=start_date,
        end=end_date,
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        fixed_hybrid_edge=stats_fx["edge"],
    )
    (out_dir / "configs").mkdir(parents=True, exist_ok=True)
    (out_dir / "configs" / f"{code}.json").write_text(
        json.dumps(cfg_out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return dict(
        code=code,
        name=name,
        out="",
        method=method,
        train_end=cfg.get("train_end"),
        train_bars=cfg.get("train_bars"),
        skipped_train=cfg.get("skipped_train"),
        up_buy=ru["up"]["buy"],
        up_exit=ru["up"]["exit"],
        up_n=ru["up"].get("n_trades"),
        up_post=ru["up"].get("posterior"),
        down_buy=ru["down"]["buy"],
        down_exit=ru["down"]["exit"],
        down_n=ru["down"].get("n_trades"),
        range_buy=ru["range"]["buy"],
        range_exit=ru["range"]["exit"],
        range_n=ru["range"].get("n_trades"),
        bars=len(px_w),
        compound=stats["compound"],
        bh=stats["bh"],
        edge=stats["edge"],
        n_closed=stats["n"],
        win=stats["win"],
        open_pos=stats["open_pos"],
        open_unrealized=stats["open_unrealized"],
        fixed_hybrid_edge=stats_fx["edge"],
        edge_vs_fixed=float(stats["edge"] - stats_fx["edge"]),
        n_seg=len(seg_df),
        baseline_method=baseline_method,
        baseline_regime_diff=baseline_diff,
        aev=aev,
        nav_last=nav_last,
        nav_by_date=nav_by_date,
    )
