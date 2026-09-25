"""EWMA residual rails for fig7–fig11 fair-path panels.

Same causal envelope as the ETF plotter: expanding P95 of ``|r|/σ_ewma``
times the current EWMA residual sigma. Production P95 rails stay in place;
these traces are an overlay.
"""
from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_ENVELOPE_Q = 0.95
DEFAULT_MIN_BARS = 63
DEFAULT_EWMA_LAMBDA = 0.94

# fig7=2y … fig11=1m. Matches incremental_from_listing.TRAIL_YAXIS.
PANEL_YAXIS: dict[str, str] = {
    "y10": "2年",
    "y11": "1年",
    "y12": "6个月",
    "y13": "3个月",
    "y14": "1个月",
}

_EWMA_MARK = "EWMA残差轨"
_DATA_START_RE = re.compile(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*(\[)')

# Escaped subplot titles in the written HTML layout.
_TITLE_PATCHES: tuple[tuple[str, str], ...] = (
    (r"\u6eda\u52a82\u5e74\uff1b", r"\u6eda\u52a82\u5e74\u00b7\u56e0\u679cP95+EWMA\u53cc\u8f68\uff1b"),
    (r"\u6eda\u52a81\u5e74\uff1b", r"\u6eda\u52a81\u5e74\u00b7\u56e0\u679cP95+EWMA\u53cc\u8f68\uff1b"),
    (r"\u6eda\u52a86\u4e2a\u6708\uff1b", r"\u6eda\u52a86\u4e2a\u6708\u00b7\u56e0\u679cP95+EWMA\u53cc\u8f68\uff1b"),
    (r"\u6eda\u52a83\u4e2a\u6708\uff1b", r"\u6eda\u52a83\u4e2a\u6708\u00b7\u56e0\u679cP95+EWMA\u53cc\u8f68\uff1b"),
    (r"\u6eda\u52a81\u4e2a\u6708\uff1b", r"\u6eda\u52a81\u4e2a\u6708\u00b7\u56e0\u679cP95+EWMA\u53cc\u8f68\uff1b"),
)


def ewma_residual_sigma(
    residual: pd.Series,
    *,
    lam: float = DEFAULT_EWMA_LAMBDA,
) -> pd.Series:
    """Causal EWMA std; ``σ_t`` uses ``r_{t-1}`` only."""
    r = pd.to_numeric(residual, errors="coerce").to_numpy(dtype=float)
    n = r.size
    out = np.full(n, np.nan, dtype=float)
    lam_f = float(lam)
    if not (0.0 < lam_f < 1.0) or n == 0:
        return pd.Series(out, index=residual.index, dtype=float)
    one_l = 1.0 - lam_f
    last_var: float | None = None
    last_r2: float | None = None
    for t in range(n):
        if last_var is None:
            if last_r2 is not None and last_r2 > 0:
                last_var = last_r2
                out[t] = float(np.sqrt(last_var))
        else:
            if last_r2 is not None and np.isfinite(last_r2):
                last_var = lam_f * last_var + one_l * last_r2
            if last_var is not None and last_var > 0:
                out[t] = float(np.sqrt(last_var))
        rt = float(r[t])
        if np.isfinite(rt):
            last_r2 = rt * rt
    return pd.Series(out, index=residual.index, dtype=float)


def ewma_scaled_envelope(
    path: pd.Series,
    close: pd.Series,
    *,
    q: float = DEFAULT_ENVELOPE_Q,
    min_bars: int = DEFAULT_MIN_BARS,
    lam: float = DEFAULT_EWMA_LAMBDA,
) -> tuple[pd.Series, pd.Series, float]:
    """Causal envelope: expanding P95 of ``|r|/σ_ewma`` times current σ."""
    p = pd.to_numeric(path, errors="coerce")
    c = pd.to_numeric(close, errors="coerce")
    upper = pd.Series(np.nan, index=p.index, dtype=float)
    lower = pd.Series(np.nan, index=p.index, dtype=float)
    qq = float(q)
    min_b = max(2, int(min_bars))
    if p.empty or not np.isfinite(qq) or not (0.0 < qq < 1.0):
        return upper, lower, float("nan")
    r = pd.Series(np.nan, index=p.index, dtype=float)
    ok = (p > 0) & (c > 0) & p.notna() & c.notna()
    r.loc[ok] = np.log(c.loc[ok].to_numpy(dtype=float) / p.loc[ok].to_numpy(dtype=float))
    sigma = ewma_residual_sigma(r, lam=lam)
    path_arr = p.to_numpy(dtype=float)
    r_arr = r.to_numpy(dtype=float)
    s_arr = sigma.to_numpy(dtype=float)
    n = len(path_arr)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)
    delta_arr = np.full(n, np.nan, dtype=float)
    abs_z: list[float] = []
    for t in range(n):
        pt = float(path_arr[t])
        rt = float(r_arr[t])
        st = float(s_arr[t])
        ok_p = np.isfinite(pt) and pt > 0
        ok_s = np.isfinite(st) and st > 0
        if ok_s and np.isfinite(rt):
            zt = rt / st
            if np.isfinite(zt):
                abs_z.append(abs(zt))
        if len(abs_z) < min_b or not (ok_p and ok_s):
            continue
        delta_t = float(np.quantile(np.asarray(abs_z, dtype=float), qq))
        if not np.isfinite(delta_t) or delta_t < 0:
            continue
        delta_arr[t] = delta_t
        width = delta_t * st
        upper_arr[t] = pt * float(np.exp(width))
        lower_arr[t] = pt * float(np.exp(-width))
    upper.iloc[:] = upper_arr
    lower.iloc[:] = lower_arr
    finite_d = delta_arr[np.isfinite(delta_arr)]
    delta_asof = float(finite_d[-1]) if finite_d.size else float("nan")
    return upper, lower, delta_asof


def _as_1d(arr: Any) -> list[Any]:
    from ashare_daily.lib.fair_path_band_signals import _decode_plotly_bdata

    if arr is None:
        return []
    if isinstance(arr, dict) and "bdata" in arr and "dtype" in arr:
        arr = _decode_plotly_bdata(arr)
    a = np.asarray(arr)
    if a.ndim == 0:
        return [a.item()]
    return a.ravel().tolist()


def _json_x(values: list[Any]) -> list[Any]:
    out: list[Any] = []
    for v in values:
        if isinstance(v, (np.datetime64, pd.Timestamp)):
            out.append(pd.Timestamp(v).strftime("%Y-%m-%d"))
        elif hasattr(v, "isoformat") and not isinstance(v, str):
            out.append(pd.Timestamp(v).strftime("%Y-%m-%d"))
        else:
            out.append(v)
    return out


def _json_y(values: np.ndarray) -> list[float | None]:
    out: list[float | None] = []
    for v in values:
        fv = float(v)
        out.append(fv if np.isfinite(fv) else None)
    return out


def _hover(upper_v: float, lower_v: float, path_v: float, *, is_upper: bool) -> str:
    if not (np.isfinite(path_v) and path_v > 0 and np.isfinite(upper_v if is_upper else lower_v)):
        return "上轨(EWMA)=na" if is_upper else "下轨(EWMA)=na"
    half = float(upper_v) / float(path_v) - 1.0
    if is_upper:
        return f"上轨(EWMA 当日{half:+.1%})={upper_v:.4f}"
    return f"下轨(EWMA 当日{-half:+.1%})={lower_v:.4f}"


def _rail_trace(
    *,
    name: str,
    yaxis: str,
    xaxis: str,
    x: list[Any],
    y: list[float | None],
    color: str,
    customdata: list[str],
) -> dict[str, Any]:
    return {
        "type": "scatter",
        "mode": "lines",
        "name": name,
        "x": x,
        "y": y,
        "xaxis": xaxis,
        "yaxis": yaxis,
        "line": {"color": color, "width": 1.3, "dash": "dashdot"},
        "hovertemplate": "%{customdata}<extra></extra>",
        "customdata": customdata,
        "showlegend": True,
    }


def build_ewma_rail_traces(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Full-length EWMA upper/lower traces for each fair-path panel present."""
    out: list[dict[str, Any]] = []
    for yaxis, hz in PANEL_YAXIS.items():
        path_tr = None
        candle = None
        for tr in traces:
            if str(tr.get("yaxis") or "") != yaxis:
                continue
            if path_tr is None and "公平路径" in str(tr.get("name") or ""):
                path_tr = tr
            if candle is None and tr.get("type") == "candlestick":
                candle = tr
        if path_tr is None or candle is None:
            continue
        path_y = np.asarray(_as_1d(path_tr.get("y")), dtype=float)
        close_y = np.asarray(_as_1d(candle.get("close")), dtype=float)
        m = min(len(path_y), len(close_y))
        if m < 2:
            continue
        upper, lower, delta = ewma_scaled_envelope(
            pd.Series(path_y[:m]),
            pd.Series(close_y[:m]),
        )
        xs = _json_x(_as_1d(path_tr.get("x"))[:m])
        xaxis = str(path_tr.get("xaxis") or "x")
        asof = float(delta)
        asof_leg = f" +{asof:.1%}(当日)" if np.isfinite(asof) else ""
        dn_leg = asof_leg.replace("+", "-", 1) if asof_leg else ""
        hovers_up: list[str] = []
        hovers_lo: list[str] = []
        for i in range(m):
            uv = float(upper.iloc[i])
            lv = float(lower.iloc[i])
            pv = float(path_y[i])
            hovers_up.append(_hover(uv, lv, pv, is_upper=True))
            hovers_lo.append(_hover(uv, lv, pv, is_upper=False))
        out.append(
            _rail_trace(
                name=f"上轨{hz} {_EWMA_MARK}{asof_leg}",
                yaxis=yaxis,
                xaxis=xaxis,
                x=xs,
                y=_json_y(upper.to_numpy(dtype=float)[:m]),
                color="#17becf",
                customdata=hovers_up,
            )
        )
        out.append(
            _rail_trace(
                name=f"下轨{hz} {_EWMA_MARK}{dn_leg}",
                yaxis=yaxis,
                xaxis=xaxis,
                x=xs,
                y=_json_y(lower.to_numpy(dtype=float)[:m]),
                color="#e377c2",
                customdata=hovers_lo,
            )
        )
    return out


def attach_ewma_rails(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop stale EWMA rails and append a fresh full-length pair per panel."""
    kept = [tr for tr in traces if _EWMA_MARK not in str(tr.get("name") or "")]
    kept.extend(build_ewma_rail_traces(kept))
    return kept


def patch_subplot_titles(html: str) -> str:
    for old, new in _TITLE_PATCHES:
        if old in html and new not in html:
            html = html.replace(old, new)
    return html


def _data_array_end(html: str) -> int:
    m = _DATA_START_RE.search(html)
    if not m:
        raise ValueError("Plotly.newPlot data array not found")
    i = m.start(1)
    depth = 0
    in_str = False
    esc = False
    while i < len(html):
        c = html[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise ValueError("unclosed Plotly data array")


def inject_ewma_rails_html(html: str, traces: list[dict[str, Any]]) -> str:
    """Splice EWMA rails into an existing Plotly HTML document."""
    if _EWMA_MARK in html:
        return patch_subplot_titles(html)
    extra = build_ewma_rail_traces(traces)
    if not extra:
        raise ValueError("no fair-path panels to attach EWMA rails")
    payload = ",".join(json.dumps(tr, ensure_ascii=False, allow_nan=False) for tr in extra)
    end = _data_array_end(html)
    html = html[:end] + "," + payload + html[end:]
    return patch_subplot_titles(html)
