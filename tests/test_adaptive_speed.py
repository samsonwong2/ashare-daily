"""Vectorized regime labels and trailing OLS stay equal to the old loops."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_daily.plots.plot_regime_transition_example import trailing_log_trend_price_path
from ashare_daily.regime.eval_causal_regime_switch_pool import (
    _ma_stack_raw_labels,
    apply_lowvol_range_bias,
    causal_smooth,
    method_adx_di_params,
    method_hybrid_v1_params,
    method_ma_stack,
)


def _smooth_reference(labs: list[str], w: int) -> list[str]:
    out = []
    for i in range(len(labs)):
        sl = labs[max(0, i - w + 1) : i + 1]
        vals, cnts = np.unique(sl, return_counts=True)
        out.append(str(vals[np.argmax(cnts)]))
    return out


def _lowvol_reference(labels, vol, p60, p20, look=15, persist=10) -> np.ndarray:
    labs = np.asarray(labels, dtype=object).copy()
    for i in range(len(labs)):
        window = vol[max(0, i - look + 1) : i + 1]
        n_low = int(np.sum(np.isfinite(window) & (window < 0.35)))
        if n_low < persist:
            continue
        a60 = abs(p60[i]) if np.isfinite(p60[i]) else 0.0
        a20 = abs(p20[i]) if np.isfinite(p20[i]) else 0.0
        if a60 >= 0.12 or a20 >= 0.08:
            continue
        labs[i] = "range"
    return labs


def _raw_reference(px: pd.DataFrame, *, soft_hold60: bool) -> list[str]:
    raw: list[str] = []
    for _, r in px.iterrows():
        if pd.isna(r["MA20"]) or pd.isna(r["MA60"]):
            raw.append("range")
            continue
        c, m20, m60 = float(r["$close"]), float(r["MA20"]), float(r["MA60"])
        p20 = float(r["prior20"]) if pd.notna(r["prior20"]) else 0.0
        p60 = float(r["prior60"]) if pd.notna(r["prior60"]) else 0.0
        band = float(r["band60"]) if pd.notna(r["band60"]) else np.nan
        hug = abs(m20 - m60) / c < 0.02
        chop = (pd.notna(band) and band < 0.22 and abs(p60) < 0.10) or (
            hug and abs(p60) < 0.12
        )
        if soft_hold60:
            up = m20 > m60 and c > m20 and c >= m60 and (p60 >= 0.02 or p20 >= 0.02)
            down = m20 < m60 and c < m20 and c <= m60 and (p60 <= -0.02 or p20 <= -0.02)
        else:
            up = (m20 > m60 and c > m20 and p60 >= 0.03) or (
                m20 > m60 and c > m20 and p20 >= 0.03
            )
            if c > m20 and p20 >= 0.06 and (m20 > m60 or p60 <= -0.08 or p60 >= 0.08):
                up = True
            down = (m20 < m60 and c < m20 and p60 <= -0.03) or (
                m20 < m60 and c < m20 and p20 <= -0.03
            )
        if up and not down:
            lab = "up"
        elif down and not up:
            lab = "down"
        else:
            lab = "range" if chop or True else "range"
        raw.append(lab)
    return raw


def _polyfit_path(y: np.ndarray, *, min_b: int, trail_bars: int, tdy: float) -> np.ndarray:
    n = len(y)
    path = np.full(n, np.nan)
    for t in range(n):
        start = 0
        if trail_bars > 0 and (t + 1) >= trail_bars:
            start = t + 1 - trail_bars
        sl = y[start : t + 1]
        ok = np.isfinite(sl) & (sl > 0)
        if int(ok.sum()) < min_b:
            continue
        jj = np.flatnonzero(ok).astype(float)
        b, a = np.polyfit(jj, np.log(sl[ok]), 1)
        path[t] = float(np.exp(a + b * float(jj[-1])))
    return path


def _feature_frame(n: int = 240, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 10 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, size=n)))
    frame = pd.DataFrame(
        {
            "$close": close,
            "$high": close * 1.01,
            "$low": close * 0.99,
            "MA20": pd.Series(close).rolling(20, min_periods=10).mean(),
            "MA60": pd.Series(close).rolling(60, min_periods=20).mean(),
            "prior20": pd.Series(close).pct_change(20),
            "prior60": pd.Series(close).pct_change(60),
            "band60": rng.uniform(0.05, 0.4, size=n),
            "adx14": rng.uniform(5, 40, size=n),
            "plus_di": rng.uniform(5, 40, size=n),
            "minus_di": rng.uniform(5, 40, size=n),
            "rv20_pct_252": rng.uniform(0, 1, size=n),
        }
    )
    frame.loc[3, "MA20"] = np.nan
    frame.loc[8, "prior20"] = np.nan
    frame.loc[12, "band60"] = np.nan
    frame.loc[15, "adx14"] = np.nan
    return frame


def test_causal_smooth_matches_unique_ties() -> None:
    labs = ["up", "down", "up", "down", "range", "up", "weird", "up"]
    assert causal_smooth(labs, 3) == _smooth_reference(labs, 3)
    tied = ["down", "up", "down", "up", "range"]
    assert causal_smooth(tied, 4) == _smooth_reference(tied, 4)


def test_lowvol_bias_matches_window_sum() -> None:
    rng = np.random.default_rng(1)
    n = 80
    labels = rng.choice(["up", "down", "range"], size=n)
    vol = rng.uniform(0, 1, size=n)
    vol[4] = np.nan
    p60 = rng.normal(0, 0.05, size=n)
    p20 = rng.normal(0, 0.04, size=n)
    got = apply_lowvol_range_bias(labels, vol, p60, p20)
    ref = _lowvol_reference(labels, vol, p60, p20)
    assert list(got) == list(ref)


def test_ma_stack_raw_matches_iterrows() -> None:
    frame = _feature_frame()
    assert _ma_stack_raw_labels(frame, soft_hold60=False) == _raw_reference(frame, soft_hold60=False)
    assert _ma_stack_raw_labels(frame, soft_hold60=True) == _raw_reference(frame, soft_hold60=True)


def test_hybrid_and_adx_are_finite_labels() -> None:
    frame = _feature_frame()
    hybrid = method_hybrid_v1_params(frame)
    adx = method_adx_di_params(frame)
    assert len(hybrid) == len(frame)
    assert set(hybrid).issubset({"up", "down", "range"})
    assert set(adx).issubset({"up", "down", "range"})
    stacked = method_ma_stack(frame)
    assert len(stacked) == len(frame)


def test_trailing_ols_matches_polyfit() -> None:
    rng = np.random.default_rng(2)
    y = pd.Series(10 * np.exp(np.cumsum(rng.normal(0, 0.01, size=400))))
    path, _g = trailing_log_trend_price_path(y, trail_years=100 / 252, min_bars=20, trading_days_per_year=252)
    ref = _polyfit_path(y.to_numpy(float), min_b=20, trail_bars=100, tdy=252)
    # Warmup bars before min_bars use CAGR, not the fit. Compare the fitted tail.
    fitted = np.isfinite(ref)
    assert fitted.sum() > 50
    np.testing.assert_allclose(path.to_numpy()[fitted], ref[fitted], rtol=1e-6, atol=1e-6)

    gappy = y.copy()
    gappy.iloc[10] = np.nan
    gappy.iloc[40] = -1
    path_g, _ = trailing_log_trend_price_path(
        gappy, trail_years=80 / 252, min_bars=15, trading_days_per_year=252
    )
    ref_g = _polyfit_path(gappy.to_numpy(float), min_b=15, trail_bars=80, tdy=252)
    fitted_g = np.isfinite(ref_g)
    np.testing.assert_allclose(path_g.to_numpy()[fitted_g], ref_g[fitted_g], rtol=1e-6, atol=1e-6)
