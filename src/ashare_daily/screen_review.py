"""Review a screen list with trailing returns.

Reads ``stock_screen_scores.csv`` and answers two questions: which unselected
names had the strongest recent return, and which selected names had the weakest.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ashare_daily.paths import QLIB_PROVIDER_URI, TEMP_DIR
from ashare_daily.pool.builder.audit import compute_trailing_return_metrics, parse_windows
from ashare_daily.screen import SCORES_NAME

FOCUS_WINDOW = 20


def review_dir(as_of: str, root: Path | None = None) -> Path:
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    base = Path(root) if root is not None else Path(TEMP_DIR) / "stock_screen_review"
    return base / tag


def attach_trailing(
    scores: pd.DataFrame,
    close: pd.DataFrame,
    as_of: str,
    windows: list[int],
) -> pd.DataFrame:
    metrics = compute_trailing_return_metrics(close, as_of, windows)
    metrics["code"] = metrics["code"].astype(str).str.upper()
    frame = scores.copy()
    frame["code"] = frame["code"].astype(str).str.upper()
    keep = ["code"] + [col for col in metrics.columns if col.startswith("fwd_ret_")]
    return frame.merge(metrics[keep], on="code", how="left")


def split_review(frame: pd.DataFrame, *, focus_window: int = FOCUS_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    col = f"fwd_ret_{int(focus_window)}d"
    selected = frame["selected"].map(lambda value: str(value).strip().lower() in {"true", "1", "yes"} or value is True)
    missed = frame.loc[~selected].sort_values(col, ascending=False, na_position="last")
    weak = frame.loc[selected].sort_values(col, ascending=True, na_position="last")
    return missed.reset_index(drop=True), weak.reset_index(drop=True)


def _pct(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return "-"
    return f"{float(number) * 100:+.2f}%"


def _cell(column: str, value: object) -> str:
    if pd.isna(value):
        return "-"
    if str(column).startswith("fwd_ret_") or column == "cum_return":
        return _pct(value)
    if column in {"score", "mean_abs_corr"}:
        number = pd.to_numeric(value, errors="coerce")
        return "-" if pd.isna(number) else f"{float(number):.3f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    lines = [header, rule]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_cell(column, row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_review(
    frame: pd.DataFrame,
    *,
    as_of: str,
    out_dir: Path,
    top_n: int,
    focus_window: int = FOCUS_WINDOW,
) -> dict[str, Path]:
    missed, weak = split_review(frame, focus_window=focus_window)
    out_dir.mkdir(parents=True, exist_ok=True)
    missed_path = out_dir / "missed_by_score.csv"
    weak_path = out_dir / "weak_selected.csv"
    missed.to_csv(missed_path, index=False, encoding="utf-8-sig")
    weak.to_csv(weak_path, index=False, encoding="utf-8-sig")
    tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    report_path = out_dir / f"screen_review_{tag}.md"
    focus = f"fwd_ret_{int(focus_window)}d"
    show_cols = ["code", "name", "score", "cum_return", "mean_abs_corr", focus]
    show_cols = [col for col in show_cols if col in frame.columns]
    lines = [
        f"# screen 复查 {as_of}",
        "",
        f"未选上 {len(missed)} 只，选上 {len(weak)} 只。近 {focus_window} 日收益按收盘价回看。",
        "",
        f"## 未选上、近 {focus_window} 日最强",
        "",
        _markdown_table(missed.head(top_n), show_cols) if not missed.empty else "（无）",
        "",
        f"## 选上、近 {focus_window} 日最弱",
        "",
        _markdown_table(weak.head(top_n), show_cols) if not weak.empty else "（无）",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {"missed": missed_path, "weak": weak_path, "report": report_path}


def run_screen_review(
    *,
    as_of: str | None = None,
    windows: str = "5,20,60",
    top_n: int = 20,
    scores_csv: Path | None = None,
    provider_uri: str | None = None,
) -> dict[str, object]:
    review_day = as_of or date.today().isoformat()
    window_list = parse_windows(windows)
    scores_path = Path(scores_csv) if scores_csv is not None else Path(TEMP_DIR) / SCORES_NAME
    if not scores_path.is_file():
        raise FileNotFoundError(f"missing screen scores: {scores_path}")
    scores = pd.read_csv(scores_path)
    if "code" not in scores.columns or "selected" not in scores.columns:
        raise RuntimeError(f"{scores_path} needs code and selected columns")
    codes = scores["code"].astype(str).str.upper().tolist()
    if provider_uri is None:
        from ashare_daily.pool.builder.config import load_shared_config

        provider_uri = str(load_shared_config().get("provider_uri") or QLIB_PROVIDER_URI)
    start = (pd.Timestamp(review_day) - pd.Timedelta(days=max(window_list) * 3 + 30)).strftime("%Y-%m-%d")
    from ashare_daily.lib.strategy_layer_audit import load_qlib_close

    close = load_qlib_close(provider_uri, codes, start, review_day)
    framed = attach_trailing(scores, close, review_day, window_list)
    paths = write_review(framed, as_of=review_day, out_dir=review_dir(review_day), top_n=int(top_n))
    missed, weak = split_review(framed)
    return {
        "as_of": review_day,
        "n_selected": int(weak.shape[0]),
        "n_missed": int(missed.shape[0]),
        **paths,
    }
