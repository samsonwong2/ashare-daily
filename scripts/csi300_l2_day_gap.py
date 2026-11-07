"""Shenwan level-2 substitution ledger, one CSV per trading day.

Not an ashare-daily command. Score-table paths stay fixed here.
A day that trips a stop rule is skipped; the run continues and exits 1
if any day failed. Dates absent from the calendar are skipped quietly.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROVIDER = Path("/home/huangtuo/data/qlib_data/cn_data")
SCORES_DIR = Path("/home/huangtuo/temp/stock")
L2_CSV = SCORES_DIR / "stock_screen_scores_csi300_l2.csv"
CONSTITUENT_CSV = SCORES_DIR / "stock_screen_scores_csi300.csv"
INDEX_CODE = "SH000300"

COLUMNS = [
    "记录类型",
    "code",
    "name",
    "申万二级名称",
    "行业权重",
    "代表股收益",
    "行业成分收益",
    "收益差",
    "贡献",
    "权重日期",
    "无法计算的行业个数",
]


class DaySkip(Exception):
    """This trading day cannot be written. Later days still run."""


def stop(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(1)


def parse_date(value: str) -> str:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        stop(f"--start/--end must be YYYY-MM-DD, got {value}")
    return value


def read_table(path: Path) -> pd.DataFrame:
    if not path.is_file():
        stop(f"missing table: {path}")
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    stop(f"cannot read {path}: {last_error}")
    raise AssertionError("unreachable")


def calendar_days() -> list[str]:
    path = PROVIDER / "calendars" / "day.txt"
    if not path.is_file():
        stop(f"missing calendar: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def closes(code: str, days: list[str]) -> dict[str, float]:
    path = PROVIDER / "features" / code.lower() / "close.day.bin"
    if not path.is_file():
        return {}
    raw = np.fromfile(path, dtype="<f4")
    if raw.size == 0:
        return {}
    start = int(raw[0])
    out: dict[str, float] = {}
    for offset, value in enumerate(raw[1:]):
        pos = start + offset
        if pos < 0 or pos >= len(days):
            continue
        number = float(value)
        if math.isfinite(number) and number > 0.0:
            out[days[pos]] = number
    return out


def day_return(series: dict[str, float], day: str, previous: str) -> float:
    if day not in series or previous not in series:
        return math.nan
    return series[day] / series[previous] - 1.0


def norm_code(value: object) -> str:
    return str(value).strip().upper()


def trim_name(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def trim_date(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def positive_weight(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0.0:
        return None
    return number


def blank_row(kind: str, industry: str, weight_date: str) -> dict[str, object]:
    row = {column: "" for column in COLUMNS}
    row["记录类型"] = kind
    row["申万二级名称"] = industry
    row["权重日期"] = weight_date
    return row


@dataclass
class DayContext:
    days: list[str]
    day_index: dict[str, int]
    body_src: list[dict[str, object]]
    raw_sum: float
    weight_date: str
    members: list[tuple[str, str, float]]
    price_cache: dict[str, dict[str, float]]

    def cached_close(self, code: str) -> dict[str, float]:
        if code not in self.price_cache:
            self.price_cache[code] = closes(code, self.days)
        return self.price_cache[code]


def load_context() -> DayContext:
    days = calendar_days()
    l2 = read_table(L2_CSV)
    constituents = read_table(CONSTITUENT_CSV)
    for frame, path in ((l2, L2_CSV), (constituents, CONSTITUENT_CSV)):
        if "code" not in frame.columns or "申万二级名称" not in frame.columns:
            stop(f"{path.name} needs code and 申万二级名称")

    constituent_codes = [norm_code(code) for code in constituents["code"] if trim_name(code)]
    if len(constituent_codes) != len(set(constituent_codes)):
        stop("duplicate code in stock_screen_scores_csi300.csv")

    body_src: list[dict[str, object]] = []
    seen_industries: set[str] = set()
    for _, record in l2.iterrows():
        industry = trim_name(record.get("申万二级名称"))
        weight = positive_weight(record.get("所有此申万二级权重"))
        if not industry or weight is None:
            continue
        if industry in seen_industries:
            stop(f"duplicate 申万二级名称 in l2 table: {industry}")
        seen_industries.add(industry)
        body_src.append(
            {
                "code": norm_code(record.get("code")),
                "name": "" if pd.isna(record.get("name")) else record.get("name"),
                "industry": industry,
                "raw_weight": weight,
                "weight_date": trim_date(record.get("权重日期")),
            }
        )
    if not body_src:
        stop("no l2 row entered the body")
    dates = {row["weight_date"] for row in body_src}
    if len(dates) != 1 or "" in dates:
        stop("body rows do not share one 权重日期")
    raw_sum = sum(float(row["raw_weight"]) for row in body_src)
    if raw_sum <= 0.0:
        stop("industry weight sum is not positive")

    members: list[tuple[str, str, float]] = []
    industries = {str(row["industry"]) for row in body_src}
    for _, record in constituents.iterrows():
        industry = trim_name(record.get("申万二级名称"))
        weight = positive_weight(record.get("权重"))
        if not industry or weight is None or industry not in industries:
            continue
        members.append((industry, norm_code(record.get("code")), weight))

    ctx = DayContext(
        days=days,
        day_index={day: i for i, day in enumerate(days)},
        body_src=body_src,
        raw_sum=raw_sum,
        weight_date=str(next(iter(dates))),
        members=members,
        price_cache={INDEX_CODE: closes(INDEX_CODE, days)},
    )
    return ctx


def previous_index_session(ctx: DayContext, day: str) -> str:
    index = ctx.price_cache[INDEX_CODE]
    for earlier in reversed(ctx.days[: ctx.day_index[day]]):
        if earlier in index:
            return earlier
    raise DaySkip(f"no earlier finite {INDEX_CODE} close before {day}")


def build_day(day: str, ctx: DayContext) -> pd.DataFrame:
    previous = previous_index_session(ctx, day)
    index_return = day_return(ctx.price_cache[INDEX_CODE], day, previous)
    baskets: dict[str, list[tuple[float, float]]] = {str(row["industry"]): [] for row in ctx.body_src}
    for industry, code, weight in ctx.members:
        ret = day_return(ctx.cached_close(code), day, previous)
        if math.isfinite(ret):
            baskets[industry].append((weight, ret))

    rows: list[dict[str, object]] = []
    for src in ctx.body_src:
        displayed = float(src["raw_weight"]) / ctx.raw_sum
        rep = day_return(ctx.cached_close(str(src["code"])), day, previous)
        basket = baskets[str(src["industry"])]
        basket_weight = sum(item[0] for item in basket)
        industry_return = (
            sum(item[0] * item[1] for item in basket) / basket_weight
            if basket_weight > 0.0
            else math.nan
        )
        row = blank_row("body", str(src["industry"]), ctx.weight_date)
        row["code"] = src["code"]
        row["name"] = src["name"]
        row["行业权重"] = displayed
        if math.isfinite(industry_return):
            row["行业成分收益"] = industry_return
        if math.isfinite(rep):
            row["代表股收益"] = rep
        if math.isfinite(rep) and math.isfinite(industry_return):
            gap = rep - industry_return
            row["收益差"] = gap
            row["贡献"] = displayed * gap
        rows.append(row)

    def sort_key(row: dict[str, object]) -> tuple[int, float, str]:
        contribution = row["贡献"]
        if isinstance(contribution, float) and math.isfinite(contribution):
            return (0, -abs(contribution), str(row["申万二级名称"]))
        return (1, 0.0, str(row["申万二级名称"]))

    rows.sort(key=sort_key)
    finite_contrib = [
        float(row["贡献"])
        for row in rows
        if isinstance(row["贡献"], float) and math.isfinite(float(row["贡献"]))
    ]
    held = [
        row
        for row in rows
        if isinstance(row["代表股收益"], float) and math.isfinite(float(row["代表股收益"]))
    ]
    held_weight = sum(float(row["行业权重"]) for row in held)
    portfolio = (
        sum(float(row["行业权重"]) * float(row["代表股收益"]) for row in held) / held_weight
        if held_weight > 0.0
        else math.nan
    )
    excess = portfolio - index_return if math.isfinite(portfolio) and math.isfinite(index_return) else math.nan
    uncomputable = sum(
        1 for row in rows if not (isinstance(row["贡献"], float) and math.isfinite(float(row["贡献"])))
    )

    total = blank_row("footer", "贡献合计", ctx.weight_date)
    if finite_contrib:
        total["贡献"] = sum(finite_contrib)
    excess_row = blank_row("footer", "组合收益减SH000300", ctx.weight_date)
    if math.isfinite(excess):
        excess_row["贡献"] = excess
    count_row = blank_row("footer", "无法计算的行业个数", ctx.weight_date)
    count_row["贡献"] = ""
    count_row["无法计算的行业个数"] = uncomputable
    return pd.DataFrame(rows + [total, excess_row, count_row], columns=COLUMNS)


def iter_span(start: str, end: str) -> list[str]:
    if start > end:
        stop(f"--start {start} is after --end {end}")
    cursor = datetime.strptime(start, "%Y-%m-%d")
    last = datetime.strptime(end, "%Y-%m-%d")
    out: list[str] = []
    while cursor <= last:
        out.append(cursor.strftime("%Y-%m-%d"))
        cursor = cursor.fromordinal(cursor.toordinal() + 1)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write one substitution CSV per trading day.")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--out-dir", type=Path, default=SCORES_DIR)
    args = parser.parse_args(argv)
    start = parse_date(args.start)
    end = parse_date(args.end)
    span = iter_span(start, end)
    ctx = load_context()
    sessions = set(ctx.days)
    trading = [day for day in span if day in sessions]
    skipped = len(span) - len(trading)
    if not trading:
        stop(f"no trading days in {start}..{end}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failed: list[str] = []
    for day in trading:
        try:
            frame = build_day(day, ctx)
        except DaySkip as exc:
            print(exc, file=sys.stderr)
            failed.append(day)
            continue
        out = args.out_dir / f"stock_screen_scores_csi300_l2_day_{day.replace('-', '')}.csv"
        frame.to_csv(out, index=False, encoding="utf-8-sig")
        written.append(out)
        print(f"wrote {out}")

    print(f"wrote {len(written)} file(s); skipped {skipped} non-trading day(s); failed {len(failed)}")
    if failed:
        print("failed: " + ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
