from __future__ import annotations

import argparse
import sys

from ashare_daily.screen import DEFAULT_SCORE_MIN


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="ashare-daily screen",
        description=(
            "Keep csi1800 names whose low-correlation / high-return score "
            f"is at least {DEFAULT_SCORE_MIN}."
        ),
    )
    command.add_argument(
        "--score-min",
        type=float,
        default=DEFAULT_SCORE_MIN,
        help=f"Keep names with score >= this value (default {DEFAULT_SCORE_MIN}).",
    )
    command.add_argument(
        "--canonical",
        choices=("score", "csi300"),
        default="score",
        help=(
            "What to write to cluster_mapping_selected.txt. "
            "score: names at or above --score-min (default). "
            "csi300: the 300 constituents on the newest end date in csi300.txt."
        ),
    )
    command.add_argument(
        "--as-of",
        default=None,
        help="Last day of the 252-session scoring window, YYYY-MM-DD. Default: today.",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv or [])
    from ashare_daily.daily_contract import parse_as_of
    from ashare_daily.screen import run_screen

    as_of = parse_as_of(args.as_of) if args.as_of else None
    try:
        result = run_screen(
            score_min=float(args.score_min),
            canonical=str(args.canonical),
            as_of=as_of,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"[OK] selected={result['n_selected']} eligible={result['n_eligible']} "
        f"universe={result['n_universe']} score_min={result['score_min']}"
    )
    print(f"[OK] window={result['test_period'][0]}..{result['test_period'][1]}")
    print(f"[OK] {result['scores_csv']}")
    print(f"[OK] {result['selected_txt']}")
    if result.get("canonical_txt"):
        suffix = f" n={result['canonical_n']}" if result.get("canonical_n") else ""
        print(f"[OK] {result['canonical_txt']}{suffix}")
    for item in result.get("index_scores") or []:
        print(f"[OK] {item['path']} rows={item['rows']}")
        if item.get("l2_path"):
            print(f"[OK] {item['l2_path']} rows={item['l2_rows']}")
    return 0
