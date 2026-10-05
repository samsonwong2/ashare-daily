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
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv or [])
    from ashare_daily.screen import run_screen

    try:
        result = run_screen(score_min=float(args.score_min))
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"[OK] selected={result['n_selected']} eligible={result['n_eligible']} "
        f"universe={result['n_universe']} score_min={result['score_min']}"
    )
    print(f"[OK] {result['scores_csv']}")
    print(f"[OK] {result['selected_txt']}")
    if result.get("canonical_txt"):
        print(f"[OK] {result['canonical_txt']}")
    return 0
