from __future__ import annotations

import argparse
import sys
from datetime import date


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="ashare-daily screen-review",
        description="Review the screen list with trailing 5/20/60-day returns.",
    )
    command.add_argument("--as-of", default=date.today().isoformat())
    command.add_argument("--windows", default="5,20,60")
    command.add_argument("--top-n", type=int, default=20)
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv or [])
    from ashare_daily.screen_review import run_screen_review

    try:
        result = run_screen_review(as_of=args.as_of, windows=args.windows, top_n=int(args.top_n))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(
        f"[OK] selected={result['n_selected']} missed={result['n_missed']} as_of={result['as_of']}"
    )
    print(f"[OK] {result['report']}")
    return 0
