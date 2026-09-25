from __future__ import annotations

import argparse

from ashare_daily.daily_contract import (
    as_of_from_listing_dir,
    next_weekday,
    resolve_listing_dir,
)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily triggers")
    p.add_argument("--listing-dir", required=True)
    p.add_argument("--as-of", default=None, help="Last close T. Default: date in the listing directory name.")
    p.add_argument("--next-day", default=None, help="Next session T+1. Default: next weekday after --as-of.")
    p.add_argument("--jobs", type=int, default=8)
    return p


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag not in argv:
        return None
    index = argv.index(flag)
    if index + 1 >= len(argv):
        return None
    return argv[index + 1]


def main(argv: list[str] | None = None) -> int:
    rest = list(argv or [])
    raw_dir = _flag_value(rest, "--listing-dir")
    if raw_dir is None:
        from ashare_daily.triggers.scan import main as scan_main

        result = scan_main(rest)
        return int(result) if isinstance(result, int) else 0

    listing = resolve_listing_dir(raw_dir)
    if not listing.is_dir():
        print(f"missing listing dir: {listing}", flush=True)
        return 1
    index = rest.index("--listing-dir")
    rest[index + 1] = str(listing)

    as_of = _flag_value(rest, "--as-of")
    if as_of is None:
        as_of = as_of_from_listing_dir(listing)
        rest += ["--as-of", as_of]
        print(f"[INFO] --as-of {as_of} from {listing.name}", flush=True)
    next_day = _flag_value(rest, "--next-day")
    if next_day is None:
        next_day = next_weekday(as_of)
        rest += ["--next-day", next_day]
        print(f"[INFO] --next-day {next_day}", flush=True)

    from ashare_daily.triggers.scan import main as scan_main

    result = scan_main(rest)
    return int(result) if isinstance(result, int) else 0
