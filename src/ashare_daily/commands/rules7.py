from __future__ import annotations

import argparse
from datetime import date

from ashare_daily.daily_contract import as_of_from_listing_dir, parse_as_of, resolve_listing_dir


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily rules7")
    p.add_argument(
        "--listing-dir",
        default=None,
        help="From-listing HTML directory. Default: {YYYYMMDD}_from_listing_stock under PLOTLY_ROOT.",
    )
    p.add_argument("--as-of", default=None, help="Last close T. Default: date in the listing directory name, or today.")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--anchor-code", default="SH000300")
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
        as_of = _flag_value(rest, "--as-of") or date.today().isoformat()
        tag = parse_as_of(as_of).replace("-", "")
        raw_dir = f"{tag}_from_listing_stock"
        rest += ["--listing-dir", raw_dir]
    listing = resolve_listing_dir(raw_dir)
    if not listing.is_dir():
        print(f"missing listing dir: {listing}", flush=True)
        return 1
    index = rest.index("--listing-dir")
    rest[index + 1] = str(listing)
    if _flag_value(rest, "--as-of") is None:
        as_of = as_of_from_listing_dir(listing)
        rest += ["--as-of", as_of]
        print(f"[INFO] --as-of {as_of} from {listing.name}", flush=True)
    from ashare_daily.plots.scan_rules7_checklist import main as scan_main

    result = scan_main(rest)
    return int(result) if isinstance(result, int) else 0
