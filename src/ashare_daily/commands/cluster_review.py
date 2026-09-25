from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="ashare-daily cluster-review",
        description="Daily cluster-mapping review. Remaining flags go to daily_review.",
    )


def main(argv: list[str] | None = None) -> int:
    from ashare_daily.pool.builder.daily_review import main as review_main

    result = review_main(argv)
    return int(result) if isinstance(result, int) else 0
