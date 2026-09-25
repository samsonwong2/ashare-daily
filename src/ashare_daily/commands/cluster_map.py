from __future__ import annotations

import argparse
import sys


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="ashare-daily cluster-map",
        description="Production cluster map. Defaults live in pool.builder.constants.",
    )


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv or [])
    from ashare_daily.pool.builder import constants as C
    from ashare_daily.pool.builder.cli import main as pool_main

    print(
        "[INFO] Production clustering defaults: "
        f"dist_t={C.DIST_T}, target={C.TARGET_SELECTED_COUNT}, "
        f"scoring={C.SCORING_MODE}, rep_pick={C.REP_PICK_MODE}, linkage={C.CLUSTER_LINKAGE}, "
        f"cross_rank={C.CROSS_CLUSTER_RANK}"
    )
    try:
        result = pool_main([])
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    return int(result) if isinstance(result, int) else 0
