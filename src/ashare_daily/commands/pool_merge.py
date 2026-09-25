from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        prog="ashare-daily pool-merge",
        description="Merge CSI800 and CSI1000 instruments into csi1800.txt.",
    )


def main(argv: list[str] | None = None) -> int:
    parser().parse_args(argv or [])
    from ashare_daily.pool.merge_csi1800 import main as merge_main

    merge_main()
    return 0
