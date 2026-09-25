from __future__ import annotations

import argparse


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ashare-daily hrp")
    p.add_argument("--asof-date", default=None)
    p.add_argument("--lookback-days", type=int, default=None)
    p.add_argument("--dist-t", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    from ashare_daily.hrp.dendrogram import main as hrp_main

    result = hrp_main(list(argv or []))
    return int(result) if isinstance(result, int) else 0
