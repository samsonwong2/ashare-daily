from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from ashare_daily.cn_data import DEFAULT_DEST


def _release_tag(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--tag must be YYYY-MM-DD") from exc
    return value


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="ashare-daily cn-data",
        description=(
            "Replace Qlib cn_data from one chenditc/investment_data release. "
            "Uses http://127.0.0.1:31181 when that proxy accepts connections."
        ),
    )
    command.add_argument(
        "--tag",
        type=_release_tag,
        help="Release tag YYYY-MM-DD. Default: the GitHub latest release.",
    )
    command.add_argument(
        "--dest",
        default=str(DEFAULT_DEST),
        help="Directory to replace. Default: /home/huangtuo/data/qlib_data/cn_data",
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv or [])
    from ashare_daily.cn_data import refresh_cn_data

    return refresh_cn_data(tag=args.tag, dest=Path(args.dest))
