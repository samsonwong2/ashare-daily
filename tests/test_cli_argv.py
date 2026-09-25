"""Subcommand name must not reach the old parser."""
from __future__ import annotations

from ashare_daily.daily_contract import slice_after_command
from ashare_daily.hrp.dendrogram import parse_args


def test_hrp_slice_drops_command_name() -> None:
    command, rest = slice_after_command(["hrp", "--dist-t", "0.8"])
    assert command == "hrp"
    args = parse_args(rest)
    assert args.dist_t == 0.8
