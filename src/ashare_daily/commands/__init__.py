"""Subcommand dispatch. Heavy modules are imported only when a command runs."""
from __future__ import annotations


def help_for(command: str) -> None:
    module = _module(command)
    module.parser().print_help()


def dispatch(command: str, rest: list[str]) -> int:
    module = _module(command)
    result = module.main(rest)
    if isinstance(result, int):
        return result
    return 0


def _module(command: str):
    if command == "pool-merge":
        from ashare_daily.commands import pool_merge as module
    elif command == "cluster-map":
        from ashare_daily.commands import cluster_map as module
    elif command == "cluster-review":
        from ashare_daily.commands import cluster_review as module
    elif command == "regime":
        from ashare_daily.commands import regime as module
    elif command == "adaptive":
        from ashare_daily.commands import adaptive as module
    elif command == "listing":
        from ashare_daily.commands import listing as module
    elif command == "hrp":
        from ashare_daily.commands import hrp as module
    elif command == "triggers":
        from ashare_daily.commands import triggers as module
    else:
        raise SystemExit(f"unknown command {command}")
    return module
