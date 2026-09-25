"""``ashare-daily`` console script. Eight subcommands, one entry."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from ashare_daily.daily_contract import parse_as_of, slice_after_command
from ashare_daily.paths import repo_root

COMMANDS = (
    "pool-merge",
    "cluster-map",
    "cluster-review",
    "regime",
    "adaptive",
    "listing",
    "hrp",
    "triggers",
)


def load_plotly_root(root: Path | None = None) -> str:
    """Environment wins. Otherwise read ``<repo>/config.env``.

    Missing file or empty key exits 1 and names ``config.env.example``.
    """
    current = os.environ.get("PLOTLY_ROOT", "").strip()
    if current:
        return current
    checkout = Path(root) if root is not None else repo_root()
    env_file = checkout / "config.env"
    example = "config.env.example"
    if not env_file.is_file():
        print(
            f"PLOTLY_ROOT is not set and {env_file.name} is missing. Copy {example}.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    value = ""
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, item = line.split("=", 1)
        if key.strip() == "PLOTLY_ROOT":
            value = item.strip().strip('"').strip("'")
    if not value:
        print(f"PLOTLY_ROOT is empty in {env_file.name}. See {example}.", file=sys.stderr)
        raise SystemExit(1)
    os.environ["PLOTLY_ROOT"] = value
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ashare-daily")
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("rest", nargs=argparse.REMAINDER)
    return parser


def _as_of_in(rest: list[str]) -> None:
    if "--as-of" not in rest:
        return
    index = rest.index("--as-of")
    if index + 1 >= len(rest):
        print("[ERROR] --as-of must be YYYY-MM-DD", file=sys.stderr)
        raise SystemExit(2)
    parse_as_of(rest[index + 1])


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if not raw or raw[0] in {"-h", "--help"}:
        print("ashare-daily commands: " + ", ".join(COMMANDS))
        _parser().print_help()
        return 0
    command, rest = slice_after_command(raw)
    if command not in COMMANDS:
        _parser().parse_args(raw)
        return 2
    if "-h" in rest or "--help" in rest:
        from ashare_daily.commands import help_for

        help_for(command)
        return 0
    if command in {"regime", "adaptive", "listing"}:
        _as_of_in(rest)
    try:
        load_plotly_root(root)
    except SystemExit:
        raise
    from ashare_daily.commands import dispatch

    result = dispatch(command, rest)
    if isinstance(result, int):
        return result
    return 0
