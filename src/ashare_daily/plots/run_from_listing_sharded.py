#!/usr/bin/env python3
"""Shard ``plot_adaptive_from_listing.py`` across N worker processes and merge.

Splits the stock pool into ``--jobs`` shards (round-robin by code order), runs
each shard into ``{html-out-dir}/_shards/{i:02d}``, then merges HTML / trades /
configs / summary CSVs into ``--html-out-dir``.

Does not change the core plotting loop; workers call the existing script with
``--code`` filters and isolated output dirs.

Examples::

    PYTHONPATH=. python decision_pack/scripts/run_from_listing_sharded.py \\
      --jobs 4 \\
      --as-of 2026-09-01 \\
      --config-source-dir workspace/plotly_outputs/20260901all_adaptive_stock \\
      --html-out-dir workspace/plotly_outputs/20260901_from_listing_stock \\
      --disable-ru-diag

    # Incremental day update (each shard reads the full previous dir)
    PYTHONPATH=. python decision_pack/scripts/run_from_listing_sharded.py \\
      --jobs 4 \\
      --as-of 2026-09-01 \\
      --config-source-dir workspace/plotly_outputs/20260901all_adaptive_stock \\
      --html-out-dir workspace/plotly_outputs/20260901_from_listing_stock \\
      --incremental-from workspace/plotly_outputs/20260831_from_listing_stock \\
      --incremental-mode html \\
      --disable-ru-diag
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from ashare_daily.lib.symbol_trend_board import load_cluster_mapping_codes  # noqa: E402
from ashare_daily.paths import CLUSTER_MAPPING_SELECTED_TXT, plotly_outputs_dir, repo_root  # noqa: E402

_PROJECT_ROOT = repo_root()

_CSV_MERGE_NAMES = (
    "listing_dates.csv",
    "batch_summary.csv",
    "from_listing_errors.csv",
)


def _resolve(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else _PROJECT_ROOT / path


def shard_codes(codes: list[str], jobs: int) -> list[list[str]]:
    """Round-robin split so long-history names are not all in one shard."""
    n = max(1, min(int(jobs), len(codes)))
    shards: list[list[str]] = [[] for _ in range(n)]
    for i, code in enumerate(codes):
        shards[i % n].append(code)
    return [s for s in shards if s]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--jobs",
        type=int,
        default=4,
        help="number of parallel worker processes (default 4)",
    )
    p.add_argument("--as-of", default=None)
    p.add_argument("--start-date", default=None)
    p.add_argument("--end-date", default=None)
    p.add_argument("--code", action="append", default=None)
    p.add_argument("--train-cutoff", default=None)
    p.add_argument("--config-source-dir", type=Path, default=None)
    p.add_argument("--html-out-dir", type=Path, default=None)
    p.add_argument("--validation-dir", type=Path, default=None)
    p.add_argument("--cluster-mapping", type=Path, default=None)
    p.add_argument("--live-snapshot", type=Path, default=None)
    p.add_argument("--max-codes", type=int, default=None)
    p.add_argument("--retrain", action="store_true")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--hrp-membership-csv", type=Path, default=None)
    p.add_argument("--disable-ru-diag", action="store_true")
    p.add_argument("--fair-path-extra-trail-years", type=str, default=None)
    p.add_argument(
        "--incremental-mode",
        choices=("off", "html"),
        default="off",
    )
    p.add_argument("--incremental-from", type=Path, default=None)
    p.add_argument("--incremental-close-tol", type=float, default=None)
    p.add_argument("--train-cache-dir", type=Path, default=None)
    p.add_argument("--listing-cache", type=Path, default=None)
    p.add_argument("--no-train-cache", action="store_true")
    p.add_argument("--no-listing-cache", action="store_true")
    p.add_argument(
        "--keep-shards",
        action="store_true",
        default=True,
        help="keep {html-out-dir}/_shards after merge (default)",
    )
    p.add_argument(
        "--rm-shards",
        action="store_true",
        help="delete {html-out-dir}/_shards after successful merge",
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="python interpreter for worker subprocesses (default: this interpreter)",
    )
    return p.parse_args(argv)


def resolve_codes(args: argparse.Namespace) -> list[str]:
    mapping = (
        Path(args.cluster_mapping)
        if args.cluster_mapping
        else Path(CLUSTER_MAPPING_SELECTED_TXT)
    )
    if not mapping.is_absolute():
        mapping = _PROJECT_ROOT / mapping
    codes = [c.upper() for c in (args.code or [])] or list(
        load_cluster_mapping_codes(mapping)
    )
    if args.max_codes:
        codes = codes[: int(args.max_codes)]
    if not codes:
        raise SystemExit("no codes to plot")
    return codes


def default_out_dir(as_of: str, start_date: str | None, end_date: str | None) -> Path:
    end_raw = end_date or as_of
    end_tag = pd.Timestamp(end_raw).strftime("%Y%m%d")
    as_of_tag = pd.Timestamp(as_of).strftime("%Y%m%d")
    if start_date:
        start_tag = pd.Timestamp(start_date).strftime("%Y%m%d")
        return plotly_outputs_dir() / f"{end_tag}_from_listing_{start_tag}_{end_tag}"
    return plotly_outputs_dir() / f"{as_of_tag}_from_listing"


def build_worker_argv(
    args: argparse.Namespace,
    *,
    shard_codes_list: list[str],
    shard_out: Path,
) -> list[str]:
    cmd: list[str] = [str(args.python), "-m", "ashare_daily.plots.plot_adaptive_from_listing"]
    if args.as_of:
        cmd += ["--as-of", str(args.as_of)]
    if args.start_date:
        cmd += ["--start-date", str(args.start_date)]
    if args.end_date:
        cmd += ["--end-date", str(args.end_date)]
    if args.train_cutoff:
        cmd += ["--train-cutoff", str(args.train_cutoff)]
    if args.config_source_dir is not None:
        cmd += ["--config-source-dir", str(args.config_source_dir)]
    cmd += ["--html-out-dir", str(shard_out)]
    if args.validation_dir is not None:
        cmd += ["--validation-dir", str(args.validation_dir)]
    if args.cluster_mapping is not None:
        cmd += ["--cluster-mapping", str(args.cluster_mapping)]
    if args.live_snapshot is not None:
        cmd += ["--live-snapshot", str(args.live_snapshot)]
    if args.retrain:
        cmd.append("--retrain")
    if args.fail_fast:
        cmd.append("--fail-fast")
    if args.hrp_membership_csv is not None:
        cmd += ["--hrp-membership-csv", str(args.hrp_membership_csv)]
    if args.disable_ru_diag:
        cmd.append("--disable-ru-diag")
    if args.fair_path_extra_trail_years is not None:
        cmd += ["--fair-path-extra-trail-years", str(args.fair_path_extra_trail_years)]
    if args.incremental_mode and args.incremental_mode != "off":
        cmd += ["--incremental-mode", str(args.incremental_mode)]
    if args.incremental_from is not None:
        cmd += ["--incremental-from", str(args.incremental_from)]
    if args.incremental_close_tol is not None:
        cmd += ["--incremental-close-tol", str(args.incremental_close_tol)]
    if args.train_cache_dir is not None:
        cmd += ["--train-cache-dir", str(args.train_cache_dir)]
    if args.listing_cache is not None:
        cmd += ["--listing-cache", str(args.listing_cache)]
    if args.no_train_cache:
        cmd.append("--no-train-cache")
    if args.no_listing_cache:
        cmd.append("--no-listing-cache")
    for code in shard_codes_list:
        cmd += ["--code", code]
    return cmd


def run_shard(
    args: argparse.Namespace,
    *,
    shard_idx: int,
    shard_codes_list: list[str],
    shard_out: Path,
    log_path: Path,
) -> tuple[int, int, Path]:
    shard_out.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = build_worker_argv(args, shard_codes_list=shard_codes_list, shard_out=shard_out)
    print(
        f"[INFO] shard {shard_idx:02d} start n={len(shard_codes_list)} "
        f"out={shard_out} log={log_path}"
    )
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            cwd=str(_PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
        )
    rc = int(proc.returncode)
    print(f"[INFO] shard {shard_idx:02d} done rc={rc} n={len(shard_codes_list)}")
    return shard_idx, rc, log_path


def _is_adaptive_html(path: Path) -> bool:
    stem = path.stem
    if not stem.startswith("regime_transition_") or not stem.endswith("_adaptive"):
        return False
    if "_fig9" in stem or "_fig10" in stem or "_oracle" in stem:
        return False
    return True


def merge_shards(shard_dirs: list[Path], out_dir: Path) -> None:
    """Copy worker artifacts into the final html-out-dir and concat CSVs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_dst = out_dir / "trades"
    configs_dst = out_dir / "configs"
    trades_dst.mkdir(parents=True, exist_ok=True)
    configs_dst.mkdir(parents=True, exist_ok=True)

    n_html = 0
    for shard in shard_dirs:
        for html in shard.glob("regime_transition_*_adaptive.html"):
            if not _is_adaptive_html(html):
                continue
            shutil.copy2(html, out_dir / html.name)
            n_html += 1
        trades_src = shard / "trades"
        if trades_src.is_dir():
            for f in trades_src.glob("*"):
                if f.is_file():
                    shutil.copy2(f, trades_dst / f.name)
        configs_src = shard / "configs"
        if configs_src.is_dir():
            for f in configs_src.glob("*.json"):
                shutil.copy2(f, configs_dst / f.name)

    for name in _CSV_MERGE_NAMES:
        frames: list[pd.DataFrame] = []
        for shard in shard_dirs:
            path = shard / name
            if not path.exists():
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] skip bad csv {path}: {exc}")
                continue
            if not df.empty:
                frames.append(df)
        dst = out_dir / name
        if not frames:
            if name == "from_listing_errors.csv" and dst.exists():
                # Successful run with no errors: drop stale error file if present.
                dst.unlink()
            continue
        merged = pd.concat(frames, ignore_index=True)
        if "code" in merged.columns:
            merged = merged.drop_duplicates(subset=["code"], keep="last")
            merged = merged.sort_values("code").reset_index(drop=True)
        merged.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"[OK] merged {name} rows={len(merged)}")
    print(f"[OK] merged adaptive html files={n_html} -> {out_dir}")


def _listing_lookup(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return {}
    if "code" not in df.columns or "listing" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        code = str(row["code"]).upper().strip()
        listing = str(row["listing"])[:10]
        if code and listing and listing.lower() not in {"nan", "none", ""}:
            out[code] = listing
    return out


def _refresh_listing_cache(args: argparse.Namespace, out_dir: Path) -> None:
    """Rewrite the persistent listing cache from the merged listing_dates.csv.

    Shard workers also write the cache; a late shard can drop dates discovered
    by an earlier shard. The merged CSV is the authority after a successful run.
    """
    if args.no_listing_cache or args.listing_cache is None:
        return
    cache = _resolve(args.listing_cache)
    listing_csv = out_dir / "listing_dates.csv"
    if cache is None or not listing_csv.exists():
        return
    merged = _listing_lookup(cache)
    merged.update(_listing_lookup(listing_csv))
    if not merged:
        return
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"code": c, "listing": merged[c]} for c in sorted(merged)]
    ).to_csv(cache, index=False, encoding="utf-8-sig")
    print(f"[INFO] wrote listing cache n={len(merged)} -> {cache}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.as_of and not args.end_date:
        raise SystemExit("need --as-of or --end-date")
    as_of = pd.Timestamp(args.as_of or args.end_date).strftime("%Y-%m-%d")
    args.as_of = as_of

    codes = resolve_codes(args)
    jobs = max(1, int(args.jobs))
    shards = shard_codes(codes, jobs)

    out_dir = _resolve(args.html_out_dir) or default_out_dir(
        as_of, args.start_date, args.end_date
    )
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    shards_root = out_dir / "_shards"
    if shards_root.exists():
        shutil.rmtree(shards_root)
    shards_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] as_of={as_of} codes={len(codes)} jobs={len(shards)} out={out_dir}")
    for i, sc in enumerate(shards):
        print(f"[INFO] shard {i:02d} codes={len(sc)} sample={sc[:3]}")

    results: list[tuple[int, int, Path]] = []
    with ThreadPoolExecutor(max_workers=len(shards)) as ex:
        futs = []
        for i, sc in enumerate(shards):
            shard_out = shards_root / f"{i:02d}"
            log_path = shards_root / f"{i:02d}.log"
            futs.append(
                ex.submit(
                    run_shard,
                    args,
                    shard_idx=i,
                    shard_codes_list=sc,
                    shard_out=shard_out,
                    log_path=log_path,
                )
            )
        for fut in as_completed(futs):
            results.append(fut.result())

    results.sort(key=lambda x: x[0])
    failed = [(i, rc, log) for i, rc, log in results if rc != 0]
    if failed:
        print("[ERROR] one or more shards failed; not rewriting merged CSV manifests")
        for i, rc, log in failed:
            print(f"[ERROR] shard {i:02d} rc={rc} log={log}")
        return 2

    shard_dirs = [shards_root / f"{i:02d}" for i, _, _ in results]
    merge_shards(shard_dirs, out_dir)
    _refresh_listing_cache(args, out_dir)

    if args.rm_shards:
        shutil.rmtree(shards_root, ignore_errors=True)
        print(f"[OK] removed shards dir {shards_root}")
    else:
        print(f"[OK] kept shards under {shards_root}")

    summary = out_dir / "batch_summary.csv"
    n_sum = -1
    if summary.exists():
        try:
            n_sum = len(pd.read_csv(summary))
        except Exception:  # noqa: BLE001
            n_sum = -1
    print(f"[OK] sharded from_listing done out={out_dir} batch_summary_rows={n_sum}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
