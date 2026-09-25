"""Top-level orchestrator for generating ``cluster_mapping_selected.txt``.

Replaces ``2_filter/0.1run_all_fund_herc.py``. Runs the two core stages:
1. ``select`` — produce ``cluster_mapping.csv`` / ``cluster_mapping_selected.csv``
   (and the benchmark variant) via :mod:`pool_builder.selector`.
2. ``filter-txt`` — turn the selected CSV into the canonical instruments
   ``cluster_mapping_selected.txt`` via :mod:`pool_builder.filter_txt`.

Optional stages that depended on ``cluster_pool_rolling_eval/`` (one-shot
forward evaluation + auto-apply best pool) are out of scope of this package;
a thin hook is provided via ``--production-eval-cmd`` / ``--production-apply-cmd``
to keep the door open without hard-coding a missing repo path.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import shlex
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from . import constants as C
from .filter_txt import filter_all_txt
from .pool_risk_paths import (
    attach_recent_returns_to_risk_csv,
    materialize_canonical_pool_risk,
    pack_pool_risk_canonical_path,
)
from .selector import select_pool

from runtime_paths import (
    ETF_RISK_PACK_DIR,
    ETF_RISK_REPORT_SCRIPT,
    ETF_RISK_SOURCE_PACK_DIR,
    QLIB_PROVIDER_URI,
    TEMP_DIR,
)

def _require_risk_path(path: Path, key: str) -> Path:
    if not str(path) or str(path) == "." or not path.is_file() and key == "etf_risk_report_script":
        if key == "etf_risk_report_script" and not path.is_file():
            raise FileNotFoundError(
                "missing paths.etf_risk_report_script in configs/production_regime_switch_ewma_shrink.json "
                f"(got {path}). Point it at generate_fat_tail_risk_report.py. See the example config."
            )
    if key != "etf_risk_report_script" and (not str(path) or str(path) == "."):
        raise FileNotFoundError(
            f"missing paths.{key} in configs/production_regime_switch_ewma_shrink.json. See the example config."
        )
    return path

RISK_REPORT_SCRIPT = ETF_RISK_REPORT_SCRIPT
RISK_REPORT_PACK_DIR = ETF_RISK_PACK_DIR
RISK_REPORT_PROVIDER_URI = str(QLIB_PROVIDER_URI)
RISK_OUTPUT_CSV = TEMP_DIR / "stock_cluster_mapping_selected_pool_risk.csv"
RISK_SOURCE_PACK_DIR = ETF_RISK_SOURCE_PACK_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate cluster_mapping_selected.txt: select → filter_txt [→ optional one-shot eval/apply]"
    )
    parser.add_argument(
        "--shared-config",
        default=None,
        help="Path to shared_filter_config.json (default: fund_pool_builder/shared_filter_config.json)",
    )
    parser.add_argument(
        "--inception-cutoff",
        default=None,
        help="Override fund-inception cutoff date (default: test_period[1] from shared config)",
    )
    parser.add_argument(
        "--skip-select",
        action="store_true",
        help="Skip the clustering/selection stage (reuse existing cluster_mapping_selected.csv)",
    )
    parser.add_argument(
        "--skip-filter-txt",
        action="store_true",
        help="Skip the canonical-txt stage",
    )
    parser.add_argument(
        "--auto-selected-suffix",
        default=None,
        metavar="YYYYMMDD",
        help=(
            "Write machine-selected rows to cluster_mapping_selected_{suffix}.csv under "
            f"{C.OUT_DIR} and do not overwrite {C.CSV_SELECTED_OUT}."
        ),
    )
    parser.add_argument(
        "--refresh-stock-list",
        action="store_true",
        help="Refresh the stock list csv from ak.stock_info_a_code_name() before selecting.",
    )
    parser.add_argument(
        "--csv-path",
        default=C.CSV_SELECTED_OUT,
        help=f"Input CSV for filter-txt stage (default: {C.CSV_SELECTED_OUT})",
    )
    parser.add_argument(
        "--all-path",
        default=C.DEFAULT_ALL_TXT,
        help=f"qlib all.txt path (default: {C.DEFAULT_ALL_TXT})",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help=(
            "Output canonical txt. Defaults to "
            f"{C.DEFAULT_CANONICAL_TXT} when --csv-path is left at its default, "
            "else <csv_path>.with_suffix('.txt')."
        ),
    )
    parser.add_argument(
        "--code-column",
        default=None,
        help="Explicit code column name in --csv-path.",
    )
    parser.add_argument(
        "--production-eval-cmd",
        default=None,
        help=(
            "Optional shell command for the one-shot production evaluation stage "
            "(e.g. 'python /path/to/production_cluster_pool_eval.py --eval-start 2025-04-01 "
            "--eval-end 2026-03-31'). If unset, the stage is skipped."
        ),
    )
    parser.add_argument(
        "--production-apply-cmd",
        default=None,
        help=(
            "Optional shell command for the production apply stage "
            "(e.g. 'python /path/to/production_cluster_pool_apply.py'). If unset, skipped."
        ),
    )
    return parser


def _run_shell(command: str, stage_name: str) -> None:
    print(f"\n== {stage_name} ==")
    print(command)
    completed = subprocess.run(shlex.split(command), check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Stage failed: {stage_name}, exit_code={completed.returncode}")


def _latest_qlib_trade_date(provider_uri: str | Path) -> str:
    calendar_path = Path(provider_uri) / "calendars" / "day.txt"
    if not calendar_path.exists():
        raise RuntimeError(f"Missing qlib calendar: {calendar_path}")
    lines = [ln.strip() for ln in calendar_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Empty qlib calendar: {calendar_path}")
    return lines[-1]


def _write_stock_portfolio_snapshot(selected_csv_path: Path, pack_dir: Path) -> None:
    selected_df = pd.read_csv(selected_csv_path, encoding="utf-8-sig")
    if "selected" in selected_df.columns:
        selected_df = selected_df.loc[selected_df["selected"].astype(bool)]
    codes = [str(code).strip() for code in selected_df["code"].tolist() if str(code).strip()]
    if not codes:
        raise RuntimeError(f"No selected codes found in {selected_csv_path}")
    weight = 1.0 / len(codes)
    pack_dir.mkdir(parents=True, exist_ok=True)
    snapshot = pd.DataFrame({"code": codes, "weight_total": [weight] * len(codes)})
    snapshot.to_csv(pack_dir / "portfolio_snapshot.csv", index=False, encoding="utf-8-sig")
    copied_as_of = False
    for name in ("tier_decision.json", "market_snapshot.json"):
        source = RISK_SOURCE_PACK_DIR / name
        if source.exists():
            shutil.copy2(source, pack_dir / name)
            copied_as_of = True
    if not copied_as_of:
        # Source decision pack is gone (e.g. temp cleanup); fall back to the
        # latest trading day in the qlib calendar so the risk report can
        # still resolve ``as_of``.
        as_of = _latest_qlib_trade_date(RISK_REPORT_PROVIDER_URI)
        (pack_dir / "market_snapshot.json").write_text(
            json.dumps({"as_of": as_of}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] source pack {RISK_SOURCE_PACK_DIR} unavailable; wrote market_snapshot.json with as_of={as_of}")


def _apply_selected_names_to_risk_csv(selected_csv_path: Path, risk_csv_path: Path) -> None:
    selected_df = pd.read_csv(selected_csv_path, encoding="utf-8-sig")
    risk_df = pd.read_csv(risk_csv_path, encoding="utf-8-sig")
    if "code" not in selected_df.columns:
        return

    if "selected" in selected_df.columns:
        selected_df = selected_df.loc[selected_df["selected"].astype(bool)]
    selected_df = selected_df.dropna(subset=["code"]).drop_duplicates(subset=["code"], keep="first")
    selected_df = selected_df.reset_index(drop=True)
    selected_df["code"] = selected_df["code"].astype(str).str.strip()

    if "code" not in risk_df.columns:
        return
    risk_df = risk_df.dropna(subset=["code"]).copy()
    risk_df["code"] = risk_df["code"].astype(str).str.strip()

    mapping_columns = [col for col in selected_df.columns]
    risk_columns = [col for col in risk_df.columns if col != "code"]
    if "name" in risk_columns:
        risk_columns.remove("name")

    risk_payload = risk_df[["code"] + risk_columns].copy()
    if "n" in selected_df.columns and "n" in risk_payload.columns:
        risk_payload = risk_payload.rename(columns={"n": "n_risk"})

    risk_payload = risk_payload.drop_duplicates(subset=["code"], keep="first")
    merged_df = selected_df[mapping_columns].merge(
        risk_payload,
        on="code",
        how="left",
    )
    available_mapping_columns = [col for col in mapping_columns if col in merged_df.columns]
    risk_columns = [col for col in merged_df.columns if col not in available_mapping_columns and col != "code"]
    if "wt" in risk_columns:
        risk_columns = ["wt"] + [col for col in risk_columns if col != "wt"]
    output_columns = available_mapping_columns + risk_columns
    merged_df = merged_df.loc[:, output_columns]
    merged_df.to_csv(risk_csv_path, index=False, encoding="utf-8-sig")
    print(f"[INFO] aligned {risk_csv_path} to {len(selected_df)} selected rows")


def _generate_stock_pool_risk_csv(selected_csv_path: Path, cluster_mapping_path: Path) -> None:
    _require_risk_path(RISK_REPORT_SCRIPT, "etf_risk_report_script")
    _require_risk_path(RISK_REPORT_PACK_DIR, "etf_risk_pack_dir")
    _require_risk_path(RISK_SOURCE_PACK_DIR, "etf_risk_source_pack_dir")
    _write_stock_portfolio_snapshot(selected_csv_path, RISK_REPORT_PACK_DIR)
    output_path = pack_pool_risk_canonical_path(RISK_REPORT_PACK_DIR)
    if output_path.exists():
        output_path.unlink()

    env = os.environ.copy()
    py_path_parts = [str(RISK_REPORT_SCRIPT.parents[2])]
    existing = env.get("PYTHONPATH")
    if existing:
        py_path_parts.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    completed = subprocess.run(
        [
            sys.executable,
            str(RISK_REPORT_SCRIPT),
            "--pack-dir",
            str(RISK_REPORT_PACK_DIR),
            "--cluster-mapping",
            str(cluster_mapping_path),
            "--provider-uri",
            RISK_REPORT_PROVIDER_URI,
        ],
        check=False,
        cwd=str(RISK_REPORT_SCRIPT.parents[2]),
        env=env,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Risk report failed, exit_code={completed.returncode}")
    try:
        output_path = materialize_canonical_pool_risk(RISK_REPORT_PACK_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError("Risk report did not produce a pool risk CSV") from exc

    _apply_selected_names_to_risk_csv(selected_csv_path, output_path)
    as_of = None
    market_snapshot = RISK_REPORT_PACK_DIR / "market_snapshot.json"
    if market_snapshot.exists():
        try:
            as_of = str(json.loads(market_snapshot.read_text(encoding="utf-8")).get("as_of") or "") or None
        except Exception:
            as_of = None
    attach_recent_returns_to_risk_csv(
        output_path,
        provider_uri=RISK_REPORT_PROVIDER_URI,
        as_of=as_of,
    )

    RISK_OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(output_path, RISK_OUTPUT_CSV)
    print(f"[INFO] stock pool risk -> {RISK_OUTPUT_CSV}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.skip_select:
        print("\n== select ==")
        select_pool(
            shared_config_path=args.shared_config,
            inception_cutoff=args.inception_cutoff,
            auto_selected_suffix=args.auto_selected_suffix,
            refresh_stock_list=args.refresh_stock_list,
        )

    selected_csv_path = Path(args.csv_path).expanduser()
    cluster_mapping_path = Path(args.output_path).expanduser() if args.output_path else Path(C.DEFAULT_CANONICAL_TXT)

    if not args.skip_filter_txt or not cluster_mapping_path.exists():
        print("\n== filter_txt ==")
        filter_all_txt(
            csv_path=str(selected_csv_path),
            all_path=args.all_path,
            output_path=str(cluster_mapping_path),
            code_column=args.code_column,
        )

    _generate_stock_pool_risk_csv(selected_csv_path=selected_csv_path, cluster_mapping_path=cluster_mapping_path)

    if args.production_eval_cmd:
        _run_shell(args.production_eval_cmd, "production_one_shot_eval")

    if args.production_apply_cmd:
        _run_shell(args.production_apply_cmd, "apply_best_pool")

    print("\n[INFO] fund-pool pipeline completed successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
