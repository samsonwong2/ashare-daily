#!/usr/bin/env python3
"""Add fig7–fig11 EWMA residual rails to from_listing adaptive HTML.

Idempotent: files that already contain ``EWMA残差轨`` only get subplot-title
patches. Intended after ``daily_adaptive_from_listing_stock.sh`` and before
fig12, because these stock plots are drawn with causal P95 rails only.

Example::

    PYTHONPATH=. python decision_pack/scripts/add_ewma_rails.py \\
      --html-dir ~/temp/stock/plotly_outputs/20260924_from_listing_stock --jobs 4
"""
from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from decision_pack.src.ewma_fair_rails import inject_ewma_rails_html  # noqa: E402
from decision_pack.src.fair_path_band_signals import _extract_plotly_traces  # noqa: E402
from decision_pack.scripts.add_fig12_six_states import (  # noqa: E402
    list_html_files,
    parse_html_label,
)

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--html-dir", type=Path, required=True)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--code", action="append", default=[])
    return p.parse_args(argv)


def add_ewma_to_html(html_path: Path) -> dict[str, str]:
    html_path = Path(html_path)
    code, _name = parse_html_label(html_path)
    raw = html_path.read_text(encoding="utf-8")
    traces = _extract_plotly_traces(raw)
    updated = inject_ewma_rails_html(raw, traces)
    if updated != raw:
        html_path.write_text(updated, encoding="utf-8")
        status = "updated"
    else:
        status = "unchanged"
    return {"ok": "1", "code": code, "status": status, "error": ""}


def _worker(path_str: str) -> dict[str, str]:
    try:
        return add_ewma_to_html(Path(path_str))
    except Exception as exc:  # noqa: BLE001
        path = Path(path_str)
        code, _ = parse_html_label(path)
        return {
            "ok": "0",
            "code": code,
            "status": "",
            "error": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    html_dir = args.html_dir.expanduser().resolve()
    files = list_html_files(html_dir, args.code)
    if not files:
        print(f"[ewma] no matching HTML in {html_dir}")
        return 0
    jobs = max(1, int(args.jobs))
    results: list[dict[str, str]] = []
    if jobs == 1 or len(files) == 1:
        for p in files:
            results.append(_worker(str(p)))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            futs = {pool.submit(_worker, str(p)): p for p in files}
            done = 0
            n = len(files)
            for fut in as_completed(futs):
                rec = fut.result()
                results.append(rec)
                done += 1
                flag = "OK" if rec["ok"] == "1" else "FAIL"
                print(f"[ewma] {done}/{n} {flag} {rec['code']} {rec.get('status', '')}")
                if rec["ok"] != "1":
                    print(rec["error"], file=sys.stderr)
    if jobs == 1 or len(files) == 1:
        for rec in results:
            flag = "OK" if rec["ok"] == "1" else "FAIL"
            print(f"[ewma] {flag} {rec['code']} {rec.get('status', '')}")
            if rec["ok"] != "1":
                print(rec["error"], file=sys.stderr)
    failed = [r for r in results if r.get("ok") != "1"]
    print(f"[ewma] done ok={len(results) - len(failed)} fail={len(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
