#!/usr/bin/env python
"""Thin launcher: orchestrator logic lives in ``pool_builder/`` (see README.md).

生产默认参数（dist_t=0.60, target=100, scoring=sharpe_mom, rep_pick=blend, linkage=average, detone=off, cross_rank=low_corr_return）在 ``pool_builder/constants.py``；
可用环境变量覆盖，回滚见 ``experiments/cluster_tuning/rollback.env.example``。

Mirrors the top-level project style (see
``etf_strategy_pipeline/从更新txt到最终monitor_最短命令清单_20260403.py``).
"""
import sys
from pathlib import Path

# Repo root (runtime_paths) and this directory (pool_builder).
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from pool_builder import constants as C
from pool_builder.cli import main


if __name__ == "__main__":
    print(
        "[INFO] Production clustering defaults: "
        f"dist_t={C.DIST_T}, target={C.TARGET_SELECTED_COUNT}, "
        f"scoring={C.SCORING_MODE}, rep_pick={C.REP_PICK_MODE}, linkage={C.CLUSTER_LINKAGE}, "
        f"cross_rank={C.CROSS_CLUSTER_RANK}"
    )
    print(f"[INFO] Outputs: {C.CSV_OUT} -> {C.CSV_SELECTED_OUT} -> {C.DEFAULT_CANONICAL_TXT}")
    raise SystemExit(main())
