#!/usr/bin/env python
"""Thin launcher: daily hand-curated stock cluster pool review.

Example::

    python 每日审查cluster_mapping_最短命令清单.py \\
      --future-end 2026-09-01 \\
      --selected-csv ~/temp/stock/stock_cluster_mapping_selected.csv \\
      --mapping-csv ~/temp/stock/stock_cluster_mapping.csv

Optional: ``--skip-diff``, ``--skip-audit``, ``--min-regret 0.01``, ``--top-n 15``.
"""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE))

from pool_builder.daily_review import main


if __name__ == "__main__":
    raise SystemExit(main())
