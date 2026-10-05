"""Screen review ranks unselected winners and selected laggards."""
from __future__ import annotations

import pandas as pd

from ashare_daily.screen_review import attach_trailing, split_review


def test_split_review_orders_by_trailing_return() -> None:
    dates = pd.bdate_range("2026-01-01", periods=30)
    close = pd.DataFrame(
        {
            "AAA": [100 + i for i in range(30)],
            "BBB": [100 - i for i in range(30)],
            "CCC": [50 + i * 2 for i in range(30)],
        },
        index=dates,
    )
    scores = pd.DataFrame(
        {
            "code": ["AAA", "BBB", "CCC"],
            "name": ["甲", "乙", "丙"],
            "score": [0.9, 0.9, 0.4],
            "cum_return": [1.0, 1.0, 0.2],
            "mean_abs_corr": [0.1, 0.1, 0.2],
            "selected": [True, True, False],
        }
    )
    framed = attach_trailing(scores, close, "2026-02-15", [5, 20])
    missed, weak = split_review(framed, focus_window=20)
    assert list(missed["code"]) == ["CCC"]
    assert list(weak["code"]) == ["BBB", "AAA"]
