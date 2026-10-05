"""Score threshold keeps names without loading qlib."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ashare_daily.screen import publish_selected_txt, score_table


def test_score_table_marks_the_threshold() -> None:
    dates = pd.bdate_range("2025-01-01", periods=40)
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(
        {
            "AAA": rng.normal(0.01, 0.01, size=len(dates)),
            "BBB": rng.normal(-0.01, 0.01, size=len(dates)),
        },
        index=dates,
    )
    corr = returns.corr()
    frame = score_table(
        returns,
        corr,
        score_min=0.0,
        valid_days=pd.Series({"AAA": 40, "BBB": 40}),
        name_map={"AAA": "甲"},
    )
    assert list(frame["code"]) == ["AAA", "BBB"]
    assert frame.loc[frame["code"] == "AAA", "name"].item() == "甲"
    assert bool(frame["selected"].all())
    tight = score_table(
        returns,
        corr,
        score_min=1.1,
        valid_days=pd.Series({"AAA": 40, "BBB": 40}),
        name_map={},
    )
    assert not bool(tight["selected"].any())


def test_publish_selected_txt_copies_the_instruments_file(tmp_path) -> None:
    source = tmp_path / "stock_screen_selected.txt"
    source.write_text("SH600000\t2000-01-04\t2026-09-24\n", encoding="utf-8")
    dest = tmp_path / "instruments" / "cluster_mapping_selected.txt"
    written = publish_selected_txt(source, dest)
    assert written == dest
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
