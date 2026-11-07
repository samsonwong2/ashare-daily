"""Industry picker keeps the highest score without loading qlib."""
from __future__ import annotations

import pandas as pd
import pytest

from ashare_daily.csi300_l2_hedge import industry_weight_representatives, pick_one_per_industry


def test_pick_one_per_industry_keeps_the_highest_score() -> None:
    frame = pd.DataFrame(
        {
            "code": ["SH600000", "SH600001", "SZ000001", "SH600519"],
            "name": ["甲", "乙", "丙", "丁"],
            "申万二级代码": ["801044.SI", "801044.SI", "801045.SI", ""],
            "申万二级名称": ["普钢", "普钢", "特钢Ⅱ", ""],
            "score": [0.2, 0.9, 0.9, 0.99],
        }
    )
    picked = pick_one_per_industry(frame)
    assert list(picked["code"]) == ["SH600001", "SZ000001"]
    assert list(picked["申万二级名称"]) == ["普钢", "特钢Ⅱ"]


def test_pick_one_per_industry_breaks_score_ties_by_code() -> None:
    frame = pd.DataFrame(
        {
            "code": ["SZ000002", "SH600000"],
            "name": ["乙", "甲"],
            "申万二级名称": ["银行", "银行"],
            "score": [0.5, 0.5],
        }
    )
    picked = pick_one_per_industry(frame)
    assert list(picked["code"]) == ["SH600000"]


def test_industry_weight_representatives_sums_the_group_and_keeps_the_winner() -> None:
    frame = pd.DataFrame(
        {
            "code": ["SH600001", "SZ000002", "SH600000", "SZ000001", "SH600519"],
            "name": ["低分", "同分乙", "同分甲", "特钢", "空白"],
            "申万二级名称": ["普钢", "银行", "银行", "特钢Ⅱ", ""],
            "score": [0.2, 0.5, 0.5, 0.9, 0.99],
            "权重": [1.0, 2.5, None, None, 9.0],
        }
    )
    picked = industry_weight_representatives(frame)
    assert list(picked["申万二级名称"]) == ["普钢", "特钢Ⅱ", "银行"]
    assert list(picked["code"]) == ["SH600001", "SZ000001", "SH600000"]
    totals = picked.set_index("申万二级名称")["所有此申万二级权重"]
    assert totals["普钢"] == pytest.approx(1.0)
    assert totals["银行"] == pytest.approx(2.5)
    assert pd.isna(totals["特钢Ⅱ"])
    assert list(picked.columns)[-1] == "所有此申万二级权重"
    assert "空白" not in set(picked["申万二级名称"])
