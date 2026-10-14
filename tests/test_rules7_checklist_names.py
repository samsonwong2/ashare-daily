"""Checklist names keep the ticker in the code column only."""
from pathlib import Path

import pandas as pd

from ashare_daily.plots.scan_rules7_checklist import (
    _format_flat_buy_section,
    _format_hrp_cluster_section,
    parse_html_label,
)


def test_html_label_name_is_chinese_only() -> None:
    code, name = parse_html_label(
        Path("regime_transition_SZ300677_英科医疗_20100101_20261013_adaptive.html")
    )
    assert code == "SZ300677"
    assert name == "英科医疗"


def test_cluster_members_do_not_repeat_the_code() -> None:
    df = pd.DataFrame(
        [
            {
                "代码": "SZ300677",
                "名称": "英科医疗",
                "今日状态": "空仓",
                "今日动作": "买",
                "推荐买入": "是",
                "簇人数": "1",
                "ret20": "1.2%",
                "_cluster": 3,
                "_is_rep": True,
                "_ret20": 0.012,
            }
        ]
    )
    text = "\n".join(_format_hrp_cluster_section(df, cluster_path=Path("cluster.csv")))
    assert "SZ300677 英科医疗" in text
    assert "SZ300677SZ300677" not in text
    assert "SZ300677 SZ300677" not in text


def test_buy_quote_section_lists_every_name_including_holdings() -> None:
    df = pd.DataFrame(
        [
            {
                "代码": "SZ300677",
                "名称": "英科医疗",
                "类型": "高波动主题",
                "今日状态": "空仓",
                "今日动作": "观望",
                "买点价": "7.1055",
                "买点状态": "快形态挡住",
                "T收盘": "8.8822",
                "图9下轨": "7.0351",
                "距图9下轨": "26.3%",
                "_html": "regime_transition_SZ300677_英科医疗_20170721_20261013_adaptive.html",
            },
            {
                "代码": "SZ002831",
                "名称": "裕同科技",
                "类型": "长期下跌或周期",
                "今日状态": "持仓",
                "今日动作": "持有",
                "买点价": "",
                "买点状态": "",
                "T收盘": "1.8183",
                "图9下轨": "1.7601",
                "距图9下轨": "3.3%",
                "买规则": "aux_edge 或 B1",
                "买入信号": "是",
                "快形态(≥20%)": "",
                "辅助买像": "是",
                "图7或图8折价≤-8%": "是",
                "图9近下轨(≤下轨×1.01)": "是",
            },
        ]
    )
    text = "\n".join(_format_flat_buy_section(df, Path("/tmp/no-listing")))
    assert "英科医疗" in text
    assert "裕同科技" in text
    assert "| 持仓 |" in text
    assert "1.7777" in text
    assert "今天已是买点" in text
