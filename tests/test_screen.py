"""Score threshold keeps names without loading qlib."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ashare_daily.screen import (
    SHENWAN_COLUMNS,
    append_unscored,
    attach_shenwan,
    publish_selected_txt,
    score_table,
    write_index_score_slices,
    write_latest_constituent_txt,
)


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


def test_append_unscored_keeps_every_universe_code() -> None:
    dates = pd.bdate_range("2025-01-01", periods=5)
    returns_raw = pd.DataFrame(
        {"AAA": [0.01, 0.01, 0.01, 0.01, None], "CCC": [None, None, None, 0.02, 0.02]},
        index=dates,
    )
    scored = pd.DataFrame(
        {
            "code": ["AAA"],
            "name": ["甲"],
            "score": [0.9],
            "cum_return": [0.04],
            "mean_abs_corr": [0.1],
            "valid_days": [4],
            "selected": [True],
        }
    )
    frame = append_unscored(
        scored,
        ["AAA", "BBB", "CCC"],
        valid_days=returns_raw.notna().sum(),
        returns_raw=returns_raw,
        name_map={"BBB": "乙"},
    )
    assert list(frame["code"]) == ["AAA", "BBB", "CCC"]
    zeros = frame[frame["code"] != "AAA"]
    assert list(zeros["score"]) == [0.0, 0.0]
    assert not bool(zeros["selected"].any())
    assert int(frame.loc[frame["code"] == "CCC", "valid_days"].item()) == 2
    assert int(frame.loc[frame["code"] == "BBB", "valid_days"].item()) == 0
    assert float(frame.loc[frame["code"] == "BBB", "cum_return"].item()) == 0.0
    assert frame.loc[frame["code"] == "BBB", "name"].item() == "乙"


def test_attach_shenwan_inserts_industry_after_name(tmp_path) -> None:
    industry = tmp_path / "sw_third_constituents.csv"
    industry.write_text(
        "申万二级代码,申万二级名称,申万三级代码,申万三级名称,code\n"
        "801081.SI,半导体,850813.SI,半导体材料,sh688146\n",
        encoding="utf-8",
    )
    second = tmp_path / "sw_second_constituents.csv"
    second.write_text(
        "申万二级代码,申万二级名称,code\n"
        "801045.SI,特钢Ⅱ,SH600282\n",
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "code": ["SH688146", "SH600282", "SH689009"],
            "name": ["中船特气", "南钢股份", "九号公司"],
            "score": [0.9, 0.5, 0.1],
            "selected": [True, False, False],
        }
    )
    attached = attach_shenwan(frame, industry, second)
    assert list(attached.columns[:6]) == ["code", "name", *SHENWAN_COLUMNS]
    row = attached.loc[attached["code"] == "SH688146"].iloc[0]
    assert row["申万三级代码"] == "850813.SI"
    assert row["申万三级名称"] == "半导体材料"
    assert row["申万二级代码"] == "801081.SI"
    steel = attached.loc[attached["code"] == "SH600282"].iloc[0]
    assert steel["申万二级代码"] == "801045.SI"
    assert steel["申万二级名称"] == "特钢Ⅱ"
    assert steel["申万三级代码"] == ""
    assert attached.loc[attached["code"] == "SH689009", "申万二级代码"].item() == ""


def test_publish_selected_txt_copies_the_instruments_file(tmp_path) -> None:
    source = tmp_path / "stock_screen_selected.txt"
    source.write_text("SH600000\t2000-01-04\t2026-09-24\n", encoding="utf-8")
    dest = tmp_path / "instruments" / "cluster_mapping_selected.txt"
    written = publish_selected_txt(source, dest)
    assert written == dest
    assert dest.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def _score_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "code": ["SH600000", "SH600001", "SZ000001", "SH600519", "SH601318", "SH999999"],
            "name": ["浦发", "邯郸", "平安", "茅台", "平安银行", "圈外"],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "cum_return": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            "mean_abs_corr": [0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
            "valid_days": [200, 200, 200, 200, 200, 200],
            "selected": [True, False, False, False, False, False],
        }
    )


def _write_index_files(inst: Path) -> None:
    inst.mkdir()
    (inst / "csi300.txt").write_text(
        "\n".join(
            [
                "SH600000\t2005-01-01\t2020-01-01",
                "SH600000\t2024-01-01\t2026-10-01",
                "sh600001\t2024-01-01\t2026-10-01",
                "SZ000001\t2024-01-01\t2020-01-01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (inst / "csi500.txt").write_text(
        "SH600519\t2020-01-01\t2019-01-01\nSH600519\t2024-01-01\t2026-09-01\n",
        encoding="utf-8",
    )
    (inst / "csi800.txt").write_text(
        "OLDCODE\t2000-01-01\t2001-01-01\nSH601318\t2024-01-01\t2026-08-01\n",
        encoding="utf-8",
    )


def test_index_slices_keep_only_the_latest_constituents(tmp_path) -> None:
    inst = tmp_path / "instruments"
    _write_index_files(inst)
    frame = _score_frame()
    frame["申万二级名称"] = ["银行", "银行", "其他", "白酒", "保险", "圈外"]
    scores = tmp_path / "stock_screen_scores.csv"
    weights = pd.DataFrame(
        {
            "日期": ["2026-08-31", "2026-09-30", "2026-10-31"],
            "成分券代码": ["600000", "600000", "000001"],
            "权重": [1.1, 1.2, 3.4],
        }
    )
    written = write_index_score_slices(frame, scores, inst, csi300_weights=weights)

    by_name = {item["name"]: item for item in written}
    assert list(by_name) == ["csi300", "csi500", "csi800"]
    csi300 = pd.read_csv(by_name["csi300"]["path"])
    assert list(csi300["code"]) == ["SH600000", "SH600001"]
    assert int(by_name["csi300"]["rows"]) == 2
    assert list(csi300.columns) == [*frame.columns, "权重日期", "权重"]
    matched = csi300.set_index("code")
    assert matched.loc["SH600000", "权重日期"] == "2026-09-30"
    assert matched.loc["SH600000", "权重"] == pytest.approx(1.2)
    assert pd.isna(matched.loc["SH600001", "权重日期"])
    assert pd.isna(matched.loc["SH600001", "权重"])
    l2 = pd.read_csv(by_name["csi300"]["l2_path"])
    assert by_name["csi300"]["l2_path"].name == "stock_screen_scores_csi300_l2.csv"
    assert int(by_name["csi300"]["l2_rows"]) == 1
    assert list(l2["code"]) == ["SH600000"]
    assert l2.loc[0, "所有此申万二级权重"] == pytest.approx(1.2)
    assert list(l2.columns)[-1] == "所有此申万二级权重"
    csi500 = pd.read_csv(by_name["csi500"]["path"])
    assert list(csi500["code"]) == ["SH600519"]
    csi800 = pd.read_csv(by_name["csi800"]["path"])
    assert list(csi800["code"]) == ["SH601318"]
    assert list(csi500.columns) == list(frame.columns)
    assert list(csi800.columns) == list(frame.columns)
    for name in ("csi300", "csi500", "csi800"):
        assert by_name[name]["path"].name == f"stock_screen_scores_{name}.csv"


def test_index_slices_write_nothing_when_one_file_is_missing(tmp_path) -> None:
    inst = tmp_path / "instruments"
    inst.mkdir()
    (inst / "csi300.txt").write_text("SH600000\t2024-01-01\t2026-10-01\n", encoding="utf-8")
    (inst / "csi800.txt").write_text("SH601318\t2024-01-01\t2026-08-01\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        write_index_score_slices(_score_frame(), tmp_path / "stock_screen_scores.csv", inst)
    assert not (tmp_path / "stock_screen_scores_csi300.csv").exists()
    assert not (tmp_path / "stock_screen_scores_csi500.csv").exists()
    assert not (tmp_path / "stock_screen_scores_csi800.csv").exists()
    assert not (tmp_path / "stock_screen_scores_csi300_l2.csv").exists()


def test_write_latest_constituent_txt_keeps_the_current_csi300_rows(tmp_path) -> None:
    inst = tmp_path / "instruments"
    inst.mkdir()
    (inst / "csi300.txt").write_text(
        "\n".join(
            [
                "SZ000002\t2005-01-01\t2020-01-01",
                "SH600000\t2024-06-01\t2026-10-26",
                "SZ000001\t2024-01-01\t2026-10-26",
                "SH600000\t2005-01-01\t2020-01-01",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dest = tmp_path / "out" / "cluster_mapping_selected.txt"
    written = write_latest_constituent_txt("csi300", dest, inst, expect=2)
    assert written.read_text(encoding="utf-8") == (
        "SH600000\t2024-06-01\t2026-10-26\n"
        "SZ000001\t2024-01-01\t2026-10-26\n"
    )


def test_write_latest_constituent_txt_rejects_a_short_csi300(tmp_path) -> None:
    inst = tmp_path / "instruments"
    inst.mkdir()
    (inst / "csi300.txt").write_text("SH600000\t2024-01-01\t2026-10-26\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="expected 300"):
        write_latest_constituent_txt("csi300", tmp_path / "out.txt", inst, expect=300)


def test_screen_canonical_flag_defaults_to_score() -> None:
    from ashare_daily.commands.screen import parser

    args = parser().parse_args([])
    assert args.canonical == "score"
    args = parser().parse_args(["--canonical", "csi300"])
    assert args.canonical == "csi300"


def test_rolling_test_period_follows_the_run_date() -> None:
    from ashare_daily.screen import rolling_test_period

    start, end = rolling_test_period("2026-10-28")
    assert end == "2026-10-28"
    # 252 sessions need roughly 365 calendar days. Holidays need slack.
    assert (pd.Timestamp(end) - pd.Timestamp(start)).days >= 400


def test_screen_as_of_flag_is_optional() -> None:
    from ashare_daily.commands.screen import parser

    assert parser().parse_args([]).as_of is None
    assert parser().parse_args(["--as-of", "2026-10-26"]).as_of == "2026-10-26"
