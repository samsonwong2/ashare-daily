"""cn-data replaces the Qlib tree only after the release check passes."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from ashare_daily.cn_data import backup_dir, refresh_cn_data
from ashare_daily.commands.cn_data import main


def test_backup_dir_adds_a_suffix_when_the_day_is_taken(tmp_path: Path) -> None:
    dest = tmp_path / "cn_data"
    taken = tmp_path / "cn_data.bak.20261004"
    taken.mkdir()
    assert backup_dir(dest, date(2026, 10, 4)) == tmp_path / "cn_data.bak.20261004.2"


def test_failed_check_leaves_the_current_tree(tmp_path: Path) -> None:
    dest = tmp_path / "cn_data"
    (dest / "instruments").mkdir(parents=True)
    (dest / "instruments" / "csi1800.txt").write_text("local\n", encoding="utf-8")

    def fake_wget(url: str, path: Path) -> int:
        path.write_text(url, encoding="utf-8")
        return 0

    code = refresh_cn_data(
        tag="2026-10-04",
        dest=dest,
        wget_file=fake_wget,
        validate=lambda *_args: 1,
        extract=lambda *_args: 0,
        today=date(2026, 10, 4),
    )
    assert code == 1
    assert (dest / "instruments" / "csi1800.txt").read_text(encoding="utf-8") == "local\n"
    assert list(tmp_path.glob("cn_data.bak.*")) == []


def test_passed_check_swaps_the_tree(tmp_path: Path) -> None:
    dest = tmp_path / "cn_data"
    dest.mkdir()
    (dest / "old.txt").write_text("old", encoding="utf-8")

    def fake_wget(url: str, path: Path) -> int:
        path.write_text(url, encoding="utf-8")
        return 0

    def fake_extract(archive: Path, target: Path) -> int:
        target.mkdir(parents=True, exist_ok=True)
        (target / "calendars").mkdir()
        (target / "calendars" / "day.txt").write_text("2026-10-03\n", encoding="utf-8")
        return 0

    code = refresh_cn_data(
        tag="2026-10-04",
        dest=dest,
        wget_file=fake_wget,
        validate=lambda *_args: 0,
        extract=fake_extract,
        today=date(2026, 10, 4),
    )
    assert code == 0
    assert (dest / "calendars" / "day.txt").read_text(encoding="utf-8") == "2026-10-03\n"
    assert not (dest / "old.txt").exists()
    backup = tmp_path / "cn_data.bak.20261004"
    assert (backup / "old.txt").read_text(encoding="utf-8") == "old"


def test_command_rejects_a_bad_tag() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--tag", "latest"])
    assert exc.value.code == 2
