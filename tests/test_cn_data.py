"""cn-data replaces the Qlib tree only after the release check passes."""
from __future__ import annotations

import json
import subprocess
from datetime import date
from pathlib import Path

import pytest

from ashare_daily import cn_data
from ashare_daily.cn_data import LATEST_URL, backup_dir, fetch_latest_tag, refresh_cn_data, wget
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


class _OpenSocket:
    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> bool:
        return False


def test_local_github_proxy_when_port_accepts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cn_data.socket, "create_connection", lambda *_args, **_kwargs: _OpenSocket())
    assert cn_data.local_github_proxy() == "http://127.0.0.1:31181"


def test_local_github_proxy_when_port_is_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args, **_kwargs):
        raise OSError("down")

    monkeypatch.setattr(cn_data.socket, "create_connection", fail)
    assert cn_data.local_github_proxy() is None


def test_wget_adds_proxy_and_ca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def fake_run(cmd, check=False):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cn_data.subprocess, "run", fake_run)
    ca = tmp_path / "dev-sidecar-bundle.crt"
    code = wget(
        "https://github.com/chenditc/investment_data/releases/download/2026-10-14/qlib_bin.tar.gz",
        tmp_path / "qlib_bin.tar.gz",
        proxy="http://127.0.0.1:31181",
        ca_file=ca,
    )
    assert code == 0
    command = captured["cmd"]
    assert "https_proxy=http://127.0.0.1:31181" in command
    assert "http_proxy=http://127.0.0.1:31181" in command
    assert f"--ca-certificate={ca}" in command
    assert command[-1].endswith("qlib_bin.tar.gz")


def test_latest_tag_goes_through_the_proxy(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_wget(url: str, path: Path, *, proxy: str | None = None, ca_file: Path | None = None) -> int:
        seen["url"] = url
        seen["proxy"] = proxy
        seen["ca_file"] = ca_file
        path.write_text(json.dumps({"tag_name": "2026-10-14"}), encoding="utf-8")
        return 0

    ca = tmp_path / "ca.crt"
    assert fetch_latest_tag(proxy="http://127.0.0.1:31181", ca_file=ca, wget_file=fake_wget) == "2026-10-14"
    assert seen == {"url": LATEST_URL, "proxy": "http://127.0.0.1:31181", "ca_file": ca}


def test_refresh_downloads_the_archive_through_the_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str | None, Path | None]] = []
    ca = tmp_path / "dev-sidecar-bundle.crt"

    def fake_wget(url: str, path: Path, *, proxy: str | None = None, ca_file: Path | None = None) -> int:
        calls.append((url, proxy, ca_file))
        path.write_text(url, encoding="utf-8")
        return 0

    def fake_extract(archive: Path, target: Path) -> int:
        target.mkdir(parents=True, exist_ok=True)
        (target / "ok.txt").write_text("ok", encoding="utf-8")
        return 0

    monkeypatch.setattr(cn_data, "wget", fake_wget)
    monkeypatch.setattr(cn_data, "local_github_proxy", lambda: "http://127.0.0.1:31181")
    monkeypatch.setattr(cn_data, "proxy_ca_file", lambda: ca)
    dest = tmp_path / "cn_data"
    code = refresh_cn_data(
        tag="2026-10-14",
        dest=dest,
        validate=lambda *_args: 0,
        extract=fake_extract,
        today=date(2026, 10, 15),
    )
    assert code == 0
    assert [item[1] for item in calls] == ["http://127.0.0.1:31181"] * 3
    assert [item[2] for item in calls] == [ca, ca, ca]
    assert calls[0][0].endswith("/2026-10-14/qlib_bin.tar.gz")
    assert "cn-data: using proxy http://127.0.0.1:31181" in capsys.readouterr().out


def test_refresh_downloads_directly_when_the_proxy_is_down(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str | None] = []

    def fake_wget(url: str, path: Path, *, proxy: str | None = None, ca_file: Path | None = None) -> int:
        calls.append(proxy)
        path.write_text(url, encoding="utf-8")
        return 0

    monkeypatch.setattr(cn_data, "wget", fake_wget)
    monkeypatch.setattr(cn_data, "local_github_proxy", lambda: None)
    code = refresh_cn_data(
        tag="2026-10-14",
        dest=tmp_path / "cn_data",
        validate=lambda *_args: 1,
        extract=lambda *_args: 0,
        today=date(2026, 10, 15),
    )
    assert code == 1
    assert calls == [None, None, None]
    assert "downloading directly" in capsys.readouterr().out
