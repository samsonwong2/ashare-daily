"""Replace the local Qlib ``cn_data`` tree from one investment_data release."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from datetime import date
from pathlib import Path

REPO = "chenditc/investment_data"
LATEST_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
VALIDATOR_URL = f"https://raw.githubusercontent.com/{REPO}/main/qlib/validate_archive.py"
DEFAULT_DEST = Path("/home/huangtuo/data/qlib_data/cn_data")


def release_base(tag: str) -> str:
    return f"https://github.com/{REPO}/releases/download/{tag}"


def backup_dir(dest: Path, today: date | None = None) -> Path:
    stamp = (today or date.today()).strftime("%Y%m%d")
    candidate = dest.with_name(f"{dest.name}.bak.{stamp}")
    if not candidate.exists():
        return candidate
    index = 2
    while True:
        numbered = dest.with_name(f"{dest.name}.bak.{stamp}.{index}")
        if not numbered.exists():
            return numbered
        index += 1


def fetch_latest_tag(urlopen=urllib.request.urlopen) -> str:
    request = urllib.request.Request(
        LATEST_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "ashare-daily",
        },
    )
    try:
        with urlopen(request, timeout=60) as response:
            payload = json.load(response)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"cn-data: cannot read latest release: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    tag = str(payload.get("tag_name", "")).strip()
    try:
        date.fromisoformat(tag)
    except ValueError:
        print(f"cn-data: latest release tag is not YYYY-MM-DD: {tag}", file=sys.stderr)
        raise SystemExit(1)
    return tag


def wget(url: str, dest: Path) -> int:
    try:
        completed = subprocess.run(["wget", "-O", str(dest), url], check=False)
    except FileNotFoundError:
        print("cn-data: wget is required", file=sys.stderr)
        return 1
    return int(completed.returncode)


def run_validate(python: str, script: Path, archive: Path, manifest: Path, tag: str) -> int:
    completed = subprocess.run(
        [
            python,
            str(script),
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
            "--expected-tag",
            tag,
            "--require-publishable",
        ],
        check=False,
    )
    return int(completed.returncode)


def extract_tar(archive: Path, target: Path) -> int:
    target.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        ["tar", "-xzf", str(archive), "-C", str(target), "--strip-components=1"],
        check=False,
    )
    return int(completed.returncode)


def swap_in(extracted: Path, dest: Path, today: date | None = None) -> Path | None:
    backup = None
    if dest.exists():
        backup = backup_dir(dest, today)
        shutil.move(str(dest), str(backup))
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.move(str(extracted), str(dest))
    except Exception:
        if backup is not None and backup.exists() and not dest.exists():
            shutil.move(str(backup), str(dest))
        raise
    return backup


def refresh_cn_data(
    *,
    tag: str | None,
    dest: Path,
    latest_tag=fetch_latest_tag,
    wget_file=wget,
    validate=run_validate,
    extract=extract_tar,
    today: date | None = None,
    python: str | None = None,
) -> int:
    resolved = tag if tag else latest_tag()
    target = Path(dest).expanduser()
    print(f"cn-data tag={resolved} dest={target}")
    stage = Path(tempfile.mkdtemp(prefix="cn-data-"))
    try:
        archive = stage / "qlib_bin.tar.gz"
        manifest = stage / "qlib_bin.manifest.json"
        script = stage / "validate_archive.py"
        base = release_base(resolved)
        for url, path in (
            (f"{base}/qlib_bin.tar.gz", archive),
            (f"{base}/qlib_bin.manifest.json", manifest),
            (VALIDATOR_URL, script),
        ):
            code = wget_file(url, path)
            if code != 0:
                return code
        code = validate(python or sys.executable, script, archive, manifest, resolved)
        if code != 0:
            return code
        extracted = stage / "cn_data"
        code = extract(archive, extracted)
        if code != 0:
            return code
        backup = swap_in(extracted, target, today)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    if backup is None:
        print(f"cn-data wrote {target}")
    else:
        print(f"cn-data wrote {target} backup={backup}")
    return 0
