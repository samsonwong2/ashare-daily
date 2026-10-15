"""Replace the local Qlib ``cn_data`` tree from one investment_data release."""
from __future__ import annotations

import json
import os
import shutil
import socket
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
PROXY_HOST = "127.0.0.1"
PROXY_PORT = 31181
GITHUB_PROXY = f"http://{PROXY_HOST}:{PROXY_PORT}"
# dev-sidecar MITMs GitHub. This bundle is what makes wget accept that local CA.
DEV_SIDECAR_CA = Path.home() / ".local" / "share" / "certs" / "dev-sidecar-bundle.crt"


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


def local_github_proxy(timeout: float = 2.0) -> str | None:
    """Return the local HTTP proxy when 127.0.0.1:31181 accepts a connection."""
    try:
        with socket.create_connection((PROXY_HOST, PROXY_PORT), timeout=timeout):
            return GITHUB_PROXY
    except OSError:
        return None


def proxy_ca_file() -> Path | None:
    if DEV_SIDECAR_CA.is_file():
        return DEV_SIDECAR_CA
    return None


def _release_tag(payload: dict) -> str:
    tag = str(payload.get("tag_name", "")).strip()
    try:
        date.fromisoformat(tag)
    except ValueError:
        print(f"cn-data: latest release tag is not YYYY-MM-DD: {tag}", file=sys.stderr)
        raise SystemExit(1)
    return tag


def fetch_latest_tag(
    urlopen=urllib.request.urlopen,
    *,
    proxy: str | None = None,
    ca_file: Path | None = None,
    wget_file=None,
) -> str:
    if proxy:
        return _fetch_latest_tag_via_wget(proxy, ca_file, wget_file or wget)
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
    return _release_tag(payload)


def _fetch_latest_tag_via_wget(proxy: str, ca_file: Path | None, wget_file) -> str:
    handle, name = tempfile.mkstemp(prefix="cn-data-release-", suffix=".json")
    os.close(handle)
    path = Path(name)
    try:
        code = wget_file(LATEST_URL, path, proxy=proxy, ca_file=ca_file)
        if code != 0:
            print("cn-data: cannot read latest release", file=sys.stderr)
            raise SystemExit(code or 1)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            print(f"cn-data: cannot read latest release: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc
    finally:
        path.unlink(missing_ok=True)
    return _release_tag(payload)


def wget(url: str, dest: Path, *, proxy: str | None = None, ca_file: Path | None = None) -> int:
    command = ["wget", "-O", str(dest)]
    if proxy:
        command.extend(
            [
                "-e",
                "use_proxy=yes",
                "-e",
                f"http_proxy={proxy}",
                "-e",
                f"https_proxy={proxy}",
            ]
        )
    if ca_file is not None:
        command.append(f"--ca-certificate={ca_file}")
    command.append(url)
    try:
        completed = subprocess.run(command, check=False)
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
    latest_tag=None,
    wget_file=None,
    validate=run_validate,
    extract=extract_tar,
    today: date | None = None,
    python: str | None = None,
) -> int:
    proxy = None
    ca_file = None
    if wget_file is None or (latest_tag is None and tag is None):
        proxy = local_github_proxy()
        ca_file = proxy_ca_file() if proxy else None
        if proxy:
            print(f"cn-data: using proxy {proxy}", flush=True)
            if ca_file is None:
                print(f"cn-data: missing dev-sidecar CA {DEV_SIDECAR_CA}", file=sys.stderr, flush=True)
        else:
            print("cn-data: proxy 127.0.0.1:31181 unavailable, downloading directly", flush=True)

    def download(url: str, path: Path) -> int:
        return wget(url, path, proxy=proxy, ca_file=ca_file)

    def resolve_latest() -> str:
        return fetch_latest_tag(proxy=proxy, ca_file=ca_file)

    downloader = wget_file or download
    tagger = latest_tag or resolve_latest
    resolved = tag if tag else tagger()
    target = Path(dest).expanduser()
    print(f"cn-data tag={resolved} dest={target}", flush=True)
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
            code = downloader(url, path)
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
