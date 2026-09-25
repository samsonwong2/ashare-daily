#!/usr/bin/env python3
"""合并 qlib instruments：csi800.txt + csi1000.txt -> csi1800.txt

只取两文件中「最新结束日」的成分股（各约 800/1000），并集后约 1800 条。
输出格式：CODE\\tSTART\\tEND（保留该最新分段的起止日）。
"""

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from runtime_paths import QLIB_INSTRUMENTS_DIR

# ===== 按需修改 =====
INSTRUMENTS_DIR = QLIB_INSTRUMENTS_DIR
SRC_FILES = ("csi800.txt", "csi1000.txt")
OUT_NAME = "csi1800.txt"
# ==================


def load_rows(path: Path) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 3:
            print(f"[WARN] skip bad line in {path.name}: {line}")
            continue
        rows.append((parts[0], parts[1], parts[2]))
    return rows


def latest_end(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        raise ValueError("empty instrument rows")
    return max(end for _, _, end in rows)


def main() -> None:
    inst_dir = INSTRUMENTS_DIR.expanduser().resolve()

    # 先扫一遍，取两文件共同的最新结束日（通常一致）
    all_by_file: dict[str, list[tuple[str, str, str]]] = {}
    for name in SRC_FILES:
        path = inst_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        all_by_file[name] = load_rows(path)

    as_of_end = max(latest_end(rows) for rows in all_by_file.values())
    print(f"[INFO] latest end date: {as_of_end}")

    # 只保留 end == 最新结束日 的行；按 code 去重（csi800 优先，再 csi1000）
    by_code: dict[str, tuple[str, str, str]] = {}
    per_file_active: dict[str, int] = {}
    for name in SRC_FILES:
        active = [row for row in all_by_file[name] if row[2] == as_of_end]
        per_file_active[name] = len({c for c, _, _ in active})
        for code, start, end in active:
            if code not in by_code:
                by_code[code] = (code, start, end)

    merged = sorted(by_code.values(), key=lambda x: x[0])
    out_path = inst_dir / OUT_NAME
    out_path.write_text(
        "\n".join(f"{c}\t{s}\t{e}" for c, s, e in merged) + ("\n" if merged else ""),
        encoding="utf-8",
    )

    overlap = sum(per_file_active.values()) - len(merged)
    print(f"[OK] wrote {out_path}")
    for name, n in per_file_active.items():
        print(f"  {name} @ {as_of_end}: {n} codes")
    print(f"  overlap: {overlap}")
    print(f"  union: {len(merged)} codes")
    if len(merged) != 1800:
        print(f"[WARN] expected 1800 codes, got {len(merged)}")


if __name__ == "__main__":
    main()
