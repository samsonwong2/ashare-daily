"""qlib initialization + instrument metadata + close/volume panel loader."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd

qlib = None  # type: ignore[assignment]
D = None  # type: ignore[assignment]
REG_CN = None  # type: ignore[assignment]
_QLIB_INIT_PROVIDER_URI: str | None = None

from .code_utils import (
    build_normalized_code_set,
    code_in_normalized_set,
)


POSSIBLE_TYPE_NAMES = ["基金类型", "类型", "fund_type", "type"]
POSSIBLE_INCEPTION_NAMES = ["成立日期", "inception_date", "上市日期", "成立时间"]
POSSIBLE_CODE_NAMES = [
    "代码",
    "code",
    "instrument",
    "symbol",
    "ticker",
    "fund_code",
    "基金代码",
    "证券代码",
    "股票代码",
    "代码ID",
]
POSSIBLE_NAME_NAMES = [
    "基金简称",
    "股票简称",
    "证券简称",
    "简称",
    "名称",
    "name",
    "short_name",
    "fund_name",
    "基金名称",
]


def _read_instrument_txt(path: str | os.PathLike) -> list[str]:
    source = Path(path).expanduser()
    if not source.exists():
        raise FileNotFoundError(f"Instrument universe txt not found: {source}")
    codes: list[str] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            code = stripped.split()[0].upper()
            if code:
                codes.append(code)
    return list(dict.fromkeys(codes))


def _build_code_alias_map(codes: Iterable[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for code in codes:
        qlib_code = str(code).strip().upper()
        if not qlib_code:
            continue
        alias_values = {qlib_code}
        if len(qlib_code) >= 8 and qlib_code[:2] in {"SH", "SZ", "BJ"}:
            digits = qlib_code[2:]
            alias_values.update({digits, f"{digits}.{qlib_code[:2]}", f"{qlib_code[:2]}.{digits}"})
        for alias in alias_values:
            aliases[str(alias).strip().upper()] = qlib_code
    return aliases


def _normalize_stock_code(raw_code: str, alias_map: dict[str, str] | None = None) -> str:
    alias_map = alias_map or {}
    value = str(raw_code).strip().upper()
    if not value:
        return ""
    if value in alias_map:
        return alias_map[value]
    if "." in value:
        left, right = value.split(".", 1)
        dotted_aliases = [value, f"{right}{left}", f"{right}.{left}"]
        for alias in dotted_aliases:
            if alias in alias_map:
                return alias_map[alias]
        if right in {"SH", "SZ", "BJ"} and left.isdigit():
            return f"{right}{left.zfill(6)}"
    if value.startswith(("SH", "SZ", "BJ")):
        return value
    digits = "".join(ch for ch in value if ch.isdigit())
    if len(digits) == 6:
        if digits in alias_map:
            return alias_map[digits]
        if digits.startswith("6"):
            return f"SH{digits}"
        if digits.startswith(("0", "3")):
            return f"SZ{digits}"
        if digits.startswith(("4", "8")):
            return f"BJ{digits}"
    return value


def _resolve_metadata_columns(df: pd.DataFrame) -> tuple[str, str, str | None, str | None]:
    code_col = next((c for c in POSSIBLE_CODE_NAMES if c in df.columns), None)
    name_col = next((c for c in POSSIBLE_NAME_NAMES if c in df.columns), None)
    if code_col is None or name_col is None:
        code_col = code_col or df.columns[0]
        name_col = name_col or (df.columns[1] if len(df.columns) > 1 else df.columns[0])
    type_col = next((c for c in POSSIBLE_TYPE_NAMES if c in df.columns), None)
    inception_col = next((c for c in POSSIBLE_INCEPTION_NAMES if c in df.columns), None)
    return code_col, name_col, type_col, inception_col


def _build_augmented_code_name_map(base_map: dict[str, str]) -> dict[str, str]:
    aug: dict[str, str] = {}
    for key, value in base_map.items():
        code = str(key).strip()
        if not code:
            continue
        name = str(value).strip() or code
        code_upper = code.upper()
        aug[code] = name
        aug[code_upper] = name
        if code_upper.startswith(("SH", "SZ", "BJ")):
            aug[code_upper[2:]] = name
        try:
            aug[str(int(code_upper[2:] if code_upper.startswith(("SH", "SZ", "BJ")) else code_upper))] = name
        except Exception:
            pass
    return aug


def _load_metadata_name_map(metadata_csv: str | os.PathLike | None) -> dict[str, str]:
    if not metadata_csv:
        return {}
    source = Path(str(metadata_csv)).expanduser()
    if not source.exists():
        return {}
    df = pd.read_csv(source, dtype=str).fillna("")
    code_col, name_col, _type_col, _inception_col = _resolve_metadata_columns(df)
    base_map = dict(
        zip(
            df[code_col].astype(str).str.strip(),
            df[name_col].astype(str).str.strip(),
        )
    )
    return _build_augmented_code_name_map(base_map)


def ensure_stock_list_csv(
    csv_path: str | os.PathLike | None,
    *,
    universe_file: str | os.PathLike | None = None,
    refresh: bool = False,
) -> str | None:
    """Create or reuse a stock metadata CSV sourced from AkShare.

    ``ak.stock_info_a_code_name()`` returns six-digit A-share codes and names.
    We normalize those codes into the qlib instrument namespace so downstream
    reports can label ``SH/SZ/BJ`` qlib codes without changing the candidate
    universe itself.
    """
    if not csv_path:
        return None
    target = Path(str(csv_path)).expanduser()
    if target.exists() and not refresh:
        return str(target)

    alias_map: dict[str, str] = {}
    universe_codes: list[str] = []
    if universe_file:
        universe_path = Path(str(universe_file)).expanduser()
        if universe_path.suffix.lower() == ".txt" and universe_path.exists():
            universe_codes = _read_instrument_txt(universe_path)
            alias_map = _build_code_alias_map(universe_codes)

    try:
        import akshare as ak
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("akshare is required to generate stock_list.csv") from exc

    frame = ak.stock_info_a_code_name()
    if frame is None or frame.empty:
        raise RuntimeError("ak.stock_info_a_code_name() returned no rows")
    frame = frame.astype(str).fillna("")
    code_col, name_col, _type_col, _inception_col = _resolve_metadata_columns(frame)
    out = pd.DataFrame(
        {
            "raw_code": frame[code_col].astype(str).str.strip(),
            "name": frame[name_col].astype(str).str.strip(),
        }
    )
    out["code"] = out["raw_code"].map(lambda c: _normalize_stock_code(c, alias_map))
    out = out[(out["code"] != "") & (out["name"] != "")].copy()
    out = out.drop_duplicates(subset="code", keep="first").sort_values("code")
    out["股票代码"] = out["code"]
    out["股票简称"] = out["name"]
    target.parent.mkdir(parents=True, exist_ok=True)
    out[["code", "name", "股票代码", "股票简称", "raw_code"]].to_csv(target, index=False)
    print(f"Wrote stock metadata from ak.stock_info_a_code_name() to {target} ({len(out)} rows)")
    return str(target)


def load_fund_list_universe(
    source_path: str | os.PathLike,
    *,
    name_csv: str | os.PathLike | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load a metadata CSV or qlib instrument txt as a reporting universe."""
    source = Path(str(source_path)).expanduser()
    if source.suffix.lower() == ".txt":
        codes = _read_instrument_txt(source)
        name_map = _load_metadata_name_map(name_csv)
        universe = pd.DataFrame(
            {
                "code": codes,
                "name": [name_map.get(code) or name_map.get(code[2:]) or code for code in codes],
                "fund_type": "",
                "inception_date": "",
            }
        )
        code_name_map = _build_augmented_code_name_map(dict(zip(universe["code"], universe["name"])))
        return universe, code_name_map

    if not source.exists():
        raise FileNotFoundError(f"Instrument metadata CSV not found: {source}")
    df = pd.read_csv(source, dtype=str).fillna("")
    code_col, name_col, type_col, inception_col = _resolve_metadata_columns(df)
    universe = pd.DataFrame(
        {
            "code": df[code_col].astype(str).str.strip(),
            "name": df[name_col].astype(str).str.strip(),
        }
    )
    universe["fund_type"] = df[type_col].astype(str).str.strip() if type_col is not None else ""
    universe["inception_date"] = df[inception_col].astype(str).str.strip() if inception_col is not None else ""
    universe = universe[universe["code"] != ""].drop_duplicates(subset="code", keep="first").reset_index(drop=True)
    code_name_map = _build_augmented_code_name_map(dict(zip(universe["code"], universe["name"])))
    return universe, code_name_map


def load_data(
    instruments: Iterable[str],
    provider_uri: str,
    test_period: tuple[str, str],
    excluded_codes: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull ``$close`` and ``$volume`` from qlib for ``instruments``.

    Mirrors ``2_filter/0.2_cluster_select_from_dendrogram.py::load_data``.
    """
    global qlib, D, REG_CN, _QLIB_INIT_PROVIDER_URI
    if qlib is None or D is None or REG_CN is None:
        try:
            import qlib as qlib_module
            from qlib.constant import REG_CN as reg_cn
            from qlib.data import D as qlib_data
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("qlib is not available; cannot load data.") from exc
        qlib = qlib_module
        D = qlib_data
        REG_CN = reg_cn
    print("Initializing qlib provider:", provider_uri)
    if _QLIB_INIT_PROVIDER_URI != provider_uri:
        qlib.init(provider_uri=provider_uri, region=REG_CN)
        _QLIB_INIT_PROVIDER_URI = provider_uri
    stockpool = [str(code).strip() for code in instruments if str(code).strip()]
    if excluded_codes:
        normalized_excluded = build_normalized_code_set(excluded_codes)
        stockpool = [code for code in stockpool if not code_in_normalized_set(code, normalized_excluded)]
    raw = D.features(
        stockpool,
        fields=["$close", "$volume"],
        start_time=test_period[0],
        end_time=test_period[1],
    )
    close = raw["$close"].unstack(level="instrument")
    vol = raw["$volume"].unstack(level="instrument")
    close.index.name = "Date"
    vol.index.name = "Date"
    close = close.sort_index().ffill()
    vol = vol.sort_index()
    if excluded_codes:
        normalized_excluded = build_normalized_code_set(excluded_codes)
        keep_cols = [c for c in close.columns if not code_in_normalized_set(c, normalized_excluded)]
        dropped = set(close.columns) - set(keep_cols)
        if dropped:
            print(f"Removed {len(dropped)} codes due to exclude-types rule.")
            close = close[keep_cols]
            vol = vol[keep_cols]
    print("Loaded close shape", close.shape, "volume shape", vol.shape)
    return close, vol


def build_code_name_map(
    csv_path: str | os.PathLike | None,
    exclude_types: Iterable[str] | None = None,
    inception_cutoff: str | None = None,
    manual_excludes: Iterable[str] | None = None,
    cluster_excludes: Iterable[str] | None = None,
    cluster_reference_map_csv: str | None = None,
    name_csv: str | os.PathLike | None = None,
) -> tuple[dict[str, str], set[str], list[str]]:
    """Read instrument metadata/universe and apply optional exclusions.

    Returns ``(augmented_code_name_map, excluded_codes, selected_codes)``.
    """
    from .code_utils import load_cluster_seed_ids  # local import to avoid cycle

    source = Path(str(csv_path)).expanduser() if csv_path else None
    if source is None:
        raise ValueError("No instrument universe path was provided")
    if source.suffix.lower() == ".txt":
        selected_codes = _read_instrument_txt(source)
        name_map = _load_metadata_name_map(name_csv)
        code_series = pd.Series(selected_codes, dtype=str)
        name_series = code_series.map(lambda code: name_map.get(code) or name_map.get(code[2:]) or code)
        type_col = None
        inception_col = None
    else:
        if not source.exists():
            raise FileNotFoundError(f"Instrument metadata CSV not found: {source}")
        df = pd.read_csv(source, dtype=str)
        df = df.fillna("")
        code_col, name_col, type_col, inception_col = _resolve_metadata_columns(df)
        code_series = df[code_col].astype(str).str.strip()
        name_series = df[name_col].astype(str).str.strip()

    excluded_codes: set[str] = set()
    if exclude_types:
        if type_col:
            type_series = df[type_col].astype(str).str.strip()
            mask = type_series.isin({t.strip() for t in exclude_types})
            if mask.any():
                excluded_codes = set(code_series[mask])
                code_series = code_series[~mask]
                name_series = name_series[~mask]

    recent_cutoff_codes: set[str] = set()
    if inception_col and inception_cutoff is not None:
        inception_series = pd.to_datetime(df[inception_col], errors="coerce")
        cutoff = pd.to_datetime(inception_cutoff, errors="coerce")
        recent_mask = inception_series > cutoff
        if cutoff is not pd.NaT and recent_mask.any():
            recent_cutoff_codes = set(code_series[recent_mask])
            code_series = code_series[~recent_mask]
            name_series = name_series[~recent_mask]

    manual_excludes = list(manual_excludes or [])
    if manual_excludes:
        normalized_manual = build_normalized_code_set(manual_excludes)
        manual_mask = code_series.apply(lambda c: code_in_normalized_set(c, normalized_manual))
        if manual_mask.any():
            excluded_codes.update(code_series[manual_mask])
            code_series = code_series[~manual_mask]
            name_series = name_series[~manual_mask]

    cluster_excludes = list(cluster_excludes or [])
    if cluster_excludes:
        cluster_seed_ids = load_cluster_seed_ids(cluster_reference_map_csv, cluster_excludes)
        if cluster_seed_ids:
            cluster_series = pd.Series("", index=code_series.index, dtype=str)
            if cluster_reference_map_csv and os.path.exists(cluster_reference_map_csv):
                ref_df = pd.read_csv(cluster_reference_map_csv, dtype=str).fillna("")
                if "code" in ref_df.columns and "cluster" in ref_df.columns:
                    ref_df["code"] = ref_df["code"].astype(str).str.strip()
                    ref_df["cluster"] = ref_df["cluster"].astype(str).str.strip()
                    ref_map = dict(zip(ref_df["code"], ref_df["cluster"]))
                    cluster_series = code_series.map(lambda c: ref_map.get(str(c).strip(), ""))
            cluster_mask = cluster_series.astype(str).isin(cluster_seed_ids)
            if cluster_mask.any():
                excluded_codes.update(code_series[cluster_mask])
                code_series = code_series[~cluster_mask]
                name_series = name_series[~cluster_mask]

    aug = _build_augmented_code_name_map(dict(zip(code_series, name_series)))
    selected_codes = code_series.astype(str).str.strip().tolist()
    return aug, excluded_codes.union(recent_cutoff_codes), list(dict.fromkeys(selected_codes))
