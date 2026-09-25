"""Bad --as-of exits 2. A real date is not rejected. Missing config stays exit 1."""
from __future__ import annotations

from pathlib import Path

import pytest

from ashare_daily.cli import main
from ashare_daily.daily_contract import parse_as_of


def test_bad_as_of_exits_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLOTLY_ROOT", raising=False)
    for value in ("2026-13-40", "20260925"):
        with pytest.raises(SystemExit) as exc:
            main(["regime", "--as-of", value], root=tmp_path)
        assert exc.value.code == 2


def test_real_date_is_not_rejected() -> None:
    assert parse_as_of("2026-09-25") == "2026-09-25"


def test_listing_dir_fills_trigger_dates() -> None:
    from pathlib import Path

    from ashare_daily.daily_contract import as_of_from_listing_dir, next_weekday

    as_of = as_of_from_listing_dir(Path("/tmp/20260924_from_listing_stock"))
    assert as_of == "2026-09-24"
    assert next_weekday(as_of) == "2026-09-25"


def test_missing_config_stays_exit_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PLOTLY_ROOT", raising=False)
    with pytest.raises(SystemExit) as exc:
        main(["regime", "--as-of", "2026-09-25"], root=tmp_path)
    assert exc.value.code == 1
