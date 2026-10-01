"""Tests for the render server's cache directory budgets."""

from __future__ import annotations

import logging
import os

from remote_display.cache_budget import BudgetWarning, prune_to_budget

NOW = 1_000_000_000.0


def _file(root, name, size, used):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (used, used))
    return path


def test_prune_deletes_least_recently_used_until_it_fits(tmp_path):
    old = _file(tmp_path, "radar_basemap/9/1/1.png", 400, NOW - 9000)
    older = _file(tmp_path, "ncaa/army.png", 400, NOW - 10000)
    newer = _file(tmp_path, "mlb_pitchers/1.png", 400, NOW - 8000)

    deleted = prune_to_budget(tmp_path, 900, clock=lambda: NOW)

    assert deleted == [older]
    assert old.exists() and newer.exists() and not older.exists()


def test_prune_keeps_recently_used_files_and_under_budget_dirs(tmp_path, caplog):
    recent = _file(tmp_path, "a.png", 500, NOW - 60)
    stale = _file(tmp_path, "b.png", 500, NOW - 7200)

    assert prune_to_budget(tmp_path, 2000, clock=lambda: NOW) == []
    with caplog.at_level(logging.WARNING):
        assert prune_to_budget(tmp_path, 100, clock=lambda: NOW) == [stale]
    assert recent.exists()
    assert "over its" in caplog.text
    assert prune_to_budget(tmp_path / "missing", 0) == []


def test_budget_warning_logs_once_per_crossing(tmp_path, caplog):
    _file(tmp_path, "history.json", 2048, NOW)
    warning = BudgetWarning(tmp_path, 1024, what="Server caches")

    with caplog.at_level(logging.WARNING):
        assert warning.check() == 2048
        assert warning.check() == 2048
    assert caplog.text.count("over its") == 1

    warning.max_bytes = 4096
    assert warning.check() == 2048 and not warning.over
