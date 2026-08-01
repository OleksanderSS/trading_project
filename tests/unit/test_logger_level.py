"""The project ran at DEBUG because nobody chose a level.

setup_logging(level: str='DEBUG') and all seven call sites invoke it with no
arguments, so DEBUG was the level the whole system ran at. Measured on the
live log before the change: 36,146 lines, of which

    INFO      16,282   45.0%
    DEBUG     15,512   42.9%
    WARNING    1,157    3.2%
    ERROR        380    1.1%
    CRITICAL      83    0.2%

The file rotates at 10 MB with 5 backups, so nearly half the retained history
was DEBUG noise and the 463 error-and-worse lines sat buried in it -- in the
logs that are the only record of what a training run actually did.

The level now comes from `system.log_level` (INFO), and an unrecognised value
says so instead of quietly becoming INFO.
"""
from __future__ import annotations

import logging

import pytest

from src.core.logging.logger import ProjectLogger


@pytest.fixture()
def unconfigured(monkeypatch):
    """setup_logging returns early once configured; each test needs a fresh
    run, and the root logger must be restored afterwards."""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    monkeypatch.setattr(ProjectLogger, "_is_configured", False)
    yield
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)
    ProjectLogger._is_configured = True


def test_the_default_level_comes_from_config_not_from_debug(unconfigured):
    ProjectLogger.setup_logging()
    assert logging.getLogger().level == logging.INFO


def test_debug_is_still_available_when_asked_for(unconfigured):
    """Tracing a specific problem must stay one argument away."""
    ProjectLogger.setup_logging(level="DEBUG")
    assert logging.getLogger().level == logging.DEBUG


@pytest.mark.parametrize("level,expected", [
    ("WARNING", logging.WARNING),
    ("error", logging.ERROR),
    ("CRITICAL", logging.CRITICAL),
])
def test_an_explicit_level_is_honoured(unconfigured, level, expected):
    ProjectLogger.setup_logging(level=level)
    assert logging.getLogger().level == expected


def test_an_unknown_level_is_reported_and_falls_back(unconfigured, capsys):
    # capsys, not caplog: setup_logging removes every root handler before
    # installing its own, which strips pytest's capture handler too. That is
    # also why the warning is emitted AFTER the handlers exist -- raised any
    # earlier it went to stderr through logging's last resort and never
    # reached system.log, which is exactly where a misconfigured level has to
    # be visible.
    ProjectLogger.setup_logging(level="WARNIGN")

    assert logging.getLogger().level == logging.INFO
    assert "Unknown log level" in capsys.readouterr().out


def test_the_config_setting_is_readable():
    assert ProjectLogger._get_config_setting("log_level", "FALLBACK") == "INFO"


def test_a_missing_setting_returns_the_default():
    assert ProjectLogger._get_config_setting(
        "no_such_setting_here", "FALLBACK"
    ) == "FALLBACK"


def test_the_path_helper_still_returns_a_path():
    """_get_config_path now delegates to _get_config_setting; it must keep
    returning a Path, since callers build log file names from it."""
    from pathlib import Path

    assert isinstance(ProjectLogger._get_config_path("logs_path", "logs"), Path)
