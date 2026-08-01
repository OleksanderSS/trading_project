"""The generic safety helpers must actually be generic.

`except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError)`
appears 668 times across 238 files here. It reads as exhaustive and is not: it
omits OSError, RuntimeError, IndexError and every library exception. Two live
failures in this audit came from that gap -- CatBoostError (which inherits
straight from Exception) took down an entire pipeline stage, and
sqlite3.IntegrityError silently lost tracked events.

For helpers whose contract IS "catch broadly" -- safe_execute wraps
"unexpected errors", graceful_degradation promises a fallback -- the narrow
tuple contradicts the documented behaviour.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.core.error_handling.error_handler import (
    ErrorHandler,
    TradingSystemError,
    safe_execute,
)


class LibraryError(Exception):
    """Stands in for CatBoostError / duckdb.Error: straight off Exception."""


NOT_IN_THE_OLD_TUPLE = [
    OSError("disk"),
    RuntimeError("runtime"),
    IndexError("index"),
    sqlite3.IntegrityError("unique constraint"),
    LibraryError("third party"),
]

IN_THE_OLD_TUPLE = [
    ValueError("value"),
    TypeError("type"),
    KeyError("key"),
]


@pytest.mark.parametrize("error", NOT_IN_THE_OLD_TUPLE + IN_THE_OLD_TUPLE,
                         ids=lambda e: type(e).__name__)
def test_safe_execute_wraps_every_unexpected_error(error):
    def boom():
        raise error

    with pytest.raises(TradingSystemError):
        safe_execute(boom)


def test_safe_execute_lets_domain_errors_through_unwrapped():
    """A TradingSystemError is already the project's own type; re-wrapping it
    would bury the original context."""
    def boom():
        raise TradingSystemError("domain")

    with pytest.raises(TradingSystemError, match="domain"):
        safe_execute(boom)


def test_safe_execute_returns_the_value_when_nothing_fails():
    assert safe_execute(lambda x: x * 2, 21) == 42


@pytest.mark.parametrize("error", NOT_IN_THE_OLD_TUPLE + IN_THE_OLD_TUPLE,
                         ids=lambda e: type(e).__name__)
def test_graceful_degradation_degrades_for_every_error(error):
    handler = ErrorHandler()

    @handler.graceful_degradation(fallback_value="fallback", context="test")
    def boom():
        raise error

    assert boom() == "fallback"


def test_graceful_degradation_can_be_narrowed_deliberately():
    """Broad by default, narrow on request -- the same shape as `retry`."""
    handler = ErrorHandler()

    @handler.graceful_degradation(fallback_value="fallback", exceptions=(ValueError,))
    def boom():
        raise OSError("not covered")

    with pytest.raises(OSError):
        boom()


def test_graceful_degradation_passes_results_through():
    handler = ErrorHandler()

    @handler.graceful_degradation(fallback_value=None)
    def fine():
        return "ok"

    assert fine() == "ok"


def test_retry_is_broad_by_default():
    """Recorded because retry was already correct: it sits in the same class
    as the narrow decorator, which is why the tuple looks mechanical."""
    handler = ErrorHandler()
    attempts = {"n": 0}

    @handler.retry(max_retries=2, delay=0)
    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise OSError("transient")
        return "recovered"

    assert flaky() == "recovered"
    assert attempts["n"] == 3
