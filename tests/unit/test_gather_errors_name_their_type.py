"""`Error fetching FRED series: ` — and then nothing.

Every one of these collectors runs its work through `asyncio.gather(...,
return_exceptions=True)` and walks the results looking for exceptions. Each
logged the exception by interpolation alone, `f"...: {res}"`, which prints
an empty string whenever `str(exc)` is empty — and that is the normal case
for the ones that matter here: `asyncio.TimeoutError`, `CancelledError`, a
bare `KeyError()`, anything raised without arguments.

Seen in the 2026-08-13 rebuild:

    fred_collector - ERROR - Error fetching FRED series:

with nothing after the colon, from the source that supplies 95,150 macro
records. The same signature previously hid 54 drift timeouts in Stage 7 and
two dead collectors in Stage 1, both found only by reading raw artifacts.

An exception with no message still has a type, and the type is what makes it
searchable. These tests pin the format rather than any single call site,
because the pattern is what keeps recurring.
"""
import ast
import pathlib

import pytest

COLLECTORS = [
    "fred_collector.py",
    "google_news_collector.py",
    "newsapi_collector.py",
    "yf_collector.py",
]

ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "data" / "collectors"


def _gather_error_logs(path: pathlib.Path) -> list[ast.Call]:
    """Log calls inside an `isinstance(x, Exception)` branch."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Call)
                and getattr(test.func, "id", "") == "isinstance"):
            continue
        if not any(getattr(a, "id", "") == "Exception" for a in test.args):
            continue
        for inner in ast.walk(node):
            if (isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and inner.func.attr in {"error", "warning", "exception"}):
                found.append(inner)
    return found


@pytest.mark.parametrize("filename", COLLECTORS)
def test_the_exception_type_is_logged_not_only_its_message(filename):
    path = ROOT / filename
    calls = _gather_error_logs(path)
    assert calls, f"no gather-exception handler found in {filename}"

    for call in calls:
        rendered = ast.unparse(call)
        assert "__name__" in rendered or ".exception(" in rendered, (
            f"{filename}: this handler prints only the message, so an "
            f"exception whose str() is empty logs as a bare colon:\n"
            f"    {rendered[:160]}"
        )


@pytest.mark.parametrize("filename", COLLECTORS)
def test_the_traceback_travels_with_it(filename):
    """`exc_info` turns "something failed" into a place in the code."""
    path = ROOT / filename
    for call in _gather_error_logs(path):
        rendered = ast.unparse(call)
        assert "exc_info" in rendered or ".exception(" in rendered, (
            f"{filename}: no traceback attached:\n    {rendered[:160]}"
        )


def test_an_exception_with_no_message_still_renders_its_type():
    """The property the format has to have, checked directly."""
    empty = TimeoutError()
    assert str(empty) == ""

    bare = f"Error fetching FRED series: {empty}"
    typed = f"Error fetching FRED series: {type(empty).__name__}: {empty}"

    assert bare.endswith(": "), "this is what the log actually showed"
    assert "TimeoutError" in typed
