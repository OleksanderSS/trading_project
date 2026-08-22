"""One line for console encoding cost roughly a thousand tests.

`dean_os/stress/test_phase8.py` did this at module level:

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

Under pytest, `sys.stdout` is a wrapper around a temporary file that pytest
owns. Taking its `.buffer`, wrapping it again and rebinding `sys.stdout` drops
the last reference to the original wrapper, whose finaliser closes the
underlying file -- pytest's capture file, for the rest of the session.

`tests/dean_os/test_builders_refuse_empty_input` imports that module by name.
After it did, every remaining test in the directory failed at setup with "I/O
operation on closed file". `pytest tests/dean_os/` reported 293 passed and
2,004 setup/teardown errors: about 1,002 tests never ran, and the directory
looked like it had a green tail while two thirds of it was not being tested.

The rebinding belongs behind `if __name__ == "__main__"`. This scans for the
shape rather than the one file, because the next copy would be just as quiet.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest

ROOTS = (Path("src"), Path("dean_os"))
SKIP_DIRS = {"archive", "draft", "__pycache__", ".venv", "venv", "node_modules"}


def _python_files() -> list[Path]:
    found: list[Path] = []
    for root in ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            if SKIP_DIRS & set(path.parts):
                continue
            found.append(path)
    return found


def _module_level_stream_rebinds(tree: ast.Module) -> list[str]:
    """`sys.stdout = ...` / `sys.stderr = ...` executed on import."""
    offenders = []
    for node in tree.body:  # module level only, not inside functions or main-guards
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if (
                isinstance(target, ast.Attribute)
                and target.attr in {"stdout", "stderr", "stdin"}
                and isinstance(target.value, ast.Name)
                and target.value.id == "sys"
            ):
                offenders.append(f"line {node.lineno}: sys.{target.attr}")
    return offenders


@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p))
def test_no_module_rebinds_a_standard_stream_on_import(path: Path):
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except SyntaxError:
        pytest.skip(f"{path} does not parse")

    offenders = _module_level_stream_rebinds(tree)
    assert not offenders, (
        f"{path} rebinds a standard stream at import time ({offenders}). Under "
        f"pytest this closes the capture file and every later test in the run "
        f"fails at setup. Put it behind `if __name__ == \"__main__\"`."
    )


def test_importing_the_module_that_did_it_leaves_stdout_alone():
    """The specific regression, named."""
    import importlib

    before = sys.stdout
    importlib.import_module("dean_os.stress.test_phase8")
    assert sys.stdout is before


def test_the_helper_still_exists_for_when_the_script_is_run():
    """The behaviour was moved, not deleted -- the console still needs it."""
    from dean_os.stress import test_phase8

    assert callable(test_phase8._use_utf8_console)


def test_the_helper_prefers_reconfigure_over_replacing_the_object():
    """Replacing is what closed the file; reconfigure changes it in place."""
    from dean_os.stress import test_phase8

    class _Stream:
        def __init__(self):
            self.encoding_set = None

        def reconfigure(self, encoding=None):
            self.encoding_set = encoding

    stream = _Stream()
    original = sys.stdout
    sys.stdout = stream
    try:
        test_phase8._use_utf8_console()
        assert stream.encoding_set == "utf-8"
        assert sys.stdout is stream, "reconfigure must not replace the object"
    finally:
        sys.stdout = original
