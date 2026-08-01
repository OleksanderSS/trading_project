"""Silent failure paths must not multiply, and the tool that mass-produced
them must stay disarmed.

This is a ratchet, not a clean-sweep. Three shapes are counted repo-wide; the
counts may fall, never rise. Each already produced a real defect:

  LOGGED_THEN_EMPTY  ModelingStage logged "Enriched data not found. Skipping
                     Modeling Stage." and returned {}. The orchestrator read
                     the return value, not the log, and reported success
                     (commit 4a8e804e).
  SWALLOWED          an except handler that neither logs nor re-raises.
  NARROW_TUPLE       except (ValueError, TypeError, AttributeError, KeyError,
                     ZeroDivisionError) -- missed CatBoostError,
                     sqlite3.IntegrityError and yaml.YAMLError, all of which
                     inherit straight from Exception.

Not every count is a bug: a collector returning [] after logging a failed
fetch is fine, because process_and_save_results distinguishes that case. The
ratchet exists so the shapes stop spreading while they are worked through.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from tests.contracts._silent_failure_scan import (
    NARROW_TUPLE,
    PROJECT_ROOT,
    by_kind,
    scan,
)

# Measured 2026-08-01. Lower these when findings are fixed; never raise them.
CEILINGS = {
    "LOGGED_THEN_EMPTY": 168,
    "SWALLOWED": 89,
    "NARROW_TUPLE": 653,
}


@pytest.fixture(scope="module")
def findings():
    return by_kind(scan())


@pytest.mark.parametrize("kind", sorted(CEILINGS))
def test_silent_failure_shapes_do_not_spread(findings, kind):
    found = findings.get(kind, [])
    ceiling = CEILINGS[kind]

    assert len(found) <= ceiling, (
        f"{kind} rose from {ceiling} to {len(found)}.\n"
        + "\n".join(f"  {f}" for f in found[:20])
    )


def test_the_ceilings_are_kept_honest(findings):
    """A count that has dropped well below its ceiling should be re-pinned,
    otherwise the ratchet stops ratcheting."""
    slack = {
        kind: CEILINGS[kind] - len(findings.get(kind, []))
        for kind in CEILINGS
    }
    stale = {kind: gap for kind, gap in slack.items() if gap > 25}

    assert not stale, (
        f"lower these ceilings in CEILINGS -- they are now this far above "
        f"the real counts: {stale}"
    )


def test_the_script_that_mass_produced_the_narrow_tuple_is_disarmed():
    """scripts/auto_refactor_exceptions.py walked all of src/ with

        re.sub(r'except Exception as e:', <the five-tuple>, content)

    and wrote every file back. It is the origin of ~653 handlers. Re-running
    it would undo every broadening fix made during this audit -- safe_execute,
    graceful_degradation, FileManager.load_yaml -- in one pass."""
    script = PROJECT_ROOT / "scripts" / "auto_refactor_exceptions.py"
    if not script.exists():
        return  # deleting it outright is also a valid answer

    tree = ast.parse(script.read_text(encoding="utf-8"))
    guarded = any(
        isinstance(node, ast.Raise)
        and node.lineno < _first_write_line(tree)
        for node in ast.walk(tree)
    )
    assert guarded, (
        "auto_refactor_exceptions.py can still rewrite src/ in place; it must "
        "refuse to run before it reaches any file write"
    )


def _first_write_line(tree: ast.AST) -> int:
    """Line of the first open(..., 'w') in the module, or a large number."""
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "open"
            and any(
                isinstance(arg, ast.Constant) and "w" in str(arg.value)
                for arg in node.args[1:]
            )
        ):
            return node.lineno
    return 10**6


def test_the_narrow_tuple_is_defined_once_here_and_matches_the_real_thing():
    """Guards against the scan silently matching nothing if the tuple is ever
    reordered or renamed -- a scanner that finds zero must mean zero."""
    sample = Path(PROJECT_ROOT, "src", "core", "file_management", "file_manager.py")
    source = sample.read_text(encoding="utf-8")

    assert all(name in source for name in NARROW_TUPLE)
