"""Silent failure paths must not multiply, and the tool that mass-produced
them must stay disarmed.

This is a ratchet, not a clean-sweep. Four shapes are counted repo-wide; the
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
  QUIET_THEN_EMPTY   the same as LOGGED_THEN_EMPTY but reported below error
                     level, which is exactly why LOGGED_THEN_EMPTY never saw
                     it. The walk-forward stability rung caught its exception,
                     wrote the reason to `logger.debug` and returned None; the
                     gate reads `if stability and not stability.get("passed",
                     True)`, so None promoted the champion. The only rung that
                     asks whether an edge holds over TIME was off for every
                     pooled context (REGISTER #189, #202).

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

# Lower these when findings are fixed; never raise them.
CEILINGS = {
    # Re-pinned DOWN from 168 on 2026-09-01, not because anything was fixed
    # but because the scan had been counting the wrong thing: 72 of the 170
    # were `logger.error(...)` followed by `return False` inside functions
    # whose every return is a boolean. There the log says what went wrong and
    # the boolean says it went wrong; that is the contract, not a hidden
    # failure. Functions whose names ask a question (is_, has_, can_) are
    # still counted -- False from `is_ready()` after an exception answers
    # "not ready" when the truth is "could not determine".
    #
    # A ceiling that counts correct code can only be met by rewriting correct
    # code, so it never gets met, and a permanently red ratchet is not a check
    # -- REGISTER #181 stood red from 22 August to 1 September and nobody
    # acted on it, which is the same failure as a check that never fires.
    "LOGGED_THEN_EMPTY": 98,
    "NARROW_TUPLE": 653,
    # Measured 2026-09-01, when the shape was added. Three of these are in
    # `_champion_feature_columns` and its helpers, and they are the one
    # legitimate form: an artifact that cannot be read declares no columns,
    # and an empty declaration makes the caller read EVERY column. The empty
    # value widens rather than narrows, so no caller can mistake it for a
    # successful narrowing. That is the test to apply to the rest of them --
    # not "is the empty value logged" but "does the empty value make the
    # caller do less than it should".
    "QUIET_THEN_EMPTY": 41,
}

#: `SWALLOWED` was one number covering four different failures, so it meant
#: none of them. Split 2026-09-01; the three parts replace it and their sum
#: (88) sits under the old ceiling of 89 without it being raised.
SWALLOWED_CEILINGS = {
    "SWALLOWED_ERASED": 48,
    "SWALLOWED_EMPTY": 28,
    "SWALLOWED_FABRICATED": 12,
}
CEILINGS.pop("SWALLOWED", None)
CEILINGS.update(SWALLOWED_CEILINGS)

#: Modules where a return value decides whether a model is promoted, whether a
#: prediction is made, or what a metric says. Elsewhere an empty return after
#: a warning is often right -- a collector that fetched nothing really did
#: fetch nothing. Here it is the difference between "refused" and "passed",
#: and those must never share a value. The target is zero.
DECISION_PATH = (
    "src/pipeline/stages/modeling",
    "src/pipeline/stages/prediction",
    "src/pipeline/stages/evaluation",
    "src/training",
    "src/metrics",
    "src/models/adapters",
)

#: 8 when first counted on 2026-09-01; 3 after the walk-forward handler, the
#: target-registry check, the artifact-write failure and the feature-importance
#: read were each given a state distinct from "passed".
DECISION_PATH_CEILING = 3

#: Sites where the empty value IS the distinct state, because the caller was
#: changed to act on it. The scanner sees one function at a time and cannot
#: know that; an exemption is the honest way to say so, and it has to name the
#: caller, because "trust me" is what every one of the counted sites would
#: also say. Line numbers are deliberate: if the code moves, the justification
#: is re-examined instead of inherited.
#:
#: Raising a ceiling would have hidden the same two entries with no
#: justification attached, and a ceiling raised once is raised again.
EXEMPT_SITES = {
    # _file_mtime_iso and _age_hours returned the CURRENT time and 0.0 when a
    # timestamp could not be read, so an artifact of unknown age passed every
    # freshness threshold. They now return None, and the caller at
    # pipeline_bridge.py:238 records `evaluation_timestamp_unreadable` as its
    # own failure -- the empty value is acted on, not absorbed.
    "src/agents/pipeline_bridge.py:370",
    "src/agents/pipeline_bridge.py:389",
    # ContextLedger._load: an unreadable ledger leaves the ledger EMPTY, and
    # an empty ledger makes the run retrain every context instead of replaying
    # any. The empty value makes the caller do more work, never less, so it
    # cannot be mistaken for a pass -- the opposite of the sites this counts.
    "src/pipeline/stages/modeling/context_ledger.py:82",
    # ModelingStage returning {} after "Enriched data not found. Skipping
    # Modeling Stage." -- the exact line this whole scanner was written for,
    # quoted in the docstring at the top of this file.
    #
    # It is exempt because the CALLER was changed and the change was verified
    # end to end on 2026-09-04, not assumed:
    #   pipeline_orchestrator.py:533  _validate_stage_output sees a falsy
    #       output from a stage in _STAGES_REQUIRING_OUTPUT and raises
    #       DataProcessingError rather than returning None.
    #   pipeline_orchestrator.py:462  _execute_stage catches it, logs, and
    #       RE-RAISES as RuntimeError, so the run dies instead of reporting
    #       {'status': 'success'} and continuing on the previous stage's data.
    # That was the actual failure once: "a run that trained nothing still
    # ended with 'Pipeline execution completed successfully'".
    #
    # Kept as an exemption rather than a fixed site because the empty dict IS
    # the distinct state here -- the scanner sees one function and cannot see
    # the caller. Raising the ceiling instead would have hidden this with no
    # justification attached, and a ceiling raised once is raised again.
    "src/pipeline/stages/modeling/orchestrator.py:276",
}


@pytest.fixture(scope="module")
def findings():
    return by_kind(scan())


@pytest.mark.parametrize("kind", sorted(CEILINGS))
def test_silent_failure_shapes_do_not_spread(findings, kind):
    found = [
        f for f in findings.get(kind, [])
        if f"{f.module}:{f.line}" not in EXEMPT_SITES
    ]
    ceiling = CEILINGS[kind]

    assert len(found) <= ceiling, (
        f"{kind} rose from {ceiling} to {len(found)}.\n"
        + "\n".join(f"  {f}" for f in found[:20])
    )


def test_a_broken_check_never_reads_as_a_passed_check(findings):
    """Zero is the target; the ceiling is only where it stood when counted.

    Each of these is a place where a failure returns the same value a success
    returns. That is not a style problem: it is how three ladder rungs ran
    dead for weeks while the gate reported they had passed.
    """
    found = [
        f for f in findings.get("QUIET_THEN_EMPTY", [])
        if f.module.startswith(DECISION_PATH)
        and f"{f.module}:{f.line}" not in EXEMPT_SITES
    ]

    assert len(found) <= DECISION_PATH_CEILING, (
        f"quiet failures in the decision path rose from "
        f"{DECISION_PATH_CEILING} to {len(found)}. A check that could not run "
        f"must return a state distinct from 'passed':\n"
        + "\n".join(f"  {f}" for f in found)
    )


def test_the_scanner_catches_the_handler_that_produced_189():
    """A scanner that cannot fail its own case proves nothing.

    This is the handler as it stood in
    `src/pipeline/stages/modeling/orchestrator.py` while the walk-forward
    stability rung was silently off for every pooled context.
    """
    from tests.contracts._silent_failure_scan import _Scanner

    source = "\n".join([
        "def f():",
        "    try:",
        "        return evaluate()",
        "    except (ValueError, TypeError) as e:",
        "        logger.debug(f'not evaluable ({e})')",
        "        return None",
        "",
    ])

    scanner = _Scanner("sample.py")
    scanner.visit(ast.parse(source))
    assert "QUIET_THEN_EMPTY" in {f.kind for f in scanner.findings}, (
        "the scanner no longer detects the shape it was written for"
    )

    loud = source.replace("logger.debug", "logger.error")
    scanner = _Scanner("sample.py")
    scanner.visit(ast.parse(loud))
    assert "QUIET_THEN_EMPTY" not in {f.kind for f in scanner.findings}, (
        "a loud report is LOGGED_THEN_EMPTY's business, not this shape's"
    )

    reraised = source.replace("        return None", "        raise")
    scanner = _Scanner("sample.py")
    scanner.visit(ast.parse(reraised))
    assert "QUIET_THEN_EMPTY" not in {f.kind for f in scanner.findings}, (
        "a handler that re-raises hides nothing from its caller"
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
