"""A rung that never fires and a rung that always passes look the same.

REGISTER #201. On 2026-08-31 three rungs of the opponent ladder were dead at
once, for three unrelated reasons:

    the clock           did not exist in the code
    the single feature  index alignment made every correlation NaN
    stability           filtered on `__POOLED__`, a name absent from the data

None of them logged anything. None of them bound. Every champion they should
have judged was promoted unopposed, and each was found days apart by a person
who happened to look in the right place.

The rule: a check that produced no answer on any context in a whole run is a
DEFECT until shown otherwise. Not suspicious -- a defect, because a check that
cannot bind is not a check.

This is not the kind of scanner refused on 2026-09-05. Those read syntax, and
the defects they hunted live in idiom. This reads what a RUN actually did, from
evidence the run already writes: refusals carry each rung's score and champions
carry the same fields, so champions plus refusals is every context.

THE STATE THE FIRST VERSION LACKED, and its first run paid for it.
`baseline_margin_sigma` came back "not measured" on 18 of 18 refusals and was
reported as a dead check. It is not absent because it could not be computed --
it is absent because the refusal schema never carried it. NOT_RECORDED is now
its own state, because sending a reader to hunt a dead check that is really an
unrecorded one is exactly the confusion this module exists to prevent.
"""
from __future__ import annotations

import inspect

import pytest

from src.pipeline.stages.modeling.check_coverage import (
    BOUND, NOT_MEASURED, NOT_RECORDED, PASSED, dead_checks, format_report,
    summarise_checks,
)


def _refusal(**overrides):
    row = {
        "baseline_constant_score": 0.50,
        "baseline_persistence_score": 0.52,
        "baseline_clock_score": 0.56,
        "single_feature_score": 0.54,
        "baseline_margin_sigma": 0.006,
        "holdout_events": 400,
        "baseline_kind": "clock",
        "reasons": "does not beat the clock baseline",
    }
    row.update(overrides)
    return row


def _champion(**overrides):
    metrics = {
        "baseline_constant_score": 0.50,
        "baseline_persistence_score": 0.51,
        "baseline_clock_score": 0.53,
        "single_feature_score": 0.52,
        "baseline_margin_sigma": 0.004,
        "holdout_events": 500,
    }
    metrics.update(overrides)
    return {"winner_holdout_metrics": metrics}


def test_a_rung_that_answers_nowhere_is_reported_as_a_defect():
    """The 31.08 shape: the clock rung present in every record, None in all."""
    refusals = [_refusal(baseline_clock_score=None) for _ in range(6)]
    champions = {f"c{i}": _champion(baseline_clock_score=None) for i in range(4)}

    coverage = summarise_checks(champions, refusals)
    assert "clock opponent" in dead_checks(coverage)
    assert "DEFECT" in format_report(coverage)


def test_a_rung_that_measures_and_never_refuses_is_not_a_defect():
    """Working and finding nothing is a result, not a fault. A check that
    binds on nothing is how a clean batch looks."""
    coverage = summarise_checks(
        {f"c{i}": _champion() for i in range(5)}, [])
    assert dead_checks(coverage) == []
    assert "finding nothing" in format_report(coverage)


def test_an_unrecorded_check_is_not_called_dead():
    """The defect this file's own first run produced."""
    refusals = [_refusal() for _ in range(3)]
    for row in refusals:
        del row["baseline_margin_sigma"]

    coverage = summarise_checks({}, refusals)
    assert coverage["margin standard error"].states[NOT_RECORDED] == 3
    assert "margin standard error" not in dead_checks(coverage)

    report = format_report(coverage)
    assert "not written into these records" in report
    assert "DEFECT" not in report, (
        "a gap in the artefact is being reported as a dead check, which sends "
        "a reader to fix code that is working"
    )


def test_not_measured_and_passed_are_never_the_same_column():
    """#202 in one line: a check that could not run must not read as one that
    was cleared."""
    refusals = [_refusal(baseline_persistence_score=None),
                _refusal(baseline_persistence_score=0.51)]
    coverage = summarise_checks({}, refusals)
    states = coverage["persistence opponent"].states
    assert states[NOT_MEASURED] == 1
    assert states[BOUND] + states[PASSED] == 1


def test_the_binding_rung_is_the_one_the_refusal_names():
    refusals = [_refusal(baseline_kind="persistence")]
    coverage = summarise_checks({}, refusals)
    assert coverage["persistence opponent"].states[BOUND] == 1
    assert coverage["clock opponent"].states[PASSED] == 1


def test_a_run_with_no_contexts_says_so_rather_than_passing():
    """Zero champions and zero refusals is not a clean run -- it is a run that
    checked nothing, and reporting it as clean is the #202 shape one level up."""
    report = format_report(summarise_checks({}, []))
    assert "NO CONTEXTS SEEN" in report
    assert "DEFECT" not in report


def test_the_stage_forms_the_verdict_at_its_boundary():
    """Per-context logging cannot show a rung that never fires: it produces no
    line to read. The count has to happen where every context has been seen."""
    from src.pipeline.stages.modeling.orchestrator import ModelingStage

    source = inspect.getsource(ModelingStage)
    assert "summarise_checks(champions, self._gate_refusals)" in source
    assert "'dead_checks': dead" in source, (
        "the verdict is logged and not returned, so nothing downstream can "
        "act on it"
    )
    marker = source.index("summarise_checks(champions")
    window = source[marker:marker + 700]
    assert "logger.error" in window, (
        "a dead check is reported at the same level as a healthy one"
    )


@pytest.mark.parametrize("check", [
    "constant opponent", "persistence opponent", "clock opponent",
    "best single feature",
])
def test_every_ladder_rung_is_covered(check):
    """The rungs are named individually: a coverage report that silently
    stopped tracking one would be the defect it exists to catch."""
    coverage = summarise_checks({"c": _champion()}, [])
    assert check in coverage
    assert coverage[check].contexts == 1


def test_both_refusal_shapes_carry_the_same_checks():
    """The coverage report reads refusal rows, and the stage writes TWO shapes
    -- a judged context and one where training never ran. A field present in
    one and absent in the other makes the report say "not recorded" on half
    the rows and "not measured" on the rest, which is a third wrong answer.

    Found by adding the margin to one of them.
    """
    import ast
    from pathlib import Path

    source = Path("src/pipeline/stages/modeling/orchestrator.py").read_text(
        encoding="utf-8")
    shapes = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and node.args and isinstance(node.args[0], ast.Dict)):
            keys = {k.value for k in node.args[0].keys
                    if isinstance(k, ast.Constant)}
            if "baseline_kind" in keys:
                shapes.append(keys)

    assert len(shapes) == 2, f"expected two refusal shapes, found {len(shapes)}"
    assert shapes[0] == shapes[1], (
        f"the two refusal rows disagree on {shapes[0] ^ shapes[1]}"
    )


def test_the_margin_and_its_error_are_recorded_not_only_narrated():
    """The reasons string quotes "margin -0.0359, sigma 0.0048", and a number
    inside prose cannot be re-aggregated or compared across runs. The coverage
    report found this on its first run: the margin's standard error was the one
    check absent from every refusal row, and it is the number #192 and #200 are
    both about."""
    from pathlib import Path

    source = Path("src/pipeline/stages/modeling/orchestrator.py").read_text(
        encoding="utf-8")
    for field in ("baseline_margin", "baseline_margin_sigma",
                  "baseline_margin_rows_per_bar"):
        assert source.count(f'"{field}":') >= 2, (
            f"{field} is written into fewer than both refusal shapes"
        )
