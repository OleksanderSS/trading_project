"""Which checks actually ran, and which never fired at all.

REGISTER #201. On 2026-08-31 three rungs of the opponent ladder were dead at
once, for three different reasons, and all three looked identical from
outside:

    the clock          did not exist in the code       -> no comparison at all
    the single feature index alignment returned nothing -> "no usable feature"
    stability          filtered on a name not in the data -> exception, None

A dead rung and a passed rung are the same thing to a reader: nothing is
logged, the gate does not bind, the champion is promoted. Each was found by a
person looking in the right place, days apart.

THE RULE THIS FILE ENFORCES: a check that never fired across an entire run is
a defect until shown otherwise. Not "suspicious" -- a defect, because a check
that cannot bind is not a check, and the promotion it did not block was
unopposed.

FOUR STATES, and the fourth is the one that matters:

    BOUND          the check is why this context was refused
    PASSED         the check was computed and the model cleared it
    NOT_MEASURED   the check could not be computed here -- a missing column, an
                   exception, too few rows. NOT the same as PASSED, and
                   conflating them is #202
    NOT_APPLICABLE the check does not apply to this kind of target -- a
                   passive-holding opponent means nothing for a class label.
                   Kept apart from NOT_MEASURED so an honest absence never
                   counts as a failure to look

WHY IT NEEDS NO NEW EVIDENCE. Every number is already recorded: refusals carry
`baseline_constant_score`, `baseline_persistence_score`, `baseline_clock_score`
and `single_feature_score`, and champions carry the same fields inside
`winner_holdout_metrics`. Champions plus refusals is every context the stage
saw, so coverage is a pure function of two things the run already writes.
This adds a reading, not a measurement.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

#: The state a single check reached on a single context.
BOUND = "bound"
PASSED = "passed"
NOT_MEASURED = "not_measured"
NOT_APPLICABLE = "not_applicable"

#: The check is not written into this record at all -- a gap in the ARTEFACT,
#: not in the run.
#:
#: The first version of this file did not have this state, and its first run
#: paid for it: `baseline_margin_sigma` came back "not measured" on 18 of 18
#: refusals and was reported as a DEFECT. It is not absent because it could
#: not be computed; it is absent because the refusal schema never carried it.
#: Sending a reader to hunt a dead check that is really an unrecorded one is
#: the same confusion this module exists to prevent, committed by the module.
NOT_RECORDED = "not_recorded"

#: Ladder rungs and where their score lives. The key is the field on a refusal
#: row; champions carry the same name inside `winner_holdout_metrics`.
LADDER = {
    "constant opponent": "baseline_constant_score",
    "persistence opponent": "baseline_persistence_score",
    "clock opponent": "baseline_clock_score",
    "best single feature": "single_feature_score",
}

#: Checks that are not ladder rungs but bind the same way.
OTHER_CHECKS = {
    "margin standard error": "baseline_margin_sigma",
    "holdout event count": "holdout_events",
}


@dataclass
class CheckCoverage:
    """What one check did across a whole run."""

    name: str
    states: Counter = field(default_factory=Counter)

    @property
    def measured(self) -> int:
        return self.states[BOUND] + self.states[PASSED]

    @property
    def contexts(self) -> int:
        return sum(self.states.values())

    @property
    def unrecorded(self) -> int:
        return self.states[NOT_RECORDED]

    @property
    def never_fired(self) -> bool:
        """No context anywhere gave this check something to judge.

        Deliberately NOT "never bound": a check that measured a hundred
        contexts and refused none of them is doing its job and finding
        nothing, which is a different and legitimate outcome. The defect is a
        check that never got as far as an answer.
        """
        # A check present in every record and answered in none is dead. A
        # check absent from every record is an ARTEFACT gap and is reported on
        # its own line -- calling it dead would be a guess about code from the
        # shape of a file.
        judged = self.contexts - self.states[NOT_RECORDED] - self.states[NOT_APPLICABLE]
        return judged > 0 and self.measured == 0




def _state(row: Any, field_name: str, bound: bool) -> str:
    if field_name not in row:
        return NOT_RECORDED
    if row.get(field_name) is None:
        return NOT_MEASURED
    return BOUND if bound else PASSED


def summarise_checks(
    champions: dict[str, Any] | None,
    refusals: list[dict[str, Any]] | None,
) -> dict[str, CheckCoverage]:
    """Coverage per check, over every context the stage saw.

    `champions` is the stage's metadata mapping; `refusals` the rows written to
    `gate_refusals_*.parquet`. Together they are every context: one of the two
    holds each.
    """
    coverage = {name: CheckCoverage(name)
                for name in list(LADDER) + list(OTHER_CHECKS)}

    for row in refusals or []:
        binding = str(row.get("baseline_kind") or "")
        reasons = str(row.get("reasons") or "")
        for name, field_name in LADDER.items():
            value = row.get(field_name)
            # A rung BOUND when it is the one the refusal names.
            bound = (binding and binding in name) or (
                name == "best single feature" and "SINGLE FEATURE" in reasons)
            coverage[name].states[_state(row, field_name, bool(bound))] += 1
        for name, field_name in OTHER_CHECKS.items():
            coverage[name].states[_state(row, field_name, False)] += 1

    for payload in (champions or {}).values():
        metrics = (payload or {}).get("winner_holdout_metrics") or {}
        if not metrics:
            continue
        for name, field_name in LADDER.items():
            coverage[name].states[_state(metrics, field_name, False)] += 1
        for name, field_name in OTHER_CHECKS.items():
            coverage[name].states[_state(metrics, field_name, False)] += 1

    return coverage


def dead_checks(coverage: dict[str, CheckCoverage]) -> list[str]:
    """Checks that saw contexts and never once produced an answer."""
    return sorted(name for name, item in coverage.items() if item.never_fired)


def format_report(coverage: dict[str, CheckCoverage]) -> str:
    """The table, and the verdict under it.

    Printed at the stage boundary rather than per context: the three dead
    rungs of 31.08 were each invisible in their own context and obvious the
    moment all of them were counted together.
    """
    if not coverage or not any(item.contexts for item in coverage.values()):
        return ("check coverage: NO CONTEXTS SEEN -- the stage produced "
                "neither champions nor refusals, so nothing was checked and "
                "this report cannot say whether the checks work")

    lines = [
        f"{'check':<24}{'contexts':>9}{'bound':>7}{'passed':>8}"
        f"{'not measured':>14}{'not recorded':>14}",
        "-" * 76,
    ]
    for name in sorted(coverage):
        item = coverage[name]
        lines.append(
            f"{name:<24}{item.contexts:>9}{item.states[BOUND]:>7}"
            f"{item.states[PASSED]:>8}{item.states[NOT_MEASURED]:>14}"
            f"{item.states[NOT_RECORDED]:>14}"
        )

    dead = dead_checks(coverage)
    unrecorded = sorted(name for name, item in coverage.items()
                        if item.unrecorded == item.contexts and item.contexts)
    lines.append("")
    if unrecorded:
        lines.append(
            f"{len(unrecorded)} check(s) are not written into these records at "
            f"all -- {', '.join(unrecorded)}. That is a gap in the ARTEFACT, "
            f"not a verdict about the check: it cannot be audited after the "
            f"fact, which for a margin is the number the promotion turned on."
        )
    if dead:
        lines.append(
            f"DEFECT: {len(dead)} check(s) never produced an answer on any "
            f"context -- {', '.join(dead)}. A check that cannot bind is not a "
            f"check, and every promotion it did not block was unopposed "
            f"(REGISTER #201)."
        )
    else:
        lines.append(
            "Every check produced an answer on at least one context. A check "
            "that measured many and refused none is working and finding "
            "nothing, which is a result rather than a fault."
        )
    return "\n".join(lines)
