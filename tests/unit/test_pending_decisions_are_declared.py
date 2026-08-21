"""Numbers that are in use without having been chosen must say so.

A placeholder in a config is indistinguishable from a decision. That shape has
cost this project repeatedly: a 0.5% trade cost copied into five files, a CIK
that resolved to a company nobody meant, a risk gate reporting "checks passed"
from two invented inputs. In each case the number was not wrong on purpose --
nobody had ever chosen it, and nothing said so.

`pending_decisions.yaml` is where those are declared. These tests keep it
honest: an entry that does not say what would resolve it is another number
nobody can act on, and it would look documented.
"""

import pytest
import yaml

from src.config.pending_decisions import (
    REQUIRED_FIELDS,
    as_report_header,
    blocking_real_money,
    is_provisional,
    log_pending_decisions,
    pending_decisions,
    provisional_note,
)


def test_the_file_parses_and_is_not_empty():
    decisions = pending_decisions()
    assert decisions, "nothing declared; if everything is decided, say so in the register"


@pytest.mark.parametrize("name", sorted(pending_decisions()))
def test_every_entry_says_what_would_resolve_it(name):
    decision = pending_decisions()[name]
    assert decision.what_resolves_it.strip(), f"{name} does not say what settles it"
    assert decision.why_provisional.strip(), f"{name} does not say why it is provisional"
    assert decision.affects, f"{name} does not name anything it affects"


def test_the_three_known_placeholders_are_declared():
    """Named explicitly: silently dropping one is how it becomes a decision."""
    declared = set(pending_decisions())
    assert {"broker_cost_profile", "position_size_rule", "starting_capital"} <= declared


def test_the_money_ones_are_marked_as_blocking():
    blocking = {decision.name for decision in blocking_real_money()}
    assert {"broker_cost_profile", "position_size_rule"} <= blocking


def test_a_half_filled_entry_is_refused(tmp_path):
    """Worse than none: it looks documented."""
    path = tmp_path / "pending.yaml"
    path.write_text(yaml.safe_dump({
        "pending": {"half": {"value_in_use": "something", "affects": ["x"]}}
    }), encoding="utf-8")

    with pytest.raises(ValueError) as caught:
        pending_decisions(str(path))
    assert "what resolves it" in str(caught.value)


def test_a_missing_file_is_not_an_error(tmp_path):
    assert pending_decisions(str(tmp_path / "absent.yaml")) == {}


def test_a_report_can_stamp_a_provisional_number():
    note = provisional_note("broker_cost_profile")
    assert note and note.startswith("PROVISIONAL")
    assert provisional_note("something_actually_decided") is None


def test_a_decided_number_is_not_flagged():
    assert is_provisional("broker_cost_profile")
    assert not is_provisional("purge_rows")


def test_the_header_names_every_entry():
    header = as_report_header()
    for name in pending_decisions():
        assert name in header


def test_the_run_is_told_once(caplog):
    import logging

    logger = logging.getLogger("pending-test")
    with caplog.at_level(logging.WARNING, logger="pending-test"):
        log_pending_decisions(logger)

    said = " ".join(record.getMessage() for record in caplog.records)
    assert "NOT decisions yet" in said
    assert "blocks real money" in said


def test_required_fields_are_what_the_file_actually_uses():
    raw = yaml.safe_load(
        open("src/config/pending_decisions.yaml", encoding="utf-8").read()
    )
    for name, body in (raw.get("pending") or {}).items():
        assert set(REQUIRED_FIELDS) <= set(body), name
