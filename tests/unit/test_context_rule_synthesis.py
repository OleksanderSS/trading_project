"""Rules from the market states where an agent actually loses.

run_hypothesis_generation used to call a method that does not exist. These
tests pin the replacement: what makes a rule, what does not, and why the
agent's own baseline has to be subtracted first.
"""
from __future__ import annotations

import pytest

from src.meta_learning.evolution.context_rule_synthesis import (
    MAX_RULES_PER_AGENT,
    synthesise_context_rules,
)

SCHEMA = ("abc123", ["state_ATR_14", "state_MACD", "state_RSI_14"])


def _rates(**by_index):
    """{index: {value: {rate, count, total}}} in the shape DiaryEngine emits."""
    return {
        index: {
            value: {"rate": rate, "count": rate * total, "total": total}
            for value, (rate, total) in per_value.items()
        }
        for index, per_value in by_index.items()
    }


def test_a_state_much_worse_than_the_agents_baseline_becomes_a_rule():
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{"2": {"1": (0.78, 50)}}),
        baseline_loss_rate=0.52,
        schema=SCHEMA,
    )

    assert len(rules) == 1
    conditions = rules[0]["conditions"]
    assert conditions["driver"] == "state_RSI_14"
    assert conditions["driver_value"] == 1
    assert conditions["excess_loss_rate"] == pytest.approx(0.26)
    assert conditions["total_trades"] == 50
    assert conditions["context_schema_id"] == "abc123"
    assert conditions["driver_named"] is True
    assert "state_RSI_14" in rules[0]["description"]
    assert rules[0]["action"] == "reduce_exposure"


def test_an_agent_that_loses_everywhere_generates_nothing():
    """78% loss looks alarming until you see the agent loses 76% of all
    trades. The state says nothing; only the excess carries information."""
    rules = synthesise_context_rules(
        "svm",
        _rates(**{"2": {"1": (0.78, 50)}}),
        baseline_loss_rate=0.76,
        schema=SCHEMA,
    )

    assert rules == []


def test_a_thin_sample_does_not_become_a_rule():
    """~90 candidate statements per agent means small samples throw up
    extreme rates by chance."""
    rules = synthesise_context_rules(
        "knn",
        _rates(**{"0": {"-1": (1.00, 3)}}),
        baseline_loss_rate=0.40,
        schema=SCHEMA,
    )

    assert rules == []


def test_rules_are_ranked_worst_first():
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{
            "0": {"1": (0.70, 40)},
            "1": {"-1": (0.90, 40)},
            "2": {"0": (0.80, 40)},
        }),
        baseline_loss_rate=0.50,
        schema=SCHEMA,
    )

    assert [r["conditions"]["driver"] for r in rules] == [
        "state_MACD", "state_RSI_14", "state_ATR_14",
    ]


def test_evidence_breaks_ties_between_equally_bad_states():
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{"0": {"1": (0.80, 20)}, "1": {"1": (0.80, 200)}}),
        baseline_loss_rate=0.50,
        schema=SCHEMA,
    )

    assert rules[0]["conditions"]["total_trades"] == 200


def test_the_number_of_rules_per_run_is_capped():
    rates = _rates(**{
        str(index): {"1": (0.95, 100)} for index in range(MAX_RULES_PER_AGENT + 4)
    })

    rules = synthesise_context_rules(
        "catboost", rates, baseline_loss_rate=0.10, schema=SCHEMA
    )

    assert len(rules) == MAX_RULES_PER_AGENT


def test_an_unknown_schema_labels_the_position_and_says_it_is_unnamed():
    """A rule naming the wrong column would be worse than one naming none."""
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{"37": {"1": (0.90, 50)}}),
        baseline_loss_rate=0.40,
        schema=("", []),
    )

    assert rules[0]["conditions"]["driver"] == "driver_37"
    assert rules[0]["conditions"]["driver_named"] is False
    assert rules[0]["conditions"]["context_schema_id"] == ""


def test_no_decomposable_components_yields_no_rules():
    assert synthesise_context_rules(
        "catboost", {}, baseline_loss_rate=0.5, schema=SCHEMA
    ) == []


def test_every_rule_carries_the_keys_TradingRule_reads():
    """The original defect was a shape mismatch, three lines after the call."""
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{"1": {"0": (0.85, 60)}}),
        baseline_loss_rate=0.45,
        schema=SCHEMA,
    )

    assert set(rules[0]) >= {"description", "conditions", "action"}
    assert isinstance(rules[0]["conditions"], dict)
    assert isinstance(rules[0]["description"], str)
    assert isinstance(rules[0]["action"], str)


def test_the_evidence_travels_with_the_rule():
    """A reviewer must never have to trust the description alone."""
    rules = synthesise_context_rules(
        "catboost",
        _rates(**{"1": {"0": (0.85, 60)}}),
        baseline_loss_rate=0.45,
        schema=SCHEMA,
    )

    conditions = rules[0]["conditions"]
    assert conditions["losing_trades"] == 51
    assert conditions["total_trades"] == 60
    assert conditions["loss_rate"] == pytest.approx(0.85)
    assert conditions["baseline_loss_rate"] == pytest.approx(0.45)
