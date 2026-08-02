"""Turn "this agent loses in these market states" into reviewable rules.

`LearningLoopsEngine.run_hypothesis_generation` called
`ContextRuleGenerator.generate_rules_from_context`, a method that was never
written. The gap was wider than the name: `ContextRuleGenerator` measures
forward returns after an indicator crosses a threshold, given price history,
and returns {indicator, condition, value, event_count, effects_on_target} --
a different question with a different answer shape from the
{description, conditions, action} the caller reads.

WHAT A RULE MEANS HERE. A context fingerprint is the concatenation of the
`state_<FEATURE>` columns, each discretised to -1/0/1 by
ContextMapEnricher. So position i at value v is a concrete, checkable
statement about the market: "state_RSI_14 is +1". A rule says: when driver
`name` sits at `value`, this agent loses `rate` of the time across `total`
resolved trades, against its own baseline of `baseline`.

WHY THE BASELINE MATTERS. An agent that loses 55% of all its trades loses
55% of the time in most states too. Without subtracting its own baseline,
every driver value looks damning and the rule set is noise. The margin below
is the excess loss rate over the agent's own average, which is the only part
that carries information about the STATE rather than the agent.

WHY EVIDENCE THRESHOLDS ARE NOT OPTIONAL. With 30+ drivers at 3 values each
there are ~90 candidate statements per agent. Testing all of them against the
same data guarantees some clear any fixed bar by chance, so the thresholds
here are deliberately blunt: enough trades to mean something, and a margin
large enough that the rule is worth a human's attention. These are
EXPERIMENTAL rules entering a promotion lifecycle, not signals -- the
lifecycle is where they earn trust.

WHY THE SCHEMA ID IS RECORDED. Driver positions come from
`sorted(state_cols)` at enrichment time. Change the feature set and position
37 becomes a different column, silently. Each rule carries the schema id it
was derived under, so a rule built against a schema that no longer applies
can be found and retired instead of quietly meaning something else.
"""
from __future__ import annotations

from typing import Any

from src.features.context_schema import driver_name, latest_schema

#: Minimum resolved trades behind a (driver, value) pair before it can become
#: a rule. Below this the rate is an anecdote; ~90 candidates per agent means
#: small samples will produce extreme rates by chance alone.
MIN_TRADES_FOR_RULE = 10

#: How much worse than the agent's own loss rate the state must be. 0.15 is a
#: judgement call, not a derived constant: large enough that a rule is worth
#: reading, small enough that a genuinely bad regime is not missed.
MIN_EXCESS_LOSS_RATE = 0.15

#: Cap on rules per agent per run, worst first. An unbounded generator that
#: fires on every review would fill the rule table faster than anyone can
#: promote or retire entries.
MAX_RULES_PER_AGENT = 5


def synthesise_context_rules(
    agent_id: str,
    component_loss_rates: dict[int, dict[str, dict[str, float]]],
    *,
    baseline_loss_rate: float,
    min_trades: int = MIN_TRADES_FOR_RULE,
    min_excess: float = MIN_EXCESS_LOSS_RATE,
    max_rules: int = MAX_RULES_PER_AGENT,
    schema: tuple[str, list[str]] | None = None,
) -> list[dict[str, Any]]:
    """Rules for the market states where `agent_id` does worst.

    `component_loss_rates` is DiaryEngine's decomposition: driver position ->
    tri-state value -> {rate, count, total}.

    Returns dicts shaped for TradingRule: description / conditions / action.
    Empty when nothing clears the evidence bar, which is the honest answer
    far more often than not.
    """
    if not component_loss_rates:
        return []

    schema_identifier, drivers = schema if schema is not None else latest_schema()

    candidates: list[dict[str, Any]] = []
    for index, per_value in component_loss_rates.items():
        for value, stats in per_value.items():
            total = float(stats.get("total", 0.0))
            rate = float(stats.get("rate", 0.0))
            if total < min_trades:
                continue
            excess = rate - float(baseline_loss_rate)
            if excess < min_excess:
                continue
            candidates.append(
                _build_rule(
                    agent_id=agent_id,
                    index=int(index),
                    value=str(value),
                    rate=rate,
                    excess=excess,
                    total=total,
                    count=float(stats.get("count", 0.0)),
                    baseline=float(baseline_loss_rate),
                    drivers=drivers,
                    schema_identifier=schema_identifier,
                )
            )

    # Worst first, then by evidence, so a tie between two equally bad states
    # is broken by the one observed more often rather than by dict order.
    candidates.sort(
        key=lambda rule: (
            -rule["conditions"]["excess_loss_rate"],
            -rule["conditions"]["total_trades"],
        )
    )
    return candidates[:max_rules]


def _build_rule(
    *,
    agent_id: str,
    index: int,
    value: str,
    rate: float,
    excess: float,
    total: float,
    count: float,
    baseline: float,
    drivers: list[str],
    schema_identifier: str,
) -> dict[str, Any]:
    name = driver_name(index, drivers)
    readable = _readable_state(value)
    named = bool(drivers) and index < len(drivers)

    return {
        "description": (
            f"{agent_id} loses {rate:.0%} of trades when {name} is "
            f"{readable} ({int(count)} of {int(total)}), against its own "
            f"baseline of {baseline:.0%}."
        ),
        "conditions": {
            # Machine-checkable form: the column and the discrete value.
            "driver": name,
            "driver_index": index,
            "driver_value": int(value) if value.lstrip("-").isdigit() else value,
            # Evidence, kept with the rule so a reviewer never has to trust
            # the description alone.
            "loss_rate": round(rate, 4),
            "baseline_loss_rate": round(baseline, 4),
            "excess_loss_rate": round(excess, 4),
            "losing_trades": int(count),
            "total_trades": int(total),
            # Provenance. driver_index is meaningless without the ordering it
            # was computed under; `driver_named` says outright whether the
            # name above is real or a positional placeholder.
            "context_schema_id": schema_identifier,
            "driver_named": named,
            "agent_id": agent_id,
        },
        # Deliberately advisory. Nothing in this project consumes rules to
        # size or block a trade yet, and a generated statistic should not
        # start doing that on its own -- promotion through the rule lifecycle
        # is the gate for that decision.
        "action": "reduce_exposure",
    }


def _readable_state(value: str) -> str:
    return {
        "1": "rising (+1)",
        "0": "flat (0)",
        "-1": "falling (-1)",
    }.get(str(value), f"at {value}")
