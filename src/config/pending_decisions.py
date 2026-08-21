"""Read the numbers that are in use but are not decisions yet.

A placeholder in a config is indistinguishable from a choice. This project has
been bitten by that shape repeatedly: a 0.5% cost copied into five files, a CIK
that resolved to the wrong company, a risk gate reporting "checks passed" from
two invented inputs. Each was a number nobody had chosen, sitting where a
chosen number goes.

`pending_decisions.yaml` is where such numbers are declared. This module makes
them visible at runtime, so a result computed on top of one carries that fact
instead of looking finished.

    from src.config.pending_decisions import pending_decisions, provisional_note

    log_pending_decisions(logger)          # once, at pipeline start
    note = provisional_note("broker_cost_profile")   # stamp on a report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PATH = Path(__file__).with_name("pending_decisions.yaml")

REQUIRED_FIELDS = (
    "value_in_use",
    "why_provisional",
    "what_resolves_it",
    "affects",
    "blocks_real_money",
)


@dataclass(frozen=True)
class PendingDecision:
    """One number that is being used without having been chosen."""

    name: str
    value_in_use: str
    why_provisional: str
    what_resolves_it: str
    affects: tuple[str, ...]
    blocks_real_money: bool

    def one_line(self) -> str:
        return f"{self.name}: {self.value_in_use.strip().splitlines()[0]}"


@lru_cache(maxsize=4)
def pending_decisions(path: str | None = None) -> dict[str, PendingDecision]:
    """Every provisional number, keyed by name. Empty if the file is absent."""
    source = Path(path) if path else DEFAULT_PATH
    if not source.exists():
        return {}

    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    entries = raw.get("pending") or {}

    decisions: dict[str, PendingDecision] = {}
    for name, body in entries.items():
        if not isinstance(body, dict):
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in body]
        if missing:
            # A half-filled entry is worse than none: it looks documented.
            raise ValueError(
                f"pending decision '{name}' is missing {missing}. Every entry "
                f"must say what resolves it and what it affects, or it becomes "
                f"another number nobody can act on."
            )
        decisions[name] = PendingDecision(
            name=name,
            value_in_use=str(body["value_in_use"]),
            why_provisional=str(body["why_provisional"]),
            what_resolves_it=str(body["what_resolves_it"]),
            affects=tuple(str(item) for item in (body["affects"] or ())),
            blocks_real_money=bool(body["blocks_real_money"]),
        )
    return decisions


def is_provisional(name: str, path: str | None = None) -> bool:
    return name in pending_decisions(path)


def provisional_note(name: str, path: str | None = None) -> str | None:
    """A line to stamp on any report computed from this number."""
    decision = pending_decisions(path).get(name)
    if decision is None:
        return None
    return (
        f"PROVISIONAL — {name} is not a decision. In use: "
        f"{decision.value_in_use.strip().splitlines()[0]} "
        f"Resolved by: {decision.what_resolves_it.strip().splitlines()[0]}"
    )


def blocking_real_money(path: str | None = None) -> tuple[PendingDecision, ...]:
    """The ones that must be settled before money moves."""
    return tuple(
        decision for decision in pending_decisions(path).values()
        if decision.blocks_real_money
    )


def log_pending_decisions(logger: logging.Logger, path: str | None = None) -> None:
    """Say, once per run, which numbers are standing in for choices."""
    decisions = pending_decisions(path)
    if not decisions:
        return
    blocking = blocking_real_money(path)
    logger.warning(
        "%d value(s) in use are NOT decisions yet; %d of them must be settled "
        "before real money moves. Anything computed from them is provisional.",
        len(decisions), len(blocking),
    )
    for decision in decisions.values():
        logger.warning(
            "    %s%s", decision.one_line(),
            "  [blocks real money]" if decision.blocks_real_money else "",
        )


def as_report_header(path: str | None = None) -> str:
    """A block a diagnostic can print above its numbers."""
    decisions = pending_decisions(path)
    if not decisions:
        return ""
    lines = ["PROVISIONAL INPUTS — these numbers were not chosen:"]
    for decision in decisions.values():
        lines.append(f"  {decision.one_line()}")
        lines.append(f"      resolved by: {decision.what_resolves_it.strip().splitlines()[0]}")
    return "\n".join(lines)


def _describe(decision: PendingDecision) -> dict[str, Any]:
    return {
        "value_in_use": decision.value_in_use,
        "why_provisional": decision.why_provisional,
        "what_resolves_it": decision.what_resolves_it,
        "affects": list(decision.affects),
        "blocks_real_money": decision.blocks_real_money,
    }
