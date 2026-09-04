"""A source declared critical must not be able to deliver nothing in silence.

`collectors.yaml` has carried `critical: true/false` for twenty collectors
since before this audit, and it was read NOWHERE in `src/`. That is the shape
of REGISTER #235 exactly: a config field promising a mechanism nobody built,
found this time by following the third failing ratchet rather than by accident
(#240, #252).

Only `yahoo_finance` declares it, and that is the right choice -- it is the
price source. The consequence of the gap is specific: collectors return an
empty result both when a source is genuinely empty and when it failed, so a
run whose prices never arrived reads identically to a run with nothing new.
Every later stage then works from whatever is already in the database and
reports success on it. That has happened here before for other reasons: ten
collectors were cancelled at once on 2026-08-17 and "the pipeline reported
success", in the stage's own words.

WHAT THIS DOES AND DOES NOT DO. The shortfall is detected and said at ERROR
with the names. It does not yet FAIL the run, because failing a run is a
behaviour change of the same class as #229 and belongs to the owner. The check
existing is what makes that decision available; without it there is nothing to
decide with.

The list is read from the collector config rather than restated here, so the
declaration and the enforcement cannot drift apart -- which is how
`family_size` came to be promised in one file and verified in none.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipeline.stages.collection.orchestrator import CollectionStage

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = PROJECT_ROOT / "src" / "config" / "collectors.yaml"


class _Config:
    def __init__(self, collectors: dict):
        self._collectors = collectors

    def get_config(self, name):
        return {"collectors": self._collectors} if name == "collectors" else {}


def _stage(collectors: dict) -> CollectionStage:
    stage = CollectionStage.__new__(CollectionStage)
    stage.config_manager = _Config(collectors)
    return stage


def test_a_critical_source_that_delivered_nothing_is_named():
    stage = _stage({
        "yahoo_finance": {"critical": True, "enabled": True},
        "fear_greed": {"critical": False, "enabled": True},
    })
    assert stage._critical_shortfall(["yahoo_finance", "fear_greed"], []) == [
        "yahoo_finance"
    ]


def test_a_critical_source_that_failed_outright_counts_too():
    """Silent and failed are different facts about the collector and the same
    fact about the run: no prices arrived."""
    stage = _stage({"yahoo_finance": {"critical": True, "enabled": True}})
    assert stage._critical_shortfall([], ["yahoo_finance"]) == ["yahoo_finance"]


def test_nothing_is_reported_when_the_critical_source_delivered():
    stage = _stage({
        "yahoo_finance": {"critical": True, "enabled": True},
        "cftc": {"critical": False, "enabled": True},
    })
    assert stage._critical_shortfall(["cftc"], []) == []


def test_a_disabled_critical_source_is_not_a_shortfall():
    """Turning a source off is a decision; it must not read as a failure."""
    stage = _stage({"yahoo_finance": {"critical": True, "enabled": False}})
    assert stage._critical_shortfall(["yahoo_finance"], []) == []


def test_an_unreadable_config_does_not_crash_collection():
    """The check must never be the reason a run dies -- it exists to make a
    failure audible, not to add one."""
    stage = CollectionStage.__new__(CollectionStage)
    stage.config_manager = None
    assert stage._critical_shortfall(["yahoo_finance"], []) == []


def test_the_price_source_is_still_the_one_declared_critical():
    """If this changes, the change was deliberate and belongs in REGISTER:
    everything downstream assumes prices arrived."""
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sources = config.get("collectors", config)
    critical = sorted(
        name for name, body in sources.items()
        if isinstance(body, dict) and body.get("critical")
    )
    assert critical == ["yahoo_finance"], (
        f"the set of critical sources is now {critical}. Adding one means a "
        f"run without it is a run about nothing; removing yahoo_finance means "
        f"a run with no prices is acceptable. Both are decisions, not edits."
    )


def test_the_field_is_read_by_the_stage_and_not_only_declared():
    """The defect this whole file exists for: twenty declarations, no reader."""
    import inspect

    source = inspect.getsource(CollectionStage)
    assert "critical" in source, (
        "the collection stage no longer reads `critical`, so the field is "
        "back to being a promise in a config file"
    )
