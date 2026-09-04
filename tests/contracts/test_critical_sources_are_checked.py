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


# ---------------------------------------------------------------------------
# The second half: what matters is the STORE, not the collector.
#
# A collector that added nothing because everything was already saved is a
# success, and this pipeline runs that way often. Failing on that would make
# the thing brittle, and a check that fires on ordinary recoverable conditions
# gets switched off -- `|| true` has been in ci.yml for six weeks for exactly
# that reason, and eight contract tests errored on a busy database until
# 2026-09-03 for the same one.
#
# So "no NEW data" is a warning and "no data" is a failure, and only the store
# can tell them apart.


class _Store:
    def __init__(self, rows):
        self._rows = rows

    def fetch_all(self, query, params=None):
        if self._rows is None:
            raise RuntimeError("database is locked by another process")
        return self._rows


def _stage_with_store(rows):
    stage = CollectionStage.__new__(CollectionStage)
    stage.config_manager = _Config({})
    stage.db_manager = _Store(rows)
    return stage


def _bar(days_old: int, interval: str = "1d"):
    import datetime as dt

    return {
        "interval": interval,
        "newest": dt.datetime.now(dt.UTC) - dt.timedelta(days=days_old),
        "rows": 719_169,
        "names": 112,
    }


def test_a_recent_store_is_usable_so_a_quiet_collector_is_only_a_warning():
    """Measured 2026-09-03: the newest daily bar was three days old. A
    threshold that fires on that would fire on every long weekend."""
    state = _stage_with_store([_bar(3)])._price_store_freshness()
    assert state["usable"] is True
    assert state["age_days"] == 3


def test_a_stale_store_is_not_usable():
    state = _stage_with_store([_bar(30)])._price_store_freshness()
    assert state["usable"] is False
    assert "30 days old" in state["reason"]


def test_an_empty_table_is_not_usable():
    state = _stage_with_store([])._price_store_freshness()
    assert state["usable"] is False
    assert "empty" in state["reason"]


def test_a_store_that_cannot_be_read_is_not_reported_as_healthy():
    """The substitution this whole file exists to prevent: an unreadable store
    is not a working one, and saying otherwise is how a failure becomes a
    success."""
    state = _stage_with_store(None)._price_store_freshness()
    assert state["usable"] is False
    assert "could not be read" in state["reason"]


def test_the_freshest_cadence_decides():
    """Intraday is collected too, and 15m being current says the feed works
    even if a daily bar lags. Taking the OLDEST would fail runs that are fine."""
    state = _stage_with_store([_bar(30, "1d"), _bar(2, "15m")])._price_store_freshness()
    assert state["usable"] is True
    assert state["interval"] == "15m"


def test_the_threshold_is_a_number_and_not_a_feeling():
    """Without a stated limit, "the store has prices" is true forever and the
    check never fires -- the mirror of firing too often."""
    assert isinstance(CollectionStage._MAX_PRICE_AGE_DAYS, int)
    assert 1 <= CollectionStage._MAX_PRICE_AGE_DAYS <= 30
