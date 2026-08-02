"""Fingerprint position i means all_state_cols[i] -- and nothing recorded it.

ContextMapEnricher builds the fingerprint from sorted(state_cols), so the
mapping exists but was implicit in a sort order computed at enrichment time
and then thrown away. Two consequences this registry addresses:

- component analysis could only say "driver 37", which no human can act on;
- the ordering shifts whenever the feature set changes, silently re-pointing
  every fingerprint ever written, because a fingerprint carries no version.
"""
from __future__ import annotations

import json

import pytest

from src.features.context_schema import (
    driver_name,
    drivers_for,
    latest_schema,
    record_schema,
    schema_id,
)


@pytest.fixture()
def registry(tmp_path):
    return tmp_path / "context_drivers.json"


DRIVERS = ["state_ATR_14", "state_MACD", "state_RSI_14"]


def test_a_recorded_ordering_can_be_read_back(registry):
    identifier = record_schema(DRIVERS, path=registry)

    assert drivers_for(identifier, path=registry) == DRIVERS
    assert latest_schema(path=registry) == (identifier, DRIVERS)


def test_order_is_part_of_the_identity():
    """The same columns in a different order produce different fingerprints,
    so they are genuinely a different schema."""
    assert schema_id(DRIVERS) != schema_id(list(reversed(DRIVERS)))


def test_recording_the_same_ordering_twice_does_not_grow_the_registry(registry):
    first = record_schema(DRIVERS, path=registry)
    second = record_schema(DRIVERS, path=registry)

    assert first == second
    assert len(json.loads(registry.read_text(encoding="utf-8"))["schemas"]) == 1


def test_a_changed_feature_set_becomes_a_new_schema(registry):
    old = record_schema(DRIVERS, path=registry)
    new = record_schema([*DRIVERS, "state_CDL_DOJI"], path=registry)

    assert old != new
    assert latest_schema(path=registry)[0] == new
    # The old ordering stays readable, which is what lets a rule derived
    # under it be recognised as stale rather than silently misread.
    assert drivers_for(old, path=registry) == DRIVERS


def test_a_position_past_the_end_is_labelled_not_guessed():
    """An out-of-range index means the fingerprint came from another schema.
    Saying so beats naming whichever column happens to sit nearby."""
    assert driver_name(2, DRIVERS) == "state_RSI_14"
    assert driver_name(99, DRIVERS) == "driver_99"
    assert driver_name(0, []) == "driver_0"


def test_an_unreadable_registry_does_not_raise(registry):
    """Feature engineering must not die because a JSON file is corrupt."""
    registry.write_text("{not json", encoding="utf-8")

    assert latest_schema(path=registry) == ("", [])
    assert record_schema(DRIVERS, path=registry)  # recovers by overwriting
    assert drivers_for("nonexistent", path=registry) == []


def test_an_empty_driver_list_records_nothing(registry):
    assert record_schema([], path=registry) == ""
    assert not registry.exists()


def test_the_enricher_registers_the_order_it_actually_concatenates():
    """The guarantee this all rests on, read from the source.

    _generate_context_features builds the fingerprint from all_state_cols and
    must register that exact list -- if the two ever diverge, every decoded
    driver name is wrong while looking perfectly plausible.
    """
    import inspect

    from src.features.enrichers.context_map_enricher import ContextMapEnricher

    source = inspect.getsource(ContextMapEnricher._generate_context_features)

    assert "record_schema(all_state_cols)" in source
    assert "res_df[all_state_cols].astype(str).agg('|'.join" in source
