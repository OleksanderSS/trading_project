"""A reported number exists from the day it was filed, not the day it describes.

Without financial facts, "undervalued" cannot be computed at all -- only
asserted. But the numbers are worth less than nothing if they are placed in
time by the period they cover: a June quarter reaches the SEC in August, so
joining on the period end puts it into June's bars, six weeks before anyone
could read it. Measured on Apple's own filings, the shortest gap between a
period ending and the filing that first reported it is 25 days.

That is the same defect as `reportDate` versus `filingDate` in the corporate
filings enricher, and the same shape as the persistence baseline that used a
value only knowable four days later: arithmetic that runs correctly and answers
a question nobody could have acted on.

The second thing these tests hold down is the identity key. One filing reports
NetIncomeLoss twice -- once for the quarter, once for the nine months, both
ending the same day -- so a key of (concept, unit, period_end, accession)
collides on 471 of Apple's 2,939 facts and deduplication silently keeps
whichever landed first.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.collectors.sec_fundamentals_collector import (
    DEFAULT_CONCEPTS,
    SECFundamentalsCollector,
)


@pytest.fixture
def collector():
    instance = SECFundamentalsCollector.__new__(SECFundamentalsCollector)
    instance.concepts = DEFAULT_CONCEPTS
    instance.taxonomies = ("us-gaap", "dei")
    instance.hash_keys = [
        "cik", "concept", "unit", "period_start", "period_end", "accession",
    ]
    return instance


def _payload(entries, concept="NetIncomeLoss"):
    return {
        "entityName": "PROBE INC",
        "facts": {"us-gaap": {concept: {"label": concept,
                                        "units": {"USD": entries}}}},
    }


def test_the_filing_date_is_carried_separately_from_the_period(collector):
    rows = collector._rows_from_facts(_payload([
        {"start": "2024-04-01", "end": "2024-06-30", "val": 1_000,
         "filed": "2024-08-05", "form": "10-Q", "accn": "A-1", "fy": 2024, "fp": "Q2"},
    ]), "PRB", "0000000001")

    assert len(rows) == 1
    row = rows[0]
    assert row["period_end"] == "2024-06-30", "the period it describes"
    assert row["filed"] == "2024-08-05", "the day it could first be read"
    assert row["period_end"] != row["filed"], (
        "if these were ever the same field, every join would be lookahead"
    )


def test_a_fact_with_no_filing_date_is_dropped_rather_than_placed(collector):
    """Guessing a date here would be inventing the moment it became public."""
    rows = collector._rows_from_facts(_payload([
        {"start": "2024-04-01", "end": "2024-06-30", "val": 1_000,
         "form": "10-Q", "accn": "A-1"},                       # no `filed`
        {"start": "2024-07-01", "end": "2024-09-30", "val": 2_000,
         "filed": "2024-11-01", "form": "10-Q", "accn": "A-2"},
    ]), "PRB", "0000000001")

    assert [row["period_end"] for row in rows] == ["2024-09-30"]


def test_two_spans_ending_the_same_day_are_two_facts(collector):
    """The collision that made 471 of Apple's 2,939 facts share a key.

    A 10-Q states the quarter AND the year to date. Both end on the same day,
    both come from the same accession, and they are different numbers.
    """
    rows = collector._rows_from_facts(_payload([
        {"start": "2024-04-01", "end": "2024-06-30", "val": 1_072,
         "filed": "2024-08-05", "form": "10-Q", "accn": "SAME"},
        {"start": "2023-10-01", "end": "2024-06-30", "val": 3_698,
         "filed": "2024-08-05", "form": "10-Q", "accn": "SAME"},
    ]), "PRB", "0000000001")

    frame = pd.DataFrame(rows)
    hashes = frame.apply(collector.generate_hash, axis=1)
    assert len(rows) == 2
    assert hashes.is_unique, (
        "both spans hashed the same; deduplication would keep only one of them"
    )


def test_a_restatement_does_not_overwrite_the_original(collector):
    """Keeping both is what makes "what was known then" answerable.

    A screen run on restated history is a screen that could never have been
    run. The accession is what separates the two.
    """
    rows = collector._rows_from_facts(_payload([
        {"start": "2024-01-01", "end": "2024-03-31", "val": 500,
         "filed": "2024-05-01", "form": "10-Q", "accn": "ORIGINAL"},
        {"start": "2024-01-01", "end": "2024-03-31", "val": 480,
         "filed": "2025-02-01", "form": "10-K", "accn": "RESTATED"},
    ]), "PRB", "0000000001")

    frame = pd.DataFrame(rows)
    assert len(frame) == 2
    assert frame.apply(collector.generate_hash, axis=1).is_unique

    # What a point-in-time query has to produce: on 2024-06-01 only the
    # original existed, and it said 500.
    as_of = pd.Timestamp("2024-06-01")
    visible = frame[pd.to_datetime(frame["filed"]) <= as_of]
    assert list(visible["value"]) == [500.0]


def test_only_the_requested_concepts_are_kept(collector):
    """companyfacts carries hundreds; a value screen needs sixteen."""
    payload = {
        "entityName": "PROBE INC",
        "facts": {"us-gaap": {
            "Assets": {"label": "Assets", "units": {"USD": [
                {"end": "2024-06-30", "val": 9, "filed": "2024-08-05",
                 "form": "10-Q", "accn": "A-1"}]}},
            "SomeObscureTag": {"label": "x", "units": {"USD": [
                {"end": "2024-06-30", "val": 1, "filed": "2024-08-05",
                 "form": "10-Q", "accn": "A-1"}]}},
        }},
    }
    rows = collector._rows_from_facts(payload, "PRB", "0000000001")
    assert [row["concept"] for row in rows] == ["Assets"]


def test_the_unit_is_kept_because_the_number_is_meaningless_without_it():
    """`shares`, `USD` and `USD/shares` appear side by side under one concept."""
    instance = SECFundamentalsCollector.__new__(SECFundamentalsCollector)
    instance.concepts = ("EarningsPerShareBasic",)
    instance.taxonomies = ("us-gaap",)
    payload = {
        "entityName": "PROBE INC",
        "facts": {"us-gaap": {"EarningsPerShareBasic": {"label": "EPS", "units": {
            "USD/shares": [{"start": "2024-04-01", "end": "2024-06-30", "val": 1.4,
                            "filed": "2024-08-05", "form": "10-Q", "accn": "A-1"}],
        }}}},
    }
    rows = instance._rows_from_facts(payload, "PRB", "0000000001")
    assert rows[0]["unit"] == "USD/shares"
