"""An insider row must pass a check on its CONTENT, not only on its width.

WHY. `insider_collector` addresses columns by fixed index -- `title: 6`,
`trade_type: 7`, `price: 8` -- and its only guard was `len(cells) >=
required_width`. Any variation in the page layout shifts a row by one and it
still passes that guard, so the shifted row is stored looking perfectly normal.

Measured on the stored table 2026-09-04: **8,349 of 9,744 rows carry a PRICE in
`trade_type`** -- 85.7%.

    whole    title='CHAIRPERSON, CEO'   trade_type='S - Sale+OE'  value='-$4,962,488'
    shifted  title='S - Sale'           trade_type='$307.75'      value=''

On shifted rows `value` is empty, so `TickerExternalEnricher` -- whose own logic
is correct, and which correctly uses `filing_date` rather than `trade_date` --
rolls up nothing. `insider_net_value_30d_1d` came out 77% zeros with 31 distinct
small integers across 29 of 110 names, and it still passed the leadingness
screen into the 46 FDR survivors (R20). Corrected for the market exposure found
in R28 it scores -0.034, negative at every horizon.

The consequence that matters most: of the 1,395 rows with a valid trade type,
**32 are in our universe and 0 are outside the seal**. Our own large caps are
predominantly the shifted ones.

WHAT IS FIXED: `trade_type` has a three-value vocabulary, so it is the one
field that can be checked cheaply. A row failing it is dropped and counted
rather than stored wrong, and the count is reported -- at ERROR once it passes
a fifth of the page, because that means the mapping no longer matches the page
and every row from it is wrong in a way nothing downstream can see.

Not fixed here: the stored 8,349 rows. Repairing them needs re-collection,
which is a decision about hitting an external source, and it would not unlock
anything measurable -- 32 in-universe events, all sealed.
"""
from __future__ import annotations

import re

from src.data.collectors.insider_collector import _TRADE_TYPE

#: The whole vocabulary this source writes, measured against the stored table.
REAL_TRADE_TYPES = ("P - Purchase", "S - Sale", "S - Sale+OE")

#: What the shift puts there instead: the price column.
SHIFTED_VALUES = ("$307.75", "$0.00", "$76.62", "$1,234.56")


def test_the_real_trade_types_are_accepted():
    for value in REAL_TRADE_TYPES:
        assert _TRADE_TYPE.match(value), (
            f"{value!r} is a trade type this source actually writes; rejecting "
            f"it would drop every good row and leave the table empty"
        )


def test_a_price_in_the_trade_type_column_is_rejected():
    """The exact corruption: 8,349 of 9,744 stored rows look like this."""
    for value in SHIFTED_VALUES:
        assert not _TRADE_TYPE.match(value), (
            f"{value!r} is a price sitting in the trade-type column and would "
            f"be stored as a shifted row"
        )


def test_empty_and_missing_are_rejected():
    """`.get(...)` returns '' when the field never arrived; an empty trade type
    is not a trade."""
    for value in ("", "   ", "-", "N/A"):
        assert not _TRADE_TYPE.match(value)


def test_the_check_is_not_a_whitelist_of_todays_three_values():
    """A literal list would reject a fourth legitimate code -- 'A - Award',
    'M - Option Exercise' -- and silently drop real trades. The pattern is the
    SHAPE of a code, so a new one passes and a price still does not."""
    for plausible in ("A - Award", "M - Exercise", "G - Gift", "F - Tax"):
        assert _TRADE_TYPE.match(plausible), (
            f"{plausible!r} is the shape of a Form 4 code and must pass, or "
            f"this check trades one silent loss for another"
        )


def test_the_collector_actually_applies_it():
    """The defect family this repository keeps finding: a mechanism declared in
    one place and read nowhere (`critical: true` in twenty collectors, #252;
    `family_size`, #235). A pattern nobody calls is a comment."""
    import inspect

    from src.data.collectors import insider_collector

    source = inspect.getsource(insider_collector)
    assert "_TRADE_TYPE.match" in source, (
        "the collector no longer checks trade_type, so a shifted row is stored "
        "looking normal again"
    )
    assert "shifted" in source, (
        "the rejected rows are no longer counted; dropping them silently is "
        "the same failure with fewer rows"
    )


def test_the_pattern_is_anchored():
    """Unanchored, '$76.62 S - Sale' would pass and a shifted row would be
    stored anyway."""
    assert _TRADE_TYPE.pattern.startswith("^")
    assert isinstance(_TRADE_TYPE, re.Pattern)
