"""Attention is one of the few leading series here, and it was 30 days deep.

A technical indicator is computed FROM the price it is meant to predict, so it
cannot lead by construction. Wikipedia pageviews are produced by people looking
something up -- a different process, running before they trade -- so they can.
Measured on 2026-08-23, the strongest of the 430 price-derived features reached
an out-of-sample IC of 0.046, and no model built from all of them beat it. New
information has to come from outside the price.

Which makes the window the whole game. `days_back: 30` on a daily frame that
spans decades leaves the column empty for all but the last month, so nothing
can be trained on it -- the feature exists and teaches nothing.

The depth is free: the pageviews API returns the entire range in ONE request
per article. Measured on Tesla's page, 3,487 daily points from 2017-02-02 to
2026-08-20 arrived in 526 KiB, the same single request a 30-day window costs.

Same defect as the 60-day SEC filings window, found the same day, which is why
this is a test and not a comment.
"""

from __future__ import annotations

import io

import yaml

#: The pageviews API starts here; asking for less is a choice, not a limit.
API_HISTORY_STARTS = "2015-07-01"


def _collector(name):
    config = yaml.safe_load(io.open("src/config/collectors.yaml", encoding="utf-8"))
    return config.get("collectors", config)[name]


def test_attention_is_collected_deep_enough_to_train_on():
    days = int(_collector("wikimedia_attention")["days_back"])
    assert days >= 1825, (
        f"attention is collected for {days} days. The daily frame spans "
        "decades, so a window this short leaves the feature empty almost "
        "everywhere -- and it costs nothing to widen, since the API returns "
        "the whole range in the same single request."
    )


def test_the_leading_sources_are_not_shallower_than_the_lagging_ones():
    """A price feature has full history by construction; these must too.

    The failure this guards against is subtle: the pipeline looks complete,
    every enricher reports success, and the only non-price columns are blank
    for 99% of the rows. Nothing errors. The model simply never sees them.
    """
    windows = {
        "wikimedia_attention": int(_collector("wikimedia_attention")["days_back"]),
        "sec_filings": _collector("sec_filings")["params"]["period"],
    }
    attention = windows["wikimedia_attention"]
    filings = windows["sec_filings"]

    unit = filings[-1]
    amount = int(filings[:-1]) if unit.isalpha() else int(filings)
    filing_days = amount * {"d": 1, "y": 365}[unit]

    assert attention >= 365 and filing_days >= 365, (
        f"leading sources are shallow: attention {attention}d, "
        f"filings {filing_days}d. Both feed columns that a model is expected "
        "to learn from across years of bars."
    )
