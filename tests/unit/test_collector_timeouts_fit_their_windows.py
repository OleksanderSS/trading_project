"""A timeout sized for a 30-day window kills a 4000-day one, and loses everything.

The attention collector was deepened from 30 days to 4000 on 2026-08-23,
because Wikipedia pageviews are one of the few genuinely leading series here
and 30 days is untrainable against a frame spanning decades.

The timeout was not touched, and it was sized for the old window. Measured on
the rebuild's own log: each of the 184 articles is a ~500 KiB response taking
about six seconds, so the collector needs roughly 18 minutes and would have
been cancelled at ten.

Cancellation is not a partial result. Collectors write once at the end, so a
cancelled one loses every row it fetched -- recorded as #32, still true. The
run would have finished 2.6 hours later with the attention column as empty as
before, and nothing in the log saying why.

Caught by watching the request rate two minutes in rather than waiting for the
outcome. This test exists so the next person does not have to.
"""

from __future__ import annotations

import io

import yaml

from src.pipeline.stages.collection.orchestrator import CollectionStage

#: Seconds per request, measured on the 2026-08-23 rebuild: 21:59:43 to
#: 21:59:48 for one article at the full window.
SECONDS_PER_ARTICLE = 6.0

#: Tickers plus the macro terms the collector expands them into. 184 on the
#: current asset list.
TYPICAL_ARTICLE_COUNT = 184


def test_the_attention_timeout_covers_its_own_window():
    config = yaml.safe_load(io.open("src/config/collectors.yaml", encoding="utf-8"))
    days = int(config.get("collectors", config)["wikimedia_attention"]["days_back"])
    timeout = CollectionStage._COLLECTOR_TIMEOUT_SECONDS["wikimedia_attention"]

    needed = TYPICAL_ARTICLE_COUNT * SECONDS_PER_ARTICLE
    assert timeout >= needed, (
        f"the collector asks for {days} days across ~{TYPICAL_ARTICLE_COUNT} "
        f"articles, which needs about {needed:.0f}s, and the timeout is "
        f"{timeout}s. Being cancelled loses every row fetched, so the column "
        "would come out empty with no error to explain it."
    )


def test_a_deep_window_and_a_shallow_timeout_cannot_coexist():
    """The pairing is the defect, not either number alone.

    Whichever is changed next, this fails if they stop matching.
    """
    config = yaml.safe_load(io.open("src/config/collectors.yaml", encoding="utf-8"))
    days = int(config.get("collectors", config)["wikimedia_attention"]["days_back"])
    timeout = CollectionStage._COLLECTOR_TIMEOUT_SECONDS["wikimedia_attention"]

    if days > 365:
        assert timeout >= 1200, (
            f"{days} days of history is a multi-minute fetch per article; "
            f"{timeout}s will cancel it partway and discard the lot"
        )
