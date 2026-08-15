"""A model predicting NVDA could not see that MSFT had just spiked.

Measured on the 2026-08-15 export: zero sector features, zero cross-ticker
features, and the only "market" columns were computed from the ticker's own
history. Every model looked at one company through a keyhole.

This is a different mechanism from pooling tickers into one model, and the two
were being conflated. Pooling shares the LEARNING — one model sees every
ticker's history. Peer features share the INFORMATION — each row knows what
its sector did in that same hour. A pooled model still cannot see, at
prediction time, that a neighbour moved an hour ago.

Excluding the ticker from its own sector average is the whole design. An
average that contains the row it describes leaks that row's return back into
its own features — for a two-name sector, half of it. Every aggregate is
(sum over sector - this ticker) / (count - 1).

`peer_divergence` carries the case this was built for: the sector rises and
one name does not. Google's servers burn down, the index shrugs, GOOGL does
not.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.peer_context_enricher import PeerContextEnricher


@pytest.fixture
def enricher():
    return PeerContextEnricher()


def _bars(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """(ticker, timestamp, close) triples, two bars per ticker minimum."""
    return pd.DataFrame(
        [{"ticker": t, "datetime": pd.Timestamp(ts, tz="UTC"), "close": c}
         for t, ts, c in rows]
    )


def _two_bars(moves: dict[str, float]) -> pd.DataFrame:
    """Each ticker gets a flat bar then a bar with the given return."""
    rows = []
    for ticker, move in moves.items():
        rows.append((ticker, "2026-07-01 10:00", 100.0))
        rows.append((ticker, "2026-07-01 11:00", 100.0 * (1 + move)))
    return _bars(rows)


def test_a_bar_is_never_part_of_the_average_that_describes_it(enricher):
    """Three semis move together; each one's peer_return excludes itself."""
    df = _two_bars({"NVDA": 0.10, "AMD": 0.02, "INTC": 0.02})

    out = enricher._enrich_impl(df)

    second = out[out["datetime"] == pd.Timestamp("2026-07-01 11:00", tz="UTC")]
    nvda = second.loc[second["ticker"] == "NVDA", "peer_return"].iloc[0]
    # NVDA's peers are AMD and INTC: (0.02 + 0.02) / 2
    assert nvda == pytest.approx(0.02, abs=1e-9)
    assert nvda != pytest.approx(0.10), "NVDA's own +10% leaked into its peer average"


def test_divergence_names_the_one_that_did_not_follow(enricher):
    """The whole sector rises; one name falls. That gap is the feature.

    Note the arithmetic that leave-one-out forces, because it is easy to
    expect the wrong number: NVDA's peers here are AMD (+5%) and INTC (-3%),
    averaging +1%, so NVDA diverges by +4% even though it moved exactly with
    AMD. In a sector of three, one outlier drags every other name's peer set.
    What must hold is the ORDERING — the name that fell is furthest below its
    peers, and the names that rose are above theirs.
    """
    df = _two_bars({"NVDA": 0.05, "AMD": 0.05, "INTC": -0.03})

    out = enricher._enrich_impl(df)

    second = out[out["datetime"] == pd.Timestamp("2026-07-01 11:00", tz="UTC")]
    div = second.set_index("ticker")["peer_divergence"]
    assert div["INTC"] < -0.07, "a name falling while its sector rises must show it"
    assert div["NVDA"] > 0 and div["AMD"] > 0
    assert div["INTC"] < div["NVDA"] and div["INTC"] < div["AMD"]


def test_breadth_counts_the_peers_that_rose(enricher):
    df = _two_bars({"NVDA": 0.05, "AMD": 0.05, "INTC": -0.03, "TSM": 0.01})

    out = enricher._enrich_impl(df)

    second = out[out["datetime"] == pd.Timestamp("2026-07-01 11:00", tz="UTC")]
    intc = second.loc[second["ticker"] == "INTC", "peer_breadth"].iloc[0]
    assert intc == pytest.approx(1.0), "all three of INTC's peers rose"


def test_a_ticker_alone_in_its_sector_gets_nothing_not_zero(enricher):
    """Zero would say 'the sector was flat', which is a different claim."""
    df = _two_bars({"XOM": 0.04})

    out = enricher._enrich_impl(df)

    assert out["peer_return"].isna().all()
    assert (out["peer_count"] == 0).all()


def test_sectors_do_not_bleed_into_each_other(enricher):
    df = _two_bars({"NVDA": 0.10, "AMD": 0.10, "JPM": -0.05, "BAC": -0.05})

    out = enricher._enrich_impl(df)

    second = out[out["datetime"] == pd.Timestamp("2026-07-01 11:00", tz="UTC")]
    nvda = second.loc[second["ticker"] == "NVDA", "peer_return"].iloc[0]
    jpm = second.loc[second["ticker"] == "JPM", "peer_return"].iloc[0]
    assert nvda == pytest.approx(0.10)
    assert jpm == pytest.approx(-0.05)


def test_only_the_same_timestamp_counts(enricher):
    """An aggregate built across timestamps would import the future."""
    df = _bars([
        ("NVDA", "2026-07-01 10:00", 100.0),
        ("NVDA", "2026-07-01 11:00", 110.0),
        ("AMD", "2026-07-01 10:00", 100.0),
        ("AMD", "2026-07-01 11:00", 101.0),
        ("AMD", "2026-07-01 12:00", 150.0),   # a later spike NVDA must not see
    ])

    out = enricher._enrich_impl(df)

    nvda_11 = out[(out["ticker"] == "NVDA")
                  & (out["datetime"] == pd.Timestamp("2026-07-01 11:00", tz="UTC"))]
    assert nvda_11["peer_return"].iloc[0] == pytest.approx(0.01, abs=1e-9), (
        "AMD's 12:00 move reached NVDA's 11:00 bar"
    )


def test_a_frame_without_prices_is_refused_not_guessed(enricher, caplog):
    import logging

    df = pd.DataFrame({
        "ticker": ["NVDA", "AMD"],
        "datetime": pd.to_datetime(["2026-07-01", "2026-07-01"], utc=True),
    })

    with caplog.at_level(logging.ERROR):
        out = enricher._enrich_impl(df)

    assert "peer_return" not in out.columns
    assert "close" in "\n".join(r.getMessage() for r in caplog.records)
