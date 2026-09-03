"""Sector aggregates must not be fourteen copies of the market.

Measured 2026-09-02 on daily bars 2010-2026 over the 14-sector partition: the
raw sector returns carry 2.97 independent dimensions out of 14. Their first
principal component is 56% of the variance and their mean pairwise correlation
is 0.505 -- fourteen columns saying one thing, and that thing is the market.
Subtracting the market leaves 8.63 dimensions at a mean correlation of -0.042
(CLAIMS.md R12).

That matters under this project's own power arithmetic: fourteen columns
carrying three dimensions is multiplicity bought without information, and the
detectable-effect threshold pays for every column tried (CLAIMS.md R8).

Two design decisions are pinned here because both are easy to get wrong later:

  * the excess is a DIFFERENCE, not a regression residual. Fitting beta over
    the whole sample scores slightly better (9.35 dimensions against 8.63) and
    uses the future to describe the past. The plain difference keeps 92% of
    the benefit with every bar computed from that bar alone.
  * dispersion is NOT residualised. It is already orthogonal to the market --
    8.09 dimensions before and after, mean correlation 0.213 against 0.214 --
    so removing a factor it does not contain would only add noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.peer_context_enricher import PeerContextEnricher

TICKERS = [
    "AAPL", "MSFT", "NVDA",      # tech_giants / additional_tech
    "JPM", "BAC", "GS",          # finance
    "XOM", "CVX", "COP",         # energy
    "JNJ", "PFE", "UNH",         # healthcare
]


@pytest.fixture
def panel() -> pd.DataFrame:
    """A panel with a deliberately strong common factor."""
    rng = np.random.default_rng(20260902)
    dates = pd.date_range("2024-01-01", periods=80, freq="B", tz="UTC")
    market = rng.normal(0, 0.012, len(dates))
    rows = []
    for ticker in TICKERS:
        returns = market + rng.normal(0, 0.006, len(dates))
        rows.append(pd.DataFrame({
            "ticker": ticker,
            "datetime": dates,
            "close": 100 * np.cumprod(1 + returns),
        }))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def enriched(panel) -> pd.DataFrame:
    out = PeerContextEnricher().enrich(panel)
    return out.dropna(subset=["peer_return", "market_return"])


def test_the_excess_is_the_difference_and_nothing_else(enriched):
    """Stated as an identity so a later 'improvement' to a fitted residual
    cannot slip in unnoticed -- that version looks at the future."""
    np.testing.assert_allclose(
        enriched["peer_return_excess"],
        enriched["peer_return"] - enriched["market_return"],
        rtol=1e-9, atol=1e-12,
    )
    np.testing.assert_allclose(
        enriched["peer_breadth_excess"],
        enriched["peer_breadth"] - enriched["market_breadth"],
        rtol=1e-9, atol=1e-12,
    )


def test_the_market_factor_is_actually_removed(enriched):
    raw = enriched["peer_return"].corr(enriched["market_return"])
    excess = enriched["peer_return_excess"].corr(enriched["market_return"])

    assert raw > 0.5, (
        f"the panel was built with a strong common factor but raw peer_return "
        f"correlates with the market at only {raw:.3f}; the fixture no longer "
        f"tests what it claims to"
    )
    assert abs(excess) < 0.15, (
        f"peer_return_excess still correlates with the market at {excess:.3f}; "
        f"the sector columns remain copies of one series"
    )


def test_a_bar_is_never_part_of_the_averages_that_describe_it(panel):
    """Leave-one-out, for the market as well as for the sector.

    Without it, a ticker's own move is inside the number that is supposed to
    say what everyone ELSE did, and `peer_divergence` -- the ticker minus its
    sector -- is shrunk toward zero by its own contribution.
    """
    out = PeerContextEnricher().enrich(panel)
    out = out.dropna(subset=["market_return"])
    one_date = out["datetime"].iloc[len(out) // 2]
    day = out[out["datetime"] == one_date]
    assert len(day) == len(TICKERS)

    # Each ticker's own move, recovered from what the enricher reports:
    # divergence is the ticker minus its sector, so adding the sector back
    # gives the move itself without re-deriving it from prices.
    moves = day.set_index("ticker")["peer_divergence"] + day.set_index("ticker")["peer_return"]
    for ticker in day["ticker"]:
        others = moves.drop(ticker)
        expected = others.mean()
        reported = day.loc[day["ticker"] == ticker, "market_return"].iloc[0]
        assert reported == pytest.approx(expected, abs=1e-9), (
            f"{ticker}'s market_return includes its own move"
        )


def test_dispersion_is_left_alone(enriched):
    """No `peer_volatility_excess`: it would subtract a factor that is not there."""
    assert "peer_volatility" in enriched.columns
    assert "peer_volatility_excess" not in enriched.columns, (
        "dispersion is already orthogonal to the market (8.09 dimensions "
        "before and after); residualising it adds noise and a column"
    )
