"""No ticker may land in a catch-all bucket and be reported as sector context.

`PeerContextEnricher` mapped tickers with `.fillna("other")`, and its built-in
groups named 31 of the 110 collected tickers. Measured on the 2026-08-29
batch: 79 tickers -- 72% of the universe, 69% of daily rows -- had "other" as
their sector, with a median of 65 neighbours. For those rows `peer_return` is
the market's move rather than a sector's, `peer_breadth` is market breadth,
and `peer_divergence` is ticker-minus-market.

Those are real quantities. The defect is that they travel under names that
claim to be sector context, and no coverage number showed it: every ticker had
values, on 99.9% of rows. Presence is not correctness, and this is the third
time in this audit that a fully-populated column turned out to be about
something other than its name (REGISTER #225).

Membership now comes from `assets.sector_partition`, which a contract test
holds to being a partition of the whole universe.
"""
from __future__ import annotations

import collections
from pathlib import Path

import pytest
import yaml

from src.features.enrichers.peer_context_enricher import (
    UNMAPPED_SECTOR,
    PeerContextEnricher,
)

CONFIG = Path(__file__).resolve().parents[2] / "src" / "config" / "assets.yaml"


@pytest.fixture(scope="module")
def collected() -> set[str]:
    assets = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["assets"]
    return {
        str(ticker).strip().upper()
        for body in assets["sectors"].values()
        for ticker in body.get("assets", [])
    }


def test_every_collected_ticker_is_mapped(collected):
    enricher = PeerContextEnricher()
    unmapped = sorted(collected - set(enricher.sector_of))
    assert not unmapped, (
        f"{len(unmapped)} of {len(collected)} collected tickers have no "
        f"sector, so their peer context would be the catch-all bucket and "
        f"would describe the market while being named for a sector: {unmapped}"
    )


def test_no_sector_swallows_most_of_the_universe(collected):
    """The bucket was 79 of 110. A sector that large is the market."""
    enricher = PeerContextEnricher()
    sizes = collections.Counter(enricher.sector_of.values())
    largest, count = sizes.most_common(1)[0]
    assert count <= len(collected) // 3, (
        f"sector {largest!r} holds {count} of {len(collected)} tickers; at "
        f"that size its aggregate is the market's, not a sector's"
    )


def test_the_catch_all_name_is_unused(collected):
    enricher = PeerContextEnricher()
    assert UNMAPPED_SECTOR not in set(enricher.sector_of.values()), (
        f"{UNMAPPED_SECTOR!r} is assigned to a ticker as though it were a "
        "sector; it exists only to name the absence of one"
    )


def test_every_sector_can_average(collected):
    """Peer aggregates exclude the ticker itself, so a two-name sector leaves
    one neighbour and a dispersion of zero."""
    enricher = PeerContextEnricher()
    sizes = collections.Counter(enricher.sector_of.values())
    thin = {name: n for name, n in sizes.items() if n < 3}
    assert not thin, (
        f"these sectors leave fewer than two neighbours once the ticker "
        f"itself is excluded, so their dispersion is degenerate: {thin}"
    )
