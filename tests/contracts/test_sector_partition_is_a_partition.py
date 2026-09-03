"""The sector layer's membership must be a partition, and stay one.

`assets.sectors` cannot serve as sector membership: it is two taxonomies
merged. Six hand-written entries are subsets of thirteen later auto-imported
ones, and `core` (32 names) is "large caps" rather than a sector, overlapping
almost everything. Measured 2026-09-02: 29 of 110 tickers appear in more than
one entry, and `ai_big_tech` shares six names of 6/7 with `tech_giants` --
their daily equal-weight returns correlate at 0.986 (REGISTER #225).

Why a partition and not just "some groups": the sector layer averages returns
within each group and then residualises them on the market. A ticker counted
in two groups is counted twice in the cross-section, and two groups that are
nearly the same set present one series as two sectors -- multiplicity with no
information, which is the worst trade available under the power arithmetic
this project now measures against (CLAIMS.md R8, R12).

`assets.sectors` is deliberately left alone: it is what the collector reads to
decide what to download. Changing it would change the universe, and that is a
different decision from how the universe is grouped.
"""
from __future__ import annotations

import collections
from pathlib import Path

import pytest
import yaml

CONFIG = Path(__file__).resolve().parents[2] / "src" / "config" / "assets.yaml"


@pytest.fixture(scope="module")
def assets() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["assets"]


@pytest.fixture(scope="module")
def partition(assets) -> dict[str, list[str]]:
    block = assets.get("sector_partition")
    assert block, "assets.sector_partition is missing"
    return {name: list(body["assets"]) for name, body in block.items()}


def test_no_ticker_belongs_to_two_sectors(partition):
    counts = collections.Counter(t for names in partition.values() for t in names)
    repeated = {t: n for t, n in counts.items() if n > 1}
    assert not repeated, (
        f"these tickers are in more than one sector, so the cross-section "
        f"counts them twice: {repeated}"
    )


def test_every_collected_ticker_has_a_sector(assets, partition):
    """A name with no sector is invisible to the layer and says so nowhere."""
    collected = {
        ticker
        for body in assets["sectors"].values()
        for ticker in body.get("assets", [])
    }
    assigned = {t for names in partition.values() for t in names}
    orphans = sorted(collected - assigned)
    assert not orphans, (
        f"{len(orphans)} collected ticker(s) belong to no sector in the "
        f"partition, so they contribute to the market factor but to no "
        f"sector: {orphans}"
    )


def test_the_partition_invents_no_tickers(assets, partition):
    collected = {
        ticker
        for body in assets["sectors"].values()
        for ticker in body.get("assets", [])
    }
    assigned = {t for names in partition.values() for t in names}
    unknown = sorted(assigned - collected)
    assert not unknown, (
        f"the partition names tickers that are not collected: {unknown}"
    )


def test_a_sector_has_enough_members_to_average(partition):
    """A one-name sector is that name, relabelled.

    Its equal-weight return IS the ticker's return, its breadth is 0 or 1, and
    its dispersion is undefined -- three columns that say what one column
    already said. `energy` was exactly this in `assets.sectors`: a single
    ticker, XOM.
    """
    thin = {name: len(members) for name, members in partition.items() if len(members) < 3}
    assert not thin, (
        f"these sectors have fewer than three members, so their aggregates "
        f"are not aggregates: {thin}"
    )
