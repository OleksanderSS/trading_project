"""The calendar enricher existed and had never run once.

Four reasons, and only the first was visible:

  * it declared neither `name` nor `priority`, so the class stayed abstract
    and FeatureOrchestrator could not instantiate it — which is why it is
    absent from features.yaml rather than merely switched off;
  * `__init__` forwarded **kwargs to BaseEnricher, which accepts none;
  * it read the calendar from a `db_manager` the orchestrator never passes,
    so with the first two repaired it would still have logged "No db_manager"
    and returned the bars untouched;
  * and the surprise was standardised over the WHOLE series —
    `transform(lambda x: (x - x.mean()) / x.std())` uses a mean and deviation
    computed from every row including the future.

That last one is the dangerous one. A leak of that shape is identical in
every fold, so walk-forward validation cannot see it; it simply raises the
score everywhere and reads as skill.

Underneath all four sat a fifth, in the collector: `actual` was empty on all
147 stored rows while 101 carried a forecast, because hash_keys was
(timestamp, country, event). An event is fetched before its release with
actual blank, and the post-release fetch hashed identically and was dropped as
a duplicate. Surprise is actual minus forecast, so it could not be computed at
all. `actual` is now part of the key and both snapshots are kept — the same
decision already taken for FRED vintages.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.enrichers.economic_calendar_enricher import (
    EconomicCalendarEnricher,
)


@pytest.fixture
def enricher():
    return EconomicCalendarEnricher()


def _calendar(n: int = 12) -> pd.DataFrame:
    actual = ["0.2%", "0.3%", "0.1%", "0.4%", "0.2%", "0.5%",
              "0.3%", "0.2%", "0.6%", "0.1%", "0.3%", "0.9%"][:n]
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=n, freq="7D", tz="UTC"),
        "country": ["us"] * n,
        "impact": ["High"] * n,
        "event": ["CPI m/m"] * n,
        "actual": actual,
        "forecast": ["0.3%"] * n,
        "previous": [""] * n,
    })


def _bars(n: int = 60) -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "datetime": pd.date_range("2026-07-01", periods=n, freq="D", tz="UTC"),
        "close": np.linspace(100, 120, n),
    })


def test_the_class_can_actually_be_instantiated(enricher):
    """Without name and priority it is abstract, and discovery skips it."""
    assert enricher.name == "economic_calendar"
    assert isinstance(enricher.priority, int)


def test_the_surprise_reaches_the_bars(enricher):
    enriched = enricher._enrich_impl(_bars(), economic_calendar=_calendar())

    surprise = pd.to_numeric(enriched["econ_surprise_index"], errors="coerce")
    assert surprise.notna().any()
    assert surprise.nunique() > 1, "a constant surprise index is no surprise"


def test_standardisation_uses_only_what_came_before(enricher):
    """Truncating the history must not change an earlier row's figure.

    A whole-series mean and deviation fails this outright, and that is what
    the enricher did.
    """
    full = enricher._daily_surprise(_calendar(12))
    prefix = enricher._daily_surprise(_calendar(8))

    overlap = full.head(len(prefix)).reset_index(drop=True)
    assert np.allclose(
        overlap["surprise_index"].fillna(-9).to_numpy(),
        prefix["surprise_index"].fillna(-9).to_numpy(),
    ), "a later release changed the figure an earlier bar carried"


def test_nothing_is_surprising_before_there_is_a_baseline(enricher):
    """Two observations cannot say what is normal for an event."""
    short = _calendar(2)

    daily = enricher._daily_surprise(short)

    assert daily["surprise_index"].isna().all() or daily.empty


def test_no_release_is_distinguishable_from_a_release_on_forecast(enricher):
    """The old code forward-filled and then filled 0, conflating the two."""
    calendar = _calendar(12)
    # Bars start a month before the first release.
    early = _bars(90)
    early["datetime"] = pd.date_range("2026-06-01", periods=90, freq="D", tz="UTC")

    enriched = enricher._enrich_impl(early, economic_calendar=calendar)

    before = enriched.loc[
        pd.to_datetime(enriched["datetime"]).dt.tz_localize(None)
        < pd.Timestamp("2026-07-01"),
        "econ_calendar_available",
    ]
    assert (before == 0).all(), "bars before any release must say so"


def test_percentages_and_suffixed_numbers_are_parsed(enricher):
    for text, expected in (("1.2%", 1.2), ("150K", 150_000.0),
                           ("2.5M", 2_500_000.0), ("-3B", -3_000_000_000.0),
                           ("1,250", 1250.0)):
        assert enricher._to_number(text) == pytest.approx(expected)
    assert np.isnan(enricher._to_number(""))
    assert np.isnan(enricher._to_number("n/a"))


def test_a_calendar_without_actuals_produces_nothing_rather_than_zeros(enricher):
    """The live table's exact shape: forecasts present, actuals blank."""
    calendar = _calendar()
    calendar["actual"] = ""

    enriched = enricher._enrich_impl(_bars(), economic_calendar=calendar)

    assert "econ_surprise_index" not in enriched.columns, (
        "no actual means no surprise; inventing 0 would read as 'on forecast'"
    )


def test_the_collector_key_keeps_both_snapshots():
    """Pre-release and post-release are two facts, as with FRED vintages."""
    import yaml

    with open("src/config/collectors.yaml", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    node = config.get("collectors", config)["economic_calendar"]

    assert "actual" in node["hash_keys"], (
        "without it the post-release fetch hashes identically to the "
        "pre-release one and is dropped as a duplicate"
    )
