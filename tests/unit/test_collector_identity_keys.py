"""An identity key must not be built out of what the row MEASURES.

Twice in one day a key made of values turned out to be a change detector
rather than an identity: the row changes, the hash changes, deduplication
cannot see the two copies, and the table quietly grows a second version of the
same fact.

  market_data_raw hashed a FORMATTED LOCAL-TIME STRING, so a bar's identity
  depended on the machine's timezone. 540 AAPL 60m rows stored twice — one
  instant written two ways.

  vix_data hashed (date, vix_close, volatility_regime). The regime is derived
  from a 20-day mean over a window whose START moves with the collection date,
  so the same trading day produced different statistics on each fetch. 22 of
  77 dates duplicated, one of them asserting that a day sat at both the 20th
  and the 80th percentile.

Both were found by hand. A hand sweep does not cover the collector nobody has
written yet, which is what this file is for.

WHAT THIS CANNOT DO, stated so nobody trusts it too far: no test can look at
the name of a column and know whether it is an identity or a measurement. So
the judgement is written down here, once, per column, with its reason — and
the tests enforce that the judgement STAYS MADE. A new collector fails until
someone classifies its key. A changed key fails until someone re-reads it.
That is the part a review cannot do, because reviews do not run again.

CONTENT IN A KEY IS NOT ALWAYS WRONG, and getting that backwards would be its
own defect. FRED deliberately keys on `value` alongside `realtime_start`,
because a revised figure IS a new fact and the pipeline wants both vintages so
consumers can read the series as it stood on any past date. The difference is
whether the second copy is wanted. `vintage` below means wanted and reasoned;
`content` means accidental.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COLLECTORS_YAML = Path('src/config/collectors.yaml')

IDENTITY = 'identity'   # what makes this row a distinct thing in the world
VINTAGE = 'vintage'     # deliberately stores a revision as a separate row
CONTENT = 'content'     # a measurement; its presence makes the key a change detector

#: Every column of every collector's identity key, classified once, on
#: purpose, with the reason it is classified that way.
KEY_CLASSIFICATION: dict[str, dict[str, tuple[str, str]]] = {
    'aaii_sentiment': {
        'date': (IDENTITY, 'the survey week'),
        'bullish': (CONTENT, 'a measured percentage; a corrected survey stores a second row'),
        'bearish': (CONTENT, 'same'),
        'neutral': (CONTENT, 'same, and it is 100 minus the other two — pure redundancy'),
    },
    'bigquery': {
        'date': (IDENTITY, 'the bar date'),
        'ticker': (IDENTITY, 'the instrument'),
    },
    'cftc': {
        'date': (IDENTITY, 'the report date'),
        'instrument': (IDENTITY, 'which contract'),
        'net_position': (CONTENT, 'the measurement itself. The COT report IS revised, '
                                  'and a revision would store a second row for one date'),
    },
    'economic_calendar': {
        'timestamp': (IDENTITY, 'when the release prints'),
        'country': (IDENTITY, 'whose figure'),
        'event': (IDENTITY, 'which figure'),
        'actual': (VINTAGE, 'deliberate, 2026-08-15. The feed is fetched before AND after '
                            'a release; keying on the timestamp alone dropped the '
                            'post-release fetch as a duplicate and every stored row had an '
                            'empty actual. Both snapshots are now kept'),
    },
    'fear_greed': {
        'date': (IDENTITY, 'the reading date'),
        'value': (CONTENT, 'the measurement; a revised index stores a second row'),
        'fear_greed_category': (CONTENT, 'derived FROM value, so it adds nothing the '
                                         'previous column has not already broken'),
    },
    'fred': {
        'series_id': (IDENTITY, 'which series'),
        'date': (IDENTITY, 'which observation period'),
        'realtime_start': (VINTAGE, 'the date this figure was PUBLISHED. This is a real '
                                    'vintage column, which is what makes keeping `value` '
                                    'below a decision rather than an accident'),
        'value': (VINTAGE, 'deliberate: a revised figure is a new fact, and point-in-time '
                           'reads need every vintage'),
    },
    'google_news': {
        'link': (IDENTITY, 'the article URL'),
    },
    'insider': {
        'filing_date': (IDENTITY, 'resolved to the second, which is what keeps two trades '
                                  'on one form distinct'),
        'ticker': (IDENTITY, 'the company'),
        'insider_name': (IDENTITY, 'who filed'),
    },
    'newsapi': {
        'url': (IDENTITY, 'the article'),
        'publishedAt': (IDENTITY, 'when it appeared'),
    },
    'put_call_ratio': {
        'date': (IDENTITY, 'the reading date'),
        'put_call_ratio': (CONTENT, 'the measurement itself'),
        'sentiment_signal': (CONTENT, 'derived FROM the ratio — the same shape that '
                                      'duplicated 22 of 77 VIX dates'),
    },
    'reddit_sentiment': {
        'date': (IDENTITY, 'the post date'),
        'subreddit': (IDENTITY, 'where'),
        'post_id': (IDENTITY, "reddit's own identifier — the right kind of key"),
    },
    'rss': {
        'link': (IDENTITY, 'the article URL'),
    },
    'sec_filings': {
        'accessionNumber': (IDENTITY, "the SEC's own filing identifier"),
        'cik': (IDENTITY, 'the filer'),
    },
    'vix': {
        'date': (IDENTITY, 'fixed 2026-08-16. Was (date, vix_close, volatility_regime); '
                           'the regime moved with the collection date and duplicated 22 '
                           'of 77 dates'),
    },
}

#: Collectors whose key carries an accidental measurement. Listed rather than
#: hidden, because each needs a MIGRATION and not merely an edit: every stored
#: row already carries the old hash, so changing the formula alone makes the
#: next collection re-hash the same record differently and store it AGAIN.
#: That is worse than the bug. See scripts/maintenance/rehash_market_bars.py
#: for the shape of the fix.
KNOWN_CONTENT_IN_KEY = {
    'aaii_sentiment': 'collector disabled — the site answers 403',
    'cftc': 'LIVE and the real risk here: COT reports are revised',
    'fear_greed': 'LIVE; the host is also gone, so it is not currently collecting',
    'put_call_ratio': 'collector disabled — blocked at the source',
}


def _configured_keys() -> dict[str, list[str]]:
    raw = yaml.safe_load(COLLECTORS_YAML.read_text(encoding='utf-8')) or {}
    found: dict[str, list[str]] = {}
    for name, cfg in (raw.get('collectors') or {}).items():
        if isinstance(cfg, dict) and cfg.get('hash_keys'):
            found[name] = list(cfg['hash_keys'])
    return found


CONFIGURED = _configured_keys()


def test_the_config_actually_declares_keys():
    # If this drops to nothing, every test below passes vacuously.
    assert len(CONFIGURED) >= 14


@pytest.mark.parametrize('collector', sorted(CONFIGURED))
def test_every_collector_key_has_been_classified(collector):
    """A new collector fails here until someone has read its key."""
    assert collector in KEY_CLASSIFICATION, (
        f"{collector} declares hash_keys but nobody has classified them. "
        f"Add each column to KEY_CLASSIFICATION as identity, vintage or content, "
        f"with the reason. Keys made of measurements have duplicated this "
        f"project's data twice."
    )


@pytest.mark.parametrize('collector', sorted(CONFIGURED))
def test_every_column_in_a_live_key_has_been_classified(collector):
    """A changed key fails here until someone has re-read it."""
    classified = set(KEY_CLASSIFICATION.get(collector, {}))
    live = set(CONFIGURED[collector])
    assert live <= classified, (
        f"{collector}'s identity key gained {sorted(live - classified)} since it was "
        f"last reviewed. Classify the new column before shipping it: if it is a "
        f"measurement, the key has become a change detector."
    )


@pytest.mark.parametrize('collector', sorted(
    c for c in CONFIGURED if c not in KNOWN_CONTENT_IN_KEY))
def test_no_identity_key_is_built_from_a_measurement(collector):
    """The invariant itself, for every collector not already known to break it."""
    offenders = [
        column for column in CONFIGURED[collector]
        if KEY_CLASSIFICATION.get(collector, {}).get(column, (CONTENT, ''))[0] == CONTENT
    ]
    assert not offenders, (
        f"{collector} keys on {offenders}, which the registry classifies as a "
        f"measurement. A key built from values is a change detector: the row "
        f"changes, the hash changes, dedup stores a second copy. Either the column "
        f"is really a vintage — say so and give the reason — or it does not belong "
        f"in the key."
    )


@pytest.mark.parametrize('collector', sorted(KNOWN_CONTENT_IN_KEY))
@pytest.mark.xfail(strict=True, reason='known content-in-key; each needs a migration, '
                                       'not merely an edit — see KNOWN_CONTENT_IN_KEY')
def test_known_offenders_still_offend(collector):
    """Strict xfail, so fixing one of these FAILS until the registry is updated.

    The point is that a fix must not silently leave a stale waiver behind — the
    stale waiver is how the next reader concludes the problem is still open and
    re-investigates it. Three items were re-investigated that way in one day.
    """
    offenders = [
        column for column in CONFIGURED[collector]
        if KEY_CLASSIFICATION.get(collector, {}).get(column, (CONTENT, ''))[0] == CONTENT
    ]
    assert not offenders


def test_a_vintage_column_carries_a_written_reason():
    """`vintage` is the escape hatch, so it must never be cheap to use."""
    for collector, columns in KEY_CLASSIFICATION.items():
        for column, (kind, reason) in columns.items():
            if kind != VINTAGE:
                continue
            assert len(reason) > 40, (
                f"{collector}.{column} is waived as a vintage on a one-line reason. "
                f"Storing a second copy of a fact is a deliberate decision and has to "
                f"read as one."
            )


def test_the_registry_does_not_describe_collectors_that_no_longer_exist():
    stale = set(KEY_CLASSIFICATION) - set(CONFIGURED)
    assert not stale, (
        f"{sorted(stale)} are classified here but declare no hash_keys in "
        f"{COLLECTORS_YAML}. A registry describing something that is gone is how a "
        f"note outlives the thing it described."
    )
