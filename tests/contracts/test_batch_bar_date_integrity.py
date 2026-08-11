"""Every exported row must carry the date of the bar it actually contains.

This is a RECONCILIATION test: it compares a produced artifact against the
source of truth it was built from, rather than checking that code ran.

It exists because the defect it guards against was invisible to every other
kind of check. On 2026-08-06 Stage 3 produced a batch in which all 327 AAPL
daily rows held genuine bars from the database — correct OHLCV, correct
indicators, correct calendar features — attached to OTHER DAYS' timestamps,
with offsets up to 686 days and not one row correct. Nothing raised. Row
counts were right, no NaNs appeared, and a later sort by the corrupted column
left the file looking perfectly orderly. Line coverage of the responsible
function was total: it ran on every batch and did exactly what it was written
to do, which was to copy a column by POSITION.

What made it visible was asking whether the numbers still meant what their
names said. That question is what this test automates.

Skips when the artifacts are absent, so a fresh clone or a CI box without the
data directory does not fail; it is a guard for real runs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BATCH = Path('data/colab/accumulated/main_database/features.parquet')
DB = Path('data/trading_data.duckdb')

#: A daily equity return standard deviation above this is not a market move,
#: it is a broken series. The corrupted batch showed 0.21 for AAPL and 1.36
#: for INTC; healthy values sit near 0.016.
MAX_PLAUSIBLE_DAILY_STD = 0.10


def _load():
    duckdb = pytest.importorskip('duckdb')
    if not BATCH.exists() or not DB.exists():
        pytest.skip(f"no batch/db to reconcile ({BATCH}, {DB})")

    exported = pd.read_parquet(
        BATCH, columns=['ticker', 'interval', 'datetime', 'close', 'hash']
    )
    try:
        con = duckdb.connect(str(DB), read_only=True)
    except duckdb.IOException as e:
        # A running pipeline holds the database. "Cannot check right now" is
        # not "the check failed" — reporting it as a failure would make this
        # test flaky against the very runs it is meant to police.
        pytest.skip(f"database is locked by another process: {e}")
    try:
        raw = con.execute(
            'select hash, datetime from market_data_raw where hash is not null'
        ).df()
    finally:
        con.close()
    return exported, raw


def test_every_exported_row_carries_its_own_bars_date():
    exported, raw = _load()
    if exported.empty or raw.empty:
        pytest.skip("batch or market_data_raw is empty")

    real_by_hash = raw.drop_duplicates('hash').set_index('hash')['datetime']
    declared = pd.to_datetime(exported['datetime'], utc=True)
    real = pd.to_datetime(exported['hash'].map(real_by_hash), utc=True)

    matched = real.notna()
    assert matched.any(), (
        "no exported row could be matched to a raw bar by hash — the batch "
        "cannot be reconciled with its source at all"
    )

    wrong = matched & (declared != real)
    if wrong.any():
        offsets = (declared[wrong] - real[wrong]).dt.days
        sample = exported.loc[wrong, ['ticker', 'interval']].head(3).to_dict('records')
        pytest.fail(
            f"{int(wrong.sum())} of {int(matched.sum())} exported rows carry a "
            f"date that belongs to a different bar. Offsets range "
            f"{offsets.min()}..{offsets.max()} days. Examples: {sample}. "
            f"A column was reattached by position instead of by identity."
        )


def test_daily_returns_are_physically_plausible():
    """A shuffled series betrays itself through impossible volatility."""
    exported, _ = _load()
    daily = exported[exported['interval'] == '1d']
    if daily.empty:
        pytest.skip("no daily rows in batch")

    offenders = {}
    for ticker, group in daily.sort_values('datetime').groupby('ticker'):
        returns = group['close'].pct_change().dropna()
        if len(returns) < 20:
            continue
        std = float(returns.std())
        if std > MAX_PLAUSIBLE_DAILY_STD:
            offenders[ticker] = round(std, 4)

    assert not offenders, (
        f"daily return volatility is impossible for {len(offenders)} ticker(s): "
        f"{offenders}. Healthy equities sit near 0.01-0.03; values this high "
        f"mean consecutive rows are not consecutive bars."
    )
