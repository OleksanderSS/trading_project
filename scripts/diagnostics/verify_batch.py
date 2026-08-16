"""Ask the finished batch the questions it has failed before.

Every check here exists because the pipeline once shipped an export that looked
fine and was not. They are cheap, they read only files, and they are meant to
run immediately after a rebuild, before anything trains on the result.

    python scripts/diagnostics/verify_batch.py

Each check prints PASS or FAIL with the number behind it. The exit code is
non-zero if any check fails, so this can gate a retrain.
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BATCH = Path('data/colab/accumulated/main_database/features.parquet')
KEY = ['ticker', 'datetime', 'interval']


def _latest_stage2() -> dict[str, Path]:
    """The per-timeframe price files stage 3 was built from."""
    found: dict[str, Path] = {}
    for timeframe in ('15m', '60m', '1d'):
        hits = sorted(glob.glob(f'data/processed/prices_{timeframe}_*.parquet'))
        if hits:
            found[timeframe] = Path(hits[-1])
    return found


def main() -> int:
    if not BATCH.exists():
        print(f'no batch at {BATCH}')
        return 1

    failures = 0

    def report(ok: bool, label: str, detail: str) -> None:
        nonlocal failures
        if not ok:
            failures += 1
        print(f'{"PASS" if ok else "FAIL"}  {label:34s} {detail}')

    frame = pd.read_parquet(BATCH, columns=KEY + ['hash'])
    print(f'batch: {len(frame):,} rows, {frame["ticker"].nunique()} tickers\n')

    # 1. Every bar carries its own date. The failure this catches: nlp_features
    #    stamped bars with the publication time of the article they matched,
    #    and 24,143 of 26,295 15m bars came out on somebody else's date while
    #    row count, row order and every hash stayed perfect.
    for timeframe, path in _latest_stage2().items():
        truth = pd.read_parquet(path, columns=['hash', 'datetime'])
        truth['datetime'] = pd.to_datetime(
            truth['datetime'], utc=True, errors='coerce'
        ).dt.tz_localize(None)
        rows = frame[frame['interval'] == timeframe][['hash', 'datetime']]
        joined = rows.merge(
            truth.rename(columns={'datetime': 'own'}), on='hash', how='left'
        )
        matched = joined['own'].notna()
        if not matched.any():
            report(False, f'dates belong to their bar ({timeframe})',
                   'no hash matched stage 2 — batch and DB are out of step')
            continue
        share = float((joined.loc[matched, 'datetime'] == joined.loc[matched, 'own']).mean())
        report(share >= 1.0, f'dates belong to their bar ({timeframe})',
               f'{share:.2%} of {int(matched.sum()):,} bars')

    # 2. One row per bar. Colab merges features to targets with
    #    validate='one_to_one'; duplicates here become a crash there.
    duplicates = int(frame.duplicated(KEY).sum())
    report(duplicates == 0, 'no duplicate (ticker, datetime, interval)',
           f'{duplicates:,} duplicate rows')

    # 3. Distinct identities. Two hashes for one bar is how 540 duplicate AAPL
    #    bars survived deduplication.
    collisions = int(len(frame) - frame['hash'].nunique())
    report(collisions == 0, 'every row has a distinct hash', f'{collisions:,} repeats')

    # 4. Sources that were connected but never arrived. Each of these has
    #    reached the feature set at zero columns at least once while the run
    #    reported success.
    full = pd.read_parquet(BATCH)
    columns = full.columns
    for prefix, label in (
        ('econ_', 'economic calendar'),
        ('cftc_', 'CFTC positioning'),
        ('fear_greed_', 'fear & greed'),
        ('wiki_attention_', 'wikipedia attention'),
        ('insider_', 'insider filings'),
        ('peer_', 'peer context'),
        ('market_context_', 'market context'),
    ):
        matched = [c for c in columns if str(c).startswith(prefix)]
        # Arriving and being informative are different things, and the second
        # is the one that matters. Every source in this list has at some point
        # reached the batch as a column of one repeated value while the run
        # reported it connected, so the constant count is reported beside the
        # column count rather than instead of it.
        varying = [c for c in matched if full[c].nunique(dropna=True) > 1]
        detail = f'{len(matched)} columns, {len(matched) - len(varying)} constant'
        report(bool(matched), f'{label} reached the batch', detail)

    constant = int((full.nunique(dropna=True) <= 1).sum())
    print(f'\n      {len(columns):,} columns, {constant} constant')

    print(f'\n{failures} check(s) failed' if failures else '\nall checks passed')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
