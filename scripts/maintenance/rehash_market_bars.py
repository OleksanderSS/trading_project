"""Give every stored bar the identity it should have had, and drop the copies.

`market_data_raw` rows were hashed from a FORMATTED LOCAL-TIME STRING, so the
same bar collected under a different timezone -- or labelled `1h` on one run
and `60m` on another -- was stored under two identities. `filter_new_records`
deduplicates on exactly that value, so it could not see them as the same row.
`bar_identity_hash` now hashes the instant instead.

That change alone is not enough and would make things worse if applied alone:
every bar already in the table carries an old hash, so the next collection
would recompute a new one for the same bar and store it AGAIN. This script is
the other half. Run it before the next collection.

    python scripts/maintenance/rehash_market_bars.py            # look only
    python scripts/maintenance/rehash_market_bars.py --apply    # rewrite

Dry run by default: it reports what would change and touches nothing. `--apply`
copies the table to `market_data_raw_prehash_<timestamp>` first, so the old
state stays recoverable.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.collectors.yf_collector import bar_identity_hash  # noqa: E402

TABLE = 'market_data_raw'
DEFAULT_DB = 'data/trading_data.duckdb'


def _recompute(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: bar_identity_hash(row['datetime'], row['ticker'], row['interval']),
        axis=1,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', default=DEFAULT_DB)
    ap.add_argument('--apply', action='store_true',
                    help='rewrite the table (default: report only)')
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f'no database at {db_path}')
        return 1

    try:
        con = duckdb.connect(str(db_path))
    except duckdb.IOException as exc:
        print(f'database is held by another process, try again later:\n  {exc}')
        return 1

    with con:
        frame = con.execute(f'SELECT * FROM {TABLE}').fetchdf()
        print(f'{TABLE}: {len(frame):,} rows')

        old = frame['hash'].astype(str)
        new = _recompute(frame)

        changed = int((old != new).sum())
        # The bars this was written for: identical rows that two different
        # hashes kept apart. Counted on the NEW hash, because that is the
        # identity they should always have shared.
        collapsed = int(new.duplicated().sum())
        print(f'  hashes that change: {changed:,}')
        print(f'  duplicate bars this reveals: {collapsed:,}')

        if collapsed:
            marked = frame.assign(_new=new)
            dupes = marked[marked['_new'].duplicated(keep=False)]
            per_ticker = dupes.groupby('ticker').size().sort_values(ascending=False)
            print('  by ticker: ' + ', '.join(
                f'{t} {n}' for t, n in per_ticker.head(6).items()))
            print(f"  date range: {dupes['datetime'].min()} .. {dupes['datetime'].max()}")

        if not args.apply:
            print('\ndry run — nothing written. Re-run with --apply to rewrite.')
            return 0

        backup = f'{TABLE}_prehash_{datetime.now():%Y%m%d_%H%M%S}'
        con.execute(f'CREATE TABLE {backup} AS SELECT * FROM {TABLE}')
        print(f'\nbacked up to {backup}')

        # Keep the first occurrence of each identity. The duplicates are
        # byte-identical apart from the hash -- verified on the pair that
        # started this -- so which one survives does not matter, only that one
        # does.
        cleaned = (
            frame.assign(hash=new)
            .drop_duplicates(subset=['hash'], keep='first')
            .reset_index(drop=True)
        )
        con.register('cleaned_bars', cleaned)
        con.execute(f'DROP TABLE {TABLE}')
        con.execute(f'CREATE TABLE {TABLE} AS SELECT * FROM cleaned_bars')
        con.unregister('cleaned_bars')

        print(f'{TABLE} rewritten: {len(frame):,} -> {len(cleaned):,} rows')

        check = con.execute(
            f'SELECT count(*) - count(DISTINCT hash) FROM {TABLE}'
        ).fetchone()[0]
        if check:
            print(f'STILL DUPLICATED: {check} — investigate before collecting')
            return 1
        print('every stored bar now has a distinct identity')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
