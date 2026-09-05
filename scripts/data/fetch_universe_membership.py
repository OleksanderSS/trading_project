"""Fetch which symbols existed when, including the ones that stopped.

REGISTER #169. `assets.yaml` holds today's names, and every measurement so far
carried that list back through thirty years. On our own panel that means ZERO
of 110 names ever stopped trading -- so a permanent long in the survivors pays
by construction, and Р41 measured that as more than half of a fitted
cross-sectional book's Sharpe.

Alpha Vantage's LISTING_STATUS returns, per symbol, `ipoDate` and
`delistingDate` -- exactly the membership interval. Verified 2026-09-05 by
calling it: the public demo key returned 14,411 active symbols, and 425
delisted ones for the single date the documentation uses.

WHAT EACH KEY CAN DO, measured rather than assumed, because a partial store
that looks complete is worse than no store:

                                 demo key            personal key
    active, no date              14,411 rows         14,411 rows
    delisted, NO date            REFUSED, `{}`       **9,458 rows,
                                                     1997-04-01..2026-09-04**
    delisted, date=2014-07-10    425 rows            425 rows
    delisted, any other date     REFUSED, `{}`       REFUSED, `{}`

The row that matters is the second one. With a personal key the WHOLE
delisting history arrives in a single request, so the `date` parameter is not
needed at all and a sweep of dates -- which the free tier's 25 requests a day
would have rationed -- was never necessary. Finding that out cost one request.

Without a key the store covers one delisting date: enough to build and test
the mechanism, not enough to measure with. `source_detail` records which it
is, so nobody later mistakes one for the other.

    python scripts/data/fetch_universe_membership.py --dry-run
    python scripts/data/fetch_universe_membership.py
    python scripts/data/fetch_universe_membership.py --dates 2005-01-03 2010-01-04
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

import pandas as pd  # noqa: E402

# The project's own .env loader, not a second copy of one. Without this the
# key sits in .env and `os.environ` does not have it, so the script silently
# takes the demo path and writes a store covering ONE delisting date while
# reporting success -- which is precisely the shape `source_detail` exists to
# make visible, reached by a different road.
from src.core.security.secure_secrets_manager import load_dotenv  # noqa: E402
from src.data.universe_membership import (  # noqa: E402
    COLUMNS, DEFAULT_STORE, coverage, save,
)

load_dotenv()

ENDPOINT = "https://www.alphavantage.co/query"

#: The one date the demo key serves, from Alpha Vantage's own documentation.
#: Named rather than inlined so the fallback path is visible in the file.
DEMO_DATE = "2014-07-10"

#: Free keys are rate limited. One request every fifteen seconds keeps a long
#: date sweep inside the published allowance without needing to know the exact
#: number, which changes.
PAUSE_SECONDS = 15.0


def _get(params: dict[str, str]) -> str:
    query = "&".join(f"{key}={value}" for key, value in params.items())
    request = urllib.request.Request(
        f"{ENDPOINT}?{query}",
        headers={"User-Agent": "trading-project universe membership fetch"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def _as_frame(raw: str, source_detail: str, fetched_at: pd.Timestamp) -> pd.DataFrame:
    """CSV to the store's columns, or an empty frame when the API refused.

    The refusal is a JSON object rather than an HTTP error, so a caller that
    only checked the status code would file `{}` as a successful fetch of zero
    symbols -- "could not measure" read as "measured, nothing there", which is
    REGISTER #202 in a new place.
    """
    first_line = raw.split("\n", 1)[0].strip()
    if not first_line.lower().startswith("symbol,"):
        return pd.DataFrame(columns=list(COLUMNS))

    frame = pd.read_csv(io.StringIO(raw))
    out = pd.DataFrame({
        "ticker": frame["symbol"].astype(str),
        "name": frame.get("name", pd.Series(dtype=str)).astype(str),
        "exchange": frame.get("exchange", pd.Series(dtype=str)).astype(str),
        "asset_type": frame.get("assetType", pd.Series(dtype=str)).astype(str),
        "ipo_date": pd.to_datetime(frame.get("ipoDate"), errors="coerce", utc=True),
        "delisting_date": pd.to_datetime(frame.get("delistingDate"),
                                         errors="coerce", utc=True),
        "status": frame.get("status", pd.Series(dtype=str)).astype(str),
        "source": "alphavantage:LISTING_STATUS",
        "source_detail": source_detail,
        "fetched_at": fetched_at,
    })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument("--dates", nargs="*", default=None,
                        help="dates to ask for delisted symbols as of; with no "
                             "key only the demo date is served")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    key = os.environ.get("ALPHAVANTAGE_API_KEY", "").strip()
    if key:
        # WITH A REAL KEY THE `date` PARAMETER IS NOT NEEDED, and finding that
        # out cost one request rather than a sweep. Asking for delisted
        # symbols with NO date returns the whole history in one call --
        # measured 2026-09-05: 9,458 rows spanning 1997-04-01 to 2026-09-04.
        # Asking with a recent date returns `{}`.
        #
        # So the loop below runs zero times unless --dates is given
        # explicitly, and the free tier's 25 requests a day are spent on two.
        detail = "personal key"
        dates = args.dates or []
    else:
        detail = (f"demo key -- delisted symbols available only for {DEMO_DATE}; "
                  "set ALPHAVANTAGE_API_KEY for other dates")
        key = "demo"
        if args.dates and args.dates != [DEMO_DATE]:
            print(f"no ALPHAVANTAGE_API_KEY set, so --dates is ignored: the "
                  f"demo key serves only {DEMO_DATE}")
        dates = [DEMO_DATE]

    print(f"key: {detail}")
    print(f"delisted snapshots to request: {dates}")

    fetched_at = pd.Timestamp.utcnow()
    parts: list[pd.DataFrame] = []

    active = _as_frame(_get({"function": "LISTING_STATUS", "apikey": key}),
                       detail, fetched_at)
    print(f"  active symbols: {len(active):,}")
    if len(active) and detail == "personal key":
        time.sleep(PAUSE_SECONDS)
        whole = _as_frame(
            _get({"function": "LISTING_STATUS", "state": "delisted",
                  "apikey": key}),
            f"{detail}; delisted, whole history in one call", fetched_at)
        if len(whole):
            print(f"  delisted, whole history: {len(whole):,}")
            parts.append(whole)
        else:
            print("  delisted, whole history: REFUSED -- falling back to "
                  "dated snapshots")
            dates = dates or [DEMO_DATE]
    if len(active):
        parts.append(active)
    else:
        print("  REFUSED -- the API returned no CSV for the active list. "
              "Nothing is written; an empty store would read as 'no name ever "
              "existed'.")
        return 1

    for index, when in enumerate(dates):
        if index:
            time.sleep(PAUSE_SECONDS)
        raw = _get({"function": "LISTING_STATUS", "date": when,
                    "state": "delisted", "apikey": key})
        block = _as_frame(raw, f"{detail}; delisted as of {when}", fetched_at)
        if len(block):
            print(f"  delisted as of {when}: {len(block):,}")
            parts.append(block)
        else:
            print(f"  delisted as of {when}: REFUSED (the API returned "
                  f"{raw.strip()[:40]!r}) -- recorded as absent, not as zero")

    frame = pd.concat(parts, ignore_index=True)
    # A symbol can appear in several snapshots. Keep the row that knows the
    # most: a delisting date beats no delisting date.
    frame = (frame.sort_values("delisting_date", na_position="first")
                  .drop_duplicates("ticker", keep="last")
                  .reset_index(drop=True))

    print()
    print(coverage(frame).describe())

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0

    save(frame, args.store)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
