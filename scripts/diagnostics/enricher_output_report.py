"""What each enricher actually produces, in seconds instead of fifty minutes.

Every feature defect found on 2026-08-12/13 was visible in the shape of the
output, not in the code:

    keyword_count       constant 0 on all 55,565 rows, at 32s per timeframe
    entity_count        constant 0, with spaCy loaded and working
    nlp_sentiment_score constant 0, after a spell as epoch nanoseconds
    sentiment_available constant 1.0, a flag that could not say no
    put_call_ratio      constant 1.0, a fabricated neutral

Each was found by rebuilding the batch — fifty minutes — and then reading
the parquet. None of them needed the rebuild. An enricher run over a few
hundred real bars produces the same shape as one run over fifty thousand,
because "this column is the same value everywhere" does not depend on how
many rows you have.

So this runs the enabled enrichers over a slice and reports, per enricher:
columns added, how many are constant, how many are entirely null, and how
long it took. A column that is constant here is worth looking at before the
next rebuild, not after it.

    python scripts/diagnostics/enricher_output_report.py [--rows 400] [--timeframe 15m]

News-dependent enrichers need the news frame. It is read from the batch's
own source when available; when the database is locked by a running
pipeline, those enrichers are reported as skipped rather than silently
producing nothing -- which is the failure this tool exists to catch.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings("ignore")

BATCH = PROJECT_ROOT / "data" / "colab" / "accumulated" / "main_database"
SERVICE = {"ticker", "datetime", "interval", "hash", "timestamp",
           "open", "high", "low", "close", "volume"}


def _load_bars(timeframe: str, rows: int) -> pd.DataFrame:
    """Real bars from the batch: the enrichers see what they normally see."""
    features = BATCH / "features.parquet"
    if not features.exists():
        raise SystemExit(f"no batch at {features}")
    frame = pd.read_parquet(
        features, columns=["ticker", "datetime", "interval",
                           "open", "high", "low", "close", "volume"]
    )
    frame = frame[frame["interval"] == timeframe]
    if frame.empty:
        raise SystemExit(f"batch has no {timeframe} rows")
    ticker = frame["ticker"].iloc[0]
    frame = frame[frame["ticker"] == ticker].tail(rows).reset_index(drop=True)
    return frame


def _load_news(limit: int = 4000) -> pd.DataFrame | None:
    try:
        from src.config.unified_config_manager import UnifiedConfigManager
        from src.data.management.data_manager import DataManager
        manager = DataManager(UnifiedConfigManager())
        for table in ("rss_news", "google_news", "news"):
            try:
                frame = manager.con.execute(
                    f'SELECT * FROM "{table}" LIMIT {limit}'
                ).fetch_df()
            except Exception:
                continue
            if not frame.empty:
                return frame
    except Exception as exc:                       # noqa: BLE001 - diagnostic
        print(f"  (news unavailable: {type(exc).__name__} — "
              f"a running pipeline holds the database)")
    return None


def _column_shapes(frame: pd.DataFrame, added: list[str]) -> dict[str, int]:
    constant, empty, live = 0, 0, 0
    dead_names = []
    for column in added:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            empty += 1
            dead_names.append(column)
        elif values.nunique(dropna=True) <= 1:
            constant += 1
            dead_names.append(column)
        else:
            live += 1
    return {"live": live, "constant": constant, "empty": empty,
            "dead_names": dead_names}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--verbose", action="store_true",
                        help="name every constant and empty column")
    args = parser.parse_args()

    from src.config.unified_config_manager import UnifiedConfigManager
    from src.features.feature_orchestrator import FeatureOrchestrator

    bars = _load_bars(args.timeframe, args.rows)
    print(f"{len(bars)} {args.timeframe} bars of {bars['ticker'].iloc[0]}\n")

    news = _load_news()
    print(f"news rows available: {0 if news is None else len(news)}\n")

    orchestrator = FeatureOrchestrator.create_from_config(UnifiedConfigManager())

    print(f"{'enricher':26s} {'+cols':>6s} {'live':>6s} {'const':>6s} "
          f"{'empty':>6s} {'seconds':>8s}")
    print("-" * 64)

    totals = {"live": 0, "constant": 0, "empty": 0}
    suspect = []
    for enricher in orchestrator.enrichers:
        frame = bars.copy()
        before = set(frame.columns)
        started = time.perf_counter()
        try:
            kwargs = {"news": news} if news is not None else {}
            result = enricher.enrich(frame, timeframe=args.timeframe, **kwargs)
        except Exception as exc:                   # noqa: BLE001 - diagnostic
            print(f"{enricher.name:26s} {'FAILED':>6s}  "
                  f"{type(exc).__name__}: {str(exc)[:40]}")
            continue
        elapsed = time.perf_counter() - started
        added = [c for c in result.columns if c not in before and c not in SERVICE]
        if not added:
            print(f"{enricher.name:26s} {0:6d} {'':>6s} {'':>6s} {'':>6s} "
                  f"{elapsed:8.1f}")
            continue

        shape = _column_shapes(result, added)
        for key in totals:
            totals[key] += shape[key]
        print(f"{enricher.name:26s} {len(added):6d} {shape['live']:6d} "
              f"{shape['constant']:6d} {shape['empty']:6d} {elapsed:8.1f}")
        if shape["live"] == 0 and added:
            suspect.append((enricher.name, len(added), elapsed))
        if args.verbose and shape["dead_names"]:
            for name in shape["dead_names"][:12]:
                print(f"    {name}")

    print("-" * 64)
    print(f"{'TOTAL':26s} {sum(totals.values()):6d} {totals['live']:6d} "
          f"{totals['constant']:6d} {totals['empty']:6d}")

    if suspect:
        print("\nProduced nothing that varies — every column constant or empty:")
        for name, count, elapsed in suspect:
            print(f"  {name}: {count} columns, {elapsed:.1f}s")
        print("\nThat is the shape keyword_entity had for months, at 32s per "
              "timeframe. Check these before the next rebuild, not after.")
        if news is None:
            # Said plainly, because a diagnostic that cries wolf gets ignored.
            # Without news, anything downstream of it is constant by
            # construction and this list cannot separate "broken" from
            # "starved". Re-run when no pipeline holds the database.
            print("\nNOTE: no news was available, so news-derived enrichers "
                  "are constant for want of input, not necessarily by defect. "
                  "Re-run when the database is free to judge them.")
        return 1

    print("\nEvery enricher produced at least one column that varies.")
    if news is None:
        print("NOTE: news was unavailable; news-derived enrichers were not "
              "exercised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
