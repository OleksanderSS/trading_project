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

Enrichers that need news or macro data get both, read from the database.
When a running pipeline holds it, the report names what was missing instead
of blaming the enricher -- the first version passed only news and duly
accused macro_features of 29 empty columns it had never been given the data
to fill, which is exactly the mistake this tool exists to catch.
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


def _load_source_tables(limit: int = 20000) -> tuple[pd.DataFrame | None,
                                                     pd.DataFrame | None]:
    """News and macro frames, or None when the database is held elsewhere.

    Both matter. The first version of this script passed only news, so
    macro_features reported 29 empty columns of 45 — and every one of them
    was a FRED series the enricher never received. That is the same mistake
    this tool exists to catch: an input that was never supplied, counted as
    an output that failed. Whatever is missing has to be named.
    """
    news = macro = None
    try:
        from src.config.unified_config_manager import UnifiedConfigManager
        from src.data.management.data_manager import DataManager
        manager = DataManager(UnifiedConfigManager())
    except Exception as exc:                       # noqa: BLE001 - diagnostic
        print(f"  (database unavailable: {type(exc).__name__} — "
              f"a running pipeline holds it)")
        return None, None

    for table in ("rss_news", "google_news", "news"):
        try:
            frame = manager.con.execute(
                f'SELECT * FROM "{table}" LIMIT {limit}'
            ).fetch_df()
        except Exception:
            continue
        if not frame.empty:
            news = frame
            break

    try:
        macro = manager.con.execute(
            f'SELECT * FROM "fred_data" LIMIT {limit}'
        ).fetch_df()
        if macro.empty:
            macro = None
    except Exception:
        macro = None

    return news, macro


def _column_shapes(frame: pd.DataFrame, added: list[str]) -> dict[str, int]:
    """Judge a column on its own terms.

    The first version ran every column through `pd.to_numeric` and called
    the result empty when it was all NaN. That is right for numbers and
    wrong for the categorical features this pipeline produces:
    MARKET_REGIME, volatility_regime, context_fingerprint and
    context_pattern_seq are strings, so they coerced to NaN and were
    reported dead while varying perfectly well. Three of the five columns
    the report first flagged were its own mistake.
    """
    constant, empty, live = 0, 0, 0
    dead_names = []
    for column in added:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series):
            values = series
        else:
            coerced = pd.to_numeric(series, errors="coerce")
            # Numeric-looking strings stay numeric; anything else is judged
            # as a category, where "how many distinct values" is the whole
            # question anyway.
            values = coerced if coerced.notna().any() else series

        non_null = values.notna().sum()
        if non_null == 0:
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

    news, macro = _load_source_tables()

    # Score the news the way Stage 3 does before any enricher sees it.
    # Without this the report ran the sentiment enricher against a corpus
    # whose `sentiment` column is 15,000 empty strings and duly reported
    # "0 columns" -- accusing the enricher of the very defect that was fixed
    # by moving scoring into the stage. A diagnostic that does not reproduce
    # the pipeline's own preparation measures a different pipeline.
    if news is not None:
        from src.pipeline.stages.stage_3_feature_engineering import (
            FeatureEngineeringStage,
        )
        news = FeatureEngineeringStage._score_news_sentiment(news)

    missing = [name for name, frame in (("news", news), ("macro", macro))
               if frame is None]
    print(f"news rows: {0 if news is None else len(news)} | "
          f"macro rows: {0 if macro is None else len(macro)}\n")

    orchestrator = FeatureOrchestrator.create_from_config(UnifiedConfigManager())

    print(f"{'enricher':26s} {'+cols':>6s} {'live':>6s} {'const':>6s} "
          f"{'empty':>6s} {'seconds':>8s}")
    print("-" * 64)

    totals = {"live": 0, "constant": 0, "empty": 0}
    suspect = []

    # Chain them, the way the pipeline does. Run in isolation, an enricher
    # that consumes an earlier one's column sees nothing: advanced_analytics
    # needs `nlp_sentiment_score`, which nlp_features adds, and reported "0
    # columns" here while working perfectly in a real run. Measuring each
    # enricher against the frame it will actually receive is the only reading
    # that transfers.
    accumulated = bars.copy()
    for enricher in orchestrator.enrichers:
        frame = accumulated.copy()
        before = set(frame.columns)
        started = time.perf_counter()
        try:
            kwargs = {}
            if news is not None:
                kwargs["news"] = news
            if macro is not None:
                kwargs["macro_data"] = macro
            result = enricher.enrich(frame, timeframe=args.timeframe, **kwargs)
        except Exception as exc:                   # noqa: BLE001 - diagnostic
            print(f"{enricher.name:26s} {'FAILED':>6s}  "
                  f"{type(exc).__name__}: {str(exc)[:40]}")
            continue
        elapsed = time.perf_counter() - started

        # Apply the orchestrator's own post-processing. Calling `enrich`
        # directly skips it, and skipping the row-label restoration let
        # duplicate labels reach market_context here -- "cannot reindex on an
        # axis with duplicate labels" -- in a pipeline where the real run has
        # no such problem. A diagnostic that runs enrichers differently from
        # production reports on a pipeline that does not exist.
        result = FeatureOrchestrator._restore_input_row_order(
            enricher, frame, result)
        result = FeatureOrchestrator._restore_input_row_labels(
            enricher, frame, result)

        added = [c for c in result.columns if c not in before and c not in SERVICE]
        # Carry the enriched frame forward, exactly as FeatureOrchestrator
        # does, so the next enricher sees what it will really be handed.
        accumulated = result
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

    if totals["constant"]:
        # A slice is a slice. `quarter` is constant across 300 fifteen-minute
        # bars because they fall in one quarter, not because it is broken.
        # Raise --rows before believing a slow-moving feature is dead.
        print("\nA constant over a short slice can be the slice: features that "
              "move slowly (quarter, month, regime) need --rows large enough "
              "to contain a change.")

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
            print(f"\nNOTE: {' and '.join(missing)} unavailable, so enrichers "
                  f"derived from them are empty for want of input, not "
                  f"necessarily by defect. Re-run when the database is free.")
        return 1

    print("\nEvery enricher produced at least one column that varies.")
    if missing:
        print(f"NOTE: {' and '.join(missing)} unavailable; enrichers derived "
              f"from them were not exercised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
