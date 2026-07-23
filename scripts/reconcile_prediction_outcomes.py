#!/usr/bin/env python3
"""Resolve due predictions from the confidence-calibrator prediction ledger
against realized prices, feeding the online-learning half of
src/models/calibration/adaptive_confidence_calibrator.py.

This is deliberately a manually-invoked / cron-invoked script, not wired
into the live pipeline — horizons elapse on their own calendar, independent
of any single Stage 5 run, so reconciliation is an operational scheduling
decision for the project owner (daily cron, a dean_os review cycle, etc.),
not something to bake into the request/response cycle of a prediction call.

Usage:
    python scripts/reconcile_prediction_outcomes.py --price-parquet data/processed/features/prices.parquet

The price file must have at least `ticker`, `date` (or `datetime`), and
`close` columns — no specific data source is assumed; point this at
whichever price snapshot the project owner trusts as ground truth.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.logging.logger import ProjectLogger
from src.models.calibration.prediction_ledger import (
    DEFAULT_LEDGER_PATH,
    PredictionOutcomeReconciler,
)

logger = ProjectLogger.get_logger("ReconcilePredictionOutcomes")


def make_parquet_price_lookup(parquet_path: str):
    """Builds a (ticker, as_of_date) -> close price callable backed by a
    single parquet snapshot. Returns the first close at/after as_of_date;
    None if no such row exists yet (horizon elapsed but data hasn't caught
    up — the reconciler will simply retry on the next invocation)."""
    import pandas as pd

    prices = pd.read_parquet(parquet_path)
    date_col = "date" if "date" in prices.columns else "datetime"
    prices[date_col] = pd.to_datetime(prices[date_col])

    def _lookup(ticker: str, as_of_date: datetime) -> float | None:
        subset = prices[(prices["ticker"] == ticker) & (prices[date_col] >= as_of_date)]
        if subset.empty:
            return None
        return float(subset.sort_values(date_col).iloc[0]["close"])

    return _lookup


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-parquet", required=True, help="Path to a parquet file with ticker/date/close columns")
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH)
    args = parser.parse_args()

    price_lookup = make_parquet_price_lookup(args.price_parquet)
    reconciler = PredictionOutcomeReconciler(price_lookup=price_lookup, ledger_path=args.ledger_path)
    summary = reconciler.reconcile_due_predictions()

    logger.info(f"Reconciliation summary: {summary}")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
