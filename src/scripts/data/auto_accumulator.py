#!/usr/bin/env python3
"""Integrity Guard: find gaps in stored market data and refetch just those.

Checks every configured (ticker, timeframe) pair against what is actually in
`market_data_raw`, and re-runs the market-data collector only for the pairs
that are missing or stale. Cheap to run before a pipeline pass.

Rewritten 2026-07-31. The previous version could never have run -- every
external call it made was against an API that does not exist:

  - `from src.data.collector_factory import create_all_collectors`: wrong
    module path (the real one is src.data.collectors.collector_factory) AND
    no such function -- the real API is the class
    `CollectorFactory(...).get_all_collectors()`.
  - `AssetUniverseManager(config_manager.get_config('asset_universe', {}))`:
    extracted the key twice, since the class itself does
    `config.get('asset_universe')`. Worse, the class is structurally
    incompatible with the current assets.yaml in two further ways -- there is
    no `asset_universe` key at all, and passing the config correctly raises
    `TypeError: SectorConfig.__init__() got an unexpected keyword argument
    'description'`. Tickers now come from `assets.active_preset`, the same
    path the live pipeline uses.
  - preset `'day_trading_tech'`: does not exist; the only preset is
    `default_volatile`.
  - `db_manager.get_all_tables()`: the real method is `get_all_table_names()`.
  - it queried a table called `market_data`, which does not exist -- the real
    one is `market_data_raw`.
  - it looked for a `5m` timeframe that nothing collects; timeframes are now
    read from the collector config (15m / 1h / 1d today).
"""

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# Додавання кореня проекту до шляху
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import get_current_config
from src.core.clients.http_client_factory import HttpClientFactory
from src.core.error_handling.error_handler import ErrorHandler
from src.core.logging.logger import ProjectLogger
from src.data.collectors.yf_collector import YFCollector
from src.data.management.data_manager import DataManager
from src.utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger("AutoAccumulatorGuard")

MARKET_TABLE = "market_data_raw"

#: How many completed trading days a series may lag before it counts stale.
DEFAULT_STALENESS_DAYS = 1


class AutoAccumulatorGuard:
    """Finds and refills gaps in stored market data."""

    def __init__(self, staleness_days: int = DEFAULT_STALENESS_DAYS):
        self.config_manager = get_current_config()
        self.error_handler = ErrorHandler()
        self.db_manager = DataManager(self.config_manager, self.error_handler)
        self.http_factory = HttpClientFactory(self.config_manager, self.error_handler)
        self.staleness_days = max(1, int(staleness_days))
        self.calendar = TradingCalendar()

        self.active_tickers = self._load_tickers()
        self.timeframes = self._load_timeframes()

        logger.info(
            f"Integrity Guard initialized for {len(self.active_tickers)} tickers "
            f"and {len(self.timeframes)} timeframes: {sorted(self.timeframes)}"
        )

    # ── configuration ────────────────────────────────────────────────────

    def _load_tickers(self) -> list[str]:
        """Tickers from the active preset, plus any benchmark tickers.

        Same source the live pipeline reads, rather than AssetUniverseManager,
        which cannot parse the current assets.yaml at all (see module docstring).
        """
        assets = self.config_manager.get_config("assets", {}) or {}
        preset_name = assets.get("active_preset")
        tickers: list[str] = []
        if preset_name:
            preset = (assets.get("presets", {}) or {}).get(preset_name, {}) or {}
            tickers = [str(t) for t in preset.get("tickers", []) or []]

        collectors = self.config_manager.get_config("collectors", {}) or {}
        yahoo = collectors.get("yahoo_finance", {}) or {}
        benchmarks = [str(t) for t in yahoo.get("benchmark_tickers", []) or []]

        merged = list(dict.fromkeys(tickers + benchmarks))
        if not merged:
            logger.warning(
                "No tickers resolved from assets.active_preset or "
                "collectors.yahoo_finance.benchmark_tickers."
            )
        return merged

    def _load_timeframes(self) -> dict[str, dict[str, Any]]:
        """Timeframes the market collector is actually configured to fetch."""
        collectors = self.config_manager.get_config("collectors", {}) or {}
        yahoo = collectors.get("yahoo_finance", {}) or {}
        timeframes = yahoo.get("timeframes") or {}
        if isinstance(timeframes, dict) and timeframes:
            return {str(k): dict(v or {}) for k, v in timeframes.items()}
        logger.warning(
            "Could not read collectors.yahoo_finance.timeframes; "
            "falling back to 15m/1h/1d."
        )
        return {"15m": {"period": "60d"}, "1h": {"period": "60d"}, "1d": {"period": "2y"}}

    # ── gap detection ────────────────────────────────────────────────────

    def _latest_timestamps(self) -> pd.DataFrame:
        """One bulk query for the newest bar per (ticker, interval).

        The old version issued a separate query per ticker per timeframe, with
        the ticker interpolated straight into the SQL string.
        """
        if not self.db_manager.table_exists(MARKET_TABLE):
            logger.warning(f"Table '{MARKET_TABLE}' does not exist yet.")
            return pd.DataFrame(columns=["ticker", "interval", "latest"])
        return self.db_manager.con.execute(
            f"""SELECT ticker, interval, MAX(datetime) AS latest
                FROM "{MARKET_TABLE}" GROUP BY ticker, interval"""
        ).fetchdf()

    def _cutoff_trading_day(self):
        """Oldest trading day a series may end on and still count as current.

        Staleness is measured in TRADING DAYS, not wall-clock minutes. An
        earlier draft of this rewrite used "3 intervals of silence", which
        marks every 15m series stale within 45 minutes -- so overnight, at
        weekends and on holidays it flagged essentially the whole universe.
        A gap detector that cries wolf every evening is worse than none.
        `TradingCalendar` already knows sessions and holidays; reuse it rather
        than inventing a second notion of market time.
        """
        previous = self.calendar.get_previous_trading_days(
            datetime.now().date(), self.staleness_days
        )
        return min(previous) if previous else datetime.now().date()

    def find_gaps(self) -> dict[str, list[str]]:
        """Return {ticker: [stale or missing timeframes]}."""
        latest = self._latest_timestamps()
        index: dict[tuple[str, str], Any] = {
            (str(r.ticker), str(r.interval)): r.latest
            for r in latest.itertuples(index=False)
        }
        cutoff = self._cutoff_trading_day()

        gaps: dict[str, list[str]] = {}
        for ticker in self.active_tickers:
            for interval in self.timeframes:
                newest = index.get((ticker, interval))
                if newest is None:
                    gaps.setdefault(ticker, []).append(interval)
                    continue
                if pd.Timestamp(newest).date() < cutoff:
                    gaps.setdefault(ticker, []).append(interval)
        return gaps

    # ── refill ───────────────────────────────────────────────────────────

    async def _refetch(self, tickers: list[str], intervals: list[str]) -> None:
        """Re-run the market collector for just these tickers/timeframes."""
        collectors = self.config_manager.get_config("collectors", {}) or {}
        yahoo = dict(collectors.get("yahoo_finance", {}) or {})
        yahoo["enabled"] = True
        yahoo["timeframes"] = {
            k: v for k, v in self.timeframes.items() if k in set(intervals)
        }

        collector = YFCollector(yahoo, self.http_factory, self.db_manager)
        await collector.run(tickers=tickers)

    def run_guard_cycle(self) -> bool:
        """Check every pair, refetch the stale ones. True if anything ran."""
        logger.info("--- Integrity Guard cycle start ---")
        gaps = self.find_gaps()

        if not gaps:
            logger.info(
                f"All {len(self.active_tickers)} tickers current across "
                f"{len(self.timeframes)} timeframes. Nothing to do."
            )
            return False

        total = sum(len(v) for v in gaps.values())
        logger.info(f"{total} stale (ticker, timeframe) pair(s) across {len(gaps)} ticker(s).")

        # Group by interval so each collector run covers many tickers at once.
        by_interval: dict[str, list[str]] = {}
        for ticker, intervals in gaps.items():
            for interval in intervals:
                by_interval.setdefault(interval, []).append(ticker)

        for interval, tickers in sorted(by_interval.items()):
            logger.info(f"Refetching '{interval}' for {len(tickers)} ticker(s): {tickers}")
            try:
                asyncio.run(self._refetch(tickers, [interval]))
            except Exception as e:
                logger.error(f"Refetch failed for '{interval}': {e}", exc_info=True)

        logger.info("--- Integrity Guard cycle complete ---")
        return True

    # ── reporting ────────────────────────────────────────────────────────

    def get_db_report(self) -> None:
        """Print what is actually stored per timeframe."""
        try:
            tables = self.db_manager.get_all_table_names()
            print(f"\n[Database: {self.db_manager.db_path}]")
            if MARKET_TABLE not in tables:
                print(f"Table '{MARKET_TABLE}' not found.")
                return
            df = self.db_manager.con.execute(
                f"""SELECT interval,
                           COUNT(*) AS rows,
                           COUNT(DISTINCT ticker) AS tickers,
                           MIN(datetime) AS start,
                           MAX(datetime) AS "end"
                    FROM "{MARKET_TABLE}" GROUP BY interval ORDER BY interval"""
            ).fetchdf()
            print(df.to_string(index=False))

            gaps = self.find_gaps()
            print(f"\nStale/missing pairs: {sum(len(v) for v in gaps.values())}")
            for ticker, intervals in sorted(gaps.items()):
                print(f"  {ticker}: {', '.join(sorted(intervals))}")
        except Exception as e:
            print(f"Error generating report: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Integrity Guard: refill gaps in stored market data"
    )
    parser.add_argument('--mode', default='once', choices=['once', 'cycle'],
                        help='once = single pass; cycle = keep monitoring')
    parser.add_argument('--interval', type=int, default=15,
                        help='Minutes between checks in cycle mode')
    parser.add_argument('--staleness-days', type=int,
                        default=DEFAULT_STALENESS_DAYS,
                        help='How many completed trading days of lag count as a gap')
    parser.add_argument('--report', action='store_true',
                        help='Print database status and detected gaps, change nothing')

    args = parser.parse_args()
    guard = AutoAccumulatorGuard(staleness_days=args.staleness_days)

    try:
        if args.report:
            guard.get_db_report()
            return

        if args.mode == 'once':
            guard.run_guard_cycle()
        else:
            logger.info(f"Monitoring every {args.interval} min.")
            while True:
                try:
                    guard.run_guard_cycle()
                    logger.info(f"Sleeping {args.interval} min.")
                    time.sleep(args.interval * 60)
                except KeyboardInterrupt:
                    logger.info("Stopped by user.")
                    break
                except Exception as e:
                    logger.error(f"Monitoring cycle error: {e}", exc_info=True)
                    time.sleep(60)
    finally:
        DataManager.close_all_connections()


if __name__ == "__main__":
    main()
