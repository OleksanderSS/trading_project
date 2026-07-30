# src/data/collectors/cftc_collector.py

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class CFTCCollector(BaseCollector):
    """Collector for CFTC Commitment of Traders Report data."""
    collector_type = "cftc"
    data_type = "alternative"
    collector_name = "cftc"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory,
                 db_manager: DataManager, cache_manager: CacheManager | None = None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.enabled = self.configs.get('enabled', True)
        self.timeout = self.configs.get('timeout', 30)
        self.table_name = self.configs.get('table_name', "cftc_data")
        self.hash_keys = self.configs.get('hash_keys', ["date", "instrument", "net_position"])
        # CFTC's official open-data (Socrata) endpoint for the Legacy
        # Commitments of Traders report, futures only.
        #
        # This replaces the old fixed-width text files under
        # https://www.cftc.gov/files/dea/history/. Verified 2026-07-30: EVERY
        # path on www.cftc.gov now answers automated requests with 403 (and the
        # specific file the collector used, deacotlf.txt, 404s) -- probed
        # deacotlf.txt, dea/newcot/{deafut,deacot,FinFutWk,FinComWk}.txt and
        # four annual archive zips, all blocked. The Socrata API returns 200
        # with fresh JSON and needs no key.
        self.api_url = self.configs.get(
            'api_url', "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
        )
        # Exact `market_and_exchange_names` values, confirmed against the live
        # dataset. The "Consolidated" series combine the full-size and E-mini
        # contracts, which is what a positioning signal actually wants.
        #
        # A value may be a LIST when a contract was renamed: CFTC does not
        # carry history forward under the new name, so the names have to be
        # unioned to get a continuous series. Confirmed case: NYMEX WTI was
        # reported as "CRUDE OIL, LIGHT SWEET" until 2022-02-01 and as
        # "WTI FINANCIAL CRUDE OIL" from then on.
        self.markets = self.configs.get('markets', {
            "S&P": "S&P 500 Consolidated - CHICAGO MERCANTILE EXCHANGE",
            "NASDAQ": "NASDAQ-100 Consolidated - CHICAGO MERCANTILE EXCHANGE",
            "DOW": "DJIA Consolidated - CHICAGO BOARD OF TRADE",
            "GOLD": "GOLD - COMMODITY EXCHANGE INC.",
            "CRUDE OIL": [
                "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE",
                "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
            ],
        })
        self.history_weeks = int(self.configs.get('history_weeks', 520))
        self.allow_sample_fallback = self.configs.get('allow_sample_fallback', False)
        self.logger.info(f"CFTCCollector initialized. Enabled: {self.enabled}, Allow Sample Fallback: {self.allow_sample_fallback}")

    def _generate_hash(self, row: pd.Series) -> str:
        """Generates a stable hash for a record."""
        hash_string = "|".join(str(row.get(key, "")) for key in self.hash_keys)
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def run(self, **kwargs) -> pd.DataFrame | None:
        """Fetches CFTC data and returns DataFrame."""
        if not self.enabled:
            self.logger.warning("CFTCCollector is disabled")
            return None

        try:
            self.logger.info("Fetching CFTC Commitment of Traders data")

            instruments = list(self.markets.keys())
            all_data = []

            for instrument in instruments:
                data = await self._fetch_cftc_data(instrument)
                if data:
                    all_data.extend(data)

            if not all_data:
                self.logger.warning("No CFTC data received")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(all_data)

            if df.empty:
                self.logger.warning("No CFTC data received")
                return None

            # Standardize columns
            df = self._standardize_columns(df)

            # Add metadata
            df['collector_type'] = self.collector_type
            df['collector_name'] = self.collector_name
            df['data_type'] = self.data_type
            df['collected_at'] = datetime.now()

            # Generate hashes for deduplication
            df['record_hash'] = df.apply(self._generate_hash, axis=1)

            self.logger.info(f"Successfully fetched {len(df)} CFTC records")
            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error in CFTCCollector: {e}")
            raise RuntimeError("CFTC collection failed") from e

    async def _fetch_cftc_data(self, instrument: str) -> list[dict[str, Any]]:
        """Fetch one market's Legacy COT history from CFTC's open-data API."""
        market = self.markets.get(instrument)
        if not market:
            self.logger.warning(f"No CFTC market name configured for '{instrument}'")
            return []

        names = [market] if isinstance(market, str) else list(market)
        # Socrata SoQL string literals escape a single quote by doubling it.
        name_list = ",".join("'" + n.replace("'", "''") + "'" for n in names)

        cutoff = (datetime.now() - timedelta(weeks=self.history_weeks)).strftime('%Y-%m-%d')
        params = {
            "$select": (
                "report_date_as_yyyy_mm_dd,market_and_exchange_names,"
                "noncomm_positions_long_all,noncomm_positions_short_all,"
                "open_interest_all"
            ),
            "$where": (
                f"market_and_exchange_names in ({name_list}) "
                f"AND report_date_as_yyyy_mm_dd > '{cutoff}'"
            ),
            "$order": "report_date_as_yyyy_mm_dd DESC",
            "$limit": self.history_weeks * len(names),
        }

        self.logger.info(
            f"Fetching CFTC COT data for {instrument} ({', '.join(names)})"
        )
        headers = {"Accept": "application/json"}

        http_client = await self.http_client_factory.get_http_client(timeout=self.timeout)
        if hasattr(http_client, 'get') and asyncio.iscoroutinefunction(http_client.get):
            response = await http_client.get(self.api_url, params=params, headers=headers)
        else:
            response = await asyncio.to_thread(
                http_client.get, self.api_url, params=params, headers=headers
            )

        status_code = getattr(response, 'status_code', None)
        if status_code is not None and status_code != 200:
            self.logger.error(
                f"Failed to fetch CFTC data for {instrument}: HTTP {status_code}"
            )
            return []

        try:
            rows = response.json()
        except (ValueError, TypeError) as e:
            self.logger.error(f"CFTC response for {instrument} was not JSON: {e}")
            return []

        if not isinstance(rows, list) or not rows:
            self.logger.warning(f"No CFTC rows returned for {instrument} ({market})")
            return []

        data = self._rows_to_records(rows, instrument, names)
        if not data:
            self.logger.error(f"CFTC rows for {instrument} yielded no usable records")
            return []

        self.logger.info(f"Fetched {len(data)} CFTC records for {instrument}")
        await asyncio.sleep(0.2)  # polite pacing for a public API
        return data

    def _rows_to_records(
        self,
        rows: list[dict[str, Any]],
        instrument: str,
        names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Map Socrata COT rows onto this collector's record shape.

        Non-commercial (speculative) long/short positions are the positioning
        signal; commercial positions are largely hedging. A row missing either
        leg is SKIPPED, never backfilled -- the previous implementation
        substituted hardcoded values (500000/350000/150000) whenever a line had
        too few fields, silently writing invented numbers into the same table
        as real observations and without any `is_synthetic` marker.

        When `names` holds several market names (a renamed contract), the two
        series OVERLAP rather than meeting end to end -- NYMEX reported both
        "CRUDE OIL, LIGHT SWEET" and "WTI FINANCIAL CRUDE OIL" for roughly 150
        weeks. Emitting both would put two different net positions on the same
        (date, instrument), and the record hash includes net_position so dedup
        would keep both. Earlier entries in `names` win per date.
        """
        priority = {name: i for i, name in enumerate(names or [])}
        data: list[dict[str, Any]] = []
        for row in rows:
            date_raw = row.get('report_date_as_yyyy_mm_dd')
            long_raw = row.get('noncomm_positions_long_all')
            short_raw = row.get('noncomm_positions_short_all')
            if date_raw is None or long_raw is None or short_raw is None:
                continue
            try:
                date_obj = datetime.fromisoformat(str(date_raw).replace('Z', '+00:00'))
                long_pos = int(float(long_raw))
                short_pos = int(float(short_raw))
            except (ValueError, TypeError):
                continue

            net_pos = long_pos - short_pos
            total_positions = long_pos + short_pos
            long_short_ratio = long_pos / short_pos if short_pos > 0 else float('inf')
            net_position_pct = (
                (net_pos / total_positions * 100) if total_positions > 0 else 0
            )

            try:
                open_interest = int(float(row.get('open_interest_all') or 0))
            except (ValueError, TypeError):
                open_interest = 0

            data.append({
                'date': date_obj.strftime('%Y-%m-%d'),
                'instrument': instrument,
                'report_type': 'legacy_cot_futures_only',
                'market_and_exchange': row.get('market_and_exchange_names', ''),
                'net_position': net_pos,
                'long_position': long_pos,
                'short_position': short_pos,
                'total_positions': total_positions,
                'long_short_ratio': long_short_ratio,
                'net_position_pct': net_position_pct,
                'open_interest': open_interest,
                'timestamp': date_obj.replace(tzinfo=None),
            })

        if len(priority) > 1:
            best: dict[str, dict[str, Any]] = {}
            for rec in data:
                key = rec['date']
                rank = priority.get(rec['market_and_exchange'], len(priority))
                incumbent = best.get(key)
                if incumbent is None or rank < priority.get(
                    incumbent['market_and_exchange'], len(priority)
                ):
                    best[key] = rec
            dropped = len(data) - len(best)
            if dropped:
                self.logger.info(
                    f"{instrument}: dropped {dropped} overlapping rows from "
                    f"superseded contract names, kept one per report date."
                )
            data = sorted(best.values(), key=lambda r: r['date'])

        return data

    def _create_sample_cftc_data(self, instrument: str) -> list[dict[str, Any]]:
        """Create sample CFTC data for demonstration."""
        data = []
        base_date = datetime.now() - timedelta(days=140)

        for i in range(20):  # Generate 20 weeks of data
            date_obj = base_date + timedelta(weeks=i)

            # Simulate realistic positioning data
            if instrument.upper() in ['S&P', 'NASDAQ']:
                # Equity indices - typically net long
                net_pos = 150000 + (i * 25000)
                long_pos = 500000 + (i * 50000)
                short_pos = 350000 + (i * 25000)
            elif instrument.upper() in ['GOLD']:
                # Gold - varies more
                net_pos = 50000 + (i * 10000) * (1 if i % 2 == 0 else -1)
                long_pos = 200000 + (i * 20000)
                short_pos = 150000 + (i * 10000)
            elif instrument.upper() in ['CRUDE OIL']:
                # Oil - can be net short
                net_pos = -30000 + (i * 15000) * (1 if i % 2 == 0 else -1)
                long_pos = 180000 + (i * 15000)
                short_pos = 210000 + (i * 20000)
            else:
                continue

            total_positions = long_pos + short_pos
            long_short_ratio = long_pos / short_pos if short_pos > 0 else float('inf')
            net_position_pct = (net_pos / total_positions * 100) if total_positions > 0 else 0

            data.append({
                'date': date_obj.strftime('%Y-%m-%d'),
                'instrument': instrument,
                'report_type': 'sample_data',
                'net_position': net_pos,
                'long_position': long_pos,
                'short_position': short_pos,
                'total_positions': total_positions,
                'long_short_ratio': long_short_ratio,
                'net_position_pct': net_position_pct,
                'timestamp': date_obj,
                'is_synthetic': True,
                'eligible_for_training': False
            })

        return data

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and data types."""
        try:
            # Ensure required columns exist
            if 'date' not in df.columns:
                df['date'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d')

            required_cols = ['net_position', 'long_position', 'short_position', 'total_positions', 'net_position_pct']
            for col in required_cols:
                if col not in df.columns:
                    self.logger.error(f"CFTC data missing '{col}' column")
                    return pd.DataFrame()

            # Convert date column
            df['date'] = pd.to_datetime(df['date'])

            # Ensure numeric types
            for col in required_cols + ['long_short_ratio']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Sort by date and instrument
            df = df.sort_values(['instrument', 'date']).reset_index(drop=True)

            # Add derived features
            df['position_signal'] = df['net_position_pct'].apply(self._get_position_signal)
            df['position_level'] = df['net_position_pct'].apply(self._categorize_position)
            df['extreme_positioning'] = (abs(df['net_position_pct']) > 20).astype(int)

            return df

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.exception(f"Error standardizing CFTC columns: {e}")
            return pd.DataFrame()

    def _categorize_position(self, net_pct: float) -> str:
        """Categorize positioning based on net position percentage."""
        if net_pct > 15:
            return "very_long"
        elif net_pct > 5:
            return "long"
        elif net_pct > -5:
            return "neutral"
        elif net_pct > -15:
            return "short"
        else:
            return "very_short"

    def _get_position_signal(self, net_pct: float) -> int:
        """Get trading signal based on net positioning."""
        if net_pct > 10:  # Strong long positioning
            return 1
        elif net_pct < -10:  # Strong short positioning
            return -1
        else:
            return 0  # Neutral

    async def collect_data(self, **kwargs) -> list[dict[str, Any]] | None:
        """
        UNIFIED data collection - retrieval only, without database storage.
        """
        df = await self.run(**kwargs)
        return df.to_dict('records') if df is not None else None

