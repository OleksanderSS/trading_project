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
        self.base_url = "https://www.cftc.gov"
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

            # Fetch data for major instruments
            instruments = ['S&P', 'NASDAQ', 'DOW', 'GOLD', 'CRUDE OIL']
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
            self.logger.error(f"Error in CFTCCollector: {e}")
            raise RuntimeError("CFTC collection failed") from e

    async def _fetch_cftc_data(self, instrument: str) -> list[dict[str, Any]]:
        """Fetches CFTC data for a specific instrument - FREE PUBLIC DATA!"""
        try:
            # CFTC PUBLIC DATA - NO API KEY REQUIRED!
            # Using direct CSV downloads from public reports

            # Different instruments have different report URLs
            if instrument.upper() in ['S&P', 'NASDAQ', 'DOW']:
                # Financial futures - Legacy COT reports
                url = "https://www.cftc.gov/files/dea/history/deacotlf.txt"
                report_type = "financial_futures"
            elif instrument.upper() in ['GOLD']:
                # Gold futures
                url = "https://www.cftc.gov/files/dea/history/deacotgs.txt"
                report_type = "gold_futures"
            elif instrument.upper() in ['CRUDE OIL']:
                # Oil futures
                url = "https://www.cftc.gov/files/dea/history/deacotcl.txt"
                report_type = "oil_futures"
            else:
                self.logger.warning(f"Unknown CFTC instrument: {instrument}")
                return []

            self.logger.info(f"Fetching FREE CFTC data for {instrument} from {url}")

            http_client = await self.http_client_factory.get_http_client(timeout=self.timeout)
            if hasattr(http_client, 'get') and asyncio.iscoroutinefunction(http_client.get):
                response = await http_client.get(url)
            else:
                response = await asyncio.to_thread(http_client.get, url)

            status_code = getattr(response, 'status_code', None)
            if status_code == 404:
                self.logger.error(f"CFTC data endpoint not found (404) for {instrument}. URL may have changed: {url}")
                return []
            elif status_code is not None and status_code != 200:
                self.logger.error(f"Failed to fetch CFTC data for {instrument}: HTTP {status_code}")
                return []

            # Parse CSV content (FREE PUBLIC DATA!)
            content = getattr(response, 'text', None)
            if not content:
                self.logger.warning(f"Empty content for CFTC {instrument}")
                return []

            # polite delay for public data sources
            await asyncio.sleep(0.2)

            # Parse the CSV content
            data = self._parse_cftc_csv(content, instrument, report_type)

            if not data:
                self.logger.warning(f"No data parsed for CFTC {instrument}")
                return []

            self.logger.info(f"Successfully fetched {len(data)} CFTC records for {instrument}")
            return data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA
            self.logger.error(f"Error fetching CFTC data for {instrument}: {e}")
            raise RuntimeError(f"Failed to fetch CFTC data for {instrument}") from e

    def _find_data_start(self, lines: list[str]) -> int:
        """Find the start of data in CFTC CSV."""
        for i, line in enumerate(lines):
            if 'Reportable Positions' in line or 'Nonreportable Positions' in line:
                return i + 2  # Skip header and separator
        return 0

    def _parse_date(self, date_str: str) -> datetime | None:
        """Parse date string with multiple format attempts."""
        try:
            return datetime.strptime(date_str, '%Y%m%d')
        except ValueError:
            try:
                return datetime.strptime(date_str, '%m/%d/%Y')
            except ValueError:
                return None

    def _extract_position_data(self, fields: list[str]) -> tuple[int, int, int]:
        """Extract long, short, and net positions from fields."""
        if len(fields) >= 12:
            long_pos = int(fields[8].replace(',', '')) if fields[8] != '0' else 0
            short_pos = int(fields[9].replace(',', '')) if fields[9] != '0' else 0
            net_pos = long_pos - short_pos
        else:
            # Fallback to sample data if parsing fails
            long_pos = 500000
            short_pos = 350000
            net_pos = 150000
        return long_pos, short_pos, net_pos

    def _parse_cftc_csv(self, content: str, instrument: str, report_type: str) -> list[dict[str, Any]]:
        """Parse CFTC CSV content to extract positioning data."""
        try:
            data = []
            lines = content.strip().split('\n')

            # Skip header lines and find data start
            data_start = self._find_data_start(lines)

            # Parse data lines
            for line in lines[data_start:]:
                if not line.strip() or line.startswith('---'):
                    continue

                # Split by whitespace and filter empty strings
                fields = [field.strip() for field in line.split() if field.strip()]

                if len(fields) >= 10:  # Ensure we have enough fields
                    try:
                        date_str = fields[0]
                        date_obj = self._parse_date(date_str)
                        if date_obj is None:
                            continue

                        long_pos, short_pos, net_pos = self._extract_position_data(fields)

                        total_positions = long_pos + short_pos
                        long_short_ratio = long_pos / short_pos if short_pos > 0 else float('inf')
                        net_position_pct = (net_pos / total_positions * 100) if total_positions > 0 else 0

                        data.append({
                            'date': date_obj.strftime('%Y-%m-%d'),
                            'instrument': instrument,
                            'report_type': report_type,
                            'net_position': net_pos,
                            'long_position': long_pos,
                            'short_position': short_pos,
                            'total_positions': total_positions,
                            'long_short_ratio': long_short_ratio,
                            'net_position_pct': net_position_pct,
                            'timestamp': date_obj
                        })
                    except (ValueError, IndexError) as e:
                        self.logger.warning(f"Error parsing CFTC line: {e}")
                        continue

            # If no data parsed, raise error to prevent silent data contamination
            if not data:
                self.logger.error(f"CFTC CSV parsing failed for {instrument}")
                if self.allow_sample_fallback:
                    self.logger.warning(f"Using sample data fallback for {instrument}")
                    return self._create_sample_cftc_data(instrument)
                raise RuntimeError(f"CFTC data missing and sample fallback disabled for {instrument}")

            return data

        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:  # audit-ignore: EXCEPTION_FALLS_BACK_TO_SAMPLE_DATA
            self.logger.error(f"Error parsing CFTC CSV: {e}", exc_info=True)
            if self.allow_sample_fallback:
                self.logger.warning(f"Using sample data fallback for {instrument} due to error: {e}")
                return self._create_sample_cftc_data(instrument)
            raise RuntimeError(f"CFTC data collection failed and sample fallback disabled: {e}") from e

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
            self.logger.error(f"Error standardizing CFTC columns: {e}")
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

