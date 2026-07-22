import asyncio
import hashlib
import logging
import math
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.pipeline.timeframe_lineage import timeframe_lineage_report

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

# yfinance keeps process-global download state.  The collector launches one
# worker per timeframe, so concurrent yf.download calls can otherwise return a
# frame belonging to another in-flight ticker/timeframe request.
_YFINANCE_DOWNLOAD_LOCK = threading.Lock()

def _configure_yfinance_cache() -> None:
    cache_dir = Path("data/cache/yfinance").resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        yf.set_tz_cache_location(str(cache_dir))
    except AttributeError:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Installed yfinance version does not support cache location override.")
    except (ValueError, TypeError, KeyError, ZeroDivisionError) as e:
        logger.warning(f"Could not configure yfinance cache directory '{cache_dir}': {e}", exc_info=True)
        raise RuntimeError(f"Could not configure yfinance cache directory '{cache_dir}'") from e

class YFCollector(BaseCollector):
    """
    Collects and caches historical price data from Yahoo Finance.
    """
    collector_type = "yahoo_finance"
    data_type = "market_data"

    def __init__(self, configs: dict[str, Any], http_client_factory: HttpClientFactory, db_manager: DataManager, **kwargs):
        super().__init__(configs, http_client_factory, db_manager, **kwargs)
        self.timeframes = self.configs.get('timeframes', {})
        if not self.timeframes:
            self.logger.warning("'timeframes' not configured. Collector will not be able to gather data.")

    def _check_cache(self, cache_key: str, cache_params: dict[str, Any], table_name: str, tickers: list[str]) -> list[dict[str, Any]] | None:
        """Check cache for existing data."""
        if self.cache_manager and self.db_manager:
            cached_data = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached_data is not None:
                self.logger.info(f"Checking cached data for {len(tickers)} tickers against database.")
                df_cached = pd.DataFrame(cached_data) if isinstance(cached_data, list) else cached_data
                new_from_cache = self.db_manager.filter_new_records(table_name, df_cached)

                if new_from_cache.empty:
                    self.logger.info("Cache hit, but all records are already in the database.")
                    return []

                self.logger.info(f"Returning {len(new_from_cache)} new records found in cache.")
                return new_from_cache.to_dict('records')
        return None

    def _adjust_intraday_dates(self, interval: str, start_date: datetime, end_date: datetime, reference_now: datetime) -> tuple[datetime, datetime]:
        """Adjust dates for intraday intervals."""
        limit_date = reference_now - timedelta(days=58)
        if start_date < limit_date:
            self.logger.warning(f"Interval '{interval}' is intraday and start_date {start_date} is too old. Adjusting to {limit_date}")
            start_date = limit_date

        # Ensure start is before end after adjustment
        if start_date >= end_date:
            self.logger.warning(f"Adjusted start_date for {interval} is after end_date. Setting end_date to now.")
            end_date = reference_now

        return start_date, end_date

    def _create_download_tasks(self, tickers: list[str], end_date: datetime, reference_now: datetime) -> list:
        """Create async download tasks for all timeframes."""
        tasks = []
        for interval, params in self.timeframes.items():
            interval_end_date = end_date
            period = params.get('period')
            start_date = self._calculate_start_date(interval_end_date, period)

            if not start_date:
                continue

            if (interval.endswith('m') or interval.endswith('h')):
                start_date, interval_end_date = self._adjust_intraday_dates(
                    interval,
                    start_date,
                    interval_end_date,
                    reference_now,
                )

            task = asyncio.to_thread(
                self._blocking_download,
                tickers,
                interval,
                start_date,
                interval_end_date,
            )
            tasks.append(task)
        return tasks

    async def run(self, tickers: list[str], end_date: datetime | None = None, **kwargs) -> list[dict[str, Any]]:
        """
        Asynchronously downloads data, filters for new records, and saves them to the database.
        Accepts an optional end_date for deterministic testing.
        """
        # --- Додаємо галузеві ETF-бенчмарки ---
        benchmarks = self.configs.get('benchmark_tickers', [])
        all_tickers = list(set((tickers or []) + benchmarks))
        
        if not self.timeframes or not all_tickers:
            self.logger.info("No tickers or timeframes to collect. Skipping.")
            return []

        tickers = all_tickers

        end_date = end_date or datetime.now()
        persist = bool(kwargs.get("persist", True))
        table_name = self.configs.get('table_name', 'market_data_raw')

        # Cache Check
        cache_key = f"{self.__class__.__name__}_run_{self.collector_type}"
        cache_params = {
            "tickers": sorted(tickers),
            "timeframes": self.timeframes,
            "end_date": end_date.isoformat()
        }

        cached_result = (
            self._check_cache(cache_key, cache_params, table_name, tickers)
            if persist
            else None
        )
        if cached_result is not None:
            if cached_result:
                issues = self._validate_collected_price_data(
                    pd.DataFrame(cached_result)
                )
                if issues:
                    raise RuntimeError(
                        "Cached Yahoo Finance data failed source gate: "
                        + "; ".join(issues)
                    )
            return cached_result

        # Use reference_now from kwargs if provided for stable testing, otherwise datetime.now()
        reference_now = kwargs.get('reference_now', datetime.now())

        self.logger.info(f"Starting collection for {len(tickers)} tickers. End date: {end_date.isoformat()}")

        tasks = self._create_download_tasks(tickers, end_date, reference_now)

        all_price_data = []
        results_from_threads = await asyncio.gather(*tasks, return_exceptions=True)

        for _i, result in enumerate(results_from_threads):
            if isinstance(result, Exception):
                self.logger.error(f"[YF] Download task failed: {result}")
            elif isinstance(result, list) and result:
                all_price_data.extend(result)

        if not all_price_data:
            self.logger.info("No data collected from Yahoo Finance.")
            return []

        self.logger.info(f"Collected {len(all_price_data)} total data points from API.")

        df_to_check = pd.DataFrame(all_price_data)
        issues = self._validate_collected_price_data(df_to_check)
        if issues:
            raise RuntimeError(
                "Yahoo Finance data failed source gate: "
                + "; ".join(issues)
            )
        if not persist:
            return all_price_data

        new_records_df = self.db_manager.filter_new_records(table_name, df_to_check)

        if new_records_df.empty:
            self.logger.info("No new records to save. All collected data is already cached.")
            return []

        self.logger.info(f"Found {len(new_records_df)} new records to save.")
        self.db_manager.upsert(table_name, new_records_df, unique_on=['hash'])

        result = new_records_df.to_dict('records')
        if self.cache_manager:
            self.cache_manager.set(cache_key, result, cache_params, ttl=self.configs.get('cache_ttl', 3600), namespace="collectors")

        return result

    def _validate_collected_price_data(
        self,
        frame: pd.DataFrame,
    ) -> list[str]:
        """Validate source identity and cadence before cache/database writes."""
        required = {
            "datetime",
            "ticker",
            "interval",
            "open",
            "high",
            "low",
            "close",
            "volume",
        }
        missing = sorted(required - set(frame.columns))
        if missing:
            return ["missing_columns=" + ",".join(missing)]
        if frame.empty:
            return ["empty_market_data"]

        issues: list[str] = []
        timestamps = pd.to_datetime(
            frame["datetime"],
            errors="coerce",
        )
        if timestamps.isna().any():
            issues.append(
                f"invalid_datetime_rows={int(timestamps.isna().sum())}"
            )
        if getattr(timestamps.dt, "tz", None) is None:
            issues.append("datetime_timezone_unresolved")

        identity_duplicates = int(
            frame.duplicated(
                ["ticker", "datetime", "interval"],
                keep=False,
            ).sum()
        )
        if identity_duplicates:
            issues.append(
                f"duplicate_identity_rows={identity_duplicates}"
            )

        price_identity_columns = [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]
        identity_frame = frame.assign(
            datetime=timestamps,
            _source_identity=(
                frame["ticker"].astype(str).str.upper()
                + "|"
                + frame["interval"].astype(str).str.lower()
            ),
        )
        duplicate_price_mask = identity_frame.duplicated(
            price_identity_columns,
            keep=False,
        )
        duplicate_price_rows = identity_frame.loc[duplicate_price_mask]
        if not duplicate_price_rows.empty:
            cross_identity = duplicate_price_rows.groupby(
                price_identity_columns,
                dropna=False,
            )["_source_identity"].transform("nunique") > 1
            contaminated_rows = int(cross_identity.sum())
            if contaminated_rows:
                issues.append(
                    f"cross_identity_ohlcv_rows={contaminated_rows}"
                )

        for interval, interval_frame in frame.groupby(
            "interval",
            dropna=False,
        ):
            report = timeframe_lineage_report(
                interval_frame,
                declared_timeframe=interval,
            )
            if report.get("status") in {
                "timeframe_cadence_mismatch",
                "timeframe_cadence_ambiguous",
            }:
                issues.append(
                    "cadence_mismatch="
                    f"{interval}:observed="
                    f"{report.get('observed_timeframe')}"
                )

        expected_minutes = {
            "15m": 15.0,
            "1h": 60.0,
            "60m": 60.0,
            "1d": 1440.0,
        }
        timing = frame.assign(_datetime=timestamps)
        for (ticker, interval), group in timing.groupby(
            ["ticker", "interval"],
            dropna=False,
        ):
            expected = expected_minutes.get(str(interval).lower())
            if expected is None:
                issues.append(f"unsupported_interval={interval}")
                continue
            deltas = (
                group.sort_values("_datetime")["_datetime"]
                .dropna()
                .diff()
                .dropna()
                .dt.total_seconds()
                .div(60.0)
            )
            if deltas.empty:
                continue
            ratios = deltas / expected
            invalid = (
                ratios.lt(1.0 - 1e-6)
                | (ratios - ratios.round()).abs().gt(1e-6)
            )
            if invalid.any():
                issues.append(
                    "cadence_mismatch="
                    f"{ticker}/{interval}:"
                    f"{int(invalid.sum())}/{len(deltas)}"
                )

        return sorted(set(issues))

    def _calculate_start_date(self, end_date: datetime, period: str) -> datetime | None:
        try:
            if 'y' in period:
                return end_date - timedelta(days=int(period.replace('y', '')) * 365)
            if 'd' in period:
                return end_date - timedelta(days=int(period.replace('d', '')))
            self.logger.error(f"Unsupported period format: {period}")
        except (ValueError, TypeError):
            self.logger.error(f"Invalid period format: {period}")
        return None

    def _blocking_download(self, tickers: list[str], interval: str, start_date: datetime, end_date: datetime) -> list[dict[str, Any]]:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Started blocking download for interval '{interval}'.")
        all_ticker_data = []
        retries = max(1, int(self.configs.get("max_retries", 3)))
        delay = max(0, int(self.configs.get("retry_delay", 5)))
        for ticker in tickers:
            df = self._single_ticker_download_with_retry(
                ticker,
                interval,
                start_date,
                end_date,
                retries=retries,
                delay=delay,
            )
            if not df.empty:
                processed_data = self._process_single_ticker_dataframe(df, ticker, interval)
                all_ticker_data.extend(processed_data)
        return all_ticker_data

    def _single_ticker_download_with_retry(self, ticker: str, interval: str, start_date: datetime, end_date: datetime, retries: int = 3, delay: int = 5) -> pd.DataFrame:
        last_error = None
        for attempt in range(retries):
            try:
                # Use objects directly for intraday or precise strings
                with _YFINANCE_DOWNLOAD_LOCK:
                    _configure_yfinance_cache()
                    df = yf.download(
                        tickers=ticker,
                        interval=interval,
                        start=start_date,
                        end=end_date,
                        auto_adjust=True,
                        progress=False,
                        threads=False,
                    )
                    # Detach the returned frame from any process-global
                    # yfinance cache/state before the next worker is allowed
                    # to start another request.
                    df = df.copy(deep=True)
                if not df.empty:
                    self.logger.info(f"Successfully downloaded {len(df)} rows for {ticker}/{interval}")
                    return df
                self.logger.warning(f"No data for {ticker}/{interval} on attempt {attempt + 1}/{retries}. Retrying in {delay}s.")
            except (ValueError, TypeError, Exception) as e:
                self.logger.error(f"Error downloading {ticker}/{interval} on attempt {attempt + 1}/{retries}: {e}", exc_info=True)
                last_error = e

            if attempt < retries - 1:
                time.sleep(delay)

        self.logger.error(f"Failed to download data for {ticker}/{interval} after {retries} attempts.")
        raise RuntimeError(f"Data download failed for {ticker}/{interval} after {retries} attempts: {last_error}") from last_error

    def _flatten_multiindex_columns(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """Validate Yahoo's source ticker and then flatten its columns."""
        if isinstance(df.columns, pd.MultiIndex):
            requested = str(ticker).strip().casefold()
            level_names = [
                str(name).strip().casefold() if name is not None else ""
                for name in df.columns.names
            ]
            symbol_level = next(
                (
                    index
                    for index, name in enumerate(level_names)
                    if name in {"ticker", "tickers", "symbol", "symbols"}
                ),
                None,
            )
            if symbol_level is None:
                market_fields = {
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "volume",
                    "price",
                }
                candidate_levels = []
                for index in range(df.columns.nlevels):
                    values = {
                        str(value).strip().casefold()
                        for value in df.columns.get_level_values(index)
                        if str(value).strip()
                    }
                    if requested in values or (
                        values and not values.issubset(market_fields)
                    ):
                        candidate_levels.append(index)
                if len(candidate_levels) == 1:
                    symbol_level = candidate_levels[0]

            if symbol_level is None:
                raise RuntimeError(
                    f"Could not resolve Yahoo source ticker identity for "
                    f"{ticker}/{interval}; refusing to relabel MultiIndex data."
                )

            source_tickers = {
                str(value).strip().casefold()
                for value in df.columns.get_level_values(symbol_level)
                if str(value).strip()
            }
            if source_tickers != {requested}:
                observed = ",".join(sorted(source_tickers)) or "<empty>"
                raise RuntimeError(
                    f"Yahoo source ticker mismatch for {ticker}/{interval}: "
                    f"observed={observed}"
                )

            df.columns = df.columns.get_level_values(0)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f"Flattened MultiIndex columns for {ticker}/{interval}")
        return df

    def _prepare_dataframe_columns(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """Prepare dataframe columns for processing."""
        df = self._flatten_multiindex_columns(df, ticker, interval)

        # Ensure index name is set before reset for clear column naming
        if df.index.name is None:
            df.index.name = 'datetime'

        df = df.reset_index()

        # Remove duplicated columns if any (e.g. after MultiIndex flattening)
        df = df.loc[:, ~df.columns.duplicated()]

        # Lowercase column names AFTER flattening
        df.rename(columns={col: str(col).lower() for col in df.columns}, inplace=True)

        return df

    def _identify_and_rename_date_column(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """Identify and rename the date column."""
        date_candidates = ['datetime', 'date', 'timestamp']
        date_col = next((c for c in date_candidates if c in df.columns), None)

        if date_col:
            df.rename(columns={date_col: 'datetime'}, inplace=True)
        else:
            raise RuntimeError(f"Could not find date/datetime column for {ticker}/{interval}. Available: {df.columns.tolist()}")

        return df

    def _parse_datetime_column(self, df: pd.DataFrame, ticker: str, interval: str) -> pd.DataFrame:
        """Parse datetime column with error handling."""
        try:
            df['datetime'] = pd.to_datetime(df['datetime'], utc=True)
            nat_count = df['datetime'].isna().sum()
            if nat_count > 0:
                self.logger.warning(f"Found {nat_count} NaT values in datetime for {ticker}/{interval}")
                # Remove rows with NaT datetime
                df = df[df['datetime'].notna()]
                if df.empty:
                    raise ValueError(f"All datetime values are NaT for {ticker}/{interval}")
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Failed to parse datetime for {ticker}/{interval}: {e}", exc_info=True)
            raise RuntimeError(f"Datetime parsing failed for {ticker}/{interval}: {e}") from e

        return df

    def _process_single_ticker_dataframe(self, df: pd.DataFrame, ticker: str, interval: str) -> list[dict[str, Any]]:
        if df.empty:
            return []

        df = self._prepare_dataframe_columns(df, ticker, interval)
        df = self._identify_and_rename_date_column(df, ticker, interval)
        df = self._parse_datetime_column(df, ticker, interval)

        df['ticker'] = ticker
        df['interval'] = interval

        # Drop rows with NaN or infinite values in numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if all(col in df.columns for col in numeric_cols):
            initial_len = len(df)
            df = df.dropna(subset=numeric_cols)
            for col in numeric_cols:
                df = df[df[col].map(math.isfinite)]
            dropped = initial_len - len(df)
            if dropped > 0:
                self.logger.info(f"Dropped {dropped} rows with non-finite values for {ticker}/{interval}")

        required_cols = ['datetime', 'ticker', 'interval']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' for {ticker}/{interval}. Cannot generate hash.")

        # Consistent hash generation using isoformat with microsecond precision
        df['hash'] = df.apply(lambda row: hashlib.sha256(f"{row['datetime'].strftime('%Y-%m-%dT%H:%M:%S.%f%z')}{row['ticker']}{row['interval']}".encode()).hexdigest(), axis=1)

        self.logger.info(f"✅ Processed {len(df)} rows for {ticker}/{interval} (after NaT and non-finite removal)")

        return df.to_dict('records')
