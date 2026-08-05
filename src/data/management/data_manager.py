import logging
import os
import re
import time
from contextlib import contextmanager
from typing import Any, ClassVar

import duckdb
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
from src.core.exceptions import DataLoadError, DataProcessingError
from src.data.validation.price_source_gate import price_source_issues

logger = logging.getLogger(__name__)

MEMORY_DB = ':memory:'


class IDatabaseManager:
    """Interface for database management."""

    def execute_query(self, query: str, params: list[Any] | None = None):
        raise NotImplementedError

    def fetch_all(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        raise NotImplementedError

    def fetch_one(self, query: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        raise NotImplementedError

    def fetch_data_from_table(self, table_name: str) -> pd.DataFrame:
        raise NotImplementedError

    def fetch_df(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def load_data_for_tickers(self, tickers: list[str], interval: str = '1d') -> pd.DataFrame:
        raise NotImplementedError

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: list[str] | None = None):
        raise NotImplementedError

    def table_exists(self, table_name: str) -> bool:
        raise NotImplementedError

    def filter_new_records(self, table_name: str, df: pd.DataFrame, unique_cols: list[str] | None = None) -> pd.DataFrame:
        raise NotImplementedError

    def get_all_table_names(self) -> list[str]:
        raise NotImplementedError

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        raise NotImplementedError


class DataManager(IDatabaseManager):
    _connections: ClassVar[dict[str, duckdb.DuckDBPyConnection]] = {}
    _connection_lock: ClassVar[dict[str, bool]] = {}

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: IErrorHandler | None = None):
        self.config_manager = config_manager
        raw_path = self.config_manager.get('paths.raw_db', MEMORY_DB)
        self.db_path = os.path.abspath(raw_path) if raw_path != MEMORY_DB else raw_path
        self.error_handler = error_handler or ErrorHandler()
        self._initialize_connection(force_new=False)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"DataManager instance configured with shared connection to '{self.db_path}'.")

    @classmethod
    def close_all_connections(cls):
        """Закриває всі відкриті з'єднання перед виходом з програми."""
        for db_path, conn in list(cls._connections.items()):
            try:
                conn.close()
                logger.info(f"Closed connection to '{db_path}'")
            except Exception as e:
                # Broad catch is intentional: the same connection is also registered
                # with ConnectionRegistry (see get_connection() below), whose own
                # atexit hook may close it again during interpreter shutdown. An
                # escaping exception there (e.g. duckdb.Error on a closed connection)
                # would make the OS-level exit code non-zero even after a clean run.
                logger.error(f"Error closing connection to '{db_path}': {e}", exc_info=True)
        cls._connections.clear()
        cls._connection_lock.clear()

    @classmethod
    def get_connection(cls, db_path: str, force_new: bool = False, retry_count: int = 3) -> duckdb.DuckDBPyConnection:
        """Повертає або створює з'єднання до DuckDB з повторними спробами (retry-логікою)."""
        db_path = os.path.abspath(db_path) if db_path != MEMORY_DB else db_path

        if not force_new and db_path in cls._connections:
            return cls._connections[db_path]

        logger.info(f"Attempting to create a new shared connection to '{db_path}'.")
        if force_new and db_path in cls._connections:
            try:
                cls._connections[db_path].close()
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f'Error closing connection: {e}', exc_info=True)
            del cls._connections[db_path]

        if force_new and os.path.exists(db_path) and db_path != MEMORY_DB:
            try:
                os.remove(db_path)
            except OSError as e:
                logger.error(f"Error removing database file '{db_path}': {e}", exc_info=True)
                raise DataLoadError(f"Error removing db file: {e}") from e

        last_error = None
        for attempt in range(retry_count):
            try:
                cls._connections[db_path] = duckdb.connect(
                    database=db_path,
                    read_only=False,
                    config={
                        'access_mode': 'READ_WRITE',
                        'threads': 4,
                        'max_memory': '2GB',
                        'temp_directory': 'data/temp',
                        'enable_object_cache': True,
                        'checkpoint_threshold': '1GB'
                    }
                )
                from src.core.system.connection_registry import ConnectionRegistry
                ConnectionRegistry.register(f'duckdb_{db_path}', cls._connections[db_path])
                logger.info(f"Successfully created DB connection to '{db_path}' (attempt {attempt + 1}/{retry_count})")
                return cls._connections[db_path]
            except duckdb.Error as e:
                logger.warning(f"Connection attempt {attempt + 1} failed for '{db_path}': {e}")
                last_error = e
                if attempt < retry_count - 1:
                    import random
                    time.sleep((2 ** attempt) + random.uniform(0, 0.5))

        try:
            # Резервна спроба зі стандартними налаштуваннями
            cls._connections[db_path] = duckdb.connect(database=db_path, read_only=False)
            return cls._connections[db_path]
        except duckdb.Error as e:
            logger.error(f"Failed to connect to database '{db_path}' after retries. Last error: {last_error}. Fallback error: {e}", exc_info=True)
            raise RuntimeError(f"Cannot connect to database '{db_path}': {last_error or e}") from e

    def _initialize_connection(self, force_new: bool = False):
        self.con = self.get_connection(self.db_path, force_new)

    @contextmanager
    def transaction(self):
        """Контекстний менеджер для транзакцій."""
        try:
            self.con.begin()
            yield self.con
            self.con.commit()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug('Transaction committed successfully')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.con.rollback()
            self.error_handler.handle_error(e, context={'action': 'transaction_rollback'})
            logger.error(f'Transaction rolled back due to error: {e}', exc_info=True)
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f'DataManager exiting with error: {exc_val}')
        return False

    def execute_query(self, query: str, params: list[Any] | None = None):
        try:
            self.con.execute(query, params)
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка виконання SQL запиту: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'query': query})
            raise DataLoadError(f"Database query execution failed: {e}") from e

    def fetch_all(self, query: str, params: list[Any] | None = None) -> list[dict[str, Any]]:
        try:
            return self.con.execute(query, params).fetchdf().to_dict('records')
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка при отриманні всіх записів: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'query': query})
            raise DataLoadError(f"Failed to fetch all records: {e}") from e

    def fetch_one(self, query: str, params: list[Any] | None = None) -> dict[str, Any] | None:
        try:
            result = self.con.execute(query, params).fetchdf().to_dict('records')
            return result[0] if result else None
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка при отриманні одного запису: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'query': query})
            raise DataLoadError(f"Failed to fetch one record: {e}") from e

    def fetch_data_from_table(self, table_name: str) -> pd.DataFrame:
        if not self.table_exists(table_name):
            raise DataLoadError(f"Table '{table_name}' does not exist.")
        try:
            # Table name is already validated and quoted by _quote_identifier
            query = f'SELECT * FROM {self._quote_identifier(table_name)}'
            df = self.fetch_df(query)
            logger.info(f"Successfully fetched {len(df)} records from '{table_name}'.")
            return df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка при читанні таблиці {table_name}: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'table_name': table_name, 'operation': 'fetch_data_from_table'})
            raise DataLoadError(f"Failed to fetch data from table '{table_name}': {e}") from e

    def fetch_df(self, query: str, params: list[Any] | None = None) -> pd.DataFrame:
        try:
            return self.con.execute(query, params).fetchdf()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка при виклику fetch_df: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'query': query})
            raise DataLoadError(f"Failed to fetch DataFrame: {e}") from e

    def _quote_identifier(self, identifier: str) -> str:
        """Екранує SQL ідентифікатори (імена таблиць/колонок) для запобігання SQL-ін'єкціям."""
        if not isinstance(identifier, str):
            raise ValueError('SQL identifier must be a string')
        if not re.match('^[A-Za-z_][A-Za-z0-9_]*$', identifier):
            raise ValueError(f'Invalid SQL identifier: {identifier!r}')
        return f'"{identifier}"'

    def load_data_for_tickers(self, tickers: list[str], interval: str = '1d') -> pd.DataFrame:
        if not tickers:
            return pd.DataFrame()
        try:
            placeholders = ', '.join(['?' for _ in tickers])
            query = f"""
                SELECT datetime, ticker, close
                FROM market_data_raw
                WHERE ticker IN ({placeholders}) AND interval = ?
                ORDER BY datetime
            """
            df = self.fetch_df(query, list(tickers) + [interval])
            if df.empty:
                msg = f'No data found for tickers {tickers} at interval {interval}'
                logger.warning(msg)
                raise DataLoadError(msg)

            pivot_df = df.pivot(index='datetime', columns='ticker', values='close')
            pivot_df.index = pd.to_datetime(pivot_df.index)
            logger.info(f'Loaded and aligned data for {len(tickers)} tickers. Shape: {pivot_df.shape}')
            return pivot_df
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка при завантаженні тікерів: {e}', exc_info=True)
            self.error_handler.handle_error(e, context={'tickers': tickers, 'operation': 'load_data_for_tickers'})
            raise DataLoadError(f"Failed to load data for tickers {tickers}: {e}") from e

    #: Tables holding OHLCV bars. Anything written here passes the price
    #: source gate first, whichever collector or import script is doing the
    #: writing.
    PRICE_TABLES: ClassVar[frozenset[str]] = frozenset({'market_data_raw'})

    def _gate_price_source(self, table_name: str, df: pd.DataFrame) -> None:
        """Refuse contaminated OHLCV at the door.

        The gate itself already existed -- as a PRIVATE METHOD on the Yahoo
        collector, one of 22. BaseCollector has no validation hook, so any
        second price source (a Kaggle dump, another API, a CSV import) would
        have written into market_data_raw with nothing in between.

        The cost of not having it here is measured. A yfinance shared-cache
        race filed one instrument's bars under another ticker; 63,038 rows
        sat in the database for four months, 4,668 of them at impossible
        prices (KO above 900, INTC above 900), until Stage 2's PriceFilter --
        three stages downstream -- rejected the entire 15m timeframe. The
        collector-side gate, once added, stopped it dead: zero contaminated
        rows after 2026-07-22.

        Raised rather than logged. A price table is the foundation every
        later stage stands on, and a bad row there is not recoverable by
        anything downstream -- it can only be discovered later and undone by
        hand, which is exactly what this cost.
        """
        if table_name not in self.PRICE_TABLES:
            return
        issues = price_source_issues(df)
        if not issues:
            return
        logger.error(
            "Refused %d row(s) for '%s': the price source gate found %s.",
            len(df), table_name, '; '.join(issues),
        )
        raise DataProcessingError(
            f"Price data for '{table_name}' failed the source gate: "
            + '; '.join(issues)
        )

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: list[str] | None = None):
        """Insert rows from `df` whose `unique_on` key does not already
        exist in `table_name`. Despite the name, this is INSERT-IF-ABSENT,
        not update-on-conflict: a row whose key already exists is silently
        skipped, never rewritten, even if its other column values differ
        from what's stored (see _prepare_upsert_df).

        This is intentional for how this project treats raw source data —
        prices, VIX readings, news, macro releases — as an immutable
        historical chronicle (see pipeline_executor.py's fingerprinting
        docstring: "Raw data ... is a permanent chronicle — it never
        expires"). Silently allowing updates here would let a later run
        rewrite what an earlier bar/reading/article looked like, which is
        exactly the kind of retroactive change that can leak
        look-ahead information into a backtest.

        If a specific table genuinely needs update-on-conflict semantics
        (e.g. correcting a known-bad value), that should be a separate,
        explicitly-named method — not a silent mode change here, since
        every other caller of `upsert()` is relying on existing keys being
        left alone.
        """
        if df.empty:
            return

        self._gate_price_source(table_name, df)

        unique_on = unique_on or self._detect_unique_columns(df)
        df = self._clean_numeric_data(df, table_name)

        try:
            if not self.table_exists(table_name):
                self._handle_new_table(table_name, df, unique_on)
            else:
                self._handle_existing_table(table_name, df, unique_on)

            if self._should_checkpoint(table_name):
                self.con.execute('CHECKPOINT')
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Checkpoint executed for critical table '{table_name}'")

            if unique_on:
                self._verify_no_duplicates(table_name, unique_on)

        except (duckdb.Error, Exception) as e:
            logger.error(f"Помилка під час виконання upsert у '{table_name}': {e}",
                exc_info=True)
            self.error_handler.handle_error(e, context={'table': table_name, 'dataframe_columns': list(df.columns)})
            raise DataLoadError(f"Failed to upsert data into '{table_name}': {e}") from e

    def _detect_unique_columns(self, df: pd.DataFrame) -> list[str]:
        if 'hash' in df.columns:
            return ['hash']
        if 'key_hash' in df.columns:
            return ['key_hash']
        return []

    def _handle_new_table(self, table_name: str, df: pd.DataFrame, unique_on: list[str]):
        self.con.register('df_to_upsert', df)
        try:
            self.con.execute(
                f'CREATE TABLE {self._quote_identifier(table_name)} AS SELECT * FROM df_to_upsert'
            )
            if unique_on:
                self._create_unique_index(table_name, unique_on)
            logger.info(f"Table '{table_name}' created and {len(df)} records inserted.")
        except (duckdb.Error, Exception) as e:
            logger.error(f"Помилка створення нової таблиці '{table_name}': {e}", exc_info=True)
            raise DataLoadError(f"Failed to create new table '{table_name}': {e}") from e
        finally:
            self.con.unregister('df_to_upsert')

    def _handle_existing_table(self, table_name: str, df: pd.DataFrame, unique_on: list[str]):
        if unique_on:
            self._ensure_unique_index(table_name, unique_on)

        table_schema = self.get_table_schema(table_name)
        existing_cols = set(table_schema.keys())

        df_insert = self._prepare_upsert_df(table_name, df, unique_on, existing_cols)

        if df_insert.empty:
            logger.info(f"No new records to insert into '{table_name}' (all duplicates filtered)")
            return

        self._execute_upsert_insert(table_name, df_insert, existing_cols)

    def _prepare_upsert_df(self, table_name: str, df: pd.DataFrame, unique_on: list[str], existing_cols: set) -> pd.DataFrame:
        df_insert = df.copy()

        if unique_on:
            # Internal deduplication
            valid_unique_on = [c for c in unique_on if c in df_insert.columns]
            if valid_unique_on:
                before = len(df_insert)
                df_insert = df_insert.drop_duplicates(subset=valid_unique_on, keep='first')
                if len(df_insert) < before:
                    logger.warning(f"Removed {before - len(df_insert)} internal duplicates before upsert into '{table_name}'")

            # External deduplication against table
            valid_unique_in_table = [c for c in unique_on if c in existing_cols and c in df_insert.columns]
            if valid_unique_in_table:
                # Use all columns in unique_on for composite key deduplication
                quoted_cols = ', '.join([self._quote_identifier(c) for c in valid_unique_in_table])
                try:
                    existing_keys_df = self.con.execute(
                        f'SELECT {quoted_cols} FROM {self._quote_identifier(table_name)}'
                    ).fetchdf()

                    # Create composite keys from all unique_on columns
                    if len(valid_unique_in_table) == 1:
                        # Single column: use simple set
                        existing_keys_set = {str(k) for k in existing_keys_df[valid_unique_in_table[0]].tolist()}
                        df_insert_keys = df_insert[valid_unique_in_table[0]].astype(str)
                    else:
                        # Multiple columns: use tuple composite keys
                        existing_keys_set = set(tuple(row) for row in existing_keys_df[valid_unique_in_table].values)
                        # BUGFIX: must share df_insert's index (not a fresh
                        # RangeIndex) — the drop_duplicates() call above does
                        # not reset the index, so a mismatched index here
                        # raises "Unalignable boolean Series provided as
                        # indexer" the moment df_insert has any non-contiguous
                        # index (e.g. after internal duplicates were dropped).
                        # Reproduced against vix_data in production (2026-07-21,
                        # 2026-07-23 runs) before this fix.
                        df_insert_keys = pd.Series(
                            [tuple(row) for row in df_insert[valid_unique_in_table].values],
                            index=df_insert.index,
                        )

                except (duckdb.Error, Exception) as e:
                    logger.error(f"Error fetching existing keys for deduplication in '{table_name}': {e}", exc_info=True)
                    self.error_handler.handle_error(e, context={'table_name': table_name, 'unique_on': valid_unique_in_table})
                    raise DataLoadError(f"Failed to fetch existing keys for '{table_name}': {e}") from e

                before = len(df_insert)
                df_insert = df_insert[~df_insert_keys.isin(existing_keys_set)]
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Filtered to {len(df_insert)} new records for '{table_name}' (excluded {before - len(df_insert)} duplicates based on {len(valid_unique_in_table)} columns: {valid_unique_in_table})")

        # Schema alignment
        common_cols = [c for c in df_insert.columns if c in existing_cols]
        if len(common_cols) < len(df_insert.columns):
            extra = set(df_insert.columns) - existing_cols
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Dropping {len(extra)} columns not in '{table_name}' schema: {extra}")
            df_insert = df_insert[common_cols]

        return df_insert

    def _execute_upsert_insert(self, table_name: str, df_insert: pd.DataFrame, existing_cols: set):
        self.con.register('df_to_upsert', df_insert)
        try:
            try:
                self.con.execute(
                    f"""
                    INSERT INTO {self._quote_identifier(table_name)} BY NAME
                    SELECT * FROM df_to_upsert
                    """
                )
                logger.info(f"Inserted {len(df_insert)} records into '{table_name}'.")
            except (duckdb.Error, Exception) as insert_error:
                logger.error(f"BY NAME insert failed for '{table_name}', attempting fallback: {insert_error}", exc_info=True)
                insert_cols = [c for c in df_insert.columns if c in existing_cols]
                if insert_cols:
                    col_list = ', '.join([self._quote_identifier(c) for c in insert_cols])
                    self.con.execute(
                        f"""
                        INSERT INTO {self._quote_identifier(table_name)} ({col_list})
                        SELECT {col_list} FROM df_to_upsert
                        """
                    )
                    logger.info(f"Fallback insert succeeded for '{table_name}' with {len(insert_cols)} columns.")
                else:
                    raise DataLoadError(f"Insert failed for '{table_name}': {insert_error}") from insert_error
        finally:
            try:
                self.con.unregister('df_to_upsert')
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.error(f"Failed to unregister df_to_upsert: {e}", exc_info=True)

    def _create_unique_index(self, table_name: str, unique_on: list[str]):
        cols = ', '.join([self._quote_identifier(c) for c in unique_on])
        index_name = f'idx_{table_name}_unique'
        try:
            self.con.execute(
                f'CREATE UNIQUE INDEX {self._quote_identifier(index_name)} ON {self._quote_identifier(table_name)} ({cols})'
            )
            logger.info(f"Created unique index for '{table_name}' on columns: {unique_on}")
        except (duckdb.Error, Exception) as idx_e:
            logger.error(f'Помилка створення індексу для нової таблиці {table_name}: {idx_e}', exc_info=True)
            raise DataLoadError(f"Could not create unique index for '{table_name}': {idx_e}") from idx_e

    def table_exists(self, table_name: str) -> bool:
        try:
            result = self.con.execute(
                'SELECT table_name FROM duckdb_tables() WHERE table_name = ?',
                [table_name]
            ).fetchone()
            return result is not None
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка при перевірці існування таблиці {table_name}: {e}', exc_info=True)
            raise DataLoadError(f"Failed to check existence of table '{table_name}': {e}") from e

    def filter_new_records(self, table_name: str, df: pd.DataFrame, unique_cols: list[str] | None = None) -> pd.DataFrame:
        if not self.table_exists(table_name) or df.empty:
            return df

        hash_col = unique_cols[0] if unique_cols else 'hash'
        if hash_col not in df.columns:
            logger.warning(f"Hash column '{hash_col}' not found in DataFrame for table '{table_name}'. Cannot filter new records.")
            return df

        try:
            existing_hashes_tuples = self.con.execute(
                f'SELECT {self._quote_identifier(hash_col)} FROM {self._quote_identifier(table_name)}'
            ).fetchall()

            if not existing_hashes_tuples:
                return df

            existing_hashes = {str(h[0]).strip().lower() for h in existing_hashes_tuples if h[0] is not None}
            incoming_hashes = df[hash_col].astype(str).str.strip().str.lower()
            new_records_mask = ~incoming_hashes.isin(existing_hashes)

            filtered_df = df[new_records_mask]
            filtered_out_count = len(df) - len(filtered_df)
            if filtered_out_count > 0:
                logger.info(f"Filtered out {filtered_out_count} duplicate records for table '{table_name}'.")
            return filtered_df
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка фільтрації нових записів для таблиці {table_name}: {e}', exc_info=True)
            raise DataLoadError(f"Failed to filter new records for '{table_name}': {e}") from e

    def get_all_table_names(self) -> list[str]:
        try:
            tables = self.con.execute(
                "SELECT table_name FROM duckdb_tables() WHERE schema_name = 'main'"
            ).fetchall()
            return [table[0] for table in tables]
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка при отриманні списку таблиць: {e}', exc_info=True)
            raise DataLoadError(f"Failed to get table names: {e}") from e

    def get_table_schema(self, table_name: str) -> dict[str, str]:
        try:
            # ✅ FIX: Use DuckDB information_schema instead of SQLite PRAGMA
            schema_info = self.con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = ?",
                [table_name]
            ).fetchall()
            return {row[0]: row[1].upper() for row in schema_info}
        except (duckdb.Error, Exception) as e:
            logger.error(f'Помилка отримання схеми таблиці {table_name}: {e}', exc_info=True)
            raise DataLoadError(f"Failed to get schema for table '{table_name}': {e}") from e

    def _clean_numeric_data(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Clean numeric data without filling across entity or timeframe boundaries."""
        import numpy as np
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return df

        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        nan_count = df[numeric_cols].isna().sum().sum()

        if nan_count > 0:
            nan_pct = nan_count / (len(df) * len(numeric_cols)) * 100
            if nan_pct > 10:
                logger.error(f"Critical: >10% NaN values in '{table_name}' ({nan_pct:.2f}%)")
            else:
                logger.warning(f"Table '{table_name}': {nan_count} NaN values ({nan_pct:.2f}%)")

            group_cols = []
            if 'ticker' in df.columns:
                group_cols.append('ticker')
            if 'interval' in df.columns:
                group_cols.append('interval')
            if not group_cols:
                series_col = next(
                    (column for column in ('series_id', 'series') if column in df.columns),
                    None,
                )
                if series_col:
                    group_cols.append(series_col)
            time_col = next(
                (
                    column
                    for column in ('datetime', 'date', 'timestamp', 'published_at')
                    if column in df.columns
                ),
                None,
            )
            working = df.copy()
            working['_fill_original_order'] = np.arange(len(working))
            sort_cols = [*group_cols, *([time_col] if time_col else [])]
            if sort_cols:
                working = working.sort_values(sort_cols, kind='stable')

            total_filled = 0
            for col in numeric_cols:
                col_nan_before = working[col].isna().sum()
                if group_cols:
                    working[col] = working.groupby(
                        group_cols,
                        dropna=False,
                        sort=False,
                    )[col].ffill()
                else:
                    working[col] = working[col].ffill()
                col_nan_after = working[col].isna().sum()
                total_filled += (col_nan_before - col_nan_after)

            working = working.sort_values('_fill_original_order', kind='stable')
            df[numeric_cols] = working[numeric_cols].to_numpy()
            remaining_nan_count = df[numeric_cols].isna().sum().sum()
            if remaining_nan_count:
                logger.warning(
                    f"Table '{table_name}': {remaining_nan_count} leading NaN values left unfilled to avoid lookahead"
                )
            logger.info(
                f"Applied entity-safe causal forward-fill to {total_filled} NaN values "
                f"in '{table_name}' using groups={group_cols or ['table']}."
            )
        return df

    def _should_checkpoint(self, table_name: str) -> bool:
        """Визначає, чи потрібно виконувати CHECKPOINT для цієї таблиці."""
        critical_tables = self.config_manager.get('db.critical_tables', [
            'enriched_features', 'targets', 'model_results', 'predictions'
        ])
        return table_name in critical_tables

    def _ensure_unique_index(self, table_name: str, unique_on: list[str]) -> None:
        try:
            index_name = f'idx_{table_name}_unique'
            existing_indexes = self.con.execute(
                'SELECT index_name FROM duckdb_indexes() WHERE table_name = ? AND index_name = ?',
                [table_name, index_name]
            ).fetchall()

            if not existing_indexes:
                cols = ', '.join([self._quote_identifier(c) for c in unique_on])
                self.con.execute(
                    f'CREATE UNIQUE INDEX {self._quote_identifier(index_name)} ON {self._quote_identifier(table_name)} ({cols})'
                )
                logger.info(f"Created unique index for '{table_name}' on columns: {unique_on}")
        except Exception as idx_e:
            logger.warning(
                f"Could not create/verify unique index for '{table_name}': {idx_e}",
                exc_info=True)

    def _clean_duplicates(self, table_name: str, unique_on: list[str]) -> None:
        """Очищує дублікати у таблиці, залишаючи перший запис."""
        try:
            quoted_table = self._quote_identifier(table_name)
            logger.info(f"Cleaning duplicates in '{table_name}'...")

            # Створюємо тимчасову таблицю з унікальними записами
            self.con.execute("DROP TABLE IF EXISTS tmp_clean")
            self.con.execute(f"CREATE TEMP TABLE tmp_clean AS SELECT DISTINCT ON ({', '.join([self._quote_identifier(c) for c in unique_on])}) * FROM {quoted_table}")

            # Очищуємо оригінальну таблицю та перезаписуємо дані
            self.con.execute(f"DELETE FROM {quoted_table}")
            self.con.execute(f"INSERT INTO {quoted_table} SELECT * FROM tmp_clean")
            self.con.execute("DROP TABLE tmp_clean")
            logger.info(f"Successfully cleaned duplicates in '{table_name}'.")
        except (duckdb.Error, Exception) as clean_e:
            logger.error(f"Failed to clean duplicates in '{table_name}': {clean_e}", exc_info=True)
            raise DataLoadError(f"Failed to clean duplicates in '{table_name}': {clean_e}") from clean_e

    def _verify_no_duplicates(self, table_name: str, unique_on: list[str]) -> None:
        """Перевіряє відсутність дублікатів після додавання записів."""
        try:
            quoted_cols = ', '.join([self._quote_identifier(c) for c in unique_on])
            duplicate_check_query = f"""
                SELECT {quoted_cols}, COUNT(*) as cnt
                FROM {self._quote_identifier(table_name)}
                GROUP BY {quoted_cols}
                HAVING COUNT(*) > 1
            """
            duplicates = self.con.execute(duplicate_check_query).fetchdf()
            if not duplicates.empty:
                logger.warning(f"⚠️ Found {len(duplicates)} duplicate groups in '{table_name}' after upsert")
                self._clean_duplicates(table_name, unique_on)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"✅ No duplicates in '{table_name}' after upsert")
        except (duckdb.Error, Exception) as check_e:
            logger.warning(f'⚠️ Could not verify duplicates: {check_e}',
                exc_info=True)
            # Not raising here as this is a verification step that shouldn't break the main upsert flow if it fails.
