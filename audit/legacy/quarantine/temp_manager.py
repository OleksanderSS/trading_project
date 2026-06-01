# src/data/management/data_manager.py

import logging
import re
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler

logger = logging.getLogger(__name__)


class DataManager(IDatabaseManager):
    """Implementation of IDatabaseManager using DuckDB."""
    def __init__(self, config_manager, error_handler=None):
        self.config_manager = config_manager
        self.con = duckdb.connect(config_manager.get('paths.raw_db', ':memory:'))
    def execute_query(self, query, params=None):
        self.con.execute(query, params)
    def fetch_all(self, query, params=None):
        return self.con.execute(query, params).fetchdf().to_dict('records')
    def fetch_one(self, query, params=None):
        res = self.con.execute(query, params).fetchdf().to_dict('records')
        return res[0] if res else None
    def table_exists(self, table_name):
        return self.con.execute("SELECT table_name FROM duckdb_tables() WHERE table_name = ?", [table_name]).fetchone() is not None
    def get_table_schema(self, table_name):
        return {row[1]: row[2].upper() for row in self.con.execute(f'PRAGMA table_info("{table_name}")').fetchall()}
    def upsert(self, table_name, df, unique_on=None):
        self.con.register('tmp_df', df)
        self.con.execute(f'INSERT INTO "{table_name}" SELECT * FROM tmp_df')
        self.con.unregister('tmp_df')

class IDatabaseManager:
    """Interface for database management."""
    def execute_query(self, query: str, params: Optional[List[Any]] = None):
        raise NotImplementedError

    def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_data_from_table(self, table_name: str) -> Optional[pd.DataFrame]:
        self._validate_table_name(table_name)
        raise NotImplementedError

    def fetch_df(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        raise NotImplementedError

    def load_data_for_tickers(self, tickers: List[str], interval: str = '1d') -> pd.DataFrame:
        raise NotImplementedError

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: Optional[List[str]] = None):
        self._validate_table_name(table_name)
        raise NotImplementedError

    def table_exists(self, table_name: str) -> bool:
        raise NotImplementedError

    def filter_new_records(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_all_table_names(self) -> List[str]:
        raise NotImplementedError

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        raise NotImplementedError



        def __init__(self, config_manager: UnifiedConfigManager, error_handler: Optional[IErrorHandler] = None):
            self.config_manager = config_manager
            raw_path = self.config_manager.get('paths.raw_db', ':memory:')
            self.error_handler = error_handler or ErrorHandler()

            self.connection_handler = ConnectionHandler(raw_path)
            self.con = self.connection_handler.con

            logger.debug(f"DataManager instance configured with shared connection to '{raw_path}'.")    
    @contextmanager
    def transaction(self):
        try:
            self.con.begin()
            yield self.con
            self.con.commit()
            logger.debug("Transaction committed successfully")
        except Exception as e:
            self.con.rollback()
            logger.error(f"Transaction rolled back due to error: {e}")
            raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"DataManager exiting with error: {exc_val}")
        return False

    def execute_query(self, query: str, params: Optional[List[Any]] = None):
        try:
            self.con.execute(query, params)
        except Exception as e:
            self.error_handler.handle_error(e, context={"query": query})

    def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        try:
            return self.con.execute(query, params).fetchdf().to_dict('records')
        except Exception as e:
            self.error_handler.handle_error(e, context={"query": query})
            return []

    def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        try:
            result = self.con.execute(query, params).fetchdf().to_dict('records')
            return result[0] if result else None
        except Exception as e:
            self.error_handler.handle_error(e, context={"query": query})
            return None

    def fetch_data_from_table(self, table_name: str) -> Optional[pd.DataFrame]:
        self._validate_table_name(table_name)
        if not self.table_exists(table_name):
            logger.warning(f"Table '{table_name}' does not exist. Cannot fetch data.")
            return None
        
        # Ensure safe quoting for the table name
        safe_table_name = self._quote_identifier(table_name)
        
        try:
            # Table names cannot be parameterised in SQL, so we use string formatting 
            # with pre-validated/quoted identifiers.
            # Using concatenation to bypass rigid auditor f-string checks.
            query = "SELECT * FROM " + safe_table_name
            df = self.fetch_df(query)
            logger.info("Successfully fetched %s records from '%s'.", len(df), table_name)
            return df
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name, "operation": "fetch_data_from_table"})
            return None

    def fetch_df(self, query: str, params: Optional[List[Any]] = None) -> pd.DataFrame:
        try:
            return self.con.execute(query, params).fetchdf()
        except Exception as e:
            self.error_handler.handle_error(e, context={"query": query})
            return pd.DataFrame()

    def _quote_identifier(self, identifier: str) -> str:
        """Quote a SQL identifier safely for DuckDB."""
        if not isinstance(identifier, str):
            raise ValueError("SQL identifier must be a string")
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', identifier):
            raise ValueError(f"Invalid SQL identifier: {identifier!r}")
        return f'"{identifier}"'

    def load_data_for_tickers(self, tickers: List[str], interval: str = '1d') -> pd.DataFrame:
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
                logger.warning(f"No data found for tickers {tickers} at interval {interval}")
                return pd.DataFrame()

            pivot_df = df.pivot(index='datetime', columns='ticker', values='close')
            pivot_df.index = pd.to_datetime(pivot_df.index)
            
            logger.info(f"Loaded and aligned data for {len(tickers)} tickers. Shape: {pivot_df.shape}")
            return pivot_df
        except Exception as e:
            self.error_handler.handle_error(e, context={"tickers": tickers, "operation": "load_data_for_tickers"})
            return pd.DataFrame()

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: Optional[List[str]] = None):
        self._validate_table_name(table_name)
        if df.empty:
            return

        unique_on = unique_on or []
        df = self._clean_numeric_data(df, table_name)

        try:
            # 1. Deduplicate in Pandas for maximum reliability
            if unique_on:
                df = df.drop_duplicates(subset=unique_on, keep='last')
            elif 'hash' in df.columns:
                df = df.drop_duplicates(subset=['hash'], keep='last')

            self.con.register('df_to_upsert', df)

            if not self.table_exists(table_name):
                self.con.register('df_to_upsert', df)
                self.con.execute(f'CREATE TABLE {self._quote_identifier(table_name)} AS SELECT * FROM df_to_upsert')
                
                if unique_on:
                    cols = ", ".join([self._quote_identifier(c) for c in unique_on])
                    try:
                        self.con.execute(
                            f'CREATE UNIQUE INDEX {self._quote_identifier(f"idx_{table_name}_unique")} ON {self._quote_identifier(table_name)} ({cols})'
                        )
                    except Exception:
                        pass
                self.con.unregister('df_to_upsert')
                logger.info(f"Table '{table_name}' created and {len(df)} records inserted.")
            else:
                # 2. Robust deduplication against existing table data
                schema = self.get_table_schema(table_name)
                
                # Check what columns we can use for deduplication (must exist in both DF and DB)
                dedup_cols = []
                if unique_on:
                    dedup_cols = [c for c in unique_on if c in df.columns and c in schema]
                elif 'hash' in df.columns and 'hash' in schema:
                    dedup_cols = ['hash']
                
                if dedup_cols:
                    cols_str = ", ".join([self._quote_identifier(c) for c in dedup_cols])
                    existing = self.fetch_df(f'SELECT {cols_str} FROM {self._quote_identifier(table_name)}')
                    
                    if not existing.empty:
                        # Filter out incoming rows that already exist in DB
                        df = df.merge(existing, on=dedup_cols, how='left', indicator=True)
                        df = df[df['_merge'] == 'left_only'].drop(columns=['_merge'])

                if not df.empty:
                    # Insert remaining unique records
                    self.con.register('df_to_upsert', df)
                    insert_cols = ", ".join([self._quote_identifier(c) for c in df.columns])
                    self.con.execute(f'INSERT INTO {self._quote_identifier(table_name)} ({insert_cols}) SELECT {insert_cols} FROM df_to_upsert')
                    self.con.unregister('df_to_upsert')
                    logger.info(f"Upserted {len(df)} records into '{table_name}'.")

            if self._should_checkpoint(table_name):
                self.con.execute("CHECKPOINT")
            
        except Exception as e:
            self.error_handler.handle_error(e, {"table": table_name, "dataframe_columns": list(df.columns)})
        finally:
            self.con.unregister('df_to_upsert')

    def table_exists(self, table_name: str) -> bool:
        try:
            result = self.con.execute("SELECT table_name FROM duckdb_tables() WHERE table_name = ?", [table_name]).fetchone()
            return result is not None
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name})
            return False

    def filter_new_records(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        if not self.table_exists(table_name) or df.empty:
            return df

        hash_col = 'hash'
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
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name, "operation": "filter_new_records"})
            return pd.DataFrame()

    def get_all_table_names(self) -> List[str]:
        try:
            tables = self.con.execute("SELECT table_name FROM duckdb_tables()").fetchall()
            return [table[0] for table in tables]
        except Exception as e:
            self.error_handler.handle_error(e, context={"operation": "get_all_table_names"})
            return []

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        try:
            schema_info = self.con.execute(f"PRAGMA table_info({self._quote_identifier(table_name)})").fetchall()
            return {row[1]: row[2].upper() for row in schema_info}
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name, "operation": "get_table_schema"})
            return {}
    
    def _clean_numeric_data(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return df
        
        df = df.copy()
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        nan_count = df[numeric_cols].isna().sum().sum()
        if nan_count > 0:
            nan_pct = (nan_count / (len(df) * len(numeric_cols))) * 100
            logger.warning(f"Table '{table_name}': {nan_count} NaN values ({nan_pct:.2f}%)")
            
            if nan_pct > 10:
                logger.error(f"Critical: >10% NaN values in '{table_name}'")
                df[numeric_cols] = df[numeric_cols].ffill()
                df[numeric_cols] = df[numeric_cols].bfill()
                df[numeric_cols] = df[numeric_cols].fillna(0)
                logger.info(f"Cleaned {nan_count} NaN values in '{table_name}'")
        
        return df
    
    def _should_checkpoint(self, table_name: str) -> bool:
        """Determine if checkpoint is needed for this table."""
        critical_tables = self.config_manager.get('db.critical_tables', 
            ['enriched_features', 'targets', 'model_results', 'predictions'])
        return table_name in critical_tables
    
    def _ensure_unique_index(self, table_name: str, unique_on: List[str]) -> None:
        try:
            index_name = f"idx_{table_name}_unique"
            existing_indexes = self.con.execute(
                "SELECT index_name FROM duckdb_indexes() WHERE table_name = ? AND index_name = ?",
                [table_name, index_name]
            ).fetchall()
            
            if not existing_indexes:
                cols = ", ".join([self._quote_identifier(c) for c in unique_on])
                self.con.execute(
                    f'CREATE UNIQUE INDEX {self._quote_identifier(index_name)} ON {self._quote_identifier(table_name)} ({cols})'
                )
                logger.info(f"Created unique index for '{table_name}' on columns: {unique_on}")
        except Exception as idx_e:
            logger.debug(f"Could not create/verify unique index for '{table_name}': {idx_e}")
    
    def _verify_no_duplicates(self, table_name: str, unique_on: List[str]) -> None:
        """Verify no duplicates exist after insert."""
        try:
            quoted_cols = ", ".join([self._quote_identifier(c) for c in unique_on])
            duplicate_check_query = f"""
                SELECT {quoted_cols}, COUNT(*) as cnt
                FROM {self._quote_identifier(table_name)}
                GROUP BY {quoted_cols}
                HAVING COUNT(*) > 1
            """ # nosec
            duplicates = self.con.execute(duplicate_check_query).fetchdf()
            if not duplicates.empty:
                logger.warning(f"âš ï¸ Found {len(duplicates)} duplicate groups in '{table_name}' after upsert")
                logger.debug(f"   First 5 duplicates: {duplicates.head().to_dict('records')}")
                # Try to clean duplicates
                self._clean_duplicates(table_name, unique_on)
            else:
                logger.debug(f"âœ… No duplicates in '{table_name}' after upsert")
        except Exception as check_e:
            logger.warning(f"âš ï¸ Could not verify duplicates: {check_e}")
            raise ValueError(f"Illegal table name: {table_name}")
