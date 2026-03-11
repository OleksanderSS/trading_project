# src/data/management/data_manager.py

import duckdb
import pandas as pd
import logging
from typing import List, Optional, Dict, Any
from contextlib import contextmanager
import os

from src.core.error_handling.error_handler import IErrorHandler, ErrorHandler
from src.config.unified_config_manager import UnifiedConfigManager

logger = logging.getLogger(__name__)

class IDatabaseManager:
    """Інтерфейс для керування базою даних."""
    def execute_query(self, query: str, params: Optional[List[Any]] = None):
        raise NotImplementedError

    def fetch_all(self, query: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_one(self, query: str, params: Optional[List[Any]] = None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    def fetch_data_from_table(self, table_name: str) -> Optional[pd.DataFrame]:
        raise NotImplementedError

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: List[str]):
        raise NotImplementedError

    def table_exists(self, table_name: str) -> bool:
        raise NotImplementedError

    def filter_new_records(self, table_name: str, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def get_all_table_names(self) -> List[str]:
        raise NotImplementedError

    def get_table_schema(self, table_name: str) -> Dict[str, str]:
        raise NotImplementedError

class DataManager(IDatabaseManager):
    _connections: Dict[str, duckdb.DuckDBPyConnection] = {}

    def __init__(self, config_manager: UnifiedConfigManager, error_handler: Optional[IErrorHandler] = None):
        self.config_manager = config_manager
        self.db_path = self.config_manager.get('paths.raw_db', ':memory:')
        self.error_handler = error_handler or ErrorHandler()
        self._initialize_connection(force_new=False)
        logger.debug(f"DataManager instance configured with shared connection to '{self.db_path}'.")

    @classmethod
    def get_connection(cls, db_path: str, force_new: bool = False) -> duckdb.DuckDBPyConnection:
        if force_new or db_path not in cls._connections:
            logger.info(f"Attempting to create a new shared connection to '{db_path}'.")
            if force_new and db_path in cls._connections:
                cls._connections[db_path].close()
                del cls._connections[db_path]

            if force_new and os.path.exists(db_path) and db_path != ':memory:':
                logger.warning(f"Performing aggressive cleanup for database: {db_path}")
                try:
                    os.remove(db_path)
                    logger.info(f"Removed stale file: {db_path}")
                except OSError as e:
                    logger.error(f"Error removing database file '{db_path}': {e}")

            cls._connections[db_path] = duckdb.connect(database=db_path, read_only=False)
            logger.info(f"Successfully created and initialized a new shared DB connection to '{db_path}'.")
        return cls._connections[db_path]

    def _initialize_connection(self, force_new: bool = False):
        self.con = self.get_connection(self.db_path, force_new)

    def execute_query(self, query: str, params: Optional[List[Any]] = None):
        try:
            self.con.execute(query, params)
            self.con.execute('CHECKPOINT;')
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
        if not self.table_exists(table_name):
            logger.warning(f"Table '{table_name}' does not exist. Cannot fetch data.")
            return None
        try:
            query = f'SELECT * FROM "{table_name}"'
            df = self.con.execute(query).fetchdf()
            logger.info(f"Successfully fetched {len(df)} records from '{table_name}'.")
            return df
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name, "operation": "fetch_data_from_table"})
            return None

    def upsert(self, table_name: str, df: pd.DataFrame, unique_on: List[str]):
        if df.empty:
            return

        try:
            self.con.register('df_to_upsert', df)
            if not self.table_exists(table_name):
                self.con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM df_to_upsert')
                if unique_on:
                    cols = ", ".join([f'"{c}"' for c in unique_on])
                    try:
                        self.con.execute(f'CREATE UNIQUE INDEX "idx_{table_name}_unique" ON "{table_name}" ({cols})')
                        logger.info(f"Created unique index for '{table_name}' on columns: {unique_on}")
                    except Exception as idx_e:
                        logger.warning(f"Could not create unique index for '{table_name}': {idx_e}")
                logger.info(f"Table '{table_name}' created and {len(df)} records inserted.")
            else:
                # Use a strategy to avoid duplicates even if filter_new_records missed some
                # DuckDB doesn't have ON CONFLICT without PK, so we use a subquery check
                if 'hash' in df.columns:
                    self.con.execute(f"""
                        INSERT INTO "{table_name}" BY NAME 
                        SELECT * FROM df_to_upsert 
                        WHERE hash NOT IN (SELECT hash FROM "{table_name}")
                    """)
                else:
                    self.con.execute(f'INSERT INTO "{table_name}" BY NAME SELECT * FROM df_to_upsert')
                logger.info(f"Upserted records into '{table_name}'.")
            
            self.con.execute('CHECKPOINT;')
            logger.debug(f"Database checkpoint forced after upsert into '{table_name}'.")

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
            existing_hashes_tuples = self.con.execute(f'SELECT "{hash_col}" FROM "{table_name}"').fetchall()
            if not existing_hashes_tuples:
                 return df

            # Robust hash comparison: strip and lowercase
            existing_hashes = {str(h[0]).strip().lower() for h in existing_hashes_tuples if h[0] is not None}
            
            # Prepare incoming hashes for comparison
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
            schema_info = self.con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
            return {row[1]: row[2].upper() for row in schema_info}
        except Exception as e:
            self.error_handler.handle_error(e, context={"table_name": table_name, "operation": "get_table_schema"})
            return {}