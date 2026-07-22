#!/usr/bin/env python3
"""
Unified Data Merge Debugger

Provides a structured way to debug data merging issues at various stages
of the data processing pipeline.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

# It's better to use a logger for structured output
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataMergeDebugger")

class DataMergeDebugger:
    """A class to encapsulate all merge-related debugging checks."""

    def __init__(self, data_path: Path = project_root / "data" / "stages"):
        """
        Initializes the debugger.

        Args:
            data_path (Path): The path to the directory containing staged data files.
        """
        self.data_dir = data_path
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def run_all_checks(self):
        """Executes all debugging checks in a structured sequence."""
        logger.info("Starting comprehensive data merge debug...")
        self.check_stage_files_integrity()
        self.check_column_consistency()
        self.check_data_types_and_values()
        self.check_time_series_gaps()
        logger.info("All debug checks completed.")

    def check_stage_files_integrity(self):
        """Checks basic integrity of each Parquet file in the stages directory."""
        logger.info("--- Checking Stage File Integrity ---")
        stage_files = sorted(self.data_dir.glob("*.parquet"))

        for file_path in stage_files:
            try:
                df = pd.read_parquet(file_path)
                logger.info(f"  - [OK] {file_path.name}: Shape={df.shape}")

                key_cols = ['published_at', 'ticker', 'date']
                present_keys = [col for col in key_cols if col in df.columns]
                if not present_keys:
                    logger.warning(f"    - No primary key columns found in {file_path.name}")
                else:
                    # Check for nulls in the first key found
                    primary_key = present_keys[0]
                    if df[primary_key].isnull().any():
                        logger.warning(f"    - Null values found in primary key '{primary_key}' of {file_path.name}")

            except Exception as e:
                logger.error(f"  - [FAIL] Could not read {file_path.name}: {e}")

    def check_column_consistency(self):
        """Analyzes column consistency across all staged Parquet files."""
        logger.info("--- Checking Column Consistency ---")
        all_files = sorted(self.data_dir.glob("*.parquet"))
        if not all_files:
            logger.warning("No Parquet files found to check for consistency.")
            return

        file_columns: dict[str, set[str]] = {}
        for file_path in all_files:
            try:
                df = pd.read_parquet(file_path, columns=[]) # Read only metadata
                file_columns[file_path.name] = set(df.columns)
            except Exception as e:
                logger.error(f"Could not get columns for {file_path.name}: {e}")

        if not file_columns:
            logger.error("Could not retrieve column info from any file.")
            return

        # Find common and unique columns
        all_cols = set.union(*file_columns.values())
        common_cols = set.intersection(*file_columns.values())

        logger.info(f"Total unique columns across all files: {len(all_cols)}")
        logger.info(f"Common columns present in all files: {len(common_cols)}")

        for file_name, columns in file_columns.items():
            unique_to_file = columns - common_cols
            if unique_to_file:
                logger.info(f"  - Unique to {file_name} ({len(unique_to_file)}): {sorted(unique_to_file)[:5]}...")

    def check_data_types_and_values(self):
        """Checks data types and value ranges in the final merged file."""
        logger.info("--- Checking Data Types and Values in Merged File ---")
        merged_path = self.data_dir / "merged_full_clean.parquet"
        if not merged_path.exists():
            logger.warning("Final merged file 'merged_full_clean.parquet' not found. Skipping this check.")
            return

        try:
            df = pd.read_parquet(merged_path)
            logger.info(f"Analyzing {merged_path.name} with shape {df.shape}...")

            # Datetime columns
            dt_cols = df.select_dtypes(include=["datetime64[ns]"]).columns
            for col in dt_cols:
                min_dt, max_dt = df[col].min(), df[col].max()
                if pd.isna(min_dt):
                    logger.warning(f"  - Datetime column '{col}' contains only NaNs.")
                else:
                    logger.info(f"  - Datetime '{col}': Range [{min_dt.date()}] to [{max_dt.date()}]")
                    if max_dt > datetime.now() + pd.Timedelta(days=1):
                        logger.warning(f"    - Future dates detected in '{col}'!")

            # Numeric columns
            num_cols = df.select_dtypes(include=np.number).columns
            for col in num_cols:
                if df[col].isnull().all():
                    logger.warning(f"  - Numeric column '{col}' is all NaN.")
                    continue
                min_val, max_val = df[col].min(), df[col].max()
                if abs(max_val) > 1e9:
                    logger.warning(f"  - Numeric '{col}': Potential extreme max value ({max_val:.2e})")

            # Object/String columns
            obj_cols = df.select_dtypes(include=["object", "category"]).columns
            for col in obj_cols:
                unique_count = df[col].nunique()
                logger.info(f"  - Object '{col}': {unique_count} unique values")
                if unique_count == 0:
                    logger.warning(f"    - Object column '{col}' is empty.")

        except Exception as e:
            logger.error(f"Failed to analyze {merged_path.name}: {e}")

    def check_time_series_gaps(self, time_col: str = 'date', interval_col: str = 'interval', ticker_col: str = 'ticker'):
        """
        Analyzes time series data for large gaps.
        Focuses on a primary price data file if available.
        """
        logger.info("--- Checking Time Series Gaps ---")
        price_data_path = self.data_dir / "stage1_price_data.parquet"
        if not price_data_path.exists():
            logger.warning("'stage1_price_data.parquet' not found. Skipping gap check.")
            return

        try:
            df = pd.read_parquet(price_data_path)
            if not all(col in df.columns for col in [time_col, interval_col, ticker_col]):
                logger.warning(f"Skipping gap check: Missing one of required columns: '{time_col}', '{interval_col}', '{ticker_col}'")
                return

            df[time_col] = pd.to_datetime(df[time_col])

            for interval, group in df.groupby(interval_col):
                logger.info(f"  - Analyzing interval: {interval}")
                for ticker, ticker_data in group.groupby(ticker_col):
                    sorted_data = ticker_data.sort_values(time_col)
                    gaps = sorted_data[time_col].diff()
                    max_gap = gaps.max()

                    # Define a reasonable max gap threshold (e.g., 2 days for daily data)
                    # This threshold might need to be adjusted per interval.
                    threshold = pd.Timedelta(days=3)
                    if interval == '15m':
                        threshold = pd.Timedelta(hours=1)

                    if max_gap and max_gap > threshold:
                        logger.warning(f"    - Large gap for {ticker}: {max_gap} (Threshold: {threshold})")

        except Exception as e:
            logger.error(f"Failed during gap analysis: {e}")


if __name__ == "__main__":
    debugger = DataMergeDebugger()
    debugger.run_all_checks()
