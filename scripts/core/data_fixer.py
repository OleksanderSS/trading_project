#!/usr/bin/env python3
"""
Unified Data Fixer

Provides a structured class to apply various fixes to the dataset,
addressing common issues like outliers, missing values, and incorrect data types.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure the project root is in the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.core.logging.logger import ProjectLogger
from src.config.unified_config_manager import get_current_config

logger = ProjectLogger.get_logger("DataFixer")

class DataFixer:
    """Encapsulates data fixing logic for staged Parquet files."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initializes the fixer.
        """
        self.config = get_current_config()
        self.data_dir = data_path or Path(self.config.get_config('paths', {}).get('stages', 'data/stages'))
        if not self.data_dir.is_absolute():
            self.data_dir = project_root / self.data_dir
            
        if not self.data_dir.exists():
            logger.error(f"Data directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def run_all_fixes(self):
        """Runs all configured data fixing routines."""
        logger.info("Starting comprehensive data fixing process...")
        self.fix_file("merged_full_clean.parquet", self._apply_general_fixes)
        self.fix_file("stage1_price_data.parquet", self._apply_price_data_fixes)
        logger.info("All data fixing routines completed.")

    def fix_file(self, filename: str, fix_function):
        """
        Applies a specific fixing function to a given file.

        Args:
            filename (str): The name of the file to fix.
            fix_function (callable): The function that contains the fixing logic.
        """
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.warning(f"File '{filename}' not found. Skipping.")
            return

        logger.info(f"--- Applying fixes to {filename} ---")
        try:
            df = pd.read_parquet(file_path)
            original_shape = df.shape

            # Create a backup before modifying
            self._create_backup(file_path)

            # Apply the specific fixes
            df_fixed = fix_function(df)

            # Save the fixed dataframe
            df_fixed.to_parquet(file_path)
            logger.info(f"Successfully saved fixed file: {filename}")
            logger.info(f"Original shape: {original_shape}, New shape: {df_fixed.shape}")

        except Exception as e:
            logger.error(f"Failed to fix file {filename}: {e}", exc_info=True)

    def _create_backup(self, file_path: Path):
        """Creates a backup of a file if one doesn't already exist."""
        backup_path = file_path.with_suffix('.backup.parquet')
        if not backup_path.exists():
            try:
                original_df = pd.read_parquet(file_path)
                original_df.to_parquet(backup_path)
                logger.info(f"Created backup: {backup_path.name}")
            except Exception as e:
                logger.error(f"Failed to create backup for {file_path.name}: {e}")

    def _apply_general_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes common issues in a general DataFrame."""
        # Fix infinite values
        for col in df.select_dtypes(include=np.number).columns:
            inf_count = np.isinf(df[col]).sum()
            if inf_count > 0:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                logger.info(f"Replaced {inf_count} infinite values with NaN in column '{col}'.")

        # Fix technical indicator ranges and NaNs
        self._fix_technical_indicators(df)
        return df

    def _apply_price_data_fixes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fixes specific to price data (e.g., date ranges, duplicates)."""
        # Remove future dates
        if 'date' in df.columns:
            now = pd.Timestamp.now()
            future_mask = df['date'] > now
            if future_mask.any():
                df = df[~future_mask]
                logger.info(f"Removed {future_mask.sum()} rows with future dates.")

        # Remove duplicates
        original_len = len(df)
        df = df.drop_duplicates(subset=['date', 'ticker', 'interval'])
        if len(df) < original_len:
            logger.info(f"Removed {original_len - len(df)} duplicate rows.")

        return df

    def _fix_technical_indicators(self, df: pd.DataFrame):
        """Standardizes common technical indicators."""
        indicator_patterns = {
            'rsi': (0, 100, 50),
            'stochastic': (0, 100, 50),
            'macd': (None, None, 0),
            'bollinger': (None, None, 0) 
        }

        for pattern, config in indicator_patterns.items():
            self._fix_indicator_pattern(df, pattern, config)
        
        return df
    
    def _fix_indicator_pattern(self, df: pd.DataFrame, pattern: str, config: tuple):
        """Fix a specific indicator pattern."""
        clip_low, clip_high, fill_val = config
        matching_cols = [c for c in df.columns if pattern in c.lower()]
        
        for col in matching_cols:
            self._fix_indicator_column(df, col, clip_low, clip_high, fill_val)
    
    def _fix_indicator_column(self, df: pd.DataFrame, col: str, clip_low: float, clip_high: float, fill_val: float):
        """Fix individual indicator column."""
        if df[col].isnull().any():
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Filled NaN in '{col}' with {fill_val}.")
        
        if clip_low is not None:
            self._clip_indicator_column(df, col, clip_low, clip_high)
    
    def _clip_indicator_column(self, df: pd.DataFrame, col: str, clip_low: float, clip_high: float):
        """Clip indicator column to valid range."""
        clipped_before = df[col].copy()
        df[col] = df[col].clip(clip_low, clip_high)
        if not clipped_before.equals(df[col]):
            logger.info(f"Clipped '{col}' to range [{clip_low}, {clip_high}].")

if __name__ == "__main__":
    fixer = DataFixer()
    fixer.run_all_fixes()
