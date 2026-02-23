# c:/trading_project/core/data_handler.py
"""
DataHandler Module

Provides reliable and configurable data accumulation and management operations.
This module is designed to prevent data loss and handle I/O operations
in a structured way.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

class DataHandler:
    """
    Handles loading, saving, and accumulating DataFrame data.
    """

    @staticmethod
    def accumulate_data(
        new_data: pd.DataFrame,
        storage_path: Path,
        deduplication_keys: List[str],
        columns_to_drop: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Safely accumulates new data with existing data from a parquet file.

        This function avoids the pitfalls of the previous implementation:
        1. It does NOT automatically drop all 'object' columns, preventing data loss.
        2. Column dropping is now explicit and configurable via 'columns_to_drop'.
        3. It ensures the storage directory exists before writing.
        4. Deduplication is based on a clear, provided list of keys.

        Args:
            new_data (pd.DataFrame): The new data to add.
            storage_path (Path): Path to the parquet file for storage.
            deduplication_keys (List[str]): A list of column names to use for identifying
                                            and removing duplicates.
            columns_to_drop (Optional[List[str]], optional): A list of columns to explicitly
                                                              drop before saving. Defaults to None.

        Returns:
            pd.DataFrame: The full, combined, and deduplicated DataFrame.
        """
        logger.info(f"Starting data accumulation for '{storage_path}'.")
        
        if new_data.empty:
            logger.warning("New data is empty. No accumulation will be performed.")
            if storage_path.exists():
                return pd.read_parquet(storage_path)
            return pd.DataFrame()

        # Ensure the parent directory for the storage path exists
        storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing data if the file exists
        if storage_path.exists():
            try:
                existing_data = pd.read_parquet(storage_path)
                logger.info(f"Loaded existing data from '{storage_path}'. Shape: {existing_data.shape}")
                
                # Ensure consistent data types before concatenation to avoid errors
                for col in new_data.columns:
                    if col in existing_data.columns and existing_data[col].dtype != new_data[col].dtype:
                        try:
                            # Attempt to cast existing data's column to new data's type
                            existing_data[col] = existing_data[col].astype(new_data[col].dtype)
                        except (ValueError, TypeError):
                             logger.warning(
                                 f"Could not cast column '{col}' to a consistent type. "
                                 f"Existing: {existing_data[col].dtype}, New: {new_data[col].dtype}. "
                                 "Proceeding with object type."
                             )
                             existing_data[col] = existing_data[col].astype('object')
                             new_data[col] = new_data[col].astype('object')

                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                logger.info(f"Combined data shape: {combined_data.shape}")
            except Exception as e:
                logger.error(f"Error reading existing data from '{storage_path}': {e}. Starting fresh.")
                combined_data = new_data.copy()
        else:
            logger.info(f"No existing data found at '{storage_path}'. Starting with new data.")
            combined_data = new_data.copy()
        
        # --- Deduplication ---
        # Ensure deduplication keys exist in the DataFrame
        valid_dedup_keys = [key for key in deduplication_keys if key in combined_data.columns]
        if not valid_dedup_keys:
            logger.error(f"None of the deduplication keys {deduplication_keys} found in the DataFrame. Cannot deduplicate.")
        else:
            if len(valid_dedup_keys) != len(deduplication_keys):
                logger.warning(f"Missing some deduplication keys. Using available keys: {valid_dedup_keys}")
            
            initial_rows = len(combined_data)
            combined_data.drop_duplicates(subset=valid_dedup_keys, keep='last', inplace=True)
            rows_dropped = initial_rows - len(combined_data)
            logger.info(f"Deduplication complete. Dropped {rows_dropped} duplicate rows.")

        # --- Explicit Column Dropping ---
        if columns_to_drop:
            actual_cols_to_drop = [col for col in columns_to_drop if col in combined_data.columns]
            if actual_cols_to_drop:
                combined_data.drop(columns=actual_cols_to_drop, inplace=True)
                logger.info(f"Dropped specified columns: {actual_cols_to_drop}")

        # --- Save the result ---
        try:
            combined_data.to_parquet(storage_path, index=False)
            logger.info(f"Successfully saved accumulated data to '{storage_path}'. Final shape: {combined_data.shape}")
        except Exception as e:
            logger.error(f"Failed to save data to '{storage_path}': {e}")
            # Depending on requirements, you might want to raise the exception here
            # raise e

        return combined_data
