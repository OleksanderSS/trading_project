#!/usr/bin/env python3
"""
Meta-Model Trainer for the Consensus Engine

This script is responsible for training the meta-model (StackedEnsemble) that forms
the core of the ConsensusEngine. It learns the optimal weights for combining predictions
from various base models.

The trained model is then serialized and used by the real-time ConsensusEngine.
"""
import sys
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import re

def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize path input to prevent path traversal attacks.
    
    Args:
        path_input: Input string that will be used in file paths
        
    Returns:
        Sanitized string safe for path construction
    """
    if not path_input:
        return ""
    
    # Remove path traversal characters
    sanitized = re.sub(r'[./\\]', '_', path_input)
    
    # Remove null bytes and other dangerous characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)
    
    # Limit length to prevent path overflow
    sanitized = sanitized[:100]
    
    return sanitized

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.config.unified_config_manager import get_current_config
from src.data.management.data_manager import DataManager
from src.core.error_handling.error_handler import get_error_handler

# Setup logger
ProjectLogger.setup_logging()
logger = ProjectLogger.get_logger(__name__)

def get_training_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Loads historical prediction data and actual outcomes from DuckDB for meta-model training.
    
    Returns:
        A tuple of (X_train, y_train).
    """
    logger.info("Loading historical prediction data for meta-model training from DataManager...")
    
    config_manager = get_current_config()
    data_manager = DataManager(config_manager)
    
    try:
        tables = data_manager.get_all_table_names()
        if 'predictions_history' not in tables:
            logger.warning("Table 'predictions_history' not found. Meta-model training cannot proceed with real data.")
            return pd.DataFrame(), pd.Series()

        # Query to fetch predictions and actual outcomes
        # We pivot the model_name to get a column for each model's prediction
        query = """
        SELECT 
            timestamp, 
            ticker, 
            model_name, 
            predicted_value, 
            actual_return
        FROM predictions_history
        WHERE actual_return IS NOT NULL
        """
        
        raw_df = data_manager.execute_query(query)
        
        if raw_df.empty:
            logger.warning("No historical prediction data with valid actual returns found.")
            return pd.DataFrame(), pd.Series()

        # Pivot to create feature matrix X (rows are events, columns are models)
        # We use mean if multiple predictions exist for same timestamp/ticker/model
        pivoted = raw_df.pivot_table(
            index=['timestamp', 'ticker'], 
            columns='model_name', 
            values='predicted_value',
            aggfunc='mean'
        )
        
        # Get corresponding actual outcomes (targets)
        # We assume actual_return is consistent for the same index
        outcomes = raw_df.groupby(['timestamp', 'ticker'])['actual_return'].first()
        
        # Align indices
        df_combined = pd.concat([pivoted, outcomes], axis=1).dropna()
        
        if df_combined.empty:
            logger.warning("Alignment between predictions and outcomes resulted in an empty dataset.")
            return pd.DataFrame(), pd.Series()

        X = df_combined.drop('actual_return', axis=1)
        y = df_combined['actual_return']
        
        logger.info(f"Loaded {len(X)} aligned samples with {len(X.columns)} base models from database.")
        return X, y

    except Exception as e:
        logger.error(f"Failed to load training data from DataManager: {e}")
        return pd.DataFrame(), pd.Series()
    # DataManager cleanup handled automatically

def main(output_path: str):
    """
    Main training and serialization function.
    """
    logger.info("Starting meta-model training for Consensus Engine...")
    
    # 1. Load data
    X_train, y_train = get_training_data()
    
    if X_train.empty or y_train.empty:
        logger.error("Training aborted: No data available for meta-model.")
        return
    
    # 2. Initialize the meta-model
    meta_model = StackedEnsemble()
    
    # 3. Train the model
    logger.info("Training the StackedEnsemble meta-model...")
    meta_model.train(X_train, y_train)
    logger.info("Training complete.")
    
    # 4. Serialize the trained model
    logger.info(f"Saving trained meta-model to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    meta_model.save(output_path)
    logger.info("Model saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consensus Engine Meta-Model Trainer")
    parser.add_argument(
        "--output",
        type=str,
        default="data/trained_models/consensus_meta_model.pkl",
        help="Path to save the trained model file."
    )
    args = parser.parse_args()
    
    try:
        sanitized_output = sanitize_path_input(args.output)
        main(sanitized_output)
    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        sys.exit(1)