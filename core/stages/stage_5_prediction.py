#!/usr/bin/env python3
"""
Stage 5: Prediction Layer

This stage is responsible for loading trained models and generating predictions
on a given feature set. It is designed to be a straightforward prediction step.
"""

import logging
import pandas as pd
import pickle
import os

logger = logging.getLogger(__name__)

def run_stage_5(features_df: pd.DataFrame, models_path: str = "data/stages/stage_4_models.pkl") -> pd.DataFrame:
    """
    Generates predictions using trained models.

    Args:
        features_df: DataFrame containing the features for prediction.
        models_path: Path to the saved models from Stage 4.

    Returns:
        A DataFrame with predictions.
    """
    logger.info("--- Starting Stage 5: Prediction ---")

    if not os.path.exists(models_path):
        logger.error(f"Models file not found at {models_path}. Aborting Stage 5.")
        return pd.DataFrame()

    try:
        with open(models_path, 'rb') as f:
            trained_models = pickle.load(f)
        logger.info(f"Successfully loaded models from {models_path}")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return pd.DataFrame()

    if not trained_models:
        logger.warning("No models were loaded. Skipping prediction.")
        return pd.DataFrame()

    predictions = {}
    for model_key, model_info in trained_models.items():
        if model_info['status'] == 'success':
            model = model_info['model']
            # This is a simplified prediction logic. It assumes the model can predict on the entire feature set.
            # A more robust implementation would match columns.
            try:
                predictions[f'prediction_{model_key}'] = model.predict(features_df)
            except Exception as e:
                logger.error(f"Failed to predict with model {model_key}: {e}")

    if not predictions:
        logger.warning("No predictions were generated.")
        return pd.DataFrame()

    predictions_df = pd.DataFrame(predictions)
    logger.info(f"Successfully generated predictions for {len(predictions)} models.")

    # Optionally, combine predictions with original features
    final_df = pd.concat([features_df, predictions_df], axis=1)
    
    output_path = "data/results/predictions.parquet"
    try:
        final_df.to_parquet(output_path, index=False)
        logger.info(f"Saved final predictions to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")

    logger.info("--- Stage 5: Prediction Finished ---")
    return final_df
