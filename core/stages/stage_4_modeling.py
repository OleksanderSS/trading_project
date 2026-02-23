#!/usr/bin/env python3
"""
Stage 4: Modeling (v5 - Simplified)

This version is drastically simplified because it now trusts Stage 2 to provide
a perfectly clean, high-quality dataset with no missing values in features
or targets. Its sole responsibility is to train and save the models.

Key Principles:
1.  **Trust in Upstream Quality:** Assumes the input DataFrame is 100% clean.
    All complex NaN handling has been removed as it's no longer needed.
2.  **Pure Modeling:** Focuses exclusively on splitting data, training models,
    evaluating them, and serializing the results.
3.  **Task-Oriented:** Continues to train separate models for regression and
    classification tasks.
4.  **Clear Feature Definition:** Uses a simple, predefined list of features
    from the configuration.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
from pathlib import Path

# --- Local Imports ---
from config.feature_config import get_features_by_layer

logger = logging.getLogger(__name__)

def run_stage_4_v2(master_df: pd.DataFrame) -> dict:
    """
    Orchestrates training on a pre-cleaned, high-quality dataset.
    """
    logger.info("--- Starting Stage 4 (v5 - Simplified): Pure Modeling ---")

    if master_df.empty:
        logger.error("Master DataFrame is empty. Cannot train models.")
        return {}

    # --- Feature Selection ---
    all_feature_layers = get_features_by_layer(['technical', 'sentiment', 'calendar', 'macro'])
    feature_columns = list(dict.fromkeys([item for sublist in all_feature_layers.values() for item in sublist]))
    
    available_features = [col for col in feature_columns if col in master_df.columns]
    logger.info(f"Using {len(available_features)} available features for training.")

    # --- Task Definitions ---
    tasks = {
        "regression": {
            "target": "target_regression_pct_change",
            "model": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        },
        "classification": {
            "target": "target_classification_direction",
            "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        },
    }

    models_and_metrics = {"features": {}}

    for task_name, config in tasks.items():
        target_col = config['target']
        if target_col not in master_df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping {task_name} task.")
            continue

        logger.info(f"--- Starting {task_name.capitalize()} Task ({target_col}) ---")
        
        X = master_df[available_features]
        y = master_df[target_col]

        if X.empty or y.empty:
            logger.warning(f"No data available for task '{task_name}'. Skipping.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        logger.info(f"Training {task_name} model on {len(X_train)} samples...")
        model = config['model']
        model.fit(X_train, y_train)
        
        logger.info("Evaluating model...")
        preds = model.predict(X_test)
        
        report_dict = {}
        if task_name == "regression":
            score = mean_squared_error(y_test, preds)
            logger.info(f"Evaluation Score (MSE): {score:.4f}")
        else: # Classification
            score = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)
            logger.info(f"Evaluation Score (Accuracy): {score:.4f}")
            logger.info(f"\nClassification Report:\n{classification_report(y_test, preds)}")
            report_dict = {"classification_report": report}

        models_and_metrics[task_name] = {
            "model": model,
            "evaluation_score": score,
            **report_dict
        }
        models_and_metrics["features"][task_name] = available_features

    if len(models_and_metrics) <= 1: # Only contains 'features' dict
        logger.error("No models were successfully trained. Skipping serialization.")
        return {}

    output_path = Path("data/stages/stage_4_models.pkl")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models_and_metrics, output_path)
    logger.info(f"✅ All models and metrics saved to {output_path}")

    logger.info("--- Stage 4 (v5) Finished ---")
    return {"models_path": str(output_path), "trained_tasks": list(models_and_metrics.keys())}
