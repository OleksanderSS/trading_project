
"""
Run Dual Simulation (v2)

This is the third module in the "Research Lab" pipeline. It runs the core
backtesting simulation based on the robust feature sets identified previously.

This version implements the "Shadow Modeling" architecture:
1.  For each experiment (Ticker, Target, Model Type), it trains two models in parallel:
    - **Main Model:** A primary predictive model (e.g., LightGBM) trained on the
      stable `technical_features`.
    - **Shadow Model:** A simpler, interpretable model (e.g., Decision Tree) trained
      on the stable `context_features`.

2.  Both models predict the same target, but from different "perspectives" (technical vs. contextual).

3.  The module logs the predictions from BOTH models for every point in the backtest.
    This allows the next module to analyze the effects of convergence and divergence
    between the technical and contextual views.

Output:
-   `data/dual_prediction_logs.parquet`: A log file where each row contains the
    timestamp, actual value, main model prediction, and shadow model prediction.
"""

import logging
import pandas as pd
import numpy as np
import json
import itertools
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from typing import List, Dict

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MASTER_FEATURESET_PATH = "data/master_featureset.parquet"
FEATURE_SETS_PATH = "data/feature_sets_v2.json"
PREDICTION_LOGS_PATH = "data/dual_prediction_logs.parquet"

# Define the models to be used
MAIN_MODEL = lgb.LGBMRegressor(random_state=42, verbosity=-1)
SHADOW_MODEL = DecisionTreeRegressor(max_depth=7, random_state=42)
MAIN_MODEL_NAME = "LGBM_main"
SHADOW_MODEL_NAME = "Tree_shadow"

def run_single_dual_backtest(df: pd.DataFrame, ticker: str, target: str, tech_features: List[str], ctx_features: List[str]) -> pd.DataFrame:
    """Runs a time-series backtest for a single experiment, training and logging predictions from both a main and a shadow model."""
    logger.debug(f"Running dual backtest for: {ticker}, {target}")

    # Prepare data for this specific experiment
    feature_cols = list(set(tech_features + ctx_features))
    experiment_data = df[feature_cols + [target]].dropna(subset=[target])
    if len(experiment_data) < 200: # Need enough data for meaningful backtest
        return pd.DataFrame()

    tscv = TimeSeriesSplit(n_splits=5)
    prediction_logs = []

    X = experiment_data[feature_cols]
    y = experiment_data[target]

    for train_idx, test_idx in tscv.split(X):
        # --- Prepare data for the fold ---
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # --- Train and Predict: Main Model ---
        X_train_tech, X_test_tech = X_train[tech_features], X_test[tech_features]
        try:
            MAIN_MODEL.fit(X_train_tech, y_train)
            main_preds = MAIN_MODEL.predict(X_test_tech)
        except Exception as e:
            logger.warning(f"Main model failed for {ticker}/{target}: {e}")
            main_preds = np.full(len(X_test), np.nan)

        # --- Train and Predict: Shadow Model ---
        X_train_ctx, X_test_ctx = X_train[ctx_features], X_test[ctx_features]
        if not ctx_features: # Handle cases with no context features
            shadow_preds = np.full(len(X_test), np.nan)
        else:
            try:
                SHADOW_MODEL.fit(X_train_ctx, y_train)
                shadow_preds = SHADOW_MODEL.predict(X_test_ctx)
            except Exception as e:
                logger.warning(f"Shadow model failed for {ticker}/{target}: {e}")
                shadow_preds = np.full(len(X_test), np.nan)

        # --- Log results for the fold ---
        fold_log = pd.DataFrame({
            'timestamp': y_test.index,
            'ticker': ticker,
            'target': target,
            'main_model_prediction': main_preds,
            'shadow_model_prediction': shadow_preds,
            'actual_value': y_test
        })
        prediction_logs.append(fold_log)

    return pd.concat(prediction_logs) if prediction_logs else pd.DataFrame()

def run_dual_simulation():
    """Main function to orchestrate the dual simulation process."""
    logger.info("====== Starting Dual Simulation Module (v2) ======")

    # 1. Load data and feature recipes
    try:
        df = pd.read_parquet(MASTER_FEATURESET_PATH).set_index('event_time')
        with open(FEATURE_SETS_PATH, 'r') as f:
            feature_sets = json.load(f)
        logger.info(f"Loaded data ({df.shape}) and {len(feature_sets)} feature sets.")
    except FileNotFoundError as e:
        logger.error(f"Data or feature sets file not found: {e}. Run previous modules.")
        return

    all_tickers = df['ticker'].unique().tolist()
    all_targets = list(feature_sets.keys())

    logger.info(f"Preparing to run {len(all_tickers) * len(all_targets)} dual experiments.")
    
    # 2. Main experiment loop
    all_prediction_logs = []
    for ticker, target in itertools.product(all_tickers, all_targets):
        
        ticker_df = df[df['ticker'] == ticker]
        
        # Get feature lists for this target
        features = feature_sets.get(target)
        if not features or not features.get('technical_features'):
            logger.warning(f"No technical features found for target '{target}'. Skipping.")
            continue

        tech_features = features['technical_features']
        ctx_features = features.get('context_features', []) # Context features are optional

        log_df = run_single_dual_backtest(ticker_df, ticker, target, tech_features, ctx_features)
        if not log_df.empty:
            all_prediction_logs.append(log_df)
    
    # 3. Save final combined log
    if not all_prediction_logs:
        logger.error("No prediction logs were generated from any experiment. Halting.")
        return

    final_logs_df = pd.concat(all_prediction_logs).reset_index()
    final_logs_df.to_parquet(PREDICTION_LOGS_PATH)

    logger.info(f"Generated {len(final_logs_df)} total dual predictions.")
    logger.info(f"Dual prediction logs saved to {PREDICTION_LOGS_PATH}")
    logger.info("====== Dual Simulation Module Finished ======")

if __name__ == "__main__":
    run_dual_simulation()
