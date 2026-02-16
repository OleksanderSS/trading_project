
"""
Run Robust Feature Selection (v2)

This module implements an advanced, ensemble-based feature selection process to
identify the most stable and predictive features for our models.

Methodology:
1.  **Feature Groups:** Splits all available features into two distinct pools:
    - `technical_features`: Standard price/indicator-based features.
    - `context_features`: Macro, sentiment, and other market-state indicators.

2.  **Time-Series Stability Check:** The data is split into multiple time-based folds.
    Feature selection is performed on each fold to ensure that selected features
    are consistently important across different market regimes.

3.  **Ensemble of Methods ("Voting"):** For each fold, we evaluate features using
    three different techniques:
    - **Statistical:** `f_regression` for fast linear dependency checks.
    - **Model-Based:** `LightGBM Feature Importance` for a non-linear view.
    - **Permutation-Based:** `Permutation Importance` for a robust, model-agnostic assessment.

4.  **Robustness Score:** Scores from all methods and folds are normalized and
    averaged to produce a final "Robustness Score" for each feature.

5.  **Final Selection:** For each target, we select the top N technical and top M
    contextual features based on this robust score.

Output:
-   `data/feature_sets_v2.json`: A JSON file mapping each target to its list of
    stable technical features (for the main model) and context features (for the
    shadow model).
"""

import logging
import pandas as pd
import numpy as np
import json
import lightgbm as lgb
from sklearn.feature_selection import f_regression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict

# --- Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
MASTER_FEATURESET_PATH = "data/master_featureset.parquet"
FEATURE_SETS_PATH = "data/feature_sets_v2.json"

N_SPLITS_TIME_VALIDATION = 5
TOP_N_TECHNICAL_FEATURES = 50
TOP_M_CONTEXT_FEATURES = 20

def get_feature_scores(X, y):
    """Calculates feature scores using an ensemble of methods."""
    scores = pd.DataFrame(index=X.columns)
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Method 1: f_regression
    f_vals, _ = f_regression(X_imputed, y)
    scores['f_regression'] = f_vals

    # Method 2: LightGBM Importance
    lgbm = lgb.LGBMRegressor(random_state=42, verbosity=-1)
    lgbm.fit(X, y)
    scores['lgbm_importance'] = lgbm.feature_importances_

    # Method 3: Permutation Importance
    perm_result = permutation_importance(lgbm, X, y, n_repeats=5, random_state=42, n_jobs=-1)
    scores['permutation_importance'] = perm_result.importances_mean
    
    return scores

def run_robust_feature_selection():
    """Main function to run the robust feature selection process."""
    logger.info("====== Starting Robust Feature Selection Module (v2) ======")

    try:
        df = pd.read_parquet(MASTER_FEATURESET_PATH)
        logger.info(f"Loaded master featureset with shape {df.shape}")
    except FileNotFoundError:
        logger.error(f"{MASTER_FEATURESET_PATH} not found. Run data pipeline.")
        return

    # Identify feature groups and targets
    targets = [col for col in df.columns if col.startswith('target_')]
    context_features = [col for col in df.columns if col.startswith('context_')]
    base_feature_cols = [col for col in df.columns if not col.startswith(('target_', 'context_')) and col not in ['event_time', 'news_published_at', 'ticker']]

    logger.info(f"Found {len(targets)} targets, {len(base_feature_cols)} technical features, and {len(context_features)} context features.")

    final_feature_sets = {}

    for target in targets:
        logger.info(f"--- Processing target: {target} ---")
        target_df = df.dropna(subset=[target])
        
        tscv = TimeSeriesSplit(n_splits=N_SPLITS_TIME_VALIDATION)
        fold_scores = {'technical': [], 'context': []}

        for train_idx, test_idx in tscv.split(target_df):
            fold_df = target_df.iloc[train_idx]
            y = fold_df[target]

            # Process technical features
            X_tech = fold_df[base_feature_cols]
            tech_scores = get_feature_scores(X_tech, y)
            fold_scores['technical'].append(tech_scores)

            # Process context features
            X_context = fold_df[context_features]
            context_scores = get_feature_scores(X_context, y)
            fold_scores['context'].append(context_scores)

        # Aggregate scores across folds
        final_scores = {}
        for feature_type in ['technical', 'context']:
            if not fold_scores[feature_type]: continue
            # Concatenate scores from all folds
            agg_scores = pd.concat(fold_scores[feature_type])
            # Scale each method's scores to be comparable (0-1 range)
            scaler = MinMaxScaler()
            scaled_scores = pd.DataFrame(scaler.fit_transform(agg_scores), columns=agg_scores.columns, index=agg_scores.index)
            # Group by feature name and calculate the mean score across all folds and methods
            final_scores[feature_type] = scaled_scores.groupby(scaled_scores.index).mean().mean(axis=1)

        # Select top N and M features
        if 'technical' in final_scores:
            top_tech = final_scores['technical'].nlargest(TOP_N_TECHNICAL_FEATURES).index.tolist()
        else: top_tech = []
        
        if 'context' in final_scores:
            top_context = final_scores['context'].nlargest(TOP_M_CONTEXT_FEATURES).index.tolist()
        else: top_context = []

        final_feature_sets[target] = {
            'technical_features': top_tech,
            'context_features': top_context
        }
        logger.info(f"Selected {len(top_tech)} technical and {len(top_context)} context features for {target}.")

    # Save the results
    with open(FEATURE_SETS_PATH, 'w') as f:
        json.dump(final_feature_sets, f, indent=4)
    
    logger.info(f"Robust feature sets saved to {FEATURE_SETS_PATH}")
    logger.info("====== Robust Feature Selection Module Finished ======")

if __name__ == "__main__":
    run_robust_feature_selection()
