#!/usr/bin/env python3
"""
Stage 5: Model Analysis and Selection Layer

This final stage is responsible for consolidating and analyzing the results from
both local ("light") and external ("heavy") model training sessions. Its goal is
to determine the best performing model for each task.

Responsibilities:
1.  Load the training summary from local light models (Stage 4).
2.  Load the results from heavy models trained externally (e.g., in Colab).
3.  Compare models based on key performance metrics.
4.  Generate a final report or "ranking" of the best model for each ticker/timeframe.
"""

import logging
import pandas as pd
import json
import os

logger = logging.getLogger(__name__)


def _load_heavy_model_results(path: str) -> dict:
    """
    Loads the results of heavy models from a JSON file (produced by Colab).
    
    Expected format is a dictionary where keys are "ticker_timeframe"
    and values contain metrics, e.g.:
    { "SPY_15m": { "model_type": "transformer", "target": "heavy_classification", 
                   "metrics": { "f1_score": 0.65, "accuracy": 0.7 } } }
    """
    if not os.path.exists(path):
        logger.warning(f"Heavy model results file not found at '{path}'. Continuing without them.")
        return {}
    try:
        with open(path, 'r') as f:
            results = json.load(f)
            logger.info(f"Successfully loaded {len(results)} heavy model results from {path}.")
            return results
    except Exception as e:
        logger.error(f"Failed to load or parse heavy model results from {path}: {e}")
        return {}

def _unify_results(light_summary: dict, heavy_results: dict) -> pd.DataFrame:
    """
    Combines light and heavy model results into a single, comparable DataFrame.
    """
    records = []
    # Process light models
    for key, data in light_summary.items():
        if data.get('status') != 'success':
            continue
        records.append({
            'model_key': key,
            'model_type': 'light_lgbm',
            'target': 'light_regression',
            'metric_name': 'r2_score', # Primary metric for regression
            'metric_value': data['metrics'].get('r2_score', -999),
            'model_path': data.get('model_path')
        })

    # Process heavy models
    for key, data in heavy_results.items():
        records.append({
            'model_key': key,
            'model_type': data.get('model_type', 'heavy_unknown'),
            'target': data.get('target', 'heavy_classification'),
            'metric_name': 'f1_score', # Primary metric for classification
            'metric_value': data['metrics'].get('f1_score', -999),
            'model_path': data.get('model_path', 'external')
        })
    
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records)


def run_stage_5(
    light_model_summary: dict,
    heavy_model_results_path: str = "./data/stages/heavy_model_results.json"
) -> pd.DataFrame:
    """
    Orchestrates the analysis and selection of the best models.

    Args:
        light_model_summary: The output dictionary from Stage 4.
        heavy_model_results_path: Path to the JSON file with heavy model results from Colab.

    Returns:
        A DataFrame ranking the best models for each task.
    """
    logger.info("--- Starting Stage 5: Model Analysis and Selection ---")

    # 1. Load results from both sources
    heavy_results = _load_heavy_model_results(heavy_model_results_path)

    # 2. Unify results into a single DataFrame for easy comparison
    unified_df = _unify_results(light_model_summary, heavy_results)

    if unified_df.empty:
        logger.error("No successful model results to analyze. Aborting Stage 5.")
        return pd.DataFrame()

    # 3. Select the best model for each key (ticker_timeframe)
    # We sort by metric value in descending order and pick the first one for each group.
    best_models = unified_df.sort_values('metric_value', ascending=False).groupby('model_key').first()
    
    logger.info("Determined best performing model for each task:")
    print(best_models)
    
    # 4. Placeholder for the final vector comparison between the top light and top heavy model
    # This is where your final, most complex comparison logic would go.
    logger.info("[Placeholder] Final vector comparison between best light and best heavy model to be implemented here.")

    # 5. Save the final ranking to a file
    output_path = "./data/stages/final_model_ranking.csv"
    try:
        best_models.to_csv(output_path)
        logger.info(f"✅ Final model ranking saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save final ranking: {e}")

    logger.info("--- Stage 5: Model Analysis and Selection Finished ---")
    return best_models


if __name__ == '__main__':
    logger.info("Running standalone test for Stage 5...")

    # 1. Create dummy Stage 4 summary
    dummy_light_summary = {
        "SPY_15m": {
            'status': 'success',
            'model_path': 'models/trained/light_models/SPY_15m_lgbm.pkl',
            'metrics': {'r2_score': 0.15, 'mean_absolute_error': 0.2}
        },
        "QQQ_15m": {
            'status': 'success',
            'model_path': 'models/trained/light_models/QQQ_15m_lgbm.pkl',
            'metrics': {'r2_score': 0.12, 'mean_absolute_error': 0.3}
        }
    }

    # 2. Create dummy Colab results file
    dummy_heavy_results_path = "./data/stages/heavy_model_results.json"
    os.makedirs(os.path.dirname(dummy_heavy_results_path), exist_ok=True)
    dummy_heavy_results = {
        "SPY_15m": {
            "model_type": "transformer",
            "target": "heavy_classification",
            "metrics": {"f1_score": 0.65, "accuracy": 0.7}
        },
        "QQQ_15m": {
            "model_type": "lstm",
            "target": "heavy_classification",
            "metrics": {"f1_score": 0.68, "accuracy": 0.72} # Better performance
        }
    }
    with open(dummy_heavy_results_path, 'w') as f:
        json.dump(dummy_heavy_results, f)

    # 3. Run Stage 5
    final_ranking = run_stage_5(dummy_light_summary, dummy_heavy_results_path)

    if not final_ranking.empty:
        logger.info("Stage 5 test successful!")
        print("\n--- Final Model Ranking ---")
        print(final_ranking)
        print("\n---------------------------")
    else:
        logger.error("Stage 5 test failed.")

