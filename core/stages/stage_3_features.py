#!/usr/bin/env python3
"""
Stage 3: Feature Engineering Layer (v4 - Simplified & Corrected)

This version corrects the logical flaws of its predecessor by acknowledging that
feature engineering on price data (technical indicators) is already handled
in Stage 1 and contextualized in Stage 2.

Key Principles:
1.  **No Re-computation:** Does NOT re-calculate technical indicators. It assumes they
    are already present in the event data from Stage 2 (e.g., 'pre_event_rsi_14').
2.  **Efficient Macro Merge:** Uses `merge_asof` to efficiently join the latest
    available macroeconomic data to each news event.
3.  **Advanced Target Generation:** Creates a comprehensive suite of target variables for
    various modeling tasks (regression, classification, etc.).
4.  **Meta-Feature Creation:** Generates high-level features like calendar features and
    the 'sentiment_correction_factor'.
5.  **Focus:** Its sole responsibility is to finalize the feature set for modeling,
    not to collect or enrich raw context.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def _generate_all_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Generates a comprehensive suite of target variables."""
    logger.info("Generating a comprehensive suite of target variables...")
    
    base_price_col = 'pre_event_close'
    if base_price_col not in df.columns:
        logger.error(f"Base price column '{base_price_col}' not found. Cannot generate targets.")
        return df

    final_price_col = 'post_event_2_close'
    if final_price_col in df.columns:
        df['target_regression_pct_change'] = ((df[final_price_col] - df[base_price_col]) / df[base_price_col]) * 100
    else:
        df['target_regression_pct_change'] = np.nan

    price_diff = df.get(final_price_col, df[base_price_col]) - df[base_price_col]
    noise_threshold = 0.05
    df['target_classification_direction'] = np.select(
        [price_diff > noise_threshold, price_diff < -noise_threshold],
        [1, -1], default=0
    )
    
    generated_targets = [col for col in df.columns if col.startswith('target_')]
    logger.info(f"Successfully generated {len(generated_targets)} target variables.")
    return df.dropna(subset=['target_regression_pct_change'])

def _add_reverse_sentiment_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the sentiment correction factor."""
    if 'news_sentiment_score' in df.columns and 'target_regression_pct_change' in df.columns:
        df['sentiment_correction_factor'] = df['target_regression_pct_change'] - (df['news_sentiment_score'] * 5.0)
    return df

def run_stage_3(stage1_data: dict, stage2_data: dict) -> dict:
    """
    Orchestrates the simplified and corrected feature engineering process.
    """
    logger.info("--- Starting Stage 3: Feature Engineering (v4 - Simplified) ---")

    macro_df = stage1_data.get('macro')
    events_df = stage2_data.get('events')

    if events_df is None or events_df.empty:
        logger.critical("Enriched events data from Stage 2 is missing. Aborting Stage 3.")
        return {}

    final_df = events_df.copy()
    final_df['event_time'] = pd.to_datetime(final_df['event_time'])
    final_df.sort_values('event_time', inplace=True)

    if macro_df is not None and not macro_df.empty:
        logger.info("Merging macroeconomic data...")
        macro_df = macro_df.copy()
        macro_df.reset_index(inplace=True)
        macro_df.rename(columns={'index': 'datetime'}, inplace=True)
        macro_df['datetime'] = pd.to_datetime(macro_df['datetime'])
        macro_df.sort_values('datetime', inplace=True)

        final_df = pd.merge_asof(
            final_df, macro_df, left_on='event_time', right_on='datetime',
            direction='backward', tolerance=pd.Timedelta('3d')
        )

    final_df['cal_day_of_week'] = final_df['event_time'].dt.dayofweek
    final_df['cal_hour_of_day'] = final_df['event_time'].dt.hour

    final_df = _generate_all_targets(final_df)
    final_df = _add_reverse_sentiment_analysis(final_df)
    
    output_path = "./data/stages/stage_3_master_featureset.parquet"
    try:
        final_df.to_parquet(output_path, index=False)
        logger.info(f"✅ Successfully saved the master feature set to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save the master feature set: {e}")
        return {}

    logger.info("--- Stage 3: Feature Engineering (v4) Finished ---")
    return {"master_dataset_path": output_path, "features_count": len(final_df.columns)}
