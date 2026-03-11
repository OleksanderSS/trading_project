import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Any, Optional
from .base import BaseEnricher
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ContextMapEnricher")

class ContextMapEnricher(BaseEnricher):
    """
    Generates a 'Context Fingerprint' (Market State) based on signal changes.
    It can use either a statically configured list of columns or a dynamic list
    provided at runtime (e.g., from a feature selector).
    """
    name = "context_map"
    priority = 80 # Runs after main feature generation, but before final selection

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        # Static thresholds for known columns
        self.static_thresholds = self.config.get('thresholds', {
            'VIX': 0.02, '10Y_yield': 0.001, 'DXY': 0.003, 'SPY': 0.005
        })
        # Static list of columns, used if no dynamic list is provided
        self.static_columns = self.config.get('context_columns', list(self.static_thresholds.keys()))
        # Default threshold for dynamically selected columns without a static one
        self.default_dynamic_threshold = self.config.get('default_dynamic_threshold', 0.005)
        self.noise_sensitivity = self.config.get('noise_sensitivity', 1.5)

        logger.info(f"ContextMapEnricher initialized. Static columns: {self.static_columns}")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generates a contextual fingerprint.
        
        The enricher now dynamically determines which columns to use:
        1. It looks for 'selected_features' in kwargs, which is expected to be a list 
           of feature names provided by a feature selection step.
        2. If not found, it falls back to the statically configured 'self.static_columns'.
        """
        if df.empty:
            return df

        res_df = df.copy()
        
        # Determine which columns to use for the context map
        context_columns = kwargs.get('selected_features', self.static_columns)
        if not context_columns:
            logger.warning("No columns available for context map generation. Skipping.")
            return df

        logger.info(f"Generating context map using columns: {context_columns}")

        state_cols = []
        for col in context_columns:
            state_col_name = f"state_{col}"
            if col not in res_df.columns:
                logger.debug(f"Column '{col}' for context map not found. Skipping.")
                continue

            # Determine the threshold for this column
            threshold = self._get_threshold(res_df, col)

            # Calculate change and apply tri-state logic
            prev_val = res_df[col].shift(1)
            change = (res_df[col] - prev_val) / prev_val.replace(0, np.nan)
            change = change.fillna(0)

            res_df[state_col_name] = np.where(change > threshold, 1,
                                        np.where(change < -threshold, -1, 0))
            state_cols.append(state_col_name)

        # Generate fingerprint and stability score
        if state_cols:
            res_df['context_fingerprint'] = res_df[state_cols].astype(str).agg('|'.join, axis=1)
            zero_counts = (res_df[state_cols] == 0).sum(axis=1)
            res_df['context_stability'] = zero_counts / len(state_cols)
            logger.info(f"Generated context fingerprint and stability for {len(res_df)} rows.")
        else:
            logger.warning("No state columns were processed for the context map.")

        return res_df

    def _get_threshold(self, df: pd.DataFrame, col: str) -> float:
        """
        Determines the appropriate noise threshold for a given column.
        
        1. Use statically defined threshold if available.
        2. Otherwise, calculate a dynamic threshold based on the feature's volatility (IQR).
        """
        if col in self.static_thresholds:
            return self.static_thresholds[col]
        
        # Dynamic threshold based on IQR of changes
        changes = df[col].diff().abs().dropna()
        if not changes.empty:
            q1, q3 = changes.quantile(0.25), changes.quantile(0.75)
            iqr = q3 - q1
            dynamic_threshold = max(iqr * self.noise_sensitivity, 1e-7)
            return dynamic_threshold
        
        return self.default_dynamic_threshold
