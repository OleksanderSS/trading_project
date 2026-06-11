from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

from .base import BaseEnricher

logger = ProjectLogger.get_logger("ContextMapEnricher")

class ContextMapEnricher(BaseEnricher):
    """
    Generates a 'Context Fingerprint' (Market State) based on signal changes.
    Loads noise filter thresholds from external YAML config.
    """

    @property
    def name(self) -> str:
        return "context_map"

    @property
    def priority(self) -> int:
        return 80

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__()
        self.config = config or {}

        # ✅ LOAD NOISE FILTER THRESHOLDS FROM CONFIG
        self.noise_filter_thresholds = {}
        self.temporal_features = set()
        self.default_dynamic_threshold = 0.005
        self.noise_sensitivity = 1.5

        # Attempt to load from noise_filter_config.yaml
        from pathlib import Path

        import yaml

        config_path = Path(__file__).parent.parent.parent / "config" / "noise_filter_config.yaml"

        try:
            if config_path.exists():
                with open(config_path, encoding='utf-8') as f:
                    noise_config = yaml.safe_load(f)
                    self.noise_filter_thresholds = noise_config.get('noise_filter_thresholds', {})
                    self.temporal_features = set(noise_config.get('temporal_features', []))
                    self.default_dynamic_threshold = noise_config.get('default_dynamic_threshold', 0.005)
                    self.noise_sensitivity = noise_config.get('noise_sensitivity', 1.5)
                    logger.info(f"✅ Loaded {len(self.noise_filter_thresholds)} noise thresholds from {config_path}")
            else:
                logger.warning(f"⚠️ Noise filter config not found: {config_path}. Using defaults.")
                self._load_defaults()
        except Exception as e:
            logger.error(f"❌ Failed to load noise config from {config_path}: {e}. Using defaults.")
            self._load_defaults()

        logger.info(f"ContextMapEnricher initialized with {len(self.noise_filter_thresholds)} noise thresholds")
        logger.info(f"Temporal features (not compared): {len(self.temporal_features)} features")

    def _load_defaults(self):
        """Loads default thresholds if config is not found."""
        self.noise_filter_thresholds = {
            'VIX': 0.02, '10Y_yield': 0.001, 'DXY': 0.003, 'SPY': 0.005,
            'RSI': 0.05, 'MACD': 0.01, 'BB_width': 0.02, 'ATR': 0.05,
            'volume': 0.1, 'close': 0.005, 'open': 0.005, 'high': 0.005, 'low': 0.005,
        }
        self.temporal_features = {
            'hour', 'day_of_week', 'day_of_month', 'day_of_year',
            'week_of_year', 'month_of_year', 'quarter', 'is_weekend'
        }
        logger.info("Loaded default noise thresholds")

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Public entry point for enrichment using the base template method."""
        return super().enrich(df, **kwargs)

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Generates a contextual fingerprint."""
        if df.empty:
            logger.warning("⚠️ Empty DataFrame received. Returning original DataFrame.")
            # ✅ FIX: Return original empty DataFrame instead of adding columns
            # This prevents validation error in BaseEnricher
            return df

        res_df = df.copy()
        context_columns = self._get_context_columns(df)

        if not context_columns:
            logger.warning("No numeric columns found for context map. Skipping enrichment.")
            # ✅ FIX: Return original DataFrame when no columns to process
            return df

        logger.info(f"Generating context map from {len(context_columns)} indicators")

        state_cols, temporal_cols = self._process_context_columns(res_df, context_columns)

        if state_cols or temporal_cols:
            self._generate_context_features(res_df, state_cols, temporal_cols)
            self._log_context_statistics(res_df, state_cols, temporal_cols)
        else:
            logger.warning("No state columns were processed. Skipping enrichment.")
            # ✅ FIX: Return original DataFrame when no state columns
            return df

        return res_df

    def _get_context_columns(self, df: pd.DataFrame) -> list[str]:
        """Get context columns for processing."""
        context_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude targets and utility columns
        # Fix: Handle None values in columns
        return [c for c in context_columns if c and isinstance(c, str) and not c.startswith('target_')
                and c not in ['hash', 'interval']]

    def _process_context_columns(self, res_df: pd.DataFrame, context_columns: list[str]) -> tuple:
        """Process all context columns and return state and temporal column lists."""
        state_cols: list[str] = []
        temporal_cols: list[str] = []

        for col in context_columns:
            if col not in res_df.columns:
                logger.debug(f"Column '{col}' not found. Skipping.")
                continue

            state_col_name = f"state_{col}"

            if col in self.temporal_features:
                self._process_temporal_column(res_df, col, state_col_name, temporal_cols)
            else:
                self._process_numeric_column(res_df, col, state_col_name, state_cols)

        return state_cols, temporal_cols

    def _process_temporal_column(self, res_df: pd.DataFrame, col: str, state_col_name: str, temporal_cols: list[str]):
        """Process temporal column - just normalize without comparison."""
        res_df[state_col_name] = res_df[col]
        temporal_cols.append(state_col_name)

    def _process_numeric_column(self, res_df: pd.DataFrame, col: str, state_col_name: str, state_cols: list[str]):
        """Process numeric column - compare with previous value."""
        threshold = self._get_threshold(res_df, col)
        prev_val = res_df[col].shift(1)
        change = (res_df[col] - prev_val) / prev_val.replace(0, np.nan)
        change = change.fillna(0)

        # Three states: -1 (falling), 0 (unchanged), 1 (rising)
        res_df[state_col_name] = np.where(change > threshold, 1,
                                    np.where(change < -threshold, -1, 0))
        state_cols.append(state_col_name)

    def _generate_context_features(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        """Generate context fingerprint and stability score."""
        state_cols + temporal_cols

        # Remove duplicate columns before generating features
        unique_state_cols = self._remove_duplicate_columns(res_df, state_cols)
        unique_temporal_cols = self._remove_duplicate_columns(res_df, temporal_cols)
        unique_all_cols = unique_state_cols + unique_temporal_cols

        # Fingerprint: combine all states using '|'
        res_df['context_fingerprint'] = res_df[unique_all_cols].astype(str).agg('|'.join, axis=1)

        # Stability: how many indicators are UNCHANGED (only for numerical, not temporal)
        if unique_state_cols:
            zero_counts = (res_df[unique_state_cols] == 0).sum(axis=1)
            res_df['context_stability'] = zero_counts / len(unique_state_cols)
        else:
            res_df['context_stability'] = 1.0

        # Update the column lists
        state_cols.clear()
        state_cols.extend(unique_state_cols)
        temporal_cols.clear()
        temporal_cols.extend(unique_temporal_cols)

    def _remove_duplicate_columns(self, df: pd.DataFrame, cols: list[str]) -> list[str]:
        """Remove duplicate columns that have identical values."""
        if not cols:
            return []

        unique_cols = []
        seen_values = set()

        for col in cols:
            if col not in df.columns:
                continue

            # Create a hashable representation of column values
            col_values = tuple(df[col].values)

            if col_values not in seen_values:
                unique_cols.append(col)
                seen_values.add(col_values)
            else:
                logger.debug(f"🗑️ Removing duplicate column: {col}")

        return unique_cols

    def _log_context_statistics(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        """Log context statistics and market state information."""
        all_state_cols = state_cols + temporal_cols

        if len(res_df) > 0 and state_cols:
            self._log_market_state(res_df, state_cols, temporal_cols)

        logger.info(f"✅ Context map: {len(state_cols)} numeric + {len(temporal_cols)} temporal = {len(all_state_cols)} total states")

    def _log_market_state(self, res_df: pd.DataFrame, state_cols: list[str], temporal_cols: list[str]):
        """Log detailed market state statistics."""
        last_idx = res_df.index[-1]
        latest_row = res_df[state_cols].iloc[-1]
        up_count = (latest_row == 1).sum()
        down_count = (latest_row == -1).sum()
        flat_count = (latest_row == 0).sum()

        logger.info(f"📊 Market State at {last_idx}: UP={up_count}, DOWN={down_count}, FLAT={flat_count}")
        logger.info(f"📊 Temporal features: {len(temporal_cols)}")
        logger.info(f"📜 Fingerprint sample: {res_df['context_fingerprint'].iloc[-1][:100]}...")

    def _get_threshold(self, df: pd.DataFrame, col: str) -> float:
        """
        Determines the noise threshold for an indicator.

        1. Uses noise_filter_thresholds if present
        2. Looks for a partial match (e.g. 'AMD_close' → 'close')
        3. Otherwise calculates a dynamic threshold based on IQR
        """
        # Direct match
        if col in self.noise_filter_thresholds:
            return float(self.noise_filter_thresholds[col])  # type: ignore

        # Partial match (e.g. 'AMD_close' contains 'close')
        for key, threshold in self.noise_filter_thresholds.items():
            if key in col:
                return float(threshold)  # type: ignore

        # Dynamic threshold based on volatility (IQR)
        changes = df[col].diff().abs().dropna()
        if not changes.empty and len(changes) > 10:
            q1, q3 = changes.quantile(0.25), changes.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                dynamic_threshold = float(max(iqr * self.noise_sensitivity, 1e-7))
                logger.debug(f"Dynamic threshold for {col}: {dynamic_threshold:.6f} (IQR={iqr:.6f})")
                return dynamic_threshold  # type: ignore

        # Fallback
        return self.default_dynamic_threshold
