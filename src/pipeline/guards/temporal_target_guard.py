

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("TemporalTargetGuard")

class TemporalTargetGuard:
    """
    Safely generates targets without lookahead bias or crashing on small data.
    """
    def __init__(self, config_manager=None):
        self.logger = logger
        self.config_manager = config_manager

    def _generate_default_targets(self, df_enriched: pd.DataFrame) -> pd.DataFrame:
        """Generate default targets when no configs supplied."""
        results = {}
        if 'close' in df_enriched.columns:
            # compute per-ticker future prices to avoid cross-ticker leakage
            for shift in [1, 5, 20]:
                name = f"target_return_{shift}d"
                if 'ticker' in df_enriched.columns:
                    future_price = df_enriched.groupby('ticker')['close'].shift(-shift)  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
                else:
                    future_price = df_enriched['close'].shift(-shift)  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
                results[name] = (future_price - df_enriched['close']) / df_enriched['close']

            # direction (binary) for 1d
            ret1 = results['target_return_1d']
            results['target_return_1d_direction'] = ret1.gt(0).astype(float).where(ret1.notna())

            # volatility as rolling std of 1d returns computed per-ticker
            ret1_series = pd.Series(ret1, index=df_enriched.index)
            if 'ticker' in df_enriched.columns:
                grouped_ret1 = ret1_series.groupby(df_enriched['ticker'])
                results['target_volatility_1d'] = grouped_ret1.transform(lambda s: s.rolling(window=5, min_periods=1).std().shift(-1))  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
                results['target_volatility_5d'] = grouped_ret1.transform(lambda s: s.rolling(window=20, min_periods=1).std().shift(-5))  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
            else:
                results['target_volatility_1d'] = ret1_series.rolling(window=5, min_periods=1).std().shift(-1)  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD
                results['target_volatility_5d'] = ret1_series.rolling(window=20, min_periods=1).std().shift(-5)  # audit-ignore: NEGATIVE_SHIFT_LOOKAHEAD

        final_df = pd.DataFrame(results, index=df_enriched.index)
        for col in ['datetime', 'ticker']:
            if col in df_enriched.columns and col not in final_df.columns:
                final_df[col] = df_enriched[col].values

        return final_df

    def _validate_config(self, config: dict, df_enriched: pd.DataFrame) -> tuple:
        """Validate config and return (is_valid, reason)."""
        name = config.get('name', 'unknown')
        shift_val = config.get('params', {}).get('shift', -1)
        window_val = config.get('params', {}).get('window', 1)
        base_col = config.get('params', {}).get('base_col', 'close')

        # Check minimum data requirement
        required_len = window_val + abs(shift_val)
        if len(df_enriched) < required_len:
            return False, f"Not enough data for {name} ({len(df_enriched)} < {required_len}). Skipping."

        if base_col not in df_enriched.columns:
            return False, f"Base column {base_col} not found for {name}. Skipping."

        return True, None

    def _get_calculator(self, calc_type: str):
        """Get calculator instance for given calculation type."""
        from src.targets.calculators.classification_calculator import ClassificationCalculator
        from src.targets.calculators.indicator_prediction_calculator import IndicatorPredictionCalculator
        from src.targets.calculators.regression_calculator import RegressionCalculator

        mapping = {
            "regression": RegressionCalculator,
            "classification_binary": ClassificationCalculator,
            "indicator_prediction": IndicatorPredictionCalculator,
        }

        calc_class = mapping.get(calc_type, RegressionCalculator)
        return calc_class()

    def _process_target_config(self, config: dict, df_enriched: pd.DataFrame) -> tuple:
        """Process a single target config and return (name, result_or_None)."""
        name = config.get('name', 'unknown')
        try:
            is_valid, reason = self._validate_config(config, df_enriched)
            if not is_valid:
                self.logger.warning(f"⚠️ {reason}")
                return name, None

            calc_type = config.get('type', 'regression')
            calc = self._get_calculator(calc_type)
            params = config.get('params', {}).copy()

            result = calc.calculate(df_enriched, **params)
            return name, result

        except Exception as e:
            self.logger.error(f"❌ Error in target {name}: {e}")
            return name, None

    def _ensure_metadata_columns(self, final_df: pd.DataFrame, df_enriched: pd.DataFrame) -> pd.DataFrame:
        """Ensure datetime and ticker columns are present."""
        for col in ['datetime', 'ticker']:
            if col in df_enriched.columns and col not in final_df.columns:
                final_df[col] = df_enriched[col].values
        return final_df

    def generate_targets_safe(self,
                           df_enriched: pd.DataFrame,
                           timeframe: str,
                           current_time: pd.Timestamp,
                           target_configs: list[dict] | None = None) -> pd.DataFrame:
        """
        Main entry point for safe target generation.
        Matches the expected signature in Stage 3.
        """
        if df_enriched is None or df_enriched.empty or not target_configs:
            # Backward-compatible default target set when no configs supplied
            if df_enriched is None or df_enriched.empty:
                return pd.DataFrame(index=df_enriched.index if df_enriched is not None else None)

            return self._generate_default_targets(df_enriched)

        self.logger.info(f"🎯 Generating safe targets for {timeframe} (Rows: {len(df_enriched)})")

        results = {}
        for config in target_configs:
            name, result = self._process_target_config(config, df_enriched)
            if result is not None:
                results[name] = result

        final_df = pd.DataFrame(results, index=df_enriched.index)
        final_df = self._ensure_metadata_columns(final_df, df_enriched)

        return final_df

def get_temporal_target_guard(config_manager=None) -> TemporalTargetGuard:
    """Factory function for TemporalTargetGuard."""
    return TemporalTargetGuard(config_manager)

def generate_targets_quick(df: pd.DataFrame, timeframe: str, configs: list[dict]) -> pd.DataFrame:
    """Quick access function for target generation."""
    guard = get_temporal_target_guard()
    return guard.generate_targets_safe(df, timeframe, pd.Timestamp.now(), configs)
