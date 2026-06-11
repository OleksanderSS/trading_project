

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

            # Compute common default targets: returns (1d,5d,20d), direction, and volatility (1d,5d)
            results = {}
            # safe access to close
            if 'close' in df_enriched.columns:
                # compute per-ticker future prices to avoid cross-ticker leakage
                for shift in [1, 5, 20]:
                    name = f"target_return_{shift}d"
                    future_price = df_enriched.groupby('ticker')['close'].shift(-shift)
                    results[name] = (future_price - df_enriched['close']) / df_enriched['close']

                # direction (binary) for 1d
                results['target_return_1d_direction'] = (results['target_return_1d'] > 0).astype(float)

                # volatility as rolling std of 1d returns computed per-ticker
                ret1 = results['target_return_1d']
                # ensure per-ticker rolling: use groupby transform on a Series
                ret1_series = pd.Series(ret1, index=df_enriched.index)
                results['target_volatility_1d'] = ret1_series.groupby(df_enriched['ticker']).transform(lambda s: s.rolling(window=1).std())
                results['target_volatility_5d'] = ret1_series.groupby(df_enriched['ticker']).transform(lambda s: s.rolling(window=5).std())

            final_df = pd.DataFrame(results, index=df_enriched.index)
            for col in ['datetime', 'ticker']:
                if col in df_enriched.columns and col not in final_df.columns:
                    final_df[col] = df_enriched[col].values

            return final_df

        self.logger.info(f"🎯 Generating safe targets for {timeframe} (Rows: {len(df_enriched)})")

        results = {}
        for config in target_configs:
            name = config.get('name', 'unknown')
            try:
                # Basic validation
                shift_val = config.get('params', {}).get('shift', -1)
                window_val = config.get('params', {}).get('window', 1)

                # Check minimum data requirement
                required_len = window_val + abs(shift_val)
                if len(df_enriched) < required_len:
                    self.logger.warning(f"⚠️ Not enough data for {name} ({len(df_enriched)} < {required_len}). Skipping.")
                    continue

                base_col = config.get('params', {}).get('base_col', 'close')
                if base_col not in df_enriched.columns:
                    continue

                # Calculation logic
                calc_type = config.get('type', 'regression')

                # Use specialized calculators for actual logic
                from src.targets.calculators.classification_calculator import ClassificationCalculator
                from src.targets.calculators.indicator_prediction_calculator import IndicatorPredictionCalculator
                from src.targets.calculators.regression_calculator import RegressionCalculator

                mapping = {
                    "regression": RegressionCalculator,
                    "classification_binary": ClassificationCalculator,
                    "indicator_prediction": IndicatorPredictionCalculator,
                }

                calc_class = mapping.get(calc_type, RegressionCalculator)
                calc = calc_class()

                # All params from config (base_col, shift, method, window, etc.)
                params = config.get('params', {}).copy()

                results[name] = calc.calculate(df_enriched, **params)

            except Exception as e:
                self.logger.error(f"❌ Error in target {name}: {e}")

        final_df = pd.DataFrame(results, index=df_enriched.index)

        # Ensure 'datetime' and 'ticker' are available if they were in input
        for col in ['datetime', 'ticker']:
            if col in df_enriched.columns and col not in final_df.columns:
                final_df[col] = df_enriched[col].values

        return final_df

def get_temporal_target_guard(config_manager=None) -> TemporalTargetGuard:
    """Factory function for TemporalTargetGuard."""
    return TemporalTargetGuard(config_manager)

def generate_targets_quick(df: pd.DataFrame, timeframe: str, configs: list[dict]) -> pd.DataFrame:
    """Quick access function for target generation."""
    guard = get_temporal_target_guard()
    return guard.generate_targets_safe(df, timeframe, pd.Timestamp.now(), configs)
