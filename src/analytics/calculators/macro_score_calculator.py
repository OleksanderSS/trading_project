import logging

import pandas as pd
from sklearn.preprocessing import minmax_scale

logger = logging.getLogger(__name__)

class MacroScoreCalculator:
    """
    Calculates a composite macroeconomic score from various weighted indicators.
    This score represents a single, normalized value for the overall macro environment.
    """

    def __init__(self, indicators_config: dict[str, dict]):
        """
        Initializes the calculator with configuration for macro indicators.

        Args:
            indicators_config (Dict[str, Dict]): A dictionary where keys are indicator names
                                                 and values are their configurations (weight, direction).
        """
        if not indicators_config:
            raise ValueError("Indicators configuration cannot be empty.")
        self.indicators_config = indicators_config
        logger.info(f"MacroScoreCalculator initialized with indicators: {list(self.indicators_config.keys())}")

    def calculate_composite_score(self, macro_data: pd.DataFrame, rolling_window: int = 252) -> pd.DataFrame:
        """
        Calculates the composite macro score over a rolling window.

        Args:
            macro_data (pd.DataFrame): DataFrame with macro indicators as columns.
            rolling_window (int): The window size for rolling normalization (e.g., 252 for 1 year).

        Returns:
            pd.DataFrame: A DataFrame with the composite macro score and its components.
        """
        if not self._validate_macro_data(macro_data):
            return pd.DataFrame()

        individual_scores = self._calculate_individual_scores(macro_data, rolling_window)

        if not individual_scores:
            return self._create_empty_composite_score(macro_data)

        scores_df = pd.DataFrame(individual_scores, index=macro_data.index)
        composite_score = self._calculate_weighted_composite(scores_df)
        final_score = self._scale_final_score(composite_score)

        scores_df['composite_macro_score'] = final_score
        logger.info("Successfully calculated the composite macro score.")

        return scores_df

    def _validate_macro_data(self, macro_data: pd.DataFrame) -> bool:
        """Validate macro data input."""
        if not isinstance(macro_data, pd.DataFrame) or macro_data.empty:
            logger.warning("Macro data is empty or not a DataFrame. Returning empty DataFrame.")
            return False
        return True

    def _calculate_individual_scores(self, macro_data: pd.DataFrame, rolling_window: int) -> dict[str, pd.Series]:
        """Calculate individual indicator scores."""
        all_scores = {}

        for indicator, config in self.indicators_config.items():
            if indicator not in macro_data.columns:
                logger.warning(f"Indicator '{indicator}' not found in macro data. Skipping.")
                continue

            score = self._process_indicator(macro_data[indicator], config, rolling_window)
            if score is not None:
                all_scores[f"{indicator}_score"] = score

        return all_scores

    def _process_indicator(self, series: pd.Series, config: dict, rolling_window: int) -> pd.Series:
        """Process a single indicator to calculate its score."""
        clean_series = series.dropna()

        transformed_series = self._transform_series(clean_series, rolling_window)
        normalized_series = self._normalize_series(transformed_series, rolling_window)
        aligned_series = self._apply_directional_alignment(normalized_series, config)

        return aligned_series

    def _transform_series(self, series: pd.Series, rolling_window: int) -> pd.Series:
        """Transform series using percentage change for momentum."""
        return series.pct_change(fill_method=None, periods=int(rolling_window / 12))

    def _normalize_series(self, series: pd.Series, rolling_window: int) -> pd.Series:
        """Normalize series using rolling Z-score."""
        min_periods = int(rolling_window * 0.8)
        mean = series.rolling(window=rolling_window, min_periods=min_periods).mean()
        std = series.rolling(window=rolling_window, min_periods=min_periods).std()
        return (series - mean) / std.replace(0, 1)

    def _apply_directional_alignment(self, series: pd.Series, config: dict) -> pd.Series:
        """Apply directional alignment based on configuration."""
        if config.get('direction', 'positive') == 'negative':
            return -series
        return series

    def _create_empty_composite_score(self, macro_data: pd.DataFrame) -> pd.DataFrame:
        """Create empty composite score DataFrame."""
        logger.error("No indicators were processed. Cannot calculate composite score.")
        return pd.DataFrame(index=macro_data.index).assign(composite_macro_score=0.0)

    def _calculate_weighted_composite(self, scores_df: pd.DataFrame) -> pd.Series:
        """Calculate weighted composite score from individual scores."""
        weighted_sum = pd.Series(0.0, index=scores_df.index)
        available_weight = pd.Series(0.0, index=scores_df.index)
        total_weight = sum(config['weight'] for config in self.indicators_config.values())

        if total_weight == 0:
            logger.warning("Total weight of indicators is zero. Composite score will be zero.")
            return pd.Series(index=scores_df.index, dtype=float)

        for indicator, config in self.indicators_config.items():
            score_col = f"{indicator}_score"
            if score_col in scores_df.columns:
                weight = config['weight'] / total_weight
                valid_mask = scores_df[score_col].notna()
                weighted_sum.loc[valid_mask] += scores_df.loc[valid_mask, score_col] * weight
                available_weight.loc[valid_mask] += weight

        composite_score = pd.Series(index=scores_df.index, dtype=float)
        valid_weight = available_weight > 0
        composite_score.loc[valid_weight] = weighted_sum.loc[valid_weight] / available_weight.loc[valid_weight]
        return composite_score

    def _scale_final_score(self, composite_score: pd.Series) -> pd.Series:
        """Scale final composite score to 0-100 range."""
        scaled_score = pd.Series(index=composite_score.index, dtype=float)
        valid_score = composite_score.dropna()
        if valid_score.empty:
            return scaled_score
        if valid_score.nunique(dropna=True) <= 1:
            scaled_score.loc[valid_score.index] = 50.0
            return scaled_score
        scaled_values = minmax_scale(valid_score, feature_range=(0, 100))
        # ✅ FIX: deduplicate index before assignment to avoid length mismatch
        if valid_score.index.duplicated().any():
            valid_score_dedup = valid_score[~valid_score.index.duplicated(keep='last')]
            scaled_values_dedup = minmax_scale(valid_score_dedup, feature_range=(0, 100))
            scaled_score = pd.Series(index=composite_score.index, dtype=float)
            scaled_score.loc[valid_score_dedup.index] = scaled_values_dedup
        else:
            scaled_score.loc[valid_score.index] = scaled_values
        return scaled_score
