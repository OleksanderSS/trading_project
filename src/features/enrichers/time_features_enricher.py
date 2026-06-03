import logging
from typing import Any
import pandas as pd
from src.features.enrichers.base import BaseEnricher
from src.features.utils.time_utils import add_time_features

class TimeFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with time-based features as configured in features.yaml.

    ✅ Phase 4 Quality: Updated to use standardized error handling from BaseEnricher.
    """

    def __init__(self):
        super().__init__()
        # TimeFeaturesEnricher is enabled via enabled_enrichers, not separate config
        self.config: dict[str, Any] = {
            'enabled': True,
            'timestamp_col': 'datetime',
            'enabled_features': [
                'hour', 'day_of_week', 'day_of_month', 'day_of_year',
                'week_of_year', 'month_of_year', 'quarter',
                'market_session',
                'hour_sin', 'hour_cos',
                'day_of_week_sin', 'day_of_week_cos'
            ]
        }
        enabled_features = self.config['enabled_features']
        self.logger.info(f"TimeFeaturesEnricher initialized with {len(enabled_features)} features.")

    @property
    def name(self) -> str:
        return "time_features"

    @property
    def priority(self) -> int:
        return 10

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Implementation of time feature enrichment.

        Error handling is provided by BaseEnricher template method.
        """
        if not self.config.get('enabled', False):
            self.logger.info("Time feature enrichment is disabled in the config.")
            return df

        df_enriched = df.copy()
        timestamp_col = str(self.config.get('timestamp_col', 'datetime'))
        enabled_features = list(self.config.get('enabled_features', []))

        if timestamp_col not in df_enriched.columns:
            if isinstance(df_enriched.index, pd.DatetimeIndex):
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"Using DatetimeIndex as temporary '{timestamp_col}'.")
                df_enriched[timestamp_col] = df_enriched.index
                temp_col_created = True
            else:
                raise ValueError(f"Required column '{timestamp_col}' not found.")
        else:
            temp_col_created = False

        # Add time features using utility function
        df_enriched = add_time_features(
            df_enriched,
            timestamp_col=timestamp_col,
            enabled_features=enabled_features
        )

        if temp_col_created:
            df_enriched = df_enriched.drop(columns=[timestamp_col])

        return df_enriched

