import logging
from typing import Any

import pandas as pd
import numpy as np

from src.features.enrichers.base import BaseEnricher

logger = logging.getLogger(__name__)

class EconomicCalendarEnricher(BaseEnricher):
    """Enriches features with Economic Calendar events and Surprise Index."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.db_manager = kwargs.get('db_manager')

    def _convert_value(self, val_str: str) -> float:
        """Converts strings like '150K', '2.5M', '1.2%' to float."""
        if pd.isna(val_str) or not isinstance(val_str, str) or not val_str.strip():
            return np.nan
        val_str = val_str.strip().replace(',', '')
        try:
            if val_str.endswith('%'):
                return float(val_str[:-1])
            elif val_str.endswith('K'):
                return float(val_str[:-1]) * 1000
            elif val_str.endswith('M'):
                return float(val_str[:-1]) * 1000000
            elif val_str.endswith('B'):
                return float(val_str[:-1]) * 1000000000
            else:
                return float(val_str)
        except ValueError:
            return np.nan

    def _enrich_impl(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if df.empty:
            return df
            
        if not self.db_manager:
            logger.warning("No db_manager provided to EconomicCalendarEnricher.")
            return df

        try:
            # Load calendar data
            query = "SELECT timestamp, event, actual, forecast, previous, impact FROM economic_calendar"
            cal_df = self.db_manager.execute_query(query)
            
            if cal_df.empty:
                return df
                
            cal_df['timestamp'] = pd.to_datetime(cal_df['timestamp'], utc=True).dt.tz_localize(None)
            
            # Clean values
            cal_df['actual_val'] = cal_df['actual'].apply(self._convert_value)
            cal_df['forecast_val'] = cal_df['forecast'].apply(self._convert_value)
            
            # Calculate surprise index: (Actual - Forecast)
            cal_df['surprise'] = cal_df['actual_val'] - cal_df['forecast_val']
            
            # Standardize surprise by event type (very basic approach)
            cal_df['surprise_index'] = cal_df.groupby('event')['surprise'].transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
            
            # Aggregate daily surprise
            daily_surprise = cal_df.groupby(cal_df['timestamp'].dt.date)['surprise_index'].mean().reset_index()
            daily_surprise['timestamp'] = pd.to_datetime(daily_surprise['timestamp'])
            daily_surprise.set_index('timestamp', inplace=True)
            
            # Merge with main df
            df_merged = df.join(daily_surprise, how='left')
            df_merged['surprise_index'] = df_merged['surprise_index'].ffill().fillna(0)
            
            logger.info(f"Enriched with surprise_index. Max value: {df_merged['surprise_index'].max()}")
            return df_merged
            
        except Exception as e:
            logger.error(f"Failed to enrich economic calendar data: {e}", exc_info=True)
            return df

    def get_feature_names(self) -> list[str]:
        return ['surprise_index']
