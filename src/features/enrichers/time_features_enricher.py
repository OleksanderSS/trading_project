import pandas as pd
from src.features.enrichers.base import BaseEnricher
from src.core.logging.logger import ProjectLogger
from src.features.utils.time_utils import add_time_features
from src.config.unified_config_manager import get_current_config

logger = ProjectLogger.get_logger("TimeFeaturesEnricher")

class TimeFeaturesEnricher(BaseEnricher):
    """
    Enriches the DataFrame with time-based features as configured in features.yaml.
    """
    
    def __init__(self):
        self.config = get_current_config().get_config('time_features') or {}
        logger.info("TimeFeaturesEnricher initialized.")

    @property
    def name(self) -> str:
        return "time_features"

    @property
    def priority(self) -> int:
        return 10

    def enrich(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Adds configured time-based features to the DataFrame.
        """
        if not self.config.get('enabled', False):
            logger.info("Time feature enrichment is disabled in the config.")
            return df

        df_enriched = df.copy()
        timestamp_col = self.config.get('timestamp_col', 'datetime')
        enabled_features = self.config.get('enabled_features', [])

        if timestamp_col not in df_enriched.columns:
            if isinstance(df_enriched.index, pd.DatetimeIndex):
                logger.debug(f"Using DatetimeIndex as temporary '{timestamp_col}'.")
                df_enriched[timestamp_col] = df_enriched.index
                temp_col_created = True
            else:
                logger.error(f"Required column '{timestamp_col}' not found.")
                return df
        else:
            temp_col_created = False

        try:
            df_enriched = add_time_features(
                df_enriched,
                timestamp_col=timestamp_col,
                enabled_features=enabled_features
            )
            if temp_col_created:
                df_enriched = df_enriched.drop(columns=[timestamp_col])
        except Exception as e:
            logger.error(f"Error during time feature generation: {e}", exc_info=True)
            return df

        return df_enriched