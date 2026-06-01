import pandas as pd
import logging
from typing import Dict, Any, Optional
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class DataSchemaMapper:
    """Handles mapping of raw table data to canonical pipeline schema keys."""

    def map_to_schema(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Map table names to schema keys and wrap in result dict."""
        result = {}
        for table_name, df in raw_data.items():
            schema_key = self._get_schema_key(table_name, df)
            if schema_key:
                result[schema_key] = df
        logger.info(f'Mapped {len(raw_data)} tables to {len(result)} schema keys')
        return result

    def _get_schema_key(self, table_name: str, df: pd.DataFrame) -> str:
        """Get schema key for table name."""
        mapping_rules = [
            ('news', 'news'), 
            (('market', 'yahoo', 'yf'), 'market_data'), 
            (('fred', 'macro'), 'macro_data'), 
            (('sentiment', 'aai'), 'sentiment_data'), 
            (('fear_greed', 'vix'), 'market_sentiment'), 
            (('sec', 'insider'), 'institutional_data'),
            (('trends', 'google'), 'trends_data'), 
            (('economic', 'calendar'), 'economic_data'), 
            (('reddit', 'social'), 'social_sentiment'),
            (('huggingface', 'ml'), 'ml_features')
        ]
        for patterns, schema_key in mapping_rules:
            if isinstance(patterns, tuple):
                if any(pattern in table_name.lower() for pattern in patterns):
                    return schema_key
            elif patterns in table_name.lower():
                return schema_key
        
        if table_name == 'raw_data' and isinstance(df, pd.DataFrame):
            logger.warning("Detected legacy 'raw_data' table name, remapping to market_data.")
            return 'market_data'
            
        return self._handle_additional_data(table_name)

    def _handle_additional_data(self, table_name: str) -> str:
        """Handle additional data mapping."""
        logger.info(f"Mapping table '{table_name}' to additional_data")
        return f'additional_{table_name}'
