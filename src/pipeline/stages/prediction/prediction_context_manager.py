import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class PredictionContextManager:
    """Manages context and data preparation for prediction stage."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger('PredictionContextManager')

    def prepare_ticker_data(self, features_df: pd.DataFrame, ticker: str) -> Optional[pd.DataFrame]:
        """Prepares ticker-specific data for prediction."""
        ticker_df = features_df[features_df['ticker'] == ticker].tail(50)
        if ticker_df.empty:
            self.logger.warning(f'⚠️ No data for ticker {ticker}')
            return None
        
        ticker_df_clean = ticker_df.copy()
        
        # Зберігаємо потрібні нам не-числові колонки
        preserved_cols = [
            'context_fingerprint',
            'context_pattern_id',
            'context_pattern_seq',
            'state_champion',
            'context_velocity',
        ]
        preserved_data = ticker_df_clean[[c for c in preserved_cols if c in ticker_df_clean.columns]].copy()

        metadata_cols = ['ticker', 'datetime', 'date', 'interval',
            'timeframe', 'hash', 'symbol']
        ticker_df_clean = ticker_df_clean.drop(columns=[c for c in
            metadata_cols if c in ticker_df_clean.columns], errors='ignore')
            
        for col in ticker_df_clean.columns:
            if col in preserved_cols: continue 
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col],
                    errors='coerce')
            except (ValueError, TypeError):
                ticker_df_clean = ticker_df_clean.drop(columns=[col],
                    errors='ignore')
                    
        ticker_df_clean = ticker_df_clean.fillna(0).replace([np.inf, -np.inf], 0)
        
        # Повертаємо збережені дані
        for c in preserved_data.columns:
            ticker_df_clean[c] = preserved_data[c]
            
        return ticker_df_clean

    def create_context_fingerprint(self, ticker_df: pd.DataFrame, market_regime: str) -> str:
        """Creates a unique fingerprint for the current context."""
        if 'context_pattern_id' in ticker_df.columns and len(ticker_df) > 0:
            return str(ticker_df['context_pattern_id'].iloc[-1])
        
        # Fallback
        try:
            regime_map = {'bull': 1, 'bear': -1, 'sideways': 0, 'volatile': 2}
            regime_val = regime_map.get(market_regime.lower(), 0)
            return f"legacy_{regime_val}"
        except Exception as e:
            logger.error(f'Помилка генерації контексту: {e}', exc_info=True)
            return 'unknown_context'
