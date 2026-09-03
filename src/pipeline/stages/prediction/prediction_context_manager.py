
import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger
from src.pipeline.modeling_context import is_pooled, rows_for_ticker

logger = ProjectLogger.get_logger(__name__)

class PredictionContextManager:
    """Manages context and data preparation for prediction stage."""

    def __init__(self):
        self.logger = ProjectLogger.get_logger('PredictionContextManager')

    def prepare_ticker_data(self, features_df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
        """Prepares ticker-specific data for prediction.

        Second copy of the filter that made Stage 5 produce zero predictions
        for every pooled champion (#210). Both copies now ask the one
        predicate in `modeling_context` instead of comparing to a literal.
        """
        rows = rows_for_ticker(features_df, ticker)
        if is_pooled(ticker) and 'ticker' in rows.columns:
            ticker_df = rows.groupby('ticker', sort=False).tail(50)
        else:
            ticker_df = rows.tail(50)
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
            if col in preserved_cols:
                continue
            try:
                ticker_df_clean[col] = pd.to_numeric(ticker_df_clean[col],
                    errors='coerce')
            except (ValueError, TypeError):
                ticker_df_clean = ticker_df_clean.drop(columns=[col],
                    errors='ignore')

        numeric_cols = [c for c in ticker_df_clean.columns if c not in preserved_cols]
        if numeric_cols:
            ticker_df_clean[numeric_cols] = ticker_df_clean[numeric_cols].replace([np.inf, -np.inf], np.nan)
            complete_rows = ticker_df_clean[numeric_cols].notna().all(axis=1)
            if not complete_rows.all():
                self.logger.warning(
                    f'âš ï¸ Dropping {int((~complete_rows).sum())} incomplete row(s) for {ticker} instead of filling zeros'
                )
                ticker_df_clean = ticker_df_clean.loc[complete_rows].copy()
                if ticker_df_clean.empty:
                    return None

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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Помилка генерації контексту: {e}')
            return 'unknown_context'
