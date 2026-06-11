"""
Improved Sentiment Features Enricher with better null handling for continue mode.
"""


import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class ImprovedSentimentEnricher:
    """Improved sentiment enricher that handles missing data better."""

    def __init__(self):
        self.sentiment_cache = {}  # Cache last known sentiment values

    def _handle_missing_sentiment(self, df: pd.DataFrame, sentiment_col: str) -> pd.DataFrame:
        """
        Handle missing sentiment data with smart interpolation.
        """
        # Instead of filling with 0, use smart interpolation
        df_enriched = df.copy()

        for ticker in df_enriched['ticker'].unique():
            ticker_mask = df_enriched['ticker'] == ticker
            ticker_data = df_enriched[ticker_mask]

            # Check if we have sentiment data
            if sentiment_col not in ticker_data.columns:
                # Add neutral sentiment
                df_enriched.loc[ticker_mask, sentiment_col] = 0.5  # Neutral sentiment
                continue

            sentiment_series = ticker_data[sentiment_col]

            # Use forward fill with limited window
            filled_sentiment = sentiment_series.ffill(limit=5)

            # For remaining NaN values, use exponential decay from last known value
            if filled_sentiment.isna().any():
                last_known = filled_sentiment.dropna().iloc[-1] if not filled_sentiment.dropna().empty else 0.5

                # Apply exponential decay
                decay_factor = 0.95
                for i, (idx, val) in enumerate(filled_sentiment.items()):
                    if pd.isna(val):
                        days_since_last = i
                        decayed_value = last_known * (decay_factor ** days_since_last)
                        filled_sentiment.loc[idx] = max(0.0, min(1.0, decayed_value))

            # Update cache
            self._update_sentiment_cache(ticker, filled_sentiment.iloc[-1])

            df_enriched.loc[ticker_mask, sentiment_col] = filled_sentiment

        return df_enriched

    def _update_sentiment_cache(self, ticker: str, last_sentiment: float) -> None:
        """Update sentiment cache with last known value."""
        self.sentiment_cache[ticker] = last_sentiment

    def _add_rolling_statistics(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Add rolling statistics with better handling of missing data."""
        # Use min_periods=1 to avoid NaN in short series
        for window in [5, 10, 20]:
            # Mean with min_periods=1
            ticker_group[f'sentiment_sma_{window}'] = (
                ticker_group[sentiment_col]
                .rolling(window=window, min_periods=1)
                .mean()
            )

            # Std with min_periods=1, fill with small value instead of 0
            ticker_group[f'sentiment_std_{window}'] = (
                ticker_group[sentiment_col]
                .rolling(window=window, min_periods=1)
                .std()
                .fillna(0.01)  # Small positive value instead of 0
            )

        # EMA with better handling
        ticker_group['sentiment_ema'] = (
            ticker_group[sentiment_col]
            .ewm(span=10, adjust=False)
            .mean()
        )

    def _add_sentiment_velocity(self, ticker_group: pd.DataFrame, sentiment_col: str) -> None:
        """Add sentiment velocity with better handling of missing data."""
        # Use .diff() with fill method
        ticker_group['sentiment_velocity'] = (
            ticker_group[sentiment_col]
            .diff()
            .fillna(0)  # No change for first value
        )

        # Add acceleration (second derivative)
        ticker_group['sentiment_acceleration'] = (
            ticker_group['sentiment_velocity']
            .diff()
            .fillna(0)
        )
