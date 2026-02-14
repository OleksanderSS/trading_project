# enrichment/reverse_impact_analyzer.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger("ReverseImpactAnalyzer")


class ReverseImpactAnalyzer:
    """
    Reverse impact analysis for news with neutral sentiment but significant price impact.
    
    Analyzes news after market close to identify:
    - Neutral sentiment news (FinBERT score -0.1 to 0.1)
    - Significant price impact (>2% change)
    - Reverse signal generation based on actual market reaction
    """
    
    def __init__(self, price_threshold: float = 0.02, sentiment_threshold: float = 0.1):
        self.price_threshold = price_threshold
        self.sentiment_threshold = sentiment_threshold
    
    def analyze_daily_impact(self, news_df: pd.DataFrame, price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Analyze reverse impact for daily news after market close.
        
        Args:
            news_df: News data with sentiment scores
            price_df: Price data with daily changes
            
        Returns:
            DataFrame with reverse impact signals
        """
        if news_df.empty or price_df.empty:
            logger.warning("Empty dataframes provided to reverse impact analyzer")
            return pd.DataFrame()
        
        try:
            # Get daily price changes
            daily_prices = self._get_daily_price_changes(price_df)
            
            # Merge news with price data
            merged = self._merge_news_with_prices(news_df, daily_prices)
            
            if merged.empty:
                logger.warning("No matching news and price data found")
                return pd.DataFrame()
            
            # Calculate reverse impact signals
            reverse_signals = self._calculate_reverse_signals(merged)
            
            logger.info(f"Generated {len(reverse_signals)} reverse impact signals")
            return reverse_signals
            
        except Exception as e:
            logger.error(f"Error in reverse impact analysis: {e}")
            return pd.DataFrame()
    
    def _get_daily_price_changes(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate daily price changes for each ticker."""
        try:
            # Filter for daily data
            daily_df = price_df[price_df["interval"] == "1d"].copy()
            
            if daily_df.empty:
                logger.warning("No daily price data found")
                return pd.DataFrame()
            
            # Calculate daily price change
            daily_df["price_change_pct"] = daily_df.groupby("ticker")["close"].pct_change()
            
            # Get previous day close for comparison
            daily_df["prev_close"] = daily_df.groupby("ticker")["close"].shift(1)
            
            return daily_df
            
        except Exception as e:
            logger.error(f"Error calculating daily price changes: {e}")
            return pd.DataFrame()
    
    def _merge_news_with_prices(self, news_df: pd.DataFrame, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """Merge news with corresponding daily price data."""
        try:
            # Ensure datetime columns are properly formatted
            news_df["date"] = pd.to_datetime(news_df["published_at"]).dt.date
            daily_prices["date"] = pd.to_datetime(daily_prices["date"]).dt.date
            
            # Merge on ticker and date
            merged = pd.merge(
                news_df,
                daily_prices[["ticker", "date", "close", "price_change_pct"]],
                on=["ticker", "date"],
                how="inner"
            )
            
            return merged
            
        except Exception as e:
            logger.error(f"Error merging news with prices: {e}")
            return pd.DataFrame()
    
    def _calculate_reverse_signals(self, merged_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate reverse impact signals based on neutral sentiment + price impact."""
        try:
            signals = []
            
            for _, row in merged_df.iterrows():
                sentiment_score = row.get("sentiment_score", 0)
                price_change = row.get("price_change_pct", 0)
                
                # Check for neutral sentiment with significant price impact
                if (abs(sentiment_score) <= self.sentiment_threshold and 
                    abs(price_change) >= self.price_threshold):
                    
                    # Generate reverse signal based on actual price direction
                    reverse_signal = {
                        "date": row["date"],
                        "ticker": row["ticker"],
                        "title": row.get("title", ""),
                        "source": row.get("source", ""),
                        "original_sentiment": sentiment_score,
                        "price_change_pct": price_change,
                        "reverse_impact": np.sign(price_change),  # 1 for up, -1 for down
                        "impact_strength": min(abs(price_change) / self.price_threshold, 3.0),  # Cap at 3x
                        "signal_type": "reverse_neutral_impact"
                    }
                    signals.append(reverse_signal)
            
            if signals:
                result_df = pd.DataFrame(signals)
                logger.info(f"Found {len(result_df)} reverse impact signals")
                return result_df
            else:
                logger.info("No reverse impact signals found")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error calculating reverse signals: {e}")
            return pd.DataFrame()
    
    def get_reverse_impact_summary(self, signals_df: pd.DataFrame) -> Dict:
        """Get summary statistics for reverse impact signals."""
        if signals_df.empty:
            return {"total_signals": 0}
        
        return {
            "total_signals": len(signals_df),
            "positive_signals": len(signals_df[signals_df["reverse_impact"] > 0]),
            "negative_signals": len(signals_df[signals_df["reverse_impact"] < 0]),
            "avg_impact_strength": signals_df["impact_strength"].mean(),
            "top_sources": signals_df["source"].value_counts().head(5).to_dict()
        }
