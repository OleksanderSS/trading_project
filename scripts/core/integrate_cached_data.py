#!/usr/bin/env python3
"""
Integration script for using cached news/feature data with new tickers.
This script shows how to leverage existing cached data for expanded ticker lists.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CachedDataIntegrator:
    """Integrate cached data for new tickers."""
    
    def __init__(self):
        self.sentiment_cache_path = "d:/trading_project/data/cache/sentiment_cache.parquet"
        self.feature_cache_dir = Path("d:/trading_project/data/cache/features")
        self.model_batches_dir = Path("d:/trading_project/models/unified/colab_batches")
        
    def load_sentiment_cache(self) -> pd.DataFrame:
        """Load sentiment cache data."""
        if not Path(self.sentiment_cache_path).exists():
            logger.warning(f"Sentiment cache not found: {self.sentiment_cache_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(self.sentiment_cache_path)
        logger.info(f"Loaded sentiment cache: {df.shape} rows")
        return df
    
    def load_feature_cache(self, cache_file: str) -> pd.DataFrame:
        """Load specific feature cache file."""
        cache_path = self.feature_cache_dir / cache_file
        if not cache_path.exists():
            logger.warning(f"Feature cache not found: {cache_path}")
            return pd.DataFrame()
        
        df = pd.read_parquet(cache_path)
        logger.info(f"Loaded feature cache {cache_file}: {df.shape} rows")
        return df
    
    def get_available_tickers(self) -> List[str]:
        """Get list of tickers with cached data."""
        tickers = set()
        
        # From feature caches
        for cache_file in self.feature_cache_dir.glob("*.parquet"):
            try:
                df = pd.read_parquet(cache_file)
                if 'ticker' in df.columns:
                    tickers.update(df['ticker'].unique())
            except Exception as e:
                logger.error(f"Error reading {cache_file}: {e}", exc_info=True)
                # Re-raise or handle as per established pattern - for now re-raise to be explicit
                raise
        
        # From model batches
        for batch_dir in self.model_batches_dir.iterdir():
            if batch_dir.is_dir():
                parts = batch_dir.name.split('_')
                if len(parts) >= 2:
                    ticker = parts[0]
                    if ticker.isupper() and len(ticker) >= 3:
                        tickers.add(ticker)
        
        return sorted(list(tickers))
    
    def create_market_sentiment_features(self, sentiment_df: pd.DataFrame, 
                                       new_tickers: List[str],
                                       date_range: Optional[pd.DatetimeIndex] = None) -> Dict[str, pd.DataFrame]:
        """Create market sentiment features for new tickers using cached sentiment data."""
        if sentiment_df.empty:
            logger.warning("No sentiment data available")
            return {}
        
        # Create date range if not provided
        if date_range is None:
            start_date = sentiment_df['timestamp'].min()
            end_date = sentiment_df['timestamp'].max()
            
            # Remove timezone if present
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
                
            date_range = pd.date_range(start=start_date, end=end_date, freq='H')
        
        results = {}
        
        # Aggregate sentiment by time windows
        sentiment_df['hour'] = sentiment_df['timestamp'].dt.floor('H')
        hourly_sentiment = sentiment_df.groupby('hour').agg({
            'sentiment': ['mean', 'std', 'count', 'min', 'max']
        }).fillna(0)
        
        hourly_sentiment.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'sentiment_min', 'sentiment_max']
        
        for ticker in new_tickers:
            logger.info(f"Creating sentiment features for {ticker}")
            
            # Create ticker-specific dataframe
            ticker_data = pd.DataFrame({
                'datetime': date_range,
                'ticker': ticker
            })
            
            # Merge with hourly sentiment
            ticker_data['hour'] = ticker_data['datetime'].dt.floor('H')
            ticker_data = ticker_data.merge(hourly_sentiment, left_on='hour', right_index=True, how='left')
            
            # Fill missing sentiment with forward fill and then mean
            sentiment_cols = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'sentiment_min', 'sentiment_max']
            ticker_data[sentiment_cols] = ticker_data[sentiment_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Add sentiment-based features
            ticker_data['sentiment_score'] = ticker_data['sentiment_mean']
            ticker_data['sentiment_volatility'] = ticker_data['sentiment_std']
            ticker_data['sentiment_momentum'] = ticker_data['sentiment_mean'].rolling(window=24).apply(lambda x: x.iloc[-1] - x.iloc[0] if len(x) >= 24 else 0)
            
            # Sentiment regime classification
            ticker_data['sentiment_regime'] = pd.cut(ticker_data['sentiment_score'], 
                                                   bins=[-1, -0.3, 0.3, 1], 
                                                   labels=['bearish', 'neutral', 'bullish'])
            
            results[ticker] = ticker_data.drop('hour', axis=1)
        
        return results
    
    def adapt_technical_features(self, feature_df: pd.DataFrame, 
                                new_tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Adapt technical features from cached data for new tickers."""
        if feature_df.empty:
            logger.warning("No feature data available")
            return {}
        
        results = {}
        
        # Get feature columns (excluding ticker-specific columns)
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['ticker', 'datetime', '_cache_ticker', '_cache_date', '_cache_config_hash']]
        
        # Create synthetic data for new tickers based on existing patterns
        for ticker in new_tickers:
            logger.info(f"Adapting technical features for {ticker}")
            
            # Sample from existing data with noise
            sample_size = min(1000, len(feature_df))  # Limit size for demo
            sampled_data = feature_df[feature_cols].sample(n=sample_size, replace=True).copy()
            
            # Add realistic noise (5-10% variation)
            noise_factor = np.random.uniform(0.05, 0.15, len(sampled_data))
            for col in feature_cols:
                if feature_df[col].dtype in ['float64', 'int64']:
                    sampled_data[col] = sampled_data[col] * (1 + noise_factor * np.random.randn(len(sampled_data)) * 0.1)
            
            # Create datetime range
            start_date = feature_df['datetime'].min()
            end_date = feature_df['datetime'].max()
            date_range = pd.date_range(start=start_date, end=end_date, periods=len(sampled_data))
            
            # Remove timezone from datetime
            if hasattr(start_date, 'tz') and start_date.tz is not None:
                start_date = start_date.tz_localize(None)
            if hasattr(end_date, 'tz') and end_date.tz is not None:
                end_date = end_date.tz_localize(None)
            
            # Assemble ticker data with timezone-naive datetime
            ticker_data = pd.DataFrame({
                'datetime': pd.date_range(start=start_date, end=end_date, periods=len(sampled_data)),
                'ticker': ticker
            })
            
            # Add adapted features
            ticker_data[feature_cols] = sampled_data[feature_cols].values
            
            # Add cache metadata
            ticker_data['_cache_ticker'] = ticker
            ticker_data['_cache_date'] = f"{ticker}_adapted_features"
            ticker_data['_cache_config_hash'] = "adapted_from_cache"
            
            results[ticker] = ticker_data
        
        return results
    
    def create_hybrid_features(self, sentiment_features: Dict[str, pd.DataFrame],
                             technical_features: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Combine sentiment and technical features for new tickers."""
        hybrid_results = {}
        
        common_tickers = set(sentiment_features.keys()) & set(technical_features.keys())
        
        for ticker in common_tickers:
            logger.info(f"Creating hybrid features for {ticker}")
            
            sentiment_df = sentiment_features[ticker]
            technical_df = technical_features[ticker]
            
            # Merge on datetime
            hybrid_df = pd.merge(sentiment_df, technical_df, on='datetime', how='inner', suffixes=('_sentiment', '_technical'))
            
            # Resolve ticker column conflicts
            hybrid_df['ticker'] = ticker
            if 'ticker_sentiment' in hybrid_df.columns:
                hybrid_df = hybrid_df.drop('ticker_sentiment', axis=1)
            if 'ticker_technical' in hybrid_df.columns:
                hybrid_df = hybrid_df.drop('ticker_technical', axis=1)
            
            # Create interaction features
            if 'sentiment_score' in hybrid_df.columns and 'RSI_14' in hybrid_df.columns:
                hybrid_df['sentiment_rsi_interaction'] = hybrid_df['sentiment_score'] * hybrid_df['RSI_14'] / 100
            
            if 'sentiment_regime' in hybrid_df.columns and 'MARKET_REGIME' in hybrid_df.columns:
                # Create combined regime indicator
                hybrid_df['combined_regime'] = (
                    hybrid_df['sentiment_regime'].astype(str) + '_' + 
                    hybrid_df['MARKET_REGIME'].astype(str)
                )
            
            hybrid_results[ticker] = hybrid_df
        
        return hybrid_results
    
    def save_adapted_features(self, features_dict: Dict[str, pd.DataFrame], 
                            output_dir: str = "d:/trading_project/data/cache/adapted_features"):
        """Save adapted features to parquet files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        for ticker, df in features_dict.items():
            file_path = output_path / f"{ticker}_adapted_features.parquet"
            df.to_parquet(file_path, index=False)
            logger.info(f"Saved adapted features for {ticker}: {df.shape} -> {file_path}")
    
    def integrate_new_tickers(self, new_tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """Main integration method for new tickers."""
        logger.info(f"Integrating {len(new_tickers)} new tickers: {new_tickers}")
        
        # Load cached data
        sentiment_df = self.load_sentiment_cache()
        
        # Load feature caches (use first available as template)
        feature_files = list(self.feature_cache_dir.glob("*.parquet"))
        feature_df = pd.DataFrame()
        if feature_files:
            feature_df = self.load_feature_cache(feature_files[0].name)
        
        # Create features for new tickers
        sentiment_features = self.create_market_sentiment_features(sentiment_df, new_tickers)
        technical_features = self.adapt_technical_features(feature_df, new_tickers)
        
        # Combine features
        hybrid_features = self.create_hybrid_features(sentiment_features, technical_features)
        
        # Save results
        self.save_adapted_features(hybrid_features)
        
        logger.info(f"Successfully integrated {len(hybrid_features)} tickers")
        return hybrid_features

def main():
    """Demonstration of cached data integration."""
    integrator = CachedDataIntegrator()
    
    # Get available tickers
    available_tickers = integrator.get_available_tickers()
    logger.info(f"Available tickers with cached data: {available_tickers}")
    
    # Example new tickers (these would be your expanded list)
    new_tickers = ['META', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']  # Example new tickers
    
    # Integrate new tickers
    adapted_features = integrator.integrate_new_tickers(new_tickers)
    
    # Summary
    print("\n=== INTEGRATION SUMMARY ===")
    print(f"Successfully created features for {len(adapted_features)} new tickers:")
    for ticker, df in adapted_features.items():
        print(f"  {ticker}: {df.shape[0]} rows, {df.shape[1]} columns")
    
    print("\n💡 NEXT STEPS:")
    print("1. Review the adapted features in data/cache/adapted_features/")
    print("2. Use these features for model training with new tickers")
    print("3. Validate feature quality before production use")
    print("4. Consider collecting real data for critical tickers")

if __name__ == "__main__":
    main()
