# core/pipeline/news_context_builder.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class NewsContextBuilder:
    """
    Builds news context that can be aligned with model training dates.
    This solves the problem: "we train models on specific dates, how do we add news context?"
    """
    
    def __init__(self):
        self.logger = logger
        self.news_cache = {}
        
    def build_news_context_by_date(self, 
                                   news_data: pd.DataFrame,
                                   target_dates: pd.DatetimeIndex,
                                   lookback_days: int = 7,
                                   forward_days: int = 1) -> pd.DataFrame:
        """
        Build news context aligned with specific training dates.
        
        Args:
            news_data: News dataframe with date and sentiment columns
            target_dates: Dates we need context for (model training dates)
            lookback_days: How many days back to look for news
            forward_days: How many days forward to look for news
            
        Returns:
            DataFrame with news context features aligned to target_dates
        """
        # Ensure news_data has datetime index
        if 'date' in news_data.columns:
            news_data = news_data.copy()
            news_data['date'] = pd.to_datetime(news_data['date'])
            news_data.set_index('date', inplace=True)
        else:
            news_data.index = pd.to_datetime(news_data.index)
            
        # Create context dataframe
        context_features = pd.DataFrame(index=target_dates)
        
        for target_date in target_dates:
            # Define window
            start_date = target_date - timedelta(days=lookback_days)
            end_date = target_date + timedelta(days=forward_days)
            
            # Filter news in window
            window_news = news_data[(news_data.index >= start_date) & 
                                  (news_data.index <= end_date)]
            
            if not window_news.empty:
                # Calculate context features
                context_features.loc[target_date, 'news_count'] = len(window_news)
                
                # Sentiment statistics
                if 'sentiment' in window_news.columns:
                    sentiments = window_news['sentiment'].dropna()
                    if not sentiments.empty:
                        context_features.loc[target_date, 'news_sentiment_mean'] = sentiments.mean()
                        context_features.loc[target_date, 'news_sentiment_std'] = sentiments.std()
                        context_features.loc[target_date, 'news_sentiment_min'] = sentiments.min()
                        context_features.loc[target_date, 'news_sentiment_max'] = sentiments.max()
                        
                        # Sentiment distribution
                        positive_count = (sentiments > 0.1).sum()
                        negative_count = (sentiments < -0.1).sum()
                        neutral_count = ((sentiments >= -0.1) & (sentiments <= 0.1)).sum()
                        
                        context_features.loc[target_date, 'news_positive_ratio'] = positive_count / len(sentiments)
                        context_features.loc[target_date, 'news_negative_ratio'] = negative_count / len(sentiments)
                        context_features.loc[target_date, 'news_neutral_ratio'] = neutral_count / len(sentiments)
                
                # Title/text analysis
                if 'title' in window_news.columns:
                    # Count mentions of specific tickers or keywords
                    context_features.loc[target_date, 'news_title_count'] = len(window_news['title'])
                    
                    # Look for specific keywords
                    keywords = ['bull', 'bear', 'crash', 'rally', 'recession', 'growth']
                    all_titles = ' '.join(window_news['title'].astype(str)).lower()
                    
                    for keyword in keywords:
                        context_features.loc[target_date, f'news_keyword_{keyword}'] = all_titles.count(keyword)
                
                # Temporal features
                if len(window_news) > 1:
                    # News frequency over time
                    news_by_day = window_news.groupby(window_news.index.date).size()
                    context_features.loc[target_date, 'news_frequency_std'] = news_by_day.std()
                    context_features.loc[target_date, 'news_frequency_max'] = news_by_day.max()
                    
                    # Sentiment trend
                    if 'sentiment' in window_news.columns:
                        sentiment_by_day = window_news.groupby(window_news.index.date)['sentiment'].mean()
                        if len(sentiment_by_day) > 1:
                            sentiment_trend = np.polyfit(range(len(sentiment_by_day)), sentiment_by_day, 1)[0]
                            context_features.loc[target_date, 'news_sentiment_trend'] = sentiment_trend
            else:
                # No news in window - set defaults
                context_features.loc[target_date, 'news_count'] = 0
                context_features.loc[target_date, 'news_sentiment_mean'] = 0.0
                context_features.loc[target_date, 'news_sentiment_std'] = 0.0
                context_features.loc[target_date, 'news_positive_ratio'] = 0.0
                context_features.loc[target_date, 'news_negative_ratio'] = 0.0
                context_features.loc[target_date, 'news_neutral_ratio'] = 1.0
                
        # Fill NaN values
        context_features.fillna(0, inplace=True)
        
        return context_features
    
    def build_macro_context_by_date(self,
                                   macro_data: pd.DataFrame,
                                   target_dates: pd.DatetimeIndex,
                                   lookback_days: int = 30) -> pd.DataFrame:
        """
        Build macro-economic context aligned with training dates.
        """
        # Ensure macro_data has datetime index
        if 'date' in macro_data.columns:
            macro_data = macro_data.copy()
            macro_data['date'] = pd.to_datetime(macro_data['date'])
            macro_data.set_index('date', inplace=True)
        else:
            macro_data.index = pd.to_datetime(macro_data.index)
            
        # Create context dataframe
        context_features = pd.DataFrame(index=target_dates)
        
        for target_date in target_dates:
            # Define window
            start_date = target_date - timedelta(days=lookback_days)
            
            # Filter macro data in window
            window_macro = macro_data[(macro_data.index >= start_date) & 
                                    (macro_data.index <= target_date)]
            
            if not window_macro.empty:
                # For each macro indicator, calculate features
                macro_cols = ['GDP', 'CPI', 'VIX', 'FEDFUNDS', 'UNRATE', 'DXY', 'TREASURY_10Y']
                available_cols = [col for col in macro_cols if col in window_macro.columns]
                
                for col in available_cols:
                    series = window_macro[col].dropna()
                    if not series.empty:
                        # Latest value
                        context_features.loc[target_date, f'macro_{col}_latest'] = series.iloc[-1]
                        
                        # Change over window
                        if len(series) > 1:
                            context_features.loc[target_date, f'macro_{col}_change'] = series.iloc[-1] - series.iloc[0]
                            context_features.loc[target_date, f'macro_{col}_pct_change'] = (series.iloc[-1] / series.iloc[0] - 1) * 100
                            
                        # Trend
                        if len(series) > 2:
                            trend = np.polyfit(range(len(series)), series, 1)[0]
                            context_features.loc[target_date, f'macro_{col}_trend'] = trend
                            
                        # Volatility
                        if len(series) > 1:
                            context_features.loc[target_date, f'macro_{col}_volatility'] = series.std()
            else:
                # No macro data - set defaults
                macro_cols = ['GDP', 'CPI', 'VIX', 'FEDFUNDS', 'UNRATE', 'DXY', 'TREASURY_10Y']
                for col in macro_cols:
                    context_features.loc[target_date, f'macro_{col}_latest'] = 0.0
                    context_features.loc[target_date, f'macro_{col}_change'] = 0.0
                    context_features.loc[target_date, f'macro_{col}_pct_change'] = 0.0
                    context_features.loc[target_date, f'macro_{col}_trend'] = 0.0
                    context_features.loc[target_date, f'macro_{col}_volatility'] = 0.0
                    
        # Fill NaN values
        context_features.fillna(0, inplace=True)
        
        return context_features
    
    def combine_context_features(self,
                                news_context: pd.DataFrame,
                                macro_context: pd.DataFrame,
                                base_features: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all context features with base features.
        """
        # Ensure all have the same index
        combined_index = base_features.index
        
        # Align context features
        news_aligned = news_context.reindex(combined_index, method='nearest')
        macro_aligned = macro_context.reindex(combined_index, method='nearest')
        
        # Combine all features
        combined_features = pd.concat([base_features, news_aligned, macro_aligned], axis=1)
        
        # Fill any remaining NaN values
        combined_features.fillna(method='ffill', inplace=True)
        combined_features.fillna(0, inplace=True)
        
        return combined_features
    
    def get_context_summary(self, context_features: pd.DataFrame) -> Dict[str, Any]:
        """Get summary of context features for analysis"""
        summary = {
            'total_features': len(context_features.columns),
            'feature_types': {
                'news': [col for col in context_features.columns if 'news_' in col],
                'macro': [col for col in context_features.columns if 'macro_' in col],
                'other': [col for col in context_features.columns if not any(prefix in col for prefix in ['news_', 'macro_'])]
            },
            'data_quality': {
                'non_null_ratio': (context_features.notna().sum() / len(context_features)).to_dict(),
                'zero_ratio': (context_features == 0).sum().to_dict()
            }
        }
        
        return summary
