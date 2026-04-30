#!/usr/bin/env python3
"""
Intelligent Data Filter - for robust model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple, Optional
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("IntelligentDataFilter")

class IntelligentDataFilter:
    """
    Performs intelligent data filtering to create a high-quality dataset for pattern learning.
    The philosophy is NOT to simply "clean" data, but to classify its imperfections
    and turn them into valuable features for the model.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if config is None:
            config = {}
        
        # Configuration with default values
        self.min_candles_per_timeframe = config.get('min_candles_per_timeframe', 2)
        self.min_data_quality_score = config.get('min_data_quality_score', 0.6)
        self.max_gap_duration = timedelta(hours=config.get('max_gap_duration_hours', 24))
        self.min_volume_threshold = config.get('min_volume_threshold', 1000)
        self.anomaly_std_dev_threshold = config.get('anomaly_std_dev_threshold', 3)
        self.news_title_min_len = config.get('news_title_min_len', 10)
        self.news_content_min_len = config.get('news_content_min_len', 50)
        self.trends_completeness_threshold = config.get('trends_completeness_threshold', 0.7)
        self.reddit_score_threshold = config.get('reddit_score_threshold', 1) # Score > 0
        self.reddit_text_min_len = config.get('reddit_text_min_len', 10)

    def filter_quality_data(self, raw_data: Dict) -> Dict:
        """
        Main filtering function to produce high-quality model training data.
        """
        filtered_data = {}
        quality_report = {}
        
        if 'prices' in raw_data:
            filtered_prices, price_quality = self._filter_price_data(raw_data['prices'])
            filtered_data['prices'] = filtered_prices
            quality_report['prices'] = price_quality
        
        if 'news' in raw_data:
            filtered_news, news_quality = self._filter_news_data(raw_data['news'])
            filtered_data['news'] = filtered_news
            quality_report['news'] = news_quality
        
        # ✅ Додаємо macro_data без фільтрації (вже очищені в Stage 2)
        if 'macro_data' in raw_data:
            filtered_data['macro_data'] = raw_data['macro_data']
            quality_report['macro_data'] = {'status': 'accepted', 'rows': len(raw_data['macro_data'])}
        
        if 'google_trends' in raw_data:
            filtered_trends, trends_quality = self._filter_trends_data(raw_data['google_trends'])
            filtered_data['google_trends'] = filtered_trends
            quality_report['google_trends'] = trends_quality
            
        if 'reddit_sentiment' in raw_data:
            filtered_reddit, reddit_quality = self._filter_reddit_data(raw_data['reddit_sentiment'])
            filtered_data['reddit_sentiment'] = filtered_reddit
            quality_report['reddit_sentiment'] = reddit_quality
        
        patterns = self._extract_patterns(filtered_data)
        
        return {
            'filtered_data': filtered_data,
            'quality_report': quality_report,
            'patterns': patterns,
            'filtering_summary': self._create_filtering_summary(quality_report)
        }
    
    def _filter_price_data(self, price_data: Dict) -> Tuple[Dict, Dict]:
        """
        Intelligently filters price data for each timeframe.
        """
        filtered_prices = {}
        quality_report = {}
        
        for timeframe, tf_data in price_data.items():
            if not isinstance(tf_data, pd.DataFrame) or tf_data.empty:
                quality_report[timeframe] = {'status': 'empty', 'reason': 'no_data'}
                continue
            
            if len(tf_data) < self.min_candles_per_timeframe:
                quality_report[timeframe] = {
                    'status': 'insufficient_data',
                    'reason': f'only_{len(tf_data)}_candles',
                    'candles': len(tf_data)
                }
                continue
            
            data_quality = self._assess_price_quality(tf_data)
            
            if data_quality['overall_score'] < self.min_data_quality_score:
                quality_report[timeframe] = {
                    'status': 'low_quality',
                    'reason': f'quality_score_{data_quality["overall_score"]:.2f}',
                    **data_quality
                }
                continue
            
            gaps = self._detect_and_classify_gaps(tf_data)
            anomalies = self._detect_and_classify_anomalies(tf_data)
            
            filtered_prices[timeframe] = {
                'data': tf_data,
                'quality': data_quality,
                'gaps': gaps,
                'anomalies': anomalies,
                'metadata': self._create_price_metadata(tf_data, timeframe)
            }
            
            quality_report[timeframe] = {
                'status': 'accepted',
                'quality_score': data_quality['overall_score'],
                'candles': len(tf_data),
                'gaps_count': len(gaps),
                'anomalies_count': len(anomalies),
                **data_quality
            }
        
        return filtered_prices, quality_report
    
    def _filter_news_data(self, news_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Intelligently filters news articles.
        """
        if not isinstance(news_data, pd.DataFrame) or news_data.empty:
            return pd.DataFrame(), {'status': 'empty', 'articles': 0}

        original_count = len(news_data)
        filtered_news = news_data.copy()
        removed_reasons = {}

        self._ensure_news_sentiment_column(filtered_news)

        timestamp_col = self._get_news_timestamp_column(filtered_news)
        quality_filters = self._build_news_quality_filters(filtered_news, timestamp_col)
        filtered_news, removed_reasons = self._apply_news_quality_filters(
            filtered_news, quality_filters
        )

        filtered_news, duplicates = self._deduplicate_news(filtered_news, timestamp_col)

        # Classify news types
        filtered_news = self._classify_news_types(filtered_news)
        
        quality_report = {
            'status': 'accepted',
            'original_articles': original_count,
            'filtered_articles': len(filtered_news),
            'removed_reasons': removed_reasons,
            'duplicates_removed': duplicates,
            'news_types': filtered_news['news_type'].value_counts().to_dict() if 'news_type' in filtered_news.columns else {}
        }
        
        return filtered_news, quality_report

    def _ensure_news_sentiment_column(self, filtered_news: pd.DataFrame) -> None:
        """Ensure sentiment column exists with neutral default."""
        if 'sentiment' in filtered_news.columns:
            return
        logger.warning("'sentiment' column not found in news_data. Adding neutral sentiment (0.0).")
        filtered_news['sentiment'] = 0.0

    def _get_news_timestamp_column(self, filtered_news: pd.DataFrame) -> Optional[str]:
        """Find the most appropriate timestamp column in news data."""
        for col in ['published_at', 'publishedAt', 'timestamp', 'date']:
            if col in filtered_news.columns:
                return col
        logger.warning("No timestamp column found in news_data. Skipping timestamp-based filtering.")
        return None

    def _build_news_quality_filters(
        self, filtered_news: pd.DataFrame, timestamp_col: Optional[str]
    ) -> List[Tuple[pd.Series, str]]:
        """Build filters for news quality checks."""
        quality_filters: List[Tuple[pd.Series, str]] = []

        if 'title' in filtered_news.columns:
            quality_filters.append(
                (filtered_news['title'].str.len() > self.news_title_min_len, 'title_too_short')
            )

        if 'content' in filtered_news.columns:
            quality_filters.append(
                (filtered_news['content'].str.len() > self.news_content_min_len, 'content_too_short')
            )

        if timestamp_col:
            quality_filters.append(
                (filtered_news[timestamp_col].notna(), 'missing_timestamp')
            )

        return quality_filters

    def _apply_news_quality_filters(
        self, filtered_news: pd.DataFrame, quality_filters: List[Tuple[pd.Series, str]]
    ) -> Tuple[pd.DataFrame, Dict[str, int]]:
        """Apply quality filters and track removed counts."""
        removed_reasons: Dict[str, int] = {}
        for filter_condition, reason in quality_filters:
            before_count = len(filtered_news)
            filtered_news = filtered_news[filter_condition]
            removed_count = before_count - len(filtered_news)
            if removed_count > 0:
                removed_reasons[reason] = removed_count
        return filtered_news, removed_reasons

    def _deduplicate_news(
        self, filtered_news: pd.DataFrame, timestamp_col: Optional[str]
    ) -> Tuple[pd.DataFrame, int]:
        """Deduplicate news data based on title and timestamp when available."""
        dedup_cols: List[str] = []
        if 'title' in filtered_news.columns:
            dedup_cols.append('title')
        if timestamp_col:
            dedup_cols.append(timestamp_col)

        if not dedup_cols:
            return filtered_news, 0

        duplicates = int(filtered_news.duplicated(subset=dedup_cols).sum())
        if duplicates > 0:
            filtered_news = filtered_news.drop_duplicates(subset=dedup_cols)

        return filtered_news, duplicates
    
    def _filter_trends_data(self, trends_data: Dict) -> Tuple[Dict, Dict]:
        """
        Filters Google Trends data.
        """
        filtered_trends = {}
        quality_report = {}
        
        for keyword, trend_data in trends_data.items():
            if isinstance(trend_data, pd.Series) and not trend_data.empty:
                completeness = trend_data.notna().sum() / len(trend_data)
                
                if completeness < self.trends_completeness_threshold:
                    quality_report[keyword] = {
                        'status': 'low_quality',
                        'reason': f'completeness_{completeness:.2f}'
                    }
                    continue
                
                trend_classification = self._classify_trend_pattern(trend_data)
                
                filtered_trends[keyword] = {
                    'data': trend_data,
                    'quality': {'completeness': completeness},
                    'pattern': trend_classification
                }
                
                quality_report[keyword] = {
                    'status': 'accepted',
                    'quality_score': completeness,
                    'pattern': trend_classification
                }
        
        return filtered_trends, quality_report
    
    def _filter_reddit_data(self, reddit_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Filters Reddit sentiment data.
        """
        if not isinstance(reddit_data, pd.DataFrame) or reddit_data.empty:
            return pd.DataFrame(), {'status': 'empty', 'posts': 0}

        original_count = len(reddit_data)
        filtered_reddit = reddit_data.copy()
        removed_reasons = {}

        quality_filters = [
            (filtered_reddit['score'] > self.reddit_score_threshold -1, 'zero_or_negative_score'),
            (filtered_reddit['created_utc'].notna(), 'missing_timestamp'),
            (filtered_reddit['text'].str.len() > self.reddit_text_min_len, 'short_text')
        ]
        if 'sentiment' in filtered_reddit.columns:
            quality_filters.append((filtered_reddit['sentiment'].notna(), 'missing_sentiment'))
        else:
            logger.warning("'sentiment' column not found in reddit_data. Skipping sentiment-based filtering.")
        
        for filter_condition, reason in quality_filters:
            before_count = len(filtered_reddit)
            filtered_reddit = filtered_reddit[filter_condition]
            removed_count = before_count - len(filtered_reddit)
            if removed_count > 0:
                removed_reasons[reason] = removed_count
        
        # Classify sentiment intensity
        filtered_reddit = self._classify_sentiment_intensity(filtered_reddit)
        
        quality_report = {
            'status': 'accepted',
            'original_posts': original_count,
            'filtered_posts': len(filtered_reddit),
            'removed_reasons': removed_reasons,
            'sentiment_distribution': filtered_reddit['sentiment_category'].value_counts().to_dict()
        }
        
        return filtered_reddit, quality_report
    
    def _assess_price_quality(self, price_data: pd.DataFrame) -> Dict:
        """
        Assesses the quality of price data.
        """
        quality_metrics = {}
        
        # Completeness
        total_cells = price_data.size
        null_cells = price_data.isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells) if total_cells > 0 else 0
        quality_metrics['completeness'] = completeness
        
        # Price consistency
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        consistency_score = 1.0
        
        for col in price_cols:
            prices = price_data[col].dropna()
            if len(prices) > 1:
                if (prices <= 0).any():
                    consistency_score -= 0.1 # Penalize for zero or negative prices
                price_changes = prices.pct_change().abs()
                extreme_changes = (price_changes > 0.5).sum() # Changes > 50%
                consistency_score -= (extreme_changes / len(prices)) * 0.2
        
        quality_metrics['consistency'] = max(0.0, consistency_score)
        
        # Volume quality
        volume_cols = [col for col in price_data.columns if 'volume' in col.lower()]
        volume_quality = 1.0
        
        for col in volume_cols:
            volumes = price_data[col].dropna()
            if len(volumes) > 0:
                zero_volume_ratio = (volumes == 0).sum() / len(volumes)
                volume_quality -= zero_volume_ratio * 0.3 # Penalize for zero volume
        
        quality_metrics['volume_quality'] = max(0.0, volume_quality)
        
        # Overall weighted score
        quality_metrics['overall_score'] = (
            quality_metrics['completeness'] * 0.4 +
            quality_metrics['consistency'] * 0.4 +
            quality_metrics['volume_quality'] * 0.2
        )
        
        return quality_metrics
    
    def _detect_and_classify_gaps(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        Detects and classifies time gaps in the price data.
        """
        gaps = []
        if 'Datetime' not in price_data.columns:
            return gaps
            
        timestamps = pd.to_datetime(price_data['Datetime']).sort_values()
        
        for i in range(1, len(timestamps)):
            gap_duration = timestamps.iloc[i] - timestamps.iloc[i-1]
            
            if gap_duration > timedelta(hours=1): # Minimum gap to be considered
                gap_info = {
                    'start_time': timestamps.iloc[i-1],
                    'end_time': timestamps.iloc[i],
                    'duration': gap_duration,
                    'gap_type': self._classify_gap_type(gap_duration),
                    'market_impact': self._assess_gap_impact(gap_duration)
                }
                gaps.append(gap_info)
        
        return gaps
    
    def _detect_and_classify_anomalies(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        Detects and classifies price anomalies based on standard deviation.
        """
        anomalies = []
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        
        for col in price_cols:
            prices = price_data[col].dropna()
            
            # Skip if insufficient data
            if len(prices) < 10:
                continue
                
            # Calculate statistics
            mean_price = prices.mean()
            std_price = prices.std()
            
            # Skip if no variation
            if std_price == 0:
                continue
            
            # Detect anomalies
            anomalies.extend(
                self._detect_anomalies_in_series(
                    prices, price_data, col, mean_price, std_price
                )
            )
        
        return anomalies
    
    def _detect_anomalies_in_series(self, prices: pd.Series, price_data: pd.DataFrame, 
                                  col: str, mean_price: float, std_price: float) -> List[Dict]:
        """Detect anomalies in a price series."""
        anomalies = []
        threshold = self.anomaly_std_dev_threshold
        
        for idx, price in prices.items():
            if self._is_anomaly(price, mean_price, std_price, threshold):
                anomaly_info = self._create_anomaly_info(
                    price_data, idx, price, col, mean_price, std_price, threshold
                )
                anomalies.append(anomaly_info)
        
        return anomalies
    
    def _is_anomaly(self, price: float, mean_price: float, std_price: float, 
                   threshold: float) -> bool:
        """Check if a price point is an anomaly."""
        return abs(price - mean_price) > threshold * std_price
    
    def _create_anomaly_info(self, price_data: pd.DataFrame, idx: Any, price: float,
                           col: str, mean_price: float, std_price: float, 
                           threshold: float) -> Dict:
        """Create anomaly information dictionary."""
        return {
            'timestamp': price_data.loc[idx, 'Datetime'] if 'Datetime' in price_data.columns else idx,
            'ticker': col.split('_')[0] if '_' in col else 'unknown',
            'price': price,
            'expected_range': (
                mean_price - threshold * std_price, 
                mean_price + threshold * std_price
            ),
            'anomaly_type': self._classify_anomaly_type(price, mean_price, std_price),
            'trading_signal': self._get_anomaly_trading_signal(price, mean_price, std_price)
        }
    
    def _classify_gap_type(self, gap_duration: timedelta) -> str:
        """Classifies the type of a time gap."""
        if gap_duration >= timedelta(days=2):
            return 'weekend_holiday'
        elif gap_duration >= timedelta(hours=16):
            return 'market_close'
        elif gap_duration >= timedelta(hours=1):
            return 'trading_halt'
        else:
            return 'data_delay'
    
    def _assess_gap_impact(self, gap_duration: timedelta) -> str:
        """Assesses the potential market impact of a gap."""
        if gap_duration >= timedelta(days=2) or gap_duration >= timedelta(hours=16):
            return 'low'  # Expected gaps
        elif gap_duration >= timedelta(hours=1):
            return 'medium'  # Trading halt is significant
        else:
            return 'high'  # Data delay could be a technical issue
    
    def _classify_anomaly_type(self, price: float, mean: float, std: float) -> str:
        """Classifies the type of a price anomaly."""
        deviation = abs(price - mean) / std
        
        if deviation > 5:
            return 'extreme_spike'
        elif deviation > 4:
            return 'significant_spike'
        else: # Covers > 3 based on initial detection
            return 'moderate_spike'

    def _get_anomaly_trading_signal(self, price: float, mean: float, std: float) -> str:
        """Generates a potential trading signal from an anomaly."""
        threshold = self.anomaly_std_dev_threshold
        if price > mean + threshold * std:
            return 'bullish_breakout_potential'
        elif price < mean - threshold * std:
            return 'bearish_breakdown_potential'
        else:
            return 'no_signal'
    
    def _classify_news_types(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Classifies news articles based on keywords in the title."""
        news_data = news_data.copy()
        
        def classify_news(title):
            title_lower = str(title).lower()
            if any(word in title_lower for word in ['earnings', 'profit', 'revenue']):
                return 'earnings'
            elif any(word in title_lower for word in ['merger', 'acquisition', 'buyout']):
                return 'm_a'
            elif any(word in title_lower for word in ['fda', 'approval', 'drug']):
                return 'regulatory_pharma'
            elif any(word in title_lower for word in ['sec', 'investigation', 'fraud', 'lawsuit']):
                return 'regulatory_legal'
            else:
                return 'general'
        
        news_data['news_type'] = news_data['title'].apply(classify_news)
        return news_data
    
    def _classify_trend_pattern(self, trend_data: pd.Series) -> str:
        """Classifies the pattern of a trend."""
        if trend_data.empty:
            return 'no_data'
        
        recent_values = trend_data.tail(10)
        if len(recent_values) < 5:
            return 'insufficient_data'
        
        if recent_values.is_monotonic_increasing:
            return 'rising_trend'
        elif recent_values.is_monotonic_decreasing:
            return 'falling_trend'
        else:
            return 'volatile'
    
    def _classify_sentiment_intensity(self, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """Classifies the intensity of sentiment scores."""
        reddit_data = reddit_data.copy()

        if 'sentiment' not in reddit_data.columns:
            logger.warning("'sentiment' column not found in reddit_data. Cannot classify sentiment intensity.")
            reddit_data['sentiment_category'] = 'unknown'
            return reddit_data
        
        def classify_sentiment(sentiment):
            if sentiment > 0.5: return 'very_bullish'
            elif sentiment > 0.2: return 'bullish'
            elif sentiment > -0.2: return 'neutral'
            elif sentiment > -0.5: return 'bearish'
            else: return 'very_bearish'
        
        reddit_data['sentiment_category'] = reddit_data['sentiment'].apply(classify_sentiment)
        return reddit_data
    
    def _create_price_metadata(self, price_data: pd.DataFrame, timeframe: str) -> Dict:
        """Creates metadata for a price dataset."""
        if 'Datetime' not in price_data.columns:
            return {'timeframe': timeframe, 'total_candles': len(price_data)}
            
        return {
            'timeframe': timeframe,
            'start_time': price_data['Datetime'].min(),
            'end_time': price_data['Datetime'].max(),
            'total_candles': len(price_data),
            'tickers': list({col.split('_')[0] for col in price_data.columns if '_' in col}),
            'data_frequency': self._estimate_data_frequency(price_data)
        }
    
    def _estimate_data_frequency(self, price_data: pd.DataFrame) -> str:
        """Estimates the data frequency from timestamps."""
        if 'Datetime' in price_data.columns and len(price_data) > 1:
            timestamps = pd.to_datetime(price_data['Datetime']).sort_values()
            avg_diff = timestamps.diff().median()
            
            if avg_diff <= timedelta(minutes=5): return 'intraday'
            elif avg_diff <= timedelta(hours=1): return 'hourly'
            elif avg_diff <= timedelta(days=1): return 'daily'
            else: return 'irregular'
        return 'unknown'
    
    def _extract_patterns(self, filtered_data: Dict) -> Dict:
        """
        Extracts higher-level patterns from the filtered data to be used as model features.
        """
        patterns = {}
        
        if 'prices' in filtered_data:
            patterns['price_patterns'] = self._extract_price_patterns(filtered_data['prices'])
        
        if 'news' in filtered_data:
            patterns['news_patterns'] = self._extract_news_patterns(filtered_data['news'])
        
        if 'reddit_sentiment' in filtered_data:
            patterns['sentiment_patterns'] = self._extract_sentiment_patterns(filtered_data['reddit_sentiment'])
        
        if 'google_trends' in filtered_data:
            patterns['trends_patterns'] = self._extract_trends_patterns(filtered_data['google_trends'])
        
        return patterns
    
    def _extract_price_patterns(self, price_data: Dict) -> Dict:
        """Extracts patterns from price data."""
        patterns = {}
        for timeframe, tf_info in price_data.items():
            patterns[timeframe] = {
                'anomaly_signals': self._create_anomaly_signals(tf_info.get('anomalies', [])),
                'gap_signals': self._create_gap_signals(tf_info.get('gaps', [])),
                'quality_indicators': tf_info.get('quality', {}),
                'trading_characteristics': self._analyze_trading_characteristics(tf_info.get('data', pd.DataFrame()))
            }
        return patterns
    
    def _extract_news_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Extracts patterns from news data."""
        if news_data.empty:
            return {}
        
        patterns = {
            'news_type_distribution': news_data['news_type'].value_counts().to_dict(),
            'temporal_patterns': self._analyze_news_temporal_patterns(news_data),
            'volume_patterns': self._analyze_news_volume_patterns(news_data)
        }
        if 'sentiment' in news_data.columns:
            patterns['sentiment_distribution'] = news_data['sentiment'].describe().to_dict()
        
        return patterns
    
    def _extract_sentiment_patterns(self, reddit_data: pd.DataFrame) -> Dict:
        """Extracts patterns from sentiment data."""
        if reddit_data.empty:
            return {}
        
        patterns = {}
        if 'sentiment_category' in reddit_data.columns:
            patterns['sentiment_distribution'] = reddit_data['sentiment_category'].value_counts().to_dict()
        if 'sentiment' in reddit_data.columns:
            patterns['intensity_patterns'] = reddit_data['sentiment'].describe().to_dict()
        if 'score' in reddit_data.columns:
            patterns['engagement_patterns'] = reddit_data['score'].describe().to_dict()

        return patterns
    
    def _extract_trends_patterns(self, trends_data: Dict) -> Dict:
        """Extracts patterns from trends data."""
        patterns = {}
        for keyword, trend_info in trends_data.items():
            patterns[keyword] = {
                'pattern_type': trend_info.get('pattern'),
                'quality_score': trend_info.get('quality', {}).get('completeness'),
                'trend_characteristics': self._analyze_trend_characteristics(trend_info.get('data', pd.Series()))
            }
        return patterns
    
    def _create_anomaly_signals(self, anomalies: List[Dict]) -> List[Dict]:
        """Creates structured signals from detected anomalies."""
        signals = []
        for anomaly in anomalies:
            signals.append({
                'timestamp': anomaly.get('timestamp'),
                'signal_type': 'anomaly',
                'signal_strength': self._calculate_anomaly_strength(anomaly),
                'trading_implication': anomaly.get('trading_signal'),
                'metadata': anomaly
            })
        return signals
    
    def _create_gap_signals(self, gaps: List[Dict]) -> List[Dict]:
        """Creates structured signals from detected gaps."""
        signals = []
        for gap in gaps:
            signals.append({
                'timestamp': gap.get('start_time'),
                'signal_type': 'gap',
                'signal_strength': self._calculate_gap_strength(gap),
                'trading_implication': self._get_gap_trading_implication(gap),
                'metadata': gap
            })
        return signals
    
    def _calculate_anomaly_strength(self, anomaly: Dict) -> str:
        """Calculates the strength of an anomaly signal."""
        expected_range = anomaly.get('expected_range', (0, 0))
        price = anomaly.get('price', 0)
        
        if expected_range[0] == 0: return 'unknown'
        
        deviation = abs(price - expected_range[0]) / expected_range[0]
        
        if deviation > 0.1: return 'very_strong'
        elif deviation > 0.05: return 'strong'
        elif deviation > 0.02: return 'moderate'
        else: return 'weak'
    
    def _calculate_gap_strength(self, gap: Dict) -> str:
        """Calculates the strength/significance of a gap signal."""
        gap_type = gap.get('gap_type')
        if gap_type in ['weekend_holiday', 'market_close']:
            return 'expected'
        elif gap_type == 'trading_halt':
            return 'significant'
        else:
            return 'attention_required'
    
    def _get_gap_trading_implication(self, gap: Dict) -> str:
        """Determines the trading implication of a gap."""
        gap_type = gap.get('gap_type')
        if gap_type == 'weekend_holiday':
            return 'expect_gap_fill_on_open'
        elif gap_type == 'trading_halt':
            return 'expect_volatility_on_resume'
        else:
            return 'monitor_for_data_quality'
    
    def _analyze_trading_characteristics(self, price_data: pd.DataFrame) -> Dict:
        """Analyzes basic trading characteristics like volatility and trend."""
        if not isinstance(price_data, pd.DataFrame) or price_data.empty:
            return {}
            
        characteristics = {}
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        volume_cols = [col for col in price_data.columns if 'volume' in col.lower()]
        
        for col in price_cols:
            prices = price_data[col].dropna()
            if len(prices) > 1:
                characteristics[f'{col}_volatility'] = prices.pct_change().std()
                characteristics[f'{col}_trend'] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] if prices.iloc[0] != 0 else 0
        
        for col in volume_cols:
            volumes = price_data[col].dropna()
            if len(volumes) > 5: # Need enough data for trend
                head_mean = volumes.head(5).mean()
                characteristics[f'{col}_avg_volume'] = volumes.mean()
                characteristics[f'{col}_volume_trend'] = (volumes.tail(5).mean() - head_mean) / head_mean if head_mean !=0 else 0
        
        return characteristics
    
    def _analyze_news_temporal_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Analyzes temporal patterns of news publication."""
        if 'published_at' not in news_data.columns:
            return {}
        
        datetimes = pd.to_datetime(news_data['published_at'])
        
        return {
            'hourly_distribution': datetimes.dt.hour.value_counts().to_dict(),
            'daily_distribution': datetimes.dt.dayofweek.value_counts().to_dict()
        }
    
    def _analyze_news_volume_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Analyzes the volume of news articles."""
        if 'published_at' not in news_data.columns:
            return {}
            
        volume_patterns = {
            'articles_per_day': news_data.groupby(pd.to_datetime(news_data['published_at']).dt.date).size().describe().to_dict()
        }
        if 'ticker' in news_data.columns:
            volume_patterns['articles_per_ticker'] = news_data.groupby('ticker').size().describe().to_dict()
        
        return volume_patterns
    
    def _analyze_trend_characteristics(self, trend_data: pd.Series) -> Dict:
        """Analyzes characteristics of a trend."""
        if not isinstance(trend_data, pd.Series) or trend_data.empty or len(trend_data) < 3:
            return {}
        
        head_mean = trend_data.head(3).mean()
        return {
            'trend_direction': 'up' if trend_data.iloc[-1] > trend_data.iloc[0] else 'down',
            'volatility': trend_data.std(),
            'recent_momentum': (trend_data.tail(3).mean() - head_mean) / head_mean if head_mean != 0 else 0
        }
    
    def _create_filtering_summary(self, quality_report: Dict) -> Dict:
        """Creates a summary of the filtering process."""
        if not quality_report:
            return {}

        total_sources = len(quality_report)
        accepted_sources = sum(1 for report in quality_report.values() if isinstance(report, dict) and report.get('status') == 'accepted')
        
        all_scores = [
            report.get('quality_score', 0) 
            for report in quality_report.values() 
            if isinstance(report, dict) and 'quality_score' in report
        ]
        
        return {
            'total_data_sources': total_sources,
            'accepted_sources': accepted_sources,
            'rejected_sources': total_sources - accepted_sources,
            'overall_quality_score': np.mean(all_scores) if all_scores else 0,
            'filtering_efficiency': accepted_sources / total_sources if total_sources > 0 else 0
        }


def filter_data_for_model_training(raw_data: Dict, config: Optional[Dict] = None) -> Dict:
    """
    High-level function to filter raw data for model training.
    
    Args:
        raw_data (Dict): A dictionary containing raw data sources like 'prices', 'news', etc.
        config (Optional[Dict]): A configuration dictionary to override default filter settings.
        
    Returns:
        Dict: A structured dictionary with filtered data, quality reports, and extracted patterns.
    """
    data_filter = IntelligentDataFilter(config=config)
    return data_filter.filter_quality_data(raw_data)


if __name__ == "__main__":
    # Example of how to use the filter
    logger.info("Intelligent Data Filter is ready for use.")
    logger.info("The main principle: Don't just delete data patterns, classify them for the model!")
