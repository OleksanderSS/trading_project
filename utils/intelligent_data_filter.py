#!/usr/bin/env python3
"""
Intelligent Data Filter - правильна фільтрація для навчання моделі
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class IntelligentDataFilter:
    """
    [START] Інтелектуальна фільтрація data для навчання моделі
    НЕ "очищуємо" дані, а створюємо якісний датасет для вивчення патернів
    """
    
    def __init__(self):
        self.min_candles_per_timeframe = 2
        self.min_data_quality_score = 0.6
        self.max_gap_duration = timedelta(hours=24)
        self.min_volume_threshold = 1000
        
    def filter_quality_data(self, raw_data: Dict) -> Dict:
        """
        [START] Фільтруємо дані для якісного навчання моделі
        """
        filtered_data = {}
        quality_report = {}
        
        # [TARGET] ФІЛЬТРУЄМО ЦІНИ
        if 'prices' in raw_data:
            filtered_prices, price_quality = self._filter_price_data(raw_data['prices'])
            filtered_data['prices'] = filtered_prices
            quality_report['prices'] = price_quality
        
        # [TARGET] ФІЛЬТРУЄМО НОВИНИ
        if 'news' in raw_data:
            filtered_news, news_quality = self._filter_news_data(raw_data['news'])
            filtered_data['news'] = filtered_news
            quality_report['news'] = news_quality
        
        # [TARGET] ФІЛЬТРУЄМО БЕЗКОШТОВНІ ДАНІ
        if 'google_trends' in raw_data:
            filtered_trends, trends_quality = self._filter_trends_data(raw_data['google_trends'])
            filtered_data['google_trends'] = filtered_trends
            quality_report['google_trends'] = trends_quality
            
        if 'reddit_sentiment' in raw_data:
            filtered_reddit, reddit_quality = self._filter_reddit_data(raw_data['reddit_sentiment'])
            filtered_data['reddit_sentiment'] = filtered_reddit
            quality_report['reddit_sentiment'] = reddit_quality
        
        # [TARGET] СТВОРЮЄМО ПАТЕРНИ ДЛЯ МОДЕЛІ
        patterns = self._extract_patterns(filtered_data)
        
        return {
            'filtered_data': filtered_data,
            'quality_report': quality_report,
            'patterns': patterns,
            'filtering_summary': self._create_filtering_summary(quality_report)
        }
    
    def _filter_price_data(self, price_data: Dict) -> Tuple[Dict, Dict]:
        """
        [START] Інтелектуальна фільтрація цін
        """
        filtered_prices = {}
        quality_report = {}
        
        for timeframe, tf_data in price_data.items():
            if tf_data.empty:
                quality_report[timeframe] = {'status': 'empty', 'reason': 'no_data'}
                continue
            
            # [TARGET] ПЕРЕВІРКА МІНІМАЛЬНОЇ КІЛЬКОСТІ СВІЧЕК
            if len(tf_data) < self.min_candles_per_timeframe:
                quality_report[timeframe] = {
                    'status': 'insufficient_data',
                    'reason': f'only_{len(tf_data)}_candles',
                    'candles': len(tf_data)
                }
                continue
            
            # [TARGET] ЯКІСНІСТЬ ДАНИХ
            data_quality = self._assess_price_quality(tf_data)
            
            if data_quality['overall_score'] < self.min_data_quality_score:
                quality_report[timeframe] = {
                    'status': 'low_quality',
                    'reason': f'quality_score_{data_quality["overall_score"]:.2f}',
                    **data_quality
                }
                continue
            
            # [TARGET] ВИЯВЛЕННЯ ТА КЛАСИФІКАЦІЯ ПРОПУСКІВ
            gaps = self._detect_and_classify_gaps(tf_data)
            
            # [TARGET] ВИЯВЛЕННЯ ТА КЛАСИФІКАЦІЯ АНОМАЛІЙ
            anomalies = self._detect_and_classify_anomalies(tf_data)
            
            # [TARGET] ЗБЕРІГАЄМО ДАНІ З МЕТАОДАНИМИ
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
        [START] Інтелектуальна фільтрація новин
        """
        if news_data.empty:
            return news_data.copy(), {'status': 'empty', 'articles': 0}
        
        # [TARGET] ФІЛЬТРУЄМО ЗА ЯКІСТЮ
        quality_filters = [
            (news_data['title'].str.len() > 10, 'title_too_short'),
            (news_data['content'].str.len() > 50, 'content_too_short'),
            (news_data['sentiment'].notna(), 'missing_sentiment'),
            (news_data['published_at'].notna(), 'missing_timestamp')
        ]
        
        filtered_news = news_data.copy()
        removed_reasons = {}
        
        for filter_condition, reason in quality_filters:
            before_count = len(filtered_news)
            filtered_news = filtered_news[filter_condition]
            removed_count = before_count - len(filtered_news)
            if removed_count > 0:
                removed_reasons[reason] = removed_count
        
        # [TARGET] ВИЯВЛЕННЯ ДУБЛІКАТІВ
        duplicates = filtered_news.duplicated(subset=['title', 'published_at']).sum()
        if duplicates > 0:
            filtered_news = filtered_news.drop_duplicates(subset=['title', 'published_at'])
        
        # [TARGET] КЛАСИФІКУЄМО НОВИНИ ЗА ТИПАМИ
        filtered_news = self._classify_news_types(filtered_news)
        
        quality_report = {
            'status': 'accepted',
            'original_articles': len(news_data),
            'filtered_articles': len(filtered_news),
            'removed_reasons': removed_reasons,
            'duplicates_removed': duplicates,
            'news_types': filtered_news['news_type'].value_counts().to_dict()
        }
        
        return filtered_news, quality_report
    
    def _filter_trends_data(self, trends_data: Dict) -> Tuple[Dict, Dict]:
        """
        [START] Фільтрація Google Trends data
        """
        filtered_trends = {}
        quality_report = {}
        
        for keyword, trend_data in trends_data.items():
            if isinstance(trend_data, pd.Series) and not trend_data.empty:
                # [TARGET] ПЕРЕВІРКА ЯКІСТІ
                non_null_ratio = trend_data.notna().sum() / len(trend_data)
                
                if non_null_ratio < 0.7:  # Менше 70% data
                    quality_report[keyword] = {
                        'status': 'low_quality',
                        'reason': f'null_ratio_{non_null_ratio:.2f}'
                    }
                    continue
                
                # [TARGET] КЛАСИФІКУЄМО ТРЕНДИ
                trend_classification = self._classify_trend_pattern(trend_data)
                
                filtered_trends[keyword] = {
                    'data': trend_data,
                    'quality': {'non_null_ratio': non_null_ratio},
                    'pattern': trend_classification
                }
                
                quality_report[keyword] = {
                    'status': 'accepted',
                    'quality_score': non_null_ratio,
                    'pattern': trend_classification
                }
        
        return filtered_trends, quality_report
    
    def _filter_reddit_data(self, reddit_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        [START] Фільтрація Reddit sentiment data
        """
        if reddit_data.empty:
            return reddit_data.copy(), {'status': 'empty', 'posts': 0}
        
        # [TARGET] ФІЛЬТРУЄМО ЗА ЯКІСТЮ
        quality_filters = [
            (reddit_data['sentiment'].notna(), 'missing_sentiment'),
            (reddit_data['score'] > 0, 'zero_or_negative_score'),
            (reddit_data['created_utc'].notna(), 'missing_timestamp'),
            (reddit_data['text'].str.len() > 10, 'short_text')
        ]
        
        filtered_reddit = reddit_data.copy()
        removed_reasons = {}
        
        for filter_condition, reason in quality_filters:
            before_count = len(filtered_reddit)
            filtered_reddit = filtered_reddit[filter_condition]
            removed_count = before_count - len(filtered_reddit)
            if removed_count > 0:
                removed_reasons[reason] = removed_count
        
        # [TARGET] КЛАСИФІКУЄМО СЕНТИМЕНТ
        filtered_reddit = self._classify_sentiment_intensity(filtered_reddit)
        
        quality_report = {
            'status': 'accepted',
            'original_posts': len(reddit_data),
            'filtered_posts': len(filtered_reddit),
            'removed_reasons': removed_reasons,
            'sentiment_distribution': filtered_reddit['sentiment_category'].value_counts().to_dict()
        }
        
        return filtered_reddit, quality_report
    
    def _assess_price_quality(self, price_data: pd.DataFrame) -> Dict:
        """
        [START] Оцінка якості data про ціни
        """
        quality_metrics = {}
        
        # [TARGET] ПОВНОТА ДАНИХ
        total_cells = price_data.size
        null_cells = price_data.isnull().sum().sum()
        completeness = 1 - (null_cells / total_cells)
        quality_metrics['completeness'] = completeness
        
        # [TARGET] КОНСИСТЕНТНІСТЬ ЦІН
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        consistency_score = 1.0
        
        for col in price_cols:
            if col in price_data.columns:
                prices = price_data[col].dropna()
                if len(prices) > 1:
                    # Негативні ціни
                    if (prices <= 0).any():
                        consistency_score -= 0.1
                    
                    # Надмірні зміни
                    price_changes = prices.pct_change().abs()
                    extreme_changes = (price_changes > 0.5).sum()
                    consistency_score -= (extreme_changes / len(prices)) * 0.2
        
        quality_metrics['consistency'] = max(0.0, consistency_score)
        
        # [TARGET] ОБСЯГИ ДАНИХ
        volume_cols = [col for col in price_data.columns if 'volume' in col.lower()]
        volume_quality = 1.0
        
        for col in volume_cols:
            if col in price_data.columns:
                volumes = price_data[col].dropna()
                if len(volumes) > 0:
                    zero_volume_ratio = (volumes == 0).sum() / len(volumes)
                    volume_quality -= zero_volume_ratio * 0.3
        
        quality_metrics['volume_quality'] = max(0.0, volume_quality)
        
        # [TARGET] ЗАГАЛЬНИЙ SCORE
        quality_metrics['overall_score'] = (
            quality_metrics['completeness'] * 0.4 +
            quality_metrics['consistency'] * 0.4 +
            quality_metrics['volume_quality'] * 0.2
        )
        
        return quality_metrics
    
    def _detect_and_classify_gaps(self, price_data: pd.DataFrame) -> List[Dict]:
        """
        [START] Виявлення та класифікація пропусків у data
        """
        gaps = []
        
        if 'Datetime' in price_data.columns:
            timestamps = pd.to_datetime(price_data['Datetime']).sort_values()
            
            for i in range(1, len(timestamps)):
                gap_duration = timestamps.iloc[i] - timestamps.iloc[i-1]
                
                # Пропуск більше 1 години = потенційна проблема
                if gap_duration > timedelta(hours=1):
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
        [START] Виявлення та класифікація аномалій
        """
        anomalies = []
        
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        
        for col in price_cols:
            if col in price_data.columns:
                prices = price_data[col].dropna()
                if len(prices) < 10:
                    continue
                
                # Статистичні аномалії
                mean_price = prices.mean()
                std_price = prices.std()
                
                for idx, price in prices.items():
                    if abs(price - mean_price) > 3 * std_price:
                        anomaly_info = {
                            'timestamp': price_data.loc[idx, 'Datetime'] if 'Datetime' in price_data.columns else idx,
                            'ticker': col.split('_')[0] if '_' in col else 'unknown',
                            'price': price,
                            'expected_range': (mean_price - 3*std_price, mean_price + 3*std_price),
                            'anomaly_type': self._classify_anomaly_type(price, mean_price, std_price),
                            'trading_signal': self._get_anomaly_trading_signal(price, mean_price, std_price)
                        }
                        anomalies.append(anomaly_info)
        
        return anomalies
    
    def _classify_gap_type(self, gap_duration: timedelta) -> str:
        """Класифікація типу пропуску"""
        if gap_duration >= timedelta(days=2):
            return 'weekend_holiday'
        elif gap_duration >= timedelta(hours=16):
            return 'market_close'
        elif gap_duration >= timedelta(hours=1):
            return 'trading_halt'
        else:
            return 'data_delay'
    
    def _assess_gap_impact(self, gap_duration: timedelta) -> str:
        """Оцінка впливу пропуску"""
        if gap_duration >= timedelta(days=2):
            return 'low'  # Weekend/holiday - очікувано
        elif gap_duration >= timedelta(hours=16):
            return 'low'  # Market close - очікувано
        elif gap_duration >= timedelta(hours=1):
            return 'medium'  # Trading halt - важливо
        else:
            return 'high'  # Data delay - потенційна проблема
    
    def _classify_anomaly_type(self, price: float, mean: float, std: float) -> str:
        """Класифікація типу аномалії"""
        deviation = abs(price - mean) / std
        
        if deviation > 5:
            return 'extreme_spike'
        elif deviation > 4:
            return 'significant_spike'
        elif deviation > 3:
            return 'moderate_spike'
        else:
            return 'minor_anomaly'
    
    def _get_anomaly_trading_signal(self, price: float, mean: float, std: float) -> str:
        """Отримання trading signal для аномалії"""
        if price > mean + 3 * std:
            return 'bullish_breakout_potential'
        elif price < mean - 3 * std:
            return 'bearish_breakdown_potential'
        else:
            return 'unknown'
    
    def _classify_news_types(self, news_data: pd.DataFrame) -> pd.DataFrame:
        """Класифікація типів новин"""
        news_data = news_data.copy()
        
        # Проста класифікація за ключовими словами
        def classify_news(title):
            title_lower = title.lower()
            if any(word in title_lower for word in ['earnings', 'profit', 'revenue']):
                return 'earnings'
            elif any(word in title_lower for word in ['merger', 'acquisition', 'buyout']):
                return 'm_a'
            elif any(word in title_lower for word in ['fda', 'approval', 'drug']):
                return 'fda'
            elif any(word in title_lower for word in ['sec', 'investigation', 'fraud']):
                return 'regulatory'
            else:
                return 'general'
        
        news_data['news_type'] = news_data['title'].apply(classify_news)
        return news_data
    
    def _classify_trend_pattern(self, trend_data: pd.Series) -> str:
        """Класифікація патерну тренду"""
        if trend_data.empty:
            return 'no_data'
        
        # Проста класифікація за формою
        recent_values = trend_data.tail(10)
        if len(recent_values) < 5:
            return 'insufficient_data'
        
        # Тренд
        if recent_values.is_monotonic_increasing:
            return 'rising_trend'
        elif recent_values.is_monotonic_decreasing:
            return 'falling_trend'
        else:
            return 'volatile'
    
    def _classify_sentiment_intensity(self, reddit_data: pd.DataFrame) -> pd.DataFrame:
        """Класифікація інтенсивності сентименту"""
        reddit_data = reddit_data.copy()
        
        def classify_sentiment(sentiment):
            if sentiment > 0.5:
                return 'very_bullish'
            elif sentiment > 0.2:
                return 'bullish'
            elif sentiment > -0.2:
                return 'neutral'
            elif sentiment > -0.5:
                return 'bearish'
            else:
                return 'very_bearish'
        
        reddit_data['sentiment_category'] = reddit_data['sentiment'].apply(classify_sentiment)
        return reddit_data
    
    def _create_price_metadata(self, price_data: pd.DataFrame, timeframe: str) -> Dict:
        """Створення метаdata для цін"""
        return {
            'timeframe': timeframe,
            'start_time': price_data['Datetime'].min() if 'Datetime' in price_data.columns else None,
            'end_time': price_data['Datetime'].max() if 'Datetime' in price_data.columns else None,
            'total_candles': len(price_data),
            'tickers': list(set([col.split('_')[0] for col in price_data.columns if '_' in col])),
            'data_frequency': self._estimate_data_frequency(price_data)
        }
    
    def _estimate_data_frequency(self, price_data: pd.DataFrame) -> str:
        """Оцінка частоти data"""
        if 'Datetime' in price_data.columns and len(price_data) > 1:
            timestamps = pd.to_datetime(price_data['Datetime']).sort_values()
            avg_diff = timestamps.diff().median()
            
            if avg_diff <= timedelta(minutes=5):
                return 'intraday'
            elif avg_diff <= timedelta(hours=1):
                return 'hourly'
            elif avg_diff <= timedelta(days=1):
                return 'daily'
            else:
                return 'irregular'
        return 'unknown'
    
    def _extract_patterns(self, filtered_data: Dict) -> Dict:
        """
        [START] Витягуємо патерни для моделі
        """
        patterns = {}
        
        # [TARGET] PRICE PATTERNS
        if 'prices' in filtered_data:
            patterns['price_patterns'] = self._extract_price_patterns(filtered_data['prices'])
        
        # [TARGET] NEWS PATTERNS
        if 'news' in filtered_data:
            patterns['news_patterns'] = self._extract_news_patterns(filtered_data['news'])
        
        # [TARGET] SENTIMENT PATTERNS
        if 'reddit_sentiment' in filtered_data:
            patterns['sentiment_patterns'] = self._extract_sentiment_patterns(filtered_data['reddit_sentiment'])
        
        # [TARGET] TRENDS PATTERNS
        if 'google_trends' in filtered_data:
            patterns['trends_patterns'] = self._extract_trends_patterns(filtered_data['google_trends'])
        
        return patterns
    
    def _extract_price_patterns(self, price_data: Dict) -> Dict:
        """Витягуємо патерни з цін"""
        patterns = {}
        
        for timeframe, tf_info in price_data.items():
            patterns[timeframe] = {
                'anomaly_signals': self._create_anomaly_signals(tf_info['anomalies']),
                'gap_signals': self._create_gap_signals(tf_info['gaps']),
                'quality_indicators': tf_info['quality'],
                'trading_characteristics': self._analyze_trading_characteristics(tf_info['data'])
            }
        
        return patterns
    
    def _extract_news_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Витягуємо патерни з новин"""
        if news_data.empty:
            return {}
        
        return {
            'news_type_distribution': news_data['news_type'].value_counts().to_dict(),
            'sentiment_distribution': news_data['sentiment'].describe().to_dict(),
            'temporal_patterns': self._analyze_news_temporal_patterns(news_data),
            'volume_patterns': self._analyze_news_volume_patterns(news_data)
        }
    
    def _extract_sentiment_patterns(self, reddit_data: pd.DataFrame) -> Dict:
        """Витягуємо патерни сентименту"""
        if reddit_data.empty:
            return {}
        
        return {
            'sentiment_distribution': reddit_data['sentiment_category'].value_counts().to_dict(),
            'intensity_patterns': reddit_data['sentiment'].describe().to_dict(),
            'engagement_patterns': reddit_data['score'].describe().to_dict()
        }
    
    def _extract_trends_patterns(self, trends_data: Dict) -> Dict:
        """Витягуємо патерни трендів"""
        patterns = {}
        
        for keyword, trend_info in trends_data.items():
            patterns[keyword] = {
                'pattern_type': trend_info['pattern'],
                'quality_score': trend_info['quality']['non_null_ratio'],
                'trend_characteristics': self._analyze_trend_characteristics(trend_info['data'])
            }
        
        return patterns
    
    def _create_anomaly_signals(self, anomalies: List[Dict]) -> List[Dict]:
        """Створюємо сигнали з аномалій"""
        signals = []
        for anomaly in anomalies:
            signals.append({
                'timestamp': anomaly['timestamp'],
                'signal_type': 'anomaly',
                'signal_strength': self._calculate_anomaly_strength(anomaly),
                'trading_implication': anomaly['trading_signal'],
                'metadata': anomaly
            })
        return signals
    
    def _create_gap_signals(self, gaps: List[Dict]) -> List[Dict]:
        """Створюємо сигнали з пропусків"""
        signals = []
        for gap in gaps:
            signals.append({
                'timestamp': gap['start_time'],
                'signal_type': 'gap',
                'signal_strength': self._calculate_gap_strength(gap),
                'trading_implication': self._get_gap_trading_implication(gap),
                'metadata': gap
            })
        return signals
    
    def _calculate_anomaly_strength(self, anomaly: Dict) -> str:
        """Розраховуємо силу аномалії"""
        deviation = abs(anomaly['price'] - anomaly['expected_range'][0]) / anomaly['expected_range'][0]
        
        if deviation > 0.1:
            return 'very_strong'
        elif deviation > 0.05:
            return 'strong'
        elif deviation > 0.02:
            return 'moderate'
        else:
            return 'weak'
    
    def _calculate_gap_strength(self, gap: Dict) -> str:
        """Розраховуємо силу пропуску"""
        if gap['gap_type'] in ['weekend_holiday', 'market_close']:
            return 'expected'
        elif gap['gap_type'] == 'trading_halt':
            return 'significant'
        else:
            return 'attention_required'
    
    def _get_gap_trading_implication(self, gap: Dict) -> str:
        """Отримуємо trading implications для пропуску"""
        if gap['gap_type'] == 'weekend_holiday':
            return 'expect_gap_fill_on_open'
        elif gap['gap_type'] == 'trading_halt':
            return 'expect_volatility_on_resume'
        else:
            return 'monitor_for_data_quality'
    
    def _analyze_trading_characteristics(self, price_data: pd.DataFrame) -> Dict:
        """Аналізуємо торгові характеристики"""
        characteristics = {}
        
        price_cols = [col for col in price_data.columns if 'close' in col.lower()]
        volume_cols = [col for col in price_data.columns if 'volume' in col.lower()]
        
        for col in price_cols:
            if col in price_data.columns:
                prices = price_data[col].dropna()
                if len(prices) > 1:
                    characteristics[f'{col}_volatility'] = prices.pct_change().std()
                    characteristics[f'{col}_trend'] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        
        for col in volume_cols:
            if col in price_data.columns:
                volumes = price_data[col].dropna()
                if len(volumes) > 0:
                    characteristics[f'{col}_avg_volume'] = volumes.mean()
                    characteristics[f'{col}_volume_trend'] = (volumes.tail(5).mean() - volumes.head(5).mean()) / volumes.head(5).mean()
        
        return characteristics
    
    def _analyze_news_temporal_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Аналізуємо часові патерни новин"""
        if 'published_at' not in news_data.columns:
            return {}
        
        news_data['hour'] = pd.to_datetime(news_data['published_at']).dt.hour
        news_data['day_of_week'] = pd.to_datetime(news_data['published_at']).dt.dayofweek
        
        return {
            'hourly_distribution': news_data['hour'].value_counts().to_dict(),
            'daily_distribution': news_data['day_of_week'].value_counts().to_dict()
        }
    
    def _analyze_news_volume_patterns(self, news_data: pd.DataFrame) -> Dict:
        """Аналізуємо патерни обсягу новин"""
        return {
            'articles_per_day': news_data.groupby(news_data['published_at'].dt.date).size().describe().to_dict(),
            'articles_per_ticker': news_data.groupby('ticker').size().describe().to_dict() if 'ticker' in news_data.columns else {}
        }
    
    def _analyze_trend_characteristics(self, trend_data: pd.Series) -> Dict:
        """Аналізуємо характеристики тренду"""
        if trend_data.empty:
            return {}
        
        return {
            'trend_direction': 'up' if trend_data.iloc[-1] > trend_data.iloc[0] else 'down',
            'volatility': trend_data.std(),
            'recent_momentum': (trend_data.tail(3).mean() - trend_data.head(3).mean()) / trend_data.head(3).mean()
        }
    
    def _create_filtering_summary(self, quality_report: Dict) -> Dict:
        """Створюємо звіт фільтрації"""
        summary = {
            'total_data_sources': len(quality_report),
            'accepted_sources': sum(1 for report in quality_report.values() if report.get('status') == 'accepted'),
            'rejected_sources': sum(1 for report in quality_report.values() if report.get('status') != 'accepted'),
            'overall_quality_score': np.mean([report.get('quality_score', 0) for report in quality_report.values() if 'quality_score' in report]),
            'filtering_efficiency': sum(1 for report in quality_report.values() if report.get('status') == 'accepted') / len(quality_report) if quality_report else 0
        }
        
        return summary


# [TARGET] ФУНКЦІЯ ДЛЯ ВИКОРИСТАННЯ
def filter_data_for_model_training(raw_data: Dict) -> Dict:
    """
    [START] Головна функція для фільтрації data для навчання моделі
    """
    filter = IntelligentDataFilter()
    return filter.filter_quality_data(raw_data)


if __name__ == "__main__":
    # [TARGET] Приклад використання
    print("Intelligent Data Filter - готовий до використання")
    print("[START] Фільтруємо дані для якісного навчання моделі")
    print("[DATA] Не видаляємо патерни, а класифікуємо їх для моделі!")
