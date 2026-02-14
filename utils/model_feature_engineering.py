#!/usr/bin/env python3
"""
Model Feature Engineering - створення фіч для моделі на основі патернів
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class ModelFeatureEngineering:
    """
    [START] Створення фіч для моделі на основі реальних патернів
    НЕ "очищуємо" дані, а створюємо інформативні фічі
    """
    
    def __init__(self):
        self.feature_categories = {
            'price_features': self._create_price_features,
            'pattern_features': self._create_pattern_features,
            'temporal_features': self._create_temporal_features,
            'sentiment_features': self._create_sentiment_features,
            'volume_features': self._create_volume_features,
            'regime_features': self._create_regime_features
        }
    
    def create_model_features(self, filtered_data: Dict, patterns: Dict) -> Dict:
        """
        [START] Створюємо всі фічі для моделі
        """
        all_features = {}
        
        # [TARGET] ЦІНОВІ ФІЧІ
        if 'prices' in filtered_data:
            all_features['price_features'] = self._create_price_features(filtered_data['prices'])
        
        # [TARGET] ПАТЕРН ФІЧІ
        if 'patterns' in patterns:
            all_features['pattern_features'] = self._create_pattern_features(patterns)
        
        # [TARGET] ЧАСОВІ ФІЧІ
        all_features['temporal_features'] = self._create_temporal_features(filtered_data)
        
        # [TARGET] СЕНТИМЕНТ ФІЧІ
        if 'reddit_sentiment' in filtered_data:
            all_features['sentiment_features'] = self._create_sentiment_features(filtered_data['reddit_sentiment'])
        
        if 'news' in filtered_data:
            all_features['news_features'] = self._create_news_features(filtered_data['news'])
        
        # [TARGET] ОБСЯГОВІ ФІЧІ
        if 'prices' in filtered_data:
            all_features['volume_features'] = self._create_volume_features(filtered_data['prices'])
        
        # [TARGET] РЕЖИМНІ ФІЧІ
        all_features['regime_features'] = self._create_regime_features(patterns)
        
        # [TARGET] ТАРГЕТИ ДЛЯ НАВЧАННЯ
        all_features['target_labels'] = self._create_target_labels(filtered_data)
        
        return all_features
    
    def _create_price_features(self, price_data: Dict) -> Dict:
        """
        [START] Створюємо цінові фічі - ЗБЕРІГАЄМО РЕАЛЬНІ ДАНІ!
        """
        features = {}
        
        for timeframe, tf_info in price_data.items():
            data = tf_info['data']
            
            if data.empty:
                continue
            
            timeframe_features = {}
            
            # [TARGET] RAW PRICE FEATURES (НЕ ЗМІНЮЄМО!)
            price_cols = [col for col in data.columns if 'close' in col.lower()]
            for col in price_cols:
                if col in data.columns:
                    ticker = col.split('_')[0] if '_' in col else 'unknown'
                    
                    # Raw prices - залишаємо як є!
                    timeframe_features[f'{ticker}_raw_price'] = data[col].values
                    
                    # Price changes - реальні зміни!
                    price_changes = data[col].pct_change()
                    timeframe_features[f'{ticker}_price_change'] = price_changes.values
                    
                    # Log returns - для моделі!
                    log_returns = np.log(data[col] / data[col].shift(1))
                    timeframe_features[f'{ticker}_log_return'] = log_returns.values
            
            # [TARGET] TECHNICAL INDICATORS (НА ОСНОВІ РЕАЛЬНИХ ДАНИХ!)
            for col in price_cols:
                if col in data.columns:
                    ticker = col.split('_')[0] if '_' in col else 'unknown'
                    prices = data[col].dropna()
                    
                    if len(prices) > 20:
                        # Moving averages
                        timeframe_features[f'{ticker}_sma_10'] = prices.rolling(10).mean().values
                        timeframe_features[f'{ticker}_sma_20'] = prices.rolling(20).mean().values
                        
                        # RSI
                        rsi = self._calculate_rsi(prices)
                        timeframe_features[f'{ticker}_rsi'] = rsi.values
                        
                        # Volatility
                        volatility = prices.rolling(20).std()
                        timeframe_features[f'{ticker}_volatility'] = volatility.values
                        
                        # Price position
                        sma_20 = prices.rolling(20).mean()
                        price_position = (prices - sma_20) / sma_20
                        timeframe_features[f'{ticker}_price_position'] = price_position.values
            
            # [TARGET] MULTI-TIMEFRAME FEATURES
            if len(timeframe_features) > 0:
                features[timeframe] = timeframe_features
        
        return features
    
    def _create_pattern_features(self, patterns: Dict) -> Dict:
        """
        [START] Створюємо фічі на основі патернів - ЦЕ НАЙВАЖЛИВІШЕ!
        """
        features = {}
        
        if 'price_patterns' in patterns:
            price_patterns = patterns['price_patterns']
            
            for timeframe, pattern_info in price_patterns.items():
                timeframe_features = {}
                
                # [TARGET] ANOMALY SIGNALS
                anomaly_signals = pattern_info.get('anomaly_signals', [])
                if anomaly_signals:
                    # Кількість аномалій
                    timeframe_features['anomaly_count'] = len(anomaly_signals)
                    
                    # Сила аномалій
                    anomaly_strengths = [self._signal_strength_to_numeric(sig['signal_strength']) for sig in anomaly_signals]
                    timeframe_features['anomaly_strength_avg'] = np.mean(anomaly_strengths) if anomaly_strengths else 0
                    timeframe_features['anomaly_strength_max'] = np.max(anomaly_strengths) if anomaly_strengths else 0
                    
                    # Trading implications
                    bullish_signals = sum(1 for sig in anomaly_signals if 'bullish' in sig['trading_implication'])
                    bearish_signals = sum(1 for sig in anomaly_signals if 'bearish' in sig['trading_implication'])
                    timeframe_features['bullish_anomaly_ratio'] = bullish_signals / len(anomaly_signals) if anomaly_signals else 0
                    timeframe_features['bearish_anomaly_ratio'] = bearish_signals / len(anomaly_signals) if anomaly_signals else 0
                
                # [TARGET] GAP SIGNALS
                gap_signals = pattern_info.get('gap_signals', [])
                if gap_signals:
                    timeframe_features['gap_count'] = len(gap_signals)
                    
                    # Типи пропусків
                    expected_gaps = sum(1 for sig in gap_signals if sig['signal_strength'] == 'expected')
                    significant_gaps = sum(1 for sig in gap_signals if sig['signal_strength'] == 'significant')
                    timeframe_features['expected_gap_ratio'] = expected_gaps / len(gap_signals) if gap_signals else 0
                    timeframe_features['significant_gap_ratio'] = significant_gaps / len(gap_signals) if gap_signals else 0
                
                # [TARGET] QUALITY INDICATORS
                quality = pattern_info.get('quality_indicators', {})
                if quality:
                    timeframe_features['data_completeness'] = quality.get('completeness', 0)
                    timeframe_features['data_consistency'] = quality.get('consistency', 0)
                    timeframe_features['volume_quality'] = quality.get('volume_quality', 0)
                    timeframe_features['overall_quality_score'] = quality.get('overall_score', 0)
                
                # [TARGET] TRADING CHARACTERISTICS
                trading_chars = pattern_info.get('trading_characteristics', {})
                for key, value in trading_chars.items():
                    timeframe_features[f'trading_{key}'] = value
                
                features[timeframe] = timeframe_features
        
        return features
    
    def _create_temporal_features(self, filtered_data: Dict) -> Dict:
        """
        [START] Створюємо часові фічі
        """
        features = {}
        
        # [TARGET] TIME OF DAY FEATURES
        features['hour_of_day'] = []  # Буде заповнено при обробці
        features['day_of_week'] = []  # Буде заповнено при обробці
        features['is_market_hours'] = []  # Буде заповнено при обробці
        features['is_pre_market'] = []  # Буде заповнено при обробці
        features['is_after_hours'] = []  # Буде заповнено при обробці
        
        # [TARGET] MARKET SESSION FEATURES
        features['session_type'] = []  # pre_market, regular, after_hours
        features['time_to_market_close'] = []  # Хвилин до закриття
        features['time_since_market_open'] = []  # Хвилин з відкриття
        
        return features
    
    def _create_sentiment_features(self, reddit_data: Dict) -> Dict:
        """
        [START] Створюємо сентимент фічі
        """
        features = {}
        
        if isinstance(reddit_data, dict) and 'data' in reddit_data:
            data = reddit_data['data']
        else:
            data = reddit_data
        
        if data.empty:
            return features
        
        # [TARGET] SENTIMENT SCORES
        features['sentiment_mean'] = data['sentiment'].mean()
        features['sentiment_std'] = data['sentiment'].std()
        features['sentiment_min'] = data['sentiment'].min()
        features['sentiment_max'] = data['sentiment'].max()
        
        # [TARGET] SENTIMENT DISTRIBUTION
        if 'sentiment_category' in data.columns:
            sentiment_counts = data['sentiment_category'].value_counts()
            total_posts = len(data)
            
            features['very_bullish_ratio'] = sentiment_counts.get('very_bullish', 0) / total_posts
            features['bullish_ratio'] = sentiment_counts.get('bullish', 0) / total_posts
            features['neutral_ratio'] = sentiment_counts.get('neutral', 0) / total_posts
            features['bearish_ratio'] = sentiment_counts.get('bearish', 0) / total_posts
            features['very_bearish_ratio'] = sentiment_counts.get('very_bearish', 0) / total_posts
        
        # [TARGET] ENGAGEMENT FEATURES
        if 'score' in data.columns:
            features['engagement_mean'] = data['score'].mean()
            features['engagement_std'] = data['score'].std()
            features['engagement_total'] = data['score'].sum()
        
        # [TARGET] TEMPORAL SENTIMENT FEATURES
        if 'created_utc' in data.columns:
            data['hour'] = pd.to_datetime(data['created_utc']).dt.hour
            hourly_sentiment = data.groupby('hour')['sentiment'].mean()
            
            for hour in range(24):
                features[f'sentiment_hour_{hour}'] = hourly_sentiment.get(hour, 0)
        
        return features
    
    def _create_news_features(self, news_data: Dict) -> Dict:
        """
        [START] Створюємо фічі з новин
        """
        features = {}
        
        if isinstance(news_data, dict) and 'data' in news_data:
            data = news_data['data']
        else:
            data = news_data
        
        if data.empty:
            return features
        
        # [TARGET] NEWS VOLUME FEATURES
        features['news_count_total'] = len(data)
        features['news_count_per_day'] = len(data) / max(1, (data['published_at'].max() - data['published_at'].min()).days)
        
        # [TARGET] NEWS SENTIMENT FEATURES
        if 'sentiment' in data.columns:
            features['news_sentiment_mean'] = data['sentiment'].mean()
            features['news_sentiment_std'] = data['sentiment'].std()
            features['news_sentiment_min'] = data['sentiment'].min()
            features['news_sentiment_max'] = data['sentiment'].max()
        
        # [TARGET] NEWS TYPE FEATURES
        if 'news_type' in data.columns:
            news_types = data['news_type'].value_counts()
            total_news = len(data)
            
            for news_type in ['earnings', 'm_a', 'fda', 'regulatory', 'general']:
                features[f'news_{news_type}_ratio'] = news_types.get(news_type, 0) / total_news
        
        # [TARGET] NEWS TEMPORAL FEATURES
        if 'published_at' in data.columns:
            data['hour'] = pd.to_datetime(data['published_at']).dt.hour
            hourly_news = data.groupby('hour').size()
            
            for hour in range(24):
                features[f'news_hour_{hour}'] = hourly_news.get(hour, 0)
        
        return features
    
    def _create_volume_features(self, price_data: Dict) -> Dict:
        """
        [START] Створюємо обсягові фічі
        """
        features = {}
        
        for timeframe, tf_info in price_data.items():
            data = tf_info['data']
            
            if data.empty:
                continue
            
            timeframe_features = {}
            
            # [TARGET] RAW VOLUME FEATURES
            volume_cols = [col for col in data.columns if 'volume' in col.lower()]
            for col in volume_cols:
                if col in data.columns:
                    ticker = col.split('_')[0] if '_' in col else 'unknown'
                    
                    # Raw volumes
                    timeframe_features[f'{ticker}_raw_volume'] = data[col].values
                    
                    # Volume changes
                    volume_changes = data[col].pct_change()
                    timeframe_features[f'{ticker}_volume_change'] = volume_changes.values
                    
                    # Volume moving averages
                    volume_ma = data[col].rolling(20).mean()
                    timeframe_features[f'{ticker}_volume_ma_20'] = volume_ma.values
                    
                    # Volume ratio to average
                    volume_ratio = data[col] / volume_ma
                    timeframe_features[f'{ticker}_volume_ratio'] = volume_ratio.values
                    
                    # Volume price relationship
                    price_col = col.replace('volume', 'close')
                    if price_col in data.columns:
                        price_volume_corr = data[col].rolling(20).corr(data[price_col])
                        timeframe_features[f'{ticker}_price_volume_corr'] = price_volume_corr.values
            
            # [TARGET] AGGREGATE VOLUME FEATURES
            if volume_cols:
                total_volume = data[volume_cols].sum(axis=1)
                timeframe_features['total_volume'] = total_volume.values
                timeframe_features['total_volume_change'] = total_volume.pct_change().values
                timeframe_features['total_volume_ma'] = total_volume.rolling(20).mean().values
            
            features[timeframe] = timeframe_features
        
        return features
    
    def _create_regime_features(self, patterns: Dict) -> Dict:
        """
        [START] Створюємо фічі режимів ринку
        """
        features = {}
        
        # [TARGET] MARKET REGIME INDICATORS
        features['volatility_regime'] = []  # Буде заповнено при обробці
        features['trend_regime'] = []  # Буде заповнено при обробці
        features['liquidity_regime'] = []  # Буде заповнено при обробці
        
        # [TARGET] ANOMALY FREQUENCY FEATURES
        if 'price_patterns' in patterns:
            total_anomalies = 0
            total_gaps = 0
            
            for timeframe, pattern_info in patterns['price_patterns'].items():
                anomalies = pattern_info.get('anomaly_signals', [])
                gaps = pattern_info.get('gap_signals', [])
                
                total_anomalies += len(anomalies)
                total_gaps += len(gaps)
            
            features['total_anomaly_frequency'] = total_anomalies
            features['total_gap_frequency'] = total_gaps
            features['anomaly_to_gap_ratio'] = total_anomalies / max(1, total_gaps)
        
        # [TARGET] QUALITY-BASED REGIME FEATURES
        if 'price_patterns' in patterns:
            quality_scores = []
            for pattern_info in patterns['price_patterns'].values():
                quality = pattern_info.get('quality_indicators', {})
                if quality:
                    quality_scores.append(quality.get('overall_quality', 0))
            
            if quality_scores:
                features['avg_data_quality'] = np.mean(quality_scores)
                features['data_quality_std'] = np.std(quality_scores)
                features['data_quality_trend'] = quality_scores[-1] - quality_scores[0] if len(quality_scores) > 1 else 0
        
        return features
    
    def _create_target_labels(self, filtered_data: Dict) -> Dict:
        """
        [START] Створюємо таргети для навчання моделі
        """
        targets = {}
        
        if 'prices' in filtered_data:
            for timeframe, tf_info in filtered_data['prices'].items():
                data = tf_info['data']
                
                if data.empty:
                    continue
                
                timeframe_targets = {}
                
                # [TARGET] FUTURE RETURNS (РЕАЛЬНІ ЦІЛІ!)
                price_cols = [col for col in data.columns if 'close' in col.lower()]
                for col in price_cols:
                    if col in data.columns:
                        ticker = col.split('_')[0] if '_' in col else 'unknown'
                        prices = data[col].dropna()
                        
                        if len(prices) > 5:
                            # Future returns 1 period ahead
                            future_returns = prices.pct_change().shift(-1)
                            timeframe_targets[f'{ticker}_future_return_1'] = future_returns.values
                            
                            # Future returns 5 periods ahead
                            future_returns_5 = prices.pct_change(5).shift(-5)
                            timeframe_targets[f'{ticker}_future_return_5'] = future_returns_5.values
                            
                            # Future volatility
                            future_vol = prices.rolling(5).std().shift(-5)
                            timeframe_targets[f'{ticker}_future_volatility'] = future_vol.values
                            
                            # Direction targets
                            future_direction = (prices.shift(-1) > prices).astype(int)
                            timeframe_targets[f'{ticker}_future_direction'] = future_direction.values
                
                # [TARGET] BREAKOUT TARGETS
                for col in price_cols:
                    if col in data.columns:
                        ticker = col.split('_')[0] if '_' in col else 'unknown'
                        prices = data[col].dropna()
                        
                        if len(prices) > 20:
                            # Breakout detection
                            sma_20 = prices.rolling(20).mean()
                            resistance = prices.rolling(20).max()
                            support = prices.rolling(20).min()
                            
                            # Future breakout
                            future_breakout_up = (prices.shift(-5) > resistance.shift(-5)).astype(int)
                            future_breakout_down = (prices.shift(-5) < support.shift(-5)).astype(int)
                            
                            timeframe_targets[f'{ticker}_future_breakout_up'] = future_breakout_up.values
                            timeframe_targets[f'{ticker}_future_breakout_down'] = future_breakout_down.values
                
                targets[timeframe] = timeframe_targets
        
        return targets
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Розраховуємо RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).abs()
        loss = (-delta.where(delta < 0, 0)).abs()
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss.where(avg_loss != 0, 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _signal_strength_to_numeric(self, strength: str) -> float:
        """Конвертуємо силу сигналу в число"""
        strength_map = {
            'very_strong': 1.0,
            'strong': 0.8,
            'moderate': 0.6,
            'weak': 0.4,
            'expected': 0.2,
            'significant': 0.7,
            'attention_required': 0.5
        }
        return strength_map.get(strength, 0.5)
    
    def align_features_and_targets(self, features: Dict, targets: Dict) -> Tuple[Dict, Dict]:
        """
        [START] Вирівнюємо фічі та таргети за часом
        """
        aligned_features = {}
        aligned_targets = {}
        
        # [TARGET] Вирівнюємо по timeframe
        for timeframe in features.keys():
            if timeframe in targets:
                # Тут буде логіка вирівнювання за часом
                aligned_features[timeframe] = features[timeframe]
                aligned_targets[timeframe] = targets[timeframe]
        
        return aligned_features, aligned_targets


# [TARGET] ГОЛОВНА ФУНКЦІЯ
def create_features_for_model_training(filtered_data: Dict, patterns: Dict) -> Dict:
    """
    [START] Створюємо фічі для навчання моделі
    """
    engineer = ModelFeatureEngineering()
    return engineer.create_model_features(filtered_data, patterns)


if __name__ == "__main__":
    print("Model Feature Engineering - готовий до використання")
    print("[START] Створюємо фічі на основі реальних патернів")
    print("[DATA] Не видаляємо аномалії, а використовуємо їх як сигнали!")
