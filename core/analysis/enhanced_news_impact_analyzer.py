# Enhanced News Impact Analyzer - Правильна логіка аналізу новин

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger("EnhancedNewsImpactAnalyzer")


class EnhancedNewsImpactAnalyzer:
    """
    Покращений аналізатор впливу новин з правильною логікою агрегування
    """
    
    def __init__(self):
        self.logger = logging.getLogger("EnhancedNewsImpactAnalyzer")
        
        # Ваги новин за важливістю
        self.news_weights = {
            'earnings': 3.0,           # Фінансові звіти
            'fed_decision': 2.8,         # Рішення ФРС
            'war_crisis': 2.5,           # Війна/кризи
            'merger_acquisition': 2.2,     # M&A
            'analyst_upgrade': 1.8,       # Апгрейди аналітиків
            'analyst_downgrade': 1.6,      # Даунгрейди аналітиків
            'sec_filing': 1.5,            # SEC документи
            'product_launch': 1.4,         # Запуск продукту
            'regulatory': 1.3,            # Регуляторні зміни
            'macro_economic': 1.2,         # Макроекономічні дані
            'market_news': 1.0,            # Загальні ринкові новини
            'general_news': 0.8             # Загальні новини
        }
        
        # Time decay параметри
        self.half_life_hours = 24  # Через скільки годин вплив зменшується вдвічі
        
        # Пороги значущості
        self.significance_thresholds = {
            'high_impact': 2.0,
            'medium_impact': 1.0,
            'low_impact': 0.5
        }
    
    def categorize_news(self, news_text: str, title: str) -> str:
        """
        Категоризує новину за ключовими словами
        """
        text = (title + " " + news_text).lower()
        
        # Ключові слова для категорій
        categories = {
            'earnings': ['earnings', 'revenue', 'profit', 'eps', 'quarterly', 'financial results'],
            'fed_decision': ['federal reserve', 'fed', 'interest rate', 'monetary policy', 'inflation'],
            'war_crisis': ['war', 'crisis', 'conflict', 'geopolitical', 'tension', 'attack'],
            'merger_acquisition': ['merger', 'acquisition', 'buyout', 'takeover', 'deal', 'm&a'],
            'analyst_upgrade': ['upgrade', 'buy', 'outperform', 'price target', 'bullish'],
            'analyst_downgrade': ['downgrade', 'sell', 'underperform', 'bearish'],
            'sec_filing': ['sec', 'filing', '10-k', '10-q', 'regulatory filing'],
            'product_launch': ['launch', 'release', 'product', 'iphone', 'tesla model'],
            'regulatory': ['regulation', 'approval', 'fda', 'compliance', 'legal'],
            'macro_economic': ['gdp', 'inflation', 'unemployment', 'manufacturing', 'consumer']
        }
        
        # Знаходимо найкращу категорію
        best_category = 'general_news'
        max_matches = 0
        
        for category, keywords in categories.items():
            matches = sum(1 for keyword in keywords if keyword in text)
            if matches > max_matches:
                max_matches = matches
                best_category = category
        
        return best_category
    
    def calculate_time_decay(self, news_time: datetime, current_time: datetime) -> float:
        """
        Розраховує коефіцієнт затухання впливу новини
        """
        if pd.isna(news_time) or pd.isna(current_time):
            return 1.0
        
        hours_passed = (current_time - news_time).total_seconds() / 3600
        if hours_passed < 0:
            return 1.0
        
        # Exponential decay
        decay_factor = 0.5 ** (hours_passed / self.half_life_hours)
        return max(decay_factor, 0.01)  # Мінімальний вплив 1%
    
    def aggregate_news_by_timeframe(self, news_data: pd.DataFrame, 
                                price_data: pd.DataFrame,
                                timeframe: str = '1H') -> Dict[str, Any]:
        """
        Агрегуємо новини за часовими проміжками
        """
        if news_data.empty:
            return {}
        
        # Переконуємось що published_at це datetime
        if 'published_at' in news_data.columns:
            news_data['published_at'] = pd.to_datetime(news_data['published_at'])
        
        # Додаємо категорії та ваги
        news_data['category'] = news_data.apply(
            lambda row: self.categorize_news(
                row.get('description', ''), 
                row.get('title', '')
            ), axis=1
        )
        
        news_data['weight'] = news_data['category'].map(self.news_weights).fillna(1.0)
        
        # Додаємо time decay
        current_time = datetime.now()
        news_data['time_decay'] = news_data['published_at'].apply(
            lambda x: self.calculate_time_decay(x, current_time)
        )
        
        # Розраховуємо зважений sentiment
        news_data['weighted_sentiment'] = (
            news_data['sentiment_score'] * 
            news_data['weight'] * 
            news_data['time_decay']
        )
        
        # Агрегуємо за timeframe
        news_data.set_index('published_at', inplace=True)
        aggregated = news_data.resample(timeframe).agg({
            'weighted_sentiment': 'sum',
            'sentiment_score': 'mean',
            'weight': 'max',
            'time_decay': 'mean',
            'category': 'count'  # Кількість новин
        }).rename(columns={'category': 'news_count'})
        
        # Заповнюємо пропуски
        aggregated.fillna(0, inplace=True)
        
        return aggregated
    
    def correlate_with_price_movements(self, aggregated_news: pd.DataFrame,
                                   price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Корелюємо агреговані новини з рухом цін
        """
        if aggregated_news.empty or price_data.empty:
            return {}
        
        # Розраховуємо зміни цін
        price_data_copy = price_data.copy()
        price_data_copy['price_change'] = price_data_copy['close'].pct_change()
        price_data_copy['abs_change'] = price_data_copy['price_change'].abs()
        price_data_copy['volume_change'] = price_data_copy['volume'].pct_change()
        
        # Вирівнюємо індекси
        common_index = aggregated_news.index.intersection(price_data_copy.index)
        if len(common_index) < 2:
            return {}
        
        news_aligned = aggregated_news.loc[common_index]
        price_aligned = price_data_copy.loc[common_index]
        
        correlations = {}
        
        # Кореляції з різними аспектами
        for news_col in ['weighted_sentiment', 'sentiment_score', 'news_count']:
            if news_col in news_aligned.columns:
                # Кореляція з абсолютними змінами ціни
                corr_abs = news_aligned[news_col].corr(price_aligned['abs_change'])
                correlations[f'{news_col}_vs_abs_price_change'] = corr_abs
                
                # Кореляція з обсягами
                corr_vol = news_aligned[news_col].corr(price_aligned['volume_change'])
                correlations[f'{news_col}_vs_volume_change'] = corr_vol
                
                # Кореляція з напрямком зміни ціни
                corr_dir = news_aligned[news_col].corr(price_aligned['price_change'])
                correlations[f'{news_col}_vs_price_direction'] = corr_dir
        
        return correlations
    
    def detect_significant_news_events(self, news_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Виявляє значущі новинні події
        """
        significant_events = []
        
        if news_data.empty:
            return significant_events
        
        for _, row in news_data.iterrows():
            sentiment_score = row.get('sentiment_score', 0)
            category = self.categorize_news(
                row.get('description', ''), 
                row.get('title', '')
            )
            weight = self.news_weights.get(category, 1.0)
            
            # Розраховуємо загальний вплив
            total_impact = abs(sentiment_score) * weight
            
            # Визначаємо рівень значущості
            if total_impact >= self.significance_thresholds['high_impact']:
                significance_level = 'high'
            elif total_impact >= self.significance_thresholds['medium_impact']:
                significance_level = 'medium'
            elif total_impact >= self.significance_thresholds['low_impact']:
                significance_level = 'low'
            else:
                continue  # Пропускаємо незначущі новини
            
            significant_events.append({
                'timestamp': row.get('published_at'),
                'title': row.get('title', ''),
                'category': category,
                'sentiment_score': sentiment_score,
                'weight': weight,
                'total_impact': total_impact,
                'significance_level': significance_level,
                'description': row.get('description', '')
            })
        
        # Сортуємо за впливом
        significant_events.sort(key=lambda x: x['total_impact'], reverse=True)
        return significant_events
    
    def analyze_comprehensive_news_impact(self, news_data: pd.DataFrame,
                                       price_data: pd.DataFrame,
                                       timeframe: str = '1H') -> Dict[str, Any]:
        """
        Комплексний аналіз впливу новин
        """
        self.logger.info(f"Starting comprehensive news impact analysis with timeframe {timeframe}")
        
        # 1. Агрегуємо новини за часом
        aggregated_news = self.aggregate_news_by_timeframe(news_data, price_data, timeframe)
        
        if aggregated_news.empty:
            return {
                'error': 'No news data available for analysis',
                'overall_impact_score': 0,
                'news_volume_effect': 0
            }
        
        # 2. Корелюємо з цінами
        correlations = self.correlate_with_price_movements(aggregated_news, price_data)
        
        # 3. Виявляємо значущі події
        significant_events = self.detect_significant_news_events(news_data)
        
        # 4. Розраховуємо загальні метрики
        total_news_count = len(news_data)
        high_impact_events = [e for e in significant_events if e['significance_level'] == 'high']
        medium_impact_events = [e for e in significant_events if e['significance_level'] == 'medium']
        
        # 5. Загальний impact score
        weighted_sentiment_sum = aggregated_news['weighted_sentiment'].sum()
        news_count_sum = aggregated_news['news_count'].sum()
        
        if news_count_sum > 0:
            overall_impact = weighted_sentiment_sum / news_count_sum
        else:
            overall_impact = 0
        
        # 6. Sentiment alignment
        avg_sentiment = news_data['sentiment_score'].mean() if 'sentiment_score' in news_data.columns else 0
        
        # 7. Time decay analysis
        current_time = datetime.now()
        recent_news = news_data[
            pd.to_datetime(news_data['published_at']) > current_time - timedelta(hours=24)
        ]
        
        result = {
            # Загальні метрики
            'overall_impact_score': overall_impact,
            'news_volume_effect': news_count_sum,
            'avg_sentiment': avg_sentiment,
            'sentiment_alignment': avg_sentiment,  # Для сумісності
            
            # Кількісні метрики
            'total_news_count': total_news_count,
            'high_impact_events': len(high_impact_events),
            'medium_impact_events': len(medium_impact_events),
            'recent_news_count': len(recent_news),
            
            # Кореляції
            'correlations': correlations,
            
            # Значущі події
            'significant_events': significant_events[:10],  # Топ-10 подій
            
            # Агреговані дані
            'aggregated_news': aggregated_news.to_dict(),
            
            # Аналіз за категоріями
            'category_analysis': self._analyze_by_categories(news_data),
            
            # Time decay analysis
            'time_decay_analysis': {
                'avg_decay_factor': aggregated_news['time_decay'].mean(),
                'recent_impact_weight': recent_news['sentiment_score'].sum() if not recent_news.empty else 0
            }
        }
        
        self.logger.info(f"News impact analysis completed. Overall impact: {overall_impact:.3f}")
        return result
    
    def _analyze_by_categories(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Аналізуємо розподіл новин за категоріями
        """
        if news_data.empty:
            return {}
        
        # Додаємо категорії
        news_data['category'] = news_data.apply(
            lambda row: self.categorize_news(
                row.get('description', ''), 
                row.get('title', '')
            ), axis=1
        )
        
        category_analysis = {}
        for category in news_data['category'].unique():
            category_news = news_data[news_data['category'] == category]
            category_analysis[category] = {
                'count': len(category_news),
                'avg_sentiment': category_news['sentiment_score'].mean(),
                'total_impact': (category_news['sentiment_score'] * 
                                category_news['category'].map(self.news_weights)).sum()
            }
        
        return category_analysis


# Створюємо екземпляр для використання
enhanced_news_analyzer = EnhancedNewsImpactAnalyzer()
