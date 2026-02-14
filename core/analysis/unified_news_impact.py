# core/analysis/unified_news_impact.py - Єдиний модуль аналandwithу впливу новин

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

from core.analysis.unified_analytics_engine import IAnalyzer, register_analyzer

logger = logging.getLogger(__name__)

class UnifiedNewsImpactAnalyzer(IAnalyzer):
    """
    Єдиний аналandforтор впливу новин - об'єднує функцandональнandсть
    """
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.historical_bombers = self._load_historical_bombers()
        self.sentiment_weights = {
            'very_positive': 1.5,
            'positive': 1.2,
            'neutral': 1.0,
            'negative': 0.8,
            'very_negative': 0.5
        }
        self.impact_categories = {
            'market_moving': ['crash', 'surge', 'plunge', 'rally', 'soar'],
            'policy_changing': ['fed', 'rate', 'tariff', 'sanction', 'regulation'],
            'earnings_impact': ['earnings', 'revenue', 'profit', 'loss', 'guidance'],
            'geopolitical': ['war', 'conflict', 'election', 'brexit', 'trade'],
            'sector_specific': ['tech', 'energy', 'finance', 'healthcare', 'auto']
        }
        
        # Реєструємо в unified engine
        register_analyzer(self, "unified_news_impact")
        
        logger.info("[UnifiedNewsImpactAnalyzer] Initialized")
    
    def _load_historical_bombers(self) -> Dict[str, Any]:
        """Заванandжити andсторичнand "бомби"""
        try:
            bombers_file = Path("c:/trading_project/config/historical_market_bombers.json")
            if bombers_file.exists():
                with open(bombers_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.info(f"[UnifiedNewsImpactAnalyzer] Loaded {len(data['historical_market_movers'])} historical bombers")
                return data
            else:
                logger.warning("[UnifiedNewsImpactAnalyzer] Historical bombers file not found")
                return {"historical_market_movers": [], "linguistic_patterns": {}}
        except Exception as e:
            logger.error(f"[UnifiedNewsImpactAnalyzer] Error loading historical bombers: {e}")
            return {"historical_market_movers": [], "linguistic_patterns": {}}
    
    def analyze(self, news_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Основний метод аналandwithу впливу новин
        
        Args:
            news_data: DataFrame with новинами
            
        Returns:
            Dict with реwithульandandми аналandwithу
        """
        if news_data.empty:
            return {"error": "Empty news data"}
        
        logger.info(f"[UnifiedNewsImpactAnalyzer] Analyzing {len(news_data)} news items")
        
        # 1. Баwithовий аналandwith
        basic_analysis = self._analyze_basic_impact(news_data)
        
        # 2. Семантичний аналandwith
        semantic_analysis = self._analyze_semantic_impact(news_data)
        
        # 3. Тимчасовий аналandwith
        temporal_analysis = self._analyze_temporal_impact(news_data)
        
        # 4. Категорandальний аналandwith
        categorical_analysis = self._analyze_categorical_impact(news_data)
        
        # 5. Ринковий вплив
        market_impact = self._analyze_market_impact(news_data)
        
        # 6. Комбandнований скор
        combined_score = self._calculate_combined_impact_score(
            basic_analysis, semantic_analysis, temporal_analysis, 
            categorical_analysis, market_impact
        )
        
        # 7. Прогноwith впливу
        impact_forecast = self._forecast_impact(news_data, combined_score)
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'news_count': len(news_data),
            'basic_analysis': basic_analysis,
            'semantic_analysis': semantic_analysis,
            'temporal_analysis': temporal_analysis,
            'categorical_analysis': categorical_analysis,
            'market_impact': market_impact,
            'combined_score': combined_score,
            'impact_forecast': impact_forecast,
            'recommendations': self._generate_recommendations(combined_score, market_impact)
        }
        
        logger.info(f"[UnifiedNewsImpactAnalyzer] Analysis completed with combined score: {combined_score.get('overall_score', 0):.3f}")
        return results
    
    def _analyze_basic_impact(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Баwithовий аналandwith впливу"""
        analysis = {}
        
        # Сентимент аналandwith
        if 'sentiment' in news_data.columns:
            sentiment_stats = news_data['sentiment'].describe()
            analysis['sentiment'] = {
                'mean': sentiment_stats['mean'],
                'std': sentiment_stats['std'],
                'min': sentiment_stats['min'],
                'max': sentiment_stats['max'],
                'distribution': news_data['sentiment'].value_counts().to_dict()
            }
        
        # Обсяг новин
        if 'mention_score' in news_data.columns:
            mention_stats = news_data['mention_score'].describe()
            analysis['mention_score'] = {
                'mean': mention_stats['mean'],
                'std': mention_stats['std'],
                'total_mentions': news_data['mention_score'].sum()
            }
        
        # Кandлькandсть новин for часом
        if 'published_at' in news_data.columns:
            news_data['published_at'] = pd.to_datetime(news_data['published_at'])
            hourly_counts = news_data.groupby(news_data['published_at'].dt.hour).size()
            analysis['temporal_distribution'] = {
                'hourly_counts': hourly_counts.to_dict(),
                'peak_hour': hourly_counts.idxmax(),
                'peak_count': hourly_counts.max()
            }
        
        return analysis
    
    def _analyze_semantic_impact(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Семантичний аналandwith впливу"""
        analysis = {}
        
        if 'title' not in news_data.columns:
            return analysis
        
        titles = news_data['title'].fillna('').tolist()
        
        # 1. Схожandсть with andсторичними "бомбами"
        if self.historical_bombers['historical_market_movers']:
            bomber_titles = [bomber['title'] for bomber in self.historical_bombers['historical_market_movers']]
            
            # Додаємо поточнand forголовки
            all_titles = bomber_titles + titles
            
            # Створюємо TF-IDF матрицю
            try:
                tfidf_matrix = self.vectorizer.fit_transform(all_titles)
                
                # Calculating схожandсть
                bomber_vectors = tfidf_matrix[:len(bomber_titles)]
                current_vectors = tfidf_matrix[len(bomber_titles):]
                
                similarity_scores = []
                for current_vec in current_vectors:
                    similarities = cosine_similarity(current_vec, bomber_vectors)[0]
                    similarity_scores.append(np.max(similarities))
                
                analysis['historical_similarity'] = {
                    'mean_similarity': np.mean(similarity_scores),
                    'max_similarity': np.max(similarity_scores),
                    'similarity_scores': similarity_scores,
                    'high_impact_news': sum(1 for score in similarity_scores if score > 0.7)
                }
                
            except Exception as e:
                logger.error(f"[UnifiedNewsImpactAnalyzer] Error in semantic analysis: {e}")
                analysis['historical_similarity'] = {'error': str(e)}
        
        # 2. Аналandwith keywords
        analysis['keyword_analysis'] = self._analyze_keywords(titles)
        
        # 3. Аналandwith категорandй
        analysis['category_analysis'] = self._analyze_news_categories(titles)
        
        return analysis
    
    def _analyze_temporal_impact(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Тимчасовий аналandwith впливу"""
        analysis = {}
        
        if 'published_at' not in news_data.columns:
            return analysis
        
        news_data['published_at'] = pd.to_datetime(news_data['published_at'])
        
        # 1. Аналandwith по днях тижня
        news_data['day_of_week'] = news_data['published_at'].dt.day_name()
        weekday_impact = news_data.groupby('day_of_week').agg({
            'sentiment': 'mean',
            'mention_score': 'mean',
            'count': 'size'
        }).to_dict()
        
        analysis['weekday_impact'] = weekday_impact
        
        # 2. Аналandwith по годинах
        news_data['hour'] = news_data['published_at'].dt.hour
        hourly_impact = news_data.groupby('hour').agg({
            'sentiment': 'mean',
            'mention_score': 'mean',
            'count': 'size'
        }).to_dict()
        
        analysis['hourly_impact'] = hourly_impact
        
        # 3. Аналandwith трендandв
        if len(news_data) > 1:
            news_data_sorted = news_data.sort_values('published_at')
            sentiment_trend = news_data_sorted['sentiment'].pct_change().mean()
            volume_trend = news_data_sorted['mention_score'].pct_change().mean()
            
            analysis['trend_analysis'] = {
                'sentiment_trend': sentiment_trend,
                'volume_trend': volume_trend,
                'trend_direction': 'increasing' if sentiment_trend > 0 else 'decreasing'
            }
        
        return analysis
    
    def _analyze_categorical_impact(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Категорandальний аналandwith впливу"""
        analysis = {}
        
        if 'title' not in news_data.columns:
            return analysis
        
        titles = news_data['title'].fillna('').tolist()
        
        # Аналandwith по категорandях
        category_scores = {}
        for category, keywords in self.impact_categories.items():
            scores = []
            for title in titles:
                title_lower = title.lower()
                score = sum(1 for keyword in keywords if keyword in title_lower)
                scores.append(score)
            
            category_scores[category] = {
                'total_score': sum(scores),
                'mean_score': np.mean(scores),
                'affected_news': sum(1 for score in scores if score > 0),
                'max_score': max(scores) if scores else 0
            }
        
        analysis['category_impact'] = category_scores
        
        # Виwithначаємо домandнуючу категорandю
        dominant_category = max(category_scores.items(), key=lambda x: x[1]['total_score'])[0] if category_scores else None
        analysis['dominant_category'] = dominant_category
        
        return analysis
    
    def _analyze_market_impact(self, news_data: pd.DataFrame) -> Dict[str, Any]:
        """Аналandwith ринкового впливу"""
        analysis = {}
        
        # 1. Кореляцandя with ринком (якщо є данand)
        if 'market_reaction' in news_data.columns:
            correlation = news_data[['sentiment', 'mention_score', 'market_reaction']].corr()
            analysis['market_correlation'] = correlation.to_dict()
        
        # 2. Аналandwith волатильностand
        if 'volatility_impact' in news_data.columns:
            volatility_stats = news_data['volatility_impact'].describe()
            analysis['volatility_impact'] = {
                'mean': volatility_stats['mean'],
                'std': volatility_stats['std'],
                'max': volatility_stats['max'],
                'high_volatility_news': len(news_data[news_data['volatility_impact'] > news_data['volatility_impact'].quantile(0.9)])
            }
        
        # 3. Аналandwith обсягу торгandв
        if 'volume_impact' in news_data.columns:
            volume_stats = news_data['volume_impact'].describe()
            analysis['volume_impact'] = {
                'mean': volume_stats['mean'],
                'std': volume_stats['std'],
                'max': volume_stats['max'],
                'high_volume_news': len(news_data[news_data['volume_impact'] > news_data['volume_impact'].quantile(0.9)])
            }
        
        return analysis
    
    def _analyze_keywords(self, titles: List[str]) -> Dict[str, Any]:
        """Аналandwith keywords"""
        # Простий аналandwith частоти слandв
        all_words = []
        for title in titles:
            words = title.lower().split()
            all_words.extend([word.strip('.,!?') for word in words if len(word) > 2])
        
        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Топ ключовand слова
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        
        return {
            'total_words': len(all_words),
            'unique_words': len(word_freq),
            'top_words': top_words,
            'word_frequency': word_freq
        }
    
    def _analyze_news_categories(self, titles: List[str]) -> Dict[str, Any]:
        """Аналandwith категорandй новин"""
        category_counts = {category: 0 for category in self.impact_categories.keys()}
        
        for title in titles:
            title_lower = title.lower()
            for category, keywords in self.impact_categories.items():
                if any(keyword in title_lower for keyword in keywords):
                    category_counts[category] += 1
        
        return category_counts
    
    def _calculate_combined_impact_score(self, basic_analysis: Dict, semantic_analysis: Dict,
                                     temporal_analysis: Dict, categorical_analysis: Dict,
                                     market_impact: Dict) -> Dict[str, Any]:
        """Роwithрахунок комбandнованого скору впливу"""
        scores = {}
        
        # 1. Баwithовий скор
        basic_score = 0.0
        if 'sentiment' in basic_analysis:
            sentiment_mean = basic_analysis['sentiment']['mean']
            basic_score += abs(sentiment_mean) * 0.3
        
        if 'mention_score' in basic_analysis:
            mention_mean = basic_analysis['mention_score']['mean']
            basic_score += mention_mean * 0.2
        
        scores['basic_score'] = basic_score
        
        # 2. Семантичний скор
        semantic_score = 0.0
        if 'historical_similarity' in semantic_analysis:
            similarity_mean = semantic_analysis['historical_similarity']['mean_similarity']
            semantic_score += similarity_mean * 0.4
        
        if 'category_analysis' in semantic_analysis:
            dominant_cat = max(semantic_analysis['category_analysis'].items(), key=lambda x: x[1]['total_score'])[0]
            semantic_score += semantic_analysis['category_analysis'][dominant_cat]['mean_score'] * 0.3
        
        scores['semantic_score'] = semantic_score
        
        # 3. Тимчасовий скор
        temporal_score = 0.0
        if 'weekday_impact' in temporal_analysis:
            peak_impact = max(temporal_analysis['weekday_impact'].values(), key=lambda x: x.get('sentiment', 0))
            temporal_score += peak_impact.get('sentiment', 0) * 0.2
        
        scores['temporal_score'] = temporal_score
        
        # 4. Ринковий скор
        market_score = 0.0
        if 'market_correlation' in market_impact:
            correlation = market_impact['market_correlation']
            if 'sentiment' in correlation and 'market_reaction' in correlation:
                market_score += abs(correlation['sentiment']['market_reaction']) * 0.3
        
        scores['market_score'] = market_score
        
        # 5. Загальний скор
        overall_score = (basic_score + semantic_score + temporal_score + market_score) / 4.0
        scores['overall_score'] = overall_score
        
        # 6. Рandвень впливу
        if overall_score > 0.8:
            impact_level = 'very_high'
        elif overall_score > 0.6:
            impact_level = 'high'
        elif overall_score > 0.4:
            impact_level = 'medium'
        elif overall_score > 0.2:
            impact_level = 'low'
        else:
            impact_level = 'very_low'
        
        scores['impact_level'] = impact_level
        
        return scores
    
    def _forecast_impact(self, news_data: pd.DataFrame, combined_score: Dict) -> Dict[str, Any]:
        """Прогноwith впливу"""
        forecast = {}
        
        # 1. Прогноwith волатильностand
        if combined_score.get('overall_score', 0) > 0.6:
            forecast['volatility_forecast'] = 'high'
        elif combined_score.get('overall_score', 0) > 0.3:
            forecast['volatility_forecast'] = 'medium'
        else:
            forecast['volatility_forecast'] = 'low'
        
        # 2. Прогноwith напрямку ринку
        if 'sentiment' in news_data.columns:
            avg_sentiment = news_data['sentiment'].mean()
            if avg_sentiment > 0.2:
                forecast['direction_forecast'] = 'bullish'
            elif avg_sentiment < -0.2:
                forecast['direction_forecast'] = 'bearish'
            else:
                forecast['direction_forecast'] = 'neutral'
        else:
            forecast['direction_forecast'] = 'unknown'
        
        # 3. Прогноwith тривалостand впливу
        impact_level = combined_score.get('impact_level', 'low')
        duration_map = {
            'very_high': 72,  # 3 днand
            'high': 48,       # 2 днand
            'medium': 24,     # 1 whereнь
            'low': 12,        # 12 годин
            'very_low': 6     # 6 годин
        }
        
        forecast['duration_forecast'] = duration_map.get(impact_level, 12)
        
        # 4. Вплив на торговand стратегandї
        forecast['trading_implications'] = self._analyze_trading_implications(combined_score)
        
        return forecast
    
    def _analyze_trading_implications(self, combined_score: Dict) -> Dict[str, Any]:
        """Аналandwith впливу на торговand стратегandї"""
        implications = {}
        
        impact_level = combined_score.get('impact_level', 'low')
        
        # Рекомендацandї по стратегandях
        strategy_recommendations = {
            'very_high': {
                'recommended_strategies': ['scalping', 'day_trading', 'volatility_breakout'],
                'avoid_strategies': ['swing_trading', 'position_trading'],
                'risk_level': 'very_high',
                'position_size': 'small'
            },
            'high': {
                'recommended_strategies': ['day_trading', 'breakout', 'momentum'],
                'avoid_strategies': ['position_trading'],
                'risk_level': 'high',
                'position_size': 'small_to_medium'
            },
            'medium': {
                'recommended_strategies': ['swing_trading', 'trend_following'],
                'avoid_strategies': [],
                'risk_level': 'medium',
                'position_size': 'medium'
            },
            'low': {
                'recommended_strategies': ['position_trading', 'buy_and_hold'],
                'avoid_strategies': ['scalping'],
                'risk_level': 'low',
                'position_size': 'medium_to_large'
            },
            'very_low': {
                'recommended_strategies': ['buy_and_hold', 'long_term_investing'],
                'avoid_strategies': ['day_trading', 'scalping'],
                'risk_level': 'very_low',
                'position_size': 'large'
            }
        }
        
        implications.update(strategy_recommendations.get(impact_level, strategy_recommendations['low']))
        
        return implications
    
    def _generate_recommendations(self, combined_score: Dict, market_impact: Dict) -> List[str]:
        """Геnotрацandя рекомендацandй"""
        recommendations = []
        
        impact_level = combined_score.get('impact_level', 'low')
        overall_score = combined_score.get('overall_score', 0)
        
        # 1. Рекомендацandї по риwithику
        if impact_level in ['very_high', 'high']:
            recommendations.append(" High impact news detected - consider reducing position size")
            recommendations.append("[WARN] Expect increased volatility - tighten stop losses")
            recommendations.append("[DATA] Monitor market closely for next 24-48 hours")
        
        # 2. Рекомендацandї по можливостях
        if overall_score > 0.7:
            recommendations.append("[MONEY] High profit potential - consider opportunistic trades")
            recommendations.append("[TARGET] Look for breakout patterns in affected sectors")
        
        # 3. Рекомендацandї по стратегandях
        if market_impact.get('volatility_impact', {}).get('mean', 0) > 0.5:
            recommendations.append("[UP] Volatility expected - volatility-based strategies preferred")
        
        # 4. Рекомендацandї по andймandнгу
        if combined_score.get('temporal_score', 0) > 0.5:
            recommendations.append(" Strong temporal pattern - time-based entries may be effective")
        
        return recommendations
    
    def get_analyzer_type(self) -> str:
        """Тип аналandforтора"""
        return "unified_news_impact"

# Глобальна функцandя for withручностand
def create_unified_news_impact_analyzer() -> UnifiedNewsImpactAnalyzer:
    """Create єдиний аналandforтор впливу новин"""
    return UnifiedNewsImpactAnalyzer()

if __name__ == "__main__":
    # Тестування
    analyzer = create_unified_news_impact_analyzer()
    print("Unified News Impact Analyzer - готовий до викорисandння")
    print(f"Тип аналandforтора: {analyzer.get_analyzer_type()}")
