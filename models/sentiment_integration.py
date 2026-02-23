# models/sentiment_integration.py - ІНТЕГРАЦІЯ SENTIMENT MODELS В PIPELINE

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from utils.logger import ProjectLogger
from models.sentiment_models import analyze_sentiment, aggregate_sentiment, get_finbert_pipeline

logger = ProjectLogger.get_logger(__name__)

class SentimentModelIntegrator:
    """Інтеграція Sentiment моделей в основний pipeline"""
    
    def __init__(self):
        self.pipeline = None
        self.is_initialized = False
        self.cache = {}
        
    def initialize(self):
        """Ініціалізація sentiment pipeline"""
        try:
            self.pipeline = get_finbert_pipeline()
            self.is_initialized = True
            logger.info("[SENTIMENT] Sentiment pipeline successfully ініціалізований")
            return True
        except Exception as e:
            logger.error(f"[SENTIMENT] Помилка ініціалізації: {e}")
            return False
    
    def analyze_news_sentiment(self, news_texts: List[str], batch_size: int = 16) -> pd.DataFrame:
        """Аналіз сентименту новин"""
        if not self.is_initialized:
            if not self.initialize():
                return self._create_fallback_sentiment(news_texts)
        
        try:
            # Аналіз сентименту
            sentiment_df = analyze_sentiment(news_texts, batch_size=batch_size)
            
            # Агрегація результатів
            aggregated = aggregate_sentiment(sentiment_df, normalize=True, method="mean")
            
            logger.info(f"[SENTIMENT] Проаналізовано {len(news_texts)} новин: {aggregated}")
            
            return sentiment_df
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Помилка аналізу сентименту: {e}")
            return self._create_fallback_sentiment(news_texts)
    
    def _create_fallback_sentiment(self, news_texts: List[str]) -> pd.DataFrame:
        """Створення резервного сентименту"""
        fallback_data = []
        for text in news_texts:
            fallback_data.append({
                'text': text,
                'label': 'neutral',
                'score': 0.5
            })
        
        return pd.DataFrame(fallback_data)
    
    def extract_sentiment_features(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Витягування sentiment фіч для моделей"""
        try:
            if news_data.empty:
                return self._create_default_sentiment_features()
            
            # Беремо останні новини
            recent_news = news_data.tail(50)  # Останні 50 новин
            
            if 'text' not in recent_news.columns or 'title' in recent_news.columns:
                # Якщо є title, використовуємо його
                texts = recent_news['title'].fillna('').tolist()
            else:
                texts = recent_news['text'].fillna('').tolist()
            
            # Фільтруємо порожні тексти
            texts = [text for text in texts if text.strip()]
            
            if not texts:
                return self._create_default_sentiment_features()
            
            # Аналізуємо сентимент
            sentiment_df = self.analyze_news_sentiment(texts)
            
            # Розраховуємо фічі
            features = self._calculate_sentiment_features(sentiment_df, price_data)
            
            return features
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Помилка витягування фіч: {e}")
            return self._create_default_sentiment_features()
    
    def _calculate_sentiment_features(self, sentiment_df: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, float]:
        """Розрахунок sentiment фіч"""
        features = {}
        
        # Базові sentiment метрики
        if not sentiment_df.empty:
            # Агрегований сентимент
            aggregated = aggregate_sentiment(sentiment_df, normalize=True, method="mean")
            
            features['sentiment_positive'] = aggregated.get('positive', 0.0)
            features['sentiment_negative'] = aggregated.get('negative', 0.0)
            features['sentiment_neutral'] = aggregated.get('neutral', 0.0)
            
            # Сентимент score (positive - negative)
            features['sentiment_score'] = features['sentiment_positive'] - features['sentiment_negative']
            
            # Середня впевненість
            features['sentiment_confidence'] = sentiment_df['score'].mean()
            
            # Кількість новин
            features['news_count'] = len(sentiment_df)
            
            # Волатильність сентименту
            if len(sentiment_df) > 1:
                sentiment_scores = []
                for _, row in sentiment_df.iterrows():
                    if row['label'] == 'positive':
                        sentiment_scores.append(row['score'])
                    elif row['label'] == 'negative':
                        sentiment_scores.append(-row['score'])
                    else:
                        sentiment_scores.append(0.0)
                
                features['sentiment_volatility'] = np.std(sentiment_scores) if sentiment_scores else 0.0
            else:
                features['sentiment_volatility'] = 0.0
            
            # Частка позитивних новин
            positive_count = (sentiment_df['label'] == 'positive').sum()
            features['positive_news_ratio'] = positive_count / len(sentiment_df) if len(sentiment_df) > 0 else 0.0
            
            # Сила сентименту (середній score для не-нейтральних)
            non_neutral = sentiment_df[sentiment_df['label'] != 'neutral']
            if not non_neutral.empty:
                features['sentiment_strength'] = non_neutral['score'].mean()
            else:
                features['sentiment_strength'] = 0.0
        
        # Інтеграція з ціновими даними
        if not price_data.empty and len(price_data) > 1:
            # Кореляція сентименту з зміною ціни
            recent_price_change = price_data['close'].pct_change().tail(5).mean()
            features['sentiment_price_correlation'] = features['sentiment_score'] * np.sign(recent_price_change)
            
            # Sentiment momentum (зміна сентименту)
            if len(sentiment_df) >= 10:
                early_sentiment = sentiment_df.head(5)
                late_sentiment = sentiment_df.tail(5)
                
                early_score = self._calculate_sentiment_score(early_sentiment)
                late_score = self._calculate_sentiment_score(late_sentiment)
                
                features['sentiment_momentum'] = late_score - early_score
            else:
                features['sentiment_momentum'] = 0.0
        
        return features
    
    def _calculate_sentiment_score(self, sentiment_df: pd.DataFrame) -> float:
        """Розрахунок загального sentiment score"""
        if sentiment_df.empty:
            return 0.0
        
        score = 0.0
        for _, row in sentiment_df.iterrows():
            if row['label'] == 'positive':
                score += row['score']
            elif row['label'] == 'negative':
                score -= row['score']
        
        return score / len(sentiment_df)
    
    def _create_default_sentiment_features(self) -> Dict[str, float]:
        """Створення sentiment фіч за замовчуванням"""
        return {
            'sentiment_positive': 0.33,
            'sentiment_negative': 0.33,
            'sentiment_neutral': 0.34,
            'sentiment_score': 0.0,
            'sentiment_confidence': 0.5,
            'news_count': 0,
            'sentiment_volatility': 0.0,
            'positive_news_ratio': 0.33,
            'sentiment_strength': 0.0,
            'sentiment_price_correlation': 0.0,
            'sentiment_momentum': 0.0
        }
    
    def enhance_features_with_sentiment(self, features_df: pd.DataFrame, news_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """Збагачення фіч sentiment даними"""
        try:
            # Отримуємо sentiment фічі
            sentiment_features = self.extract_sentiment_features(news_data, price_data)
            
            # Додаємо до основних фіч
            enhanced_df = features_df.copy()
            
            for feature_name, feature_value in sentiment_features.items():
                enhanced_df[feature_name] = feature_value
            
            logger.info(f"[SENTIMENT] Додано {len(sentiment_features)} sentiment фіч до {len(features_df)} записів")
            
            return enhanced_df
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Помилка збагачення фіч: {e}")
            return features_df
    
    def get_sentiment_signal(self, news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Отримання торгового сигналу на основі сентименту"""
        try:
            sentiment_features = self.extract_sentiment_features(news_data, price_data)
            
            # Логіка сигналу
            sentiment_score = sentiment_features['sentiment_score']
            sentiment_strength = sentiment_features['sentiment_strength']
            news_count = sentiment_features['news_count']
            
            # Пороги для сигналу
            signal_strength = 0.0
            signal_type = 'hold'
            reasoning = []
            
            if sentiment_score > 0.2 and sentiment_strength > 0.6 and news_count > 5:
                signal_type = 'buy'
                signal_strength = min(sentiment_score * sentiment_strength, 1.0)
                reasoning.append(f"Strong positive sentiment: {sentiment_score:.2f}")
                reasoning.append(f"High sentiment strength: {sentiment_strength:.2f}")
                reasoning.append(f"Sufficient news volume: {news_count}")
            
            elif sentiment_score < -0.2 and sentiment_strength > 0.6 and news_count > 5:
                signal_type = 'sell'
                signal_strength = min(abs(sentiment_score) * sentiment_strength, 1.0)
                reasoning.append(f"Strong negative sentiment: {sentiment_score:.2f}")
                reasoning.append(f"High sentiment strength: {sentiment_strength:.2f}")
                reasoning.append(f"Sufficient news volume: {news_count}")
            
            elif abs(sentiment_score) < 0.1:
                signal_type = 'hold'
                reasoning.append(f"Neutral sentiment: {sentiment_score:.2f}")
            
            else:
                signal_type = 'hold'
                reasoning.append(f"Weak sentiment signal: {sentiment_score:.2f}")
                reasoning.append(f"Low confidence: {sentiment_strength:.2f}")
            
            return {
                'signal_type': signal_type,
                'signal_strength': signal_strength,
                'confidence': signal_strength,
                'reasoning': ' | '.join(reasoning),
                'sentiment_score': sentiment_score,
                'sentiment_strength': sentiment_strength,
                'news_count': news_count,
                'model_type': 'sentiment'
            }
            
        except Exception as e:
            logger.error(f"[SENTIMENT] Помилка генерації сигналу: {e}")
            return {
                'signal_type': 'hold',
                'signal_strength': 0.0,
                'confidence': 0.0,
                'reasoning': f'Sentiment analysis error: {str(e)}',
                'model_type': 'sentiment_error'
            }

# Глобальний інтегратор
_sentiment_integrator = None

def get_sentiment_integrator() -> SentimentModelIntegrator:
    """Отримання глобального інтегратора sentiment моделей"""
    global _sentiment_integrator
    if _sentiment_integrator is None:
        _sentiment_integrator = SentimentModelIntegrator()
    return _sentiment_integrator

def integrate_sentiment_in_pipeline(features_df: pd.DataFrame, news_data: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
    """Інтеграція sentiment моделей в pipeline"""
    integrator = get_sentiment_integrator()
    return integrator.enhance_features_with_sentiment(features_df, news_data, price_data)

def get_sentiment_trading_signal(news_data: pd.DataFrame, price_data: pd.DataFrame) -> Dict[str, Any]:
    """Отримання торгового сигналу від sentiment моделей"""
    integrator = get_sentiment_integrator()
    return integrator.get_sentiment_signal(news_data, price_data)
