# src/models/model_selector/enhanced_context_analyzer.py

"""
Enhanced context analyzer with dynamic feature weighting based on market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from src.core.logging.logger import ProjectLogger
from src.analytics.context.causal_engine import CausalEngine

logger = ProjectLogger.get_logger("EnhancedContextAnalyzer")

class EnhancedContextAnalyzer:
    """Enhanced context analyzer with dynamic feature weighting"""
    
    def __init__(self, causal_engine: Optional[CausalEngine] = None):
        self.context_weights = {}
        self.feature_importance = {}
        self.market_regime_history = []
        self.volatility_history = []
        self.causal_engine = causal_engine or CausalEngine()
        
    def analyze_dynamic_context(self, df: pd.DataFrame, ticker: str, trigger_event: Optional[str] = None) -> Dict[str, Any]:
        """Analyze market context with dynamic weighting and causal factors"""
        
        context = {}
        
        # 1. Базовий аналіз волатильності
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Динамічна вага волатильності
        if volatility < 0.01:
            volatility_weight = 1.5  # Низька волатильність -> підвищуємо вагу
        elif volatility < 0.02:
            volatility_weight = 1.0  # Середня волатильність
        elif volatility < 0.04:
            volatility_weight = 0.8  # Висока волатильність
        else:
            volatility_weight = 0.5  # Дуже висока волатильність
        
        context['volatility'] = {
            'value': volatility,
            'weight': volatility_weight,
            'category': self._categorize_volatility(volatility)
        }
        
        # 2. Аналіз тренду з динамічними вагами
        if len(df) >= 20:
            short_trend = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
            medium_trend = (df['close'].iloc[-1] / df['close'].iloc[-10]) - 1
            long_trend = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1
            
            # Динамічна вага тренду
            if abs(short_trend) > abs(medium_trend) > abs(long_trend):
                trend_weight = 1.3  # Короткостроковий тренд домінує
            elif abs(medium_trend) > abs(long_trend):
                trend_weight = 1.1  # Середньостроковий тренд домінує
            else:
                trend_weight = 1.0  # Довгостроковий тренд домінує
            
            context['trend'] = {
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'weight': trend_weight,
                'dominant_period': self._get_dominant_trend(short_trend, medium_trend, long_trend)
            }
        
        # 3. Аналіз ринкового режиму з динамічними порогами
        if 'VIX_SIGNAL' in df.columns:
            vix = df['VIX_SIGNAL'].mean()
            
            # Динамічні пороги VIX
            if vix > 30:
                market_regime = "extreme_fear"
                regime_weight = 0.3  # Екстремальний страх - знижуємо вагу
            elif vix > 20:
                market_regime = "fear"
                regime_weight = 0.6  # Страх
            elif vix > 15:
                market_regime = "neutral"
                regime_weight = 1.0  # Нейтральність
            elif vix > 10:
                market_regime = "greed"
                regime_weight = 1.2  # Жадібність
            else:
                market_regime = "extreme_greed"
                regime_weight = 0.8  # Екстремальна жадібність
            
            context['market_regime'] = {
                'vix_value': vix,
                'regime': market_regime,
                'weight': regime_weight,
                'confidence': self._calculate_vix_confidence(vix)
            }
        
        # 4. Аналіз обсягу data
        data_volume = len(df)
        if data_volume < 100:
            volume_weight = 0.8  # Маленький обсяг
        elif data_volume < 500:
            volume_weight = 1.0  # Середній обсяг
        elif data_volume < 1000:
            volume_weight = 1.2  # Великий обсяг
        else:
            volume_weight = 1.5  # Дуже великий обсяг
        
        context['data_volume'] = {
            'count': data_volume,
            'weight': volume_weight,
            'category': self._categorize_data_volume(data_volume)
        }
        
        # 5. Аналіз якості data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct < 0.01:
            quality_weight = 1.2  # Високоякість data
        elif missing_pct < 0.05:
            quality_weight = 1.0  # Середня якість
        elif missing_pct < 0.1:
            quality_weight = 0.8  # Низька якість
        else:
            quality_weight = 0.5  # Погана якість
        
        context['data_quality'] = {
            'missing_pct': missing_pct,
            'weight': quality_weight,
            'category': self._categorize_data_quality(missing_pct)
        }
        
        # 6. Аналіз часових патернів
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Аналіз внутрішньоденної волатильності
            df['intraday_vol'] = df.groupby(df['datetime'].dt.date)['close'].transform(lambda x: x.std())
            avg_intraday_vol = df['intraday_vol'].mean()
            
            # Аналіз міжденної волатильності
            daily_returns = df.groupby(df['datetime'].dt.date)['close'].last() / df.groupby(df['datetime'].dt.date)['close'].first() - 1
            overnight_gap = daily_returns.mean()
            
            # Динамічна вага часових патернів
            if avg_intraday_vol > 0.02:
                time_weight = 1.2  # Висока внутрішньоденна волатильність
            elif overnight_gap > 0.005:
                time_weight = 1.3  # Значний геп між днями
            else:
                time_weight = 1.0  # Звичайні умови
            
            context['time_patterns'] = {
                'intraday_vol': avg_intraday_vol,
                'overnight_gap': overnight_gap,
                'weight': time_weight,
                'category': self._categorize_time_patterns(avg_intraday_vol, overnight_gap)
            }

        # 7. Каузальний аналіз (Causal Vectors)
        causal_weight = 1.0
        if trigger_event:
            projections = self.causal_engine.generate_causal_projections(trigger_event)
            if projections:
                context['causal_projections'] = projections
                # Зменшуємо впевненість/бал, якщо прогнозується ланцюжок негативних подій
                impact_sum = sum(p['expected_impact'] for p in projections)
                causal_weight = 1.0 + (impact_sum * 0.2) # Impact can be negative

        # 8. Динамічна оцінка важливості показників
        feature_importance = self._calculate_dynamic_feature_importance(df, context)
        
        # 9. Комплексна оцінка контексту з урахуванням каузальності
        context['overall_score'] = self._calculate_overall_context_score(context, feature_importance) * causal_weight
        context['overall_score'] = min(1.0, max(0.1, context['overall_score']))
        
        # 10. Збереження історії
        self._update_context_history(ticker, context)
        
        return context
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Категоризує волатильність"""
        if volatility < 0.01:
            return "very_low"
        elif volatility < 0.02:
            return "low"
        elif volatility < 0.04:
            return "medium"
        else:
            return "high"
    
    def _categorize_data_volume(self, volume: int) -> str:
        """Категоризує обсяг data"""
        if volume < 100:
            return "small"
        elif volume < 500:
            return "medium"
        elif volume < 1000:
            return "large"
        else:
            return "very_large"
    
    def _categorize_data_quality(self, missing_pct: float) -> str:
        """Категоризує якість data"""
        if missing_pct < 0.01:
            return "excellent"
        elif missing_pct < 0.05:
            return "good"
        elif missing_pct < 0.1:
            return "fair"
        else:
            return "poor"

    def _categorize_time_patterns(self, intraday_vol: float, overnight_gap: float) -> str:
        """Категоризує часові патерни"""
        if intraday_vol > 0.04:
            return "volatile_intraday"
        if abs(overnight_gap) > 0.01:
            return "gap_risk"
        return "stable"
    
    def _get_dominant_trend(self, short: float, medium: float, long: float) -> str:
        """Визначає домінуючий період тренду"""
        if abs(short) > abs(medium) and abs(short) > abs(long):
            return "short_term"
        elif abs(medium) > abs(long):
            return "medium_term"
        else:
            return "long_term"
    
    def _calculate_vix_confidence(self, vix: float) -> float:
        """Розраховує впевненість у режимі ринку"""
        if vix < 10:
            return 0.9  # Дуже низький VIX - висока впевненість
        elif vix < 20:
            return 0.7  # Низький VIX - помірна впевненість
        elif vix < 30:
            return 0.5  # Середній VIX - низька впевненість
        else:
            return 0.3  # Високий VIX - дуже низька впевненість
    
    def _calculate_dynamic_feature_importance(self, df: pd.DataFrame, context: Dict) -> Dict[str, float]:
        """Розраховує динамічну важливість показників"""
        
        importance = {}
        
        # Базові технічні показники
        technical_indicators = ['rsi', 'macd', 'bb_width', 'volume_ratio']
        for indicator in technical_indicators:
            if indicator in df.columns:
                # Коригуємо важливість на основі контексту
                base_importance = self._get_base_importance(indicator)
                
                # Підвищуємо важливість у волатильних ринках
                if context['volatility']['category'] == 'high':
                    importance[indicator] = base_importance * 1.5
                elif context['volatility']['category'] == 'low':
                    importance[indicator] = base_importance * 0.7
                
                # Підвищуємо важливість при трендових ринках
                if context.get('trend') and context['trend']['dominant_period'] == 'short_term':
                    importance[indicator] = importance.get(indicator, base_importance) * 1.3
                elif context.get('trend') and context['trend']['dominant_period'] == 'long_term':
                    importance[indicator] = importance.get(indicator, base_importance) * 0.8
        
        # Показники обсягу
        volume_indicators = ['volume', 'trades_count']
        for indicator in volume_indicators:
            if indicator in df.columns:
                base_importance = self._get_base_importance(indicator)
                
                # Підвищуємо важливість при малому обсязі data
                if context['data_volume']['category'] == 'small':
                    importance[indicator] = base_importance * 1.4
                elif context['data_volume']['category'] == 'large':
                    importance[indicator] = base_importance * 0.9
        
        # Показники якості data
        quality_indicators = ['close', 'high', 'low']
        for indicator in quality_indicators:
            if indicator in df.columns:
                base_importance = self._get_base_importance(indicator)
                
                # Підвищуємо важливість при поганій якості data
                if context['data_quality']['category'] == 'poor':
                    importance[indicator] = base_importance * 1.3
                elif context['data_quality']['category'] == 'excellent':
                    importance[indicator] = base_importance * 0.8
        
        return importance
    
    def _get_base_importance(self, indicator: str) -> float:
        """Базова важливість показника"""
        importance_map = {
            'rsi': 0.8,
            'macd': 0.7,
            'bb_width': 0.6,
            'volume_ratio': 0.9,
            'volume': 0.7,
            'trades_count': 0.6,
            'close': 1.0,
            'high': 1.0,
            'low': 1.0
        }
        return importance_map.get(indicator, 0.5)
    
    def _calculate_overall_context_score(self, context: Dict, feature_importance: Dict) -> float:
        """Розраховує загальний контекстний бал"""
        
        # Ваги контексту
        context_weights = {
            'volatility': context['volatility']['weight'],
            'trend': context.get('trend', {}).get('weight', 1.0),
            'market_regime': context.get('market_regime', {}).get('weight', 1.0),
            'data_volume': context['data_volume']['weight'],
            'data_quality': context['data_quality']['weight'],
            'time_patterns': context.get('time_patterns', {}).get('weight', 1.0)
        }
        
        # Середня зважена вага контексту
        avg_context_weight = np.mean(list(context_weights.values()))
        
        # Середня зважена вага важливості показників
        avg_importance_weight = np.mean(list(feature_importance.values())) if feature_importance else 0.5
        
        # Загальний бал
        overall_score = avg_context_weight * avg_importance_weight
        
        return overall_score
    
    def _update_context_history(self, ticker: str, context: Dict):
        """Оновлює історію контексту"""
        if ticker not in self.context_weights:
            self.context_weights[ticker] = []
        
        # Додаємо новий контекст
        self.context_weights[ticker].append({
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'overall_score': context['overall_score']
        })
        
        # Обмежуємо історію (останні 100 записів)
        if len(self.context_weights[ticker]) > 100:
            self.context_weights[ticker] = self.context_weights[ticker][-100:]
    
    def get_context_weighted_features(self, df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
        """Отримуємо зважені показники"""
        
        context = self.analyze_dynamic_context(df, ticker)
        
        # Створюємо зважені показники
        weighted_df = df.copy()
        
        # Застосовуємо ваги до показників
        for column in weighted_df.columns:
            if column in context.get('feature_importance', {}):
                weight = context['feature_importance'][column]
                weighted_df[f'{column}_weighted'] = weighted_df[column] * weight
        
        return weighted_df, context
    
    def get_context_recommendations(self, df: pd.DataFrame, ticker: str) -> Dict[str, List[str]]:
        """Отримуємо рекомендації на основі контексту"""
        
        context = self.analyze_dynamic_context(df, ticker)
        recommendations = []
        
        # Рекомендації на основі волатильності
        if context['volatility']['category'] == 'high':
            recommendations.extend([
                "Використовувати короткострокові індикатори",
                "Збільшувати стоп-лосси",
                "Зменшувати розмір позиції",
                "Використовувати волатильність як фільтр сигналів"
            ])
        elif context['volatility']['category'] == 'low':
            recommendations.extend([
                "Використовувати довгострокові індикатори",
                "Збільшувати розмір позиції",
                "Використовувати трендові стратегії",
                "Розглядати менш волатильні активи"
            ])
        
        # Рекомендації на основі тренду
        if context.get('trend') and context['trend']['dominant_period'] == 'short_term':
            recommendations.extend([
                "Використовувати швидкі індикатори",
                "Моментум торгівля",
                "Слідкувати новини та події",
                "Використовувати ковзаючі середні"
            ])
        elif context.get('trend') and context['trend']['dominant_period'] == 'long_term':
            recommendations.extend([
                "Використовувати довгострокові індикатори",
                "Купувати та утримувати",
                "Використовувати фундаментальний аналіз",
                "Розглядати дивіденди"
            ])
        
        # Рекомендації на основі ринкового режиму
        if context.get('market_regime') and context['market_regime']['regime'] == 'fear':
            recommendations.extend([
                "Перейти в оборонні активи",
                "Збільшувати ліквідність",
                "Використовувати опціони",
                "Зменшувати ризик"
            ])
        elif context.get('market_regime') and context['market_regime']['regime'] == 'greed':
            recommendations.extend([
                "Збільшувати ризик",
                "Використовувати кредитне плече",
                "Розглядати ростові акції",
                "Використовувати важкі активи"
            ])
        
        # Рекомендації на основі якості data
        if context['data_quality']['category'] == 'poor':
            recommendations.extend([
                "Збільшувати очищення data",
                "Використовувати більш надійні методи",
                "Зменшувати складність моделей",
                "Перевіряти джерела data"
            ])
        
        return {
            'context': context,
            'recommendations': recommendations,
            'priority_actions': self._get_priority_actions(context)
        }
    
    def _get_priority_actions(self, context: Dict) -> List[str]:
        """Отримуємо пріоритетні дії"""
        
        actions = []
        
        # Критичні умови
        if context['overall_score'] < 0.3:
            actions.extend([
                "НЕ ТОРГУВАТИ - високий ризик",
                "Check якість data",
                "Збільшити аналіз перед торгівлею"
            ])
        elif context['overall_score'] > 0.8:
            actions.extend([
                "АКТИВНА ТОРГІВЛЯ - сприятливі умови",
                "Збільшити розмір позиції",
                "Розглядати нові можливості"
            ])
        
        # Умовні дії
        if context['volatility']['category'] == 'high':
            actions.append("Зменшити час horizon")
        if context['data_quality']['category'] == 'poor':
            actions.append("Провести додаткову валідацію")
        
        return actions