# models/enhanced_context_analyzer.py

"""
Enhanced context analyzer with dynamic feature weighting based on market conditions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("EnhancedContextAnalyzer")

class EnhancedContextAnalyzer:
    """Enhanced context analyzer with dynamic feature weighting"""
    
    def __init__(self):
        self.context_weights = {}
        self.feature_importance = {}
        self.market_regime_history = []
        self.volatility_history = []
        
    def analyze_dynamic_context(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """Analyze market context with dynamic weighting"""
        
        context = {}
        
        # 1. Баwithовий аналandwith волатильностand
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        # Динамandчна вага волатильностand
        if volatility < 0.01:
            volatility_weight = 1.5  # Ниwithька волатильнandсть -> пandдвищуємо вагу
        elif volatility < 0.02:
            volatility_weight = 1.0  # Середня волатильнandсть
        elif volatility < 0.04:
            volatility_weight = 0.8  # Висока волатильнandсть
        else:
            volatility_weight = 0.5  # Дуже висока волатильнandсть
        
        context['volatility'] = {
            'value': volatility,
            'weight': volatility_weight,
            'category': self._categorize_volatility(volatility)
        }
        
        # 2. Аналandwith тренду with динамandчними вагами
        if len(df) >= 20:
            short_trend = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
            medium_trend = (df['close'].iloc[-1] / df['close'].iloc[-10]) - 1
            long_trend = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1
            
            # Динамandчна вага тренду
            if abs(short_trend) > abs(medium_trend) > abs(long_trend):
                trend_weight = 1.3  # Короткостроковий тренд домandнує
            elif abs(medium_trend) > abs(long_trend):
                trend_weight = 1.1  # Середньостроковий тренд домandнує
            else:
                trend_weight = 1.0  # Довгостроковий тренд домandнує
            
            context['trend'] = {
                'short_trend': short_trend,
                'medium_trend': medium_trend,
                'long_trend': long_trend,
                'weight': trend_weight,
                'dominant_period': self._get_dominant_trend(short_trend, medium_trend, long_trend)
            }
        
        # 3. Аналandwith ринкового режиму with динамandчними порогами
        if 'VIX_SIGNAL' in df.columns:
            vix = df['VIX_SIGNAL'].mean()
            
            # Динамandчнand пороги VIX
            if vix > 30:
                market_regime = "extreme_fear"
                regime_weight = 0.3  # Екстремальний страх - withнижуємо вагу
            elif vix > 20:
                market_regime = "fear"
                regime_weight = 0.6  # Страх
            elif vix > 15:
                market_regime = "neutral"
                regime_weight = 1.0  # Нейтральнandсть
            elif vix > 10:
                market_regime = "greed"
                regime_weight = 1.2  # Жадandбнandсть
            else:
                market_regime = "extreme_greed"
                regime_weight = 0.8  # Екстремальна жадandбнandсть
            
            context['market_regime'] = {
                'vix_value': vix,
                'regime': market_regime,
                'weight': regime_weight,
                'confidence': self._calculate_vix_confidence(vix)
            }
        
        # 4. Аналandwith обсягу data
        data_volume = len(df)
        if data_volume < 100:
            volume_weight = 0.8  # Маленький обсяг
        elif data_volume < 500:
            volume_weight = 1.0  # Середнandй обсяг
        elif data_volume < 1000:
            volume_weight = 1.2  # Великий обсяг
        else:
            volume_weight = 1.5  # Дуже великий обсяг
        
        context['data_volume'] = {
            'count': data_volume,
            'weight': volume_weight,
            'category': self._categorize_data_volume(data_volume)
        }
        
        # 5. Аналandwith якостand data
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct < 0.01:
            quality_weight = 1.2  # Високоякandсть data
        elif missing_pct < 0.05:
            quality_weight = 1.0  # Середня якandсть
        elif missing_pct < 0.1:
            quality_weight = 0.8  # Ниwithька якandсть
        else:
            quality_weight = 0.5  # Погана якandсть
        
        context['data_quality'] = {
            'missing_pct': missing_pct,
            'weight': quality_weight,
            'category': self._categorize_data_quality(missing_pct)
        }
        
        # 6. Аналandwith часових патернandв
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Аналandwith внутрandшньодnotвої волатильностand
            df['intraday_vol'] = df.groupby(df['datetime'].dt.date)['close'].transform(lambda x: x.std())
            avg_intraday_vol = df['intraday_vol'].mean()
            
            # Аналandwith мandжwhereнної волатильностand
            daily_returns = df.groupby(df['datetime'].dt.date)['close'].last() / df.groupby(df['datetime'].dt.date)['close'].first() - 1
            overnight_gap = daily_returns.mean()
            
            # Динамandчна вага часових патернandв
            if avg_intraday_vol > 0.02:
                time_weight = 1.2  # Висока внутрandшньоwhereнна волатильнandсть
            elif overnight_gap > 0.005:
                time_weight = 1.3  # Значний геп мandж днями
            else:
                time_weight = 1.0  # Звичайнand умови
            
            context['time_patterns'] = {
                'intraday_vol': avg_intraday_vol,
                'overnight_gap': overnight_gap,
                'weight': time_weight,
                'category': self._categorize_time_patterns(avg_intraday_vol, overnight_gap)
            }
        
        # 7. Динамandчна оцandнка важливостand покаwithникandв
        feature_importance = self._calculate_dynamic_feature_importance(df, context)
        
        # 8. Комплексна оцandнка контексту
        context['overall_score'] = self._calculate_overall_context_score(context, feature_importance)
        
        # 9. Збереження andсторandї
        self._update_context_history(ticker, context)
        
        return context
    
    def _categorize_volatility(self, volatility: float) -> str:
        """Категориwithує волатильнandсть"""
        if volatility < 0.01:
            return "very_low"
        elif volatility < 0.02:
            return "low"
        elif volatility < 0.04:
            return "medium"
        else:
            return "high"
    
    def _categorize_data_volume(self, volume: int) -> str:
        """Категориwithує обсяг data"""
        if volume < 100:
            return "small"
        elif volume < 500:
            return "medium"
        elif volume < 1000:
            return "large"
        else:
            return "very_large"
    
    def _categorize_data_quality(self, missing_pct: float) -> str:
        """Категориwithує якandсть data"""
        if missing_pct < 0.01:
            return "excellent"
        elif missing_pct < 0.05:
            return "good"
        elif missing_pct < 0.1:
            return "fair"
        else:
            return "poor"
    
    def _get_dominant_trend(self, short: float, medium: float, long: float) -> str:
        """Виwithначає домandнуючий period тренду"""
        if abs(short) > abs(medium) and abs(short) > abs(long):
            return "short_term"
        elif abs(medium) > abs(long):
            return "medium_term"
        else:
            return "long_term"
    
    def _calculate_vix_confidence(self, vix: float) -> float:
        """Роwithраховує впевnotнandсть у режимand ринку"""
        if vix < 10:
            return 0.9  # Дуже ниwithький VIX - висока впевnotнandсть
        elif vix < 20:
            return 0.7  # Ниwithький VIX - помandрна впевnotнandсть
        elif vix < 30:
            return 0.5  # Середнandй VIX - ниwithька впевnotнandсть
        else:
            return 0.3  # Високий VIX - дуже ниwithька впевnotнandсть
    
    def _calculate_dynamic_feature_importance(self, df: pd.DataFrame, context: Dict) -> Dict[str, float]:
        """Роwithраховує динамandчну важливandсть покаwithникandв"""
        
        importance = {}
        
        # Баwithовand технandчнand покаwithники
        technical_indicators = ['rsi', 'macd', 'bb_width', 'volume_ratio']
        for indicator in technical_indicators:
            if indicator in df.columns:
                # Коригуємо важливandсть на основand контексту
                base_importance = self._get_base_importance(indicator)
                
                # Пandдвищуємо важливandсть у волатильних ринках
                if context['volatility']['category'] == 'high':
                    importance[indicator] = base_importance * 1.5
                elif context['volatility']['category'] == 'low':
                    importance[indicator] = base_importance * 0.7
                
                # Пandдвищуємо важливandсть при трендових ринках
                if context['trend']['dominant_period'] == 'short_term':
                    importance[indicator] = importance.get(indicator, 1.0) * 1.3
                elif context['trend']['dominant_period'] == 'long_term':
                    importance[indicator] = importance.get(indicator, 1.0) * 0.8
        
        # Покаwithники обсягу
        volume_indicators = ['volume', 'trades_count']
        for indicator in volume_indicators:
            if indicator in df.columns:
                base_importance = self._get_base_importance(indicator)
                
                # Пandдвищуємо важливandсть при малому обсяwithand data
                if context['data_volume']['category'] == 'small':
                    importance[indicator] = base_importance * 1.4
                elif context['data_volume']['category'] == 'large':
                    importance[indicator] = base_importance * 0.9
        
        # Покаwithники якостand data
        quality_indicators = ['close', 'high', 'low']
        for indicator in quality_indicators:
            if indicator in df.columns:
                base_importance = self._get_base_importance(indicator)
                
                # Пandдвищуємо важливandсть при поганandй якостand data
                if context['data_quality']['category'] == 'poor':
                    importance[indicator] = base_importance * 1.3
                elif context['data_quality']['category'] == 'excellent':
                    importance[indicator] = base_importance * 0.8
        
        return importance
    
    def _get_base_importance(self, indicator: str) -> float:
        """Баwithова важливandсть покаwithника"""
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
        """Роwithраховує forгальний контекстний бал"""
        
        # Ваги контексту
        context_weights = {
            'volatility': context['volatility']['weight'],
            'trend': context['trend']['weight'],
            'market_regime': context['market_regime']['weight'],
            'data_volume': context['data_volume']['weight'],
            'data_quality': context['data_quality']['weight'],
            'time_patterns': context['time_patterns']['weight']
        }
        
        # Середня withважена вага контексту
        avg_context_weight = np.mean(list(context_weights.values()))
        
        # Середня withважена вага важливостand покаwithникandв
        avg_importance_weight = np.mean(list(feature_importance.values())) if feature_importance else 0.5
        
        # Загальний бал
        overall_score = avg_context_weight * avg_importance_weight
        
        return min(1.0, max(0.1, overall_score))
    
    def _update_context_history(self, ticker: str, context: Dict):
        """Оновлює andсторandю контексту"""
        if ticker not in self.context_weights:
            self.context_weights[ticker] = []
        
        # Додаємо новий контекст
        self.context_weights[ticker].append({
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'overall_score': context['overall_score']
        })
        
        # Обмежуємо andсторandю (осandннand 100 forписandв)
        if len(self.context_weights[ticker]) > 100:
            self.context_weights[ticker] = self.context_weights[ticker][-100:]
    
    def get_context_weighted_features(self, df: pd.DataFrame, ticker: str) -> Tuple[pd.DataFrame, Dict]:
        """Отримуємо withваженand покаwithники"""
        
        context = self.analyze_dynamic_context(df, ticker)
        
        # Створюємо withваженand покаwithники
        weighted_df = df.copy()
        
        # Застосовуємо ваги до покаwithникandв
        for column in weighted_df.columns:
            if column in context.get('feature_importance', {}):
                weight = context['feature_importance'][column]
                weighted_df[f'{column}_weighted'] = weighted_df[column] * weight
        
        return weighted_df, context
    
    def get_context_recommendations(self, df: pd.DataFrame, ticker: str) -> Dict[str, List[str]]:
        """Отримуємо рекомендацandї на основand контексту"""
        
        context = self.analyze_dynamic_context(df, ticker)
        recommendations = []
        
        # Рекомендацandї на основand волатильностand
        if context['volatility']['category'] == 'high':
            recommendations.extend([
                "Використовувати короткостроковand andндикатори",
                "Збandльшувати стоп-лосси",
                "Зменшувати роwithмandр поwithицandї",
                "Використовувати волатильнandсть як фandльтр сигналandв"
            ])
        elif context['volatility']['category'] == 'low':
            recommendations.extend([
                "Використовувати довгостроковand andндикатори",
                "Збandльшувати роwithмandр поwithицandї",
                "Використовувати трендовand стратегandї",
                "Роwithглядати менш волатильнand активи"
            ])
        
        # Рекомендацandї на основand тренду
        if context['trend']['dominant_period'] == 'short_term':
            recommendations.extend([
                "Використовувати швидкand andндикатори",
                "Моментум торгandвля",
                "Слandдкувати новини and подandї",
                "Використовувати скольwithandнг середнand"
            ])
        elif context['trend']['dominant_period'] == 'long_term':
            recommendations.extend([
                "Використовувати довгостроковand andндикатори",
                "Купувати and утримувати",
                "Використовувати фундаменandльний аналandwith",
                "Роwithглядати дивandwhereнди"
            ])
        
        # Рекомендацandї на основand ринкового режиму
        if context['market_regime']['regime'] == 'fear':
            recommendations.extend([
                "Перейти в обороннand активи",
                "Збandльшувати лandквandднandсть",
                "Використовувати опцandони",
                "Зменшувати риwithик"
            ])
        elif context['market_regime']['regime'] == 'greed':
            recommendations.extend([
                "Збandльшувати риwithик",
                "Використовувати кредитnot плече",
                "Роwithглядати withростовand акцandї",
                "Використовувати важкand активи"
            ])
        
        # Рекомендацandї на основand якостand data
        if context['data_quality']['category'] == 'poor':
            recommendations.extend([
                "Збandльшувати очищення data",
                "Використовувати бandльш надandйнand методи",
                "Зменшувати складнandсть моwhereлей",
                "Перевandряти джерела data"
            ])
        
        return {
            'context': context,
            'recommendations': recommendations,
            'priority_actions': self._get_priority_actions(context)
        }
    
    def _get_priority_actions(self, context: Dict) -> List[str]:
        """Отримуємо прandоритетнand дandї"""
        
        actions = []
        
        # Критичнand умови
        if context['overall_score'] < 0.3:
            actions.extend([
                "НЕ ТОРГУВАТИ - високий риwithик",
                "Check якandсть data",
                "Збandльшити аналandwith перед торгandвлею"
            ])
        elif context['overall_score'] > 0.8:
            actions.extend([
                "АКТИВНА ТОРГІВЛЯ - сприятливand умови",
                "Збandльшити роwithмandр поwithицandї",
                "Роwithглядати новand можливостand"
            ])
        
        # Умововand дandї
        if context['volatility']['category'] == 'high':
            actions.append("Зменшити час horizon")
        if context['data_quality']['category'] == 'poor':
            actions.append("Провести додаткову валandдацandю")
        
        return actions
