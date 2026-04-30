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
        
        # 1. Volatility analysis
        self._analyze_volatility(df, context)
        
        # 2. Trend analysis
        self._analyze_trend(df, context)
        
        # 3. Market regime analysis
        self._analyze_market_regime(df, context)
        
        # 4. Data volume analysis
        self._analyze_data_volume(df, context)
        
        # 5. Data quality analysis
        self._analyze_data_quality(df, context)
        
        # 6. Time patterns analysis
        self._analyze_time_patterns(df, context)
        
        # 7. Causal analysis
        causal_weight = self._analyze_causal_factors(trigger_event)
        
        # 8. Feature importance
        feature_importance = self._calculate_dynamic_feature_importance(df, context)
        
        # 9. Overall context score
        context['overall_score'] = self._calculate_overall_context_score(context, feature_importance) * causal_weight
        context['overall_score'] = min(1.0, max(0.1, context['overall_score']))
        
        # 10. Update history
        self._update_context_history(ticker, context)
        
        return context
    
    def _analyze_volatility(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze volatility with dynamic weighting"""
        returns = df['close'].pct_change().dropna()
        volatility = returns.std()
        
        if volatility < 0.01:
            volatility_weight = 1.5
        elif volatility < 0.02:
            volatility_weight = 1.0
        elif volatility < 0.04:
            volatility_weight = 0.8
        else:
            volatility_weight = 0.5
        
        context['volatility'] = {
            'value': volatility,
            'weight': volatility_weight,
            'category': self._categorize_volatility(volatility)
        }
    
    def _analyze_trend(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze trend with dynamic weighting"""
        if len(df) < 20:
            return
        
        short_trend = (df['close'].iloc[-1] / df['close'].iloc[-5]) - 1
        medium_trend = (df['close'].iloc[-1] / df['close'].iloc[-10]) - 1
        long_trend = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1
        
        if abs(short_trend) > abs(medium_trend) > abs(long_trend):
            trend_weight = 1.3
        elif abs(medium_trend) > abs(long_trend):
            trend_weight = 1.1
        else:
            trend_weight = 1.0
        
        context['trend'] = {
            'short_trend': short_trend,
            'medium_trend': medium_trend,
            'long_trend': long_trend,
            'weight': trend_weight,
            'dominant_period': self._get_dominant_trend(short_trend, medium_trend, long_trend)
        }
    
    def _analyze_market_regime(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze market regime based on VIX"""
        if 'VIX_SIGNAL' not in df.columns:
            return
        
        vix = df['VIX_SIGNAL'].mean()
        
        if vix > 30:
            market_regime, regime_weight = "extreme_fear", 0.3
        elif vix > 20:
            market_regime, regime_weight = "fear", 0.6
        elif vix > 15:
            market_regime, regime_weight = "neutral", 1.0
        elif vix > 10:
            market_regime, regime_weight = "greed", 1.2
        else:
            market_regime, regime_weight = "extreme_greed", 0.8
        
        context['market_regime'] = {
            'vix_value': vix,
            'regime': market_regime,
            'weight': regime_weight,
            'confidence': self._calculate_vix_confidence(vix)
        }
    
    def _analyze_data_volume(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze data volume with dynamic weighting"""
        data_volume = len(df)
        
        if data_volume < 100:
            volume_weight = 0.8
        elif data_volume < 500:
            volume_weight = 1.0
        elif data_volume < 1000:
            volume_weight = 1.2
        else:
            volume_weight = 1.5
        
        context['data_volume'] = {
            'count': data_volume,
            'weight': volume_weight,
            'category': self._categorize_data_volume(data_volume)
        }
    
    def _analyze_data_quality(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze data quality with dynamic weighting"""
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        if missing_pct < 0.01:
            quality_weight = 1.2
        elif missing_pct < 0.05:
            quality_weight = 1.0
        elif missing_pct < 0.1:
            quality_weight = 0.8
        else:
            quality_weight = 0.5
        
        context['data_quality'] = {
            'missing_pct': missing_pct,
            'weight': quality_weight,
            'category': self._categorize_data_quality(missing_pct)
        }
    
    def _analyze_time_patterns(self, df: pd.DataFrame, context: Dict) -> None:
        """Analyze time patterns"""
        if 'datetime' not in df.columns:
            return
        
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        df['intraday_vol'] = df.groupby(df['datetime'].dt.date)['close'].transform(lambda x: x.std())
        avg_intraday_vol = df['intraday_vol'].mean()
        
        daily_returns = df.groupby(df['datetime'].dt.date)['close'].last() / df.groupby(df['datetime'].dt.date)['close'].first() - 1
        overnight_gap = daily_returns.mean()
        
        if avg_intraday_vol > 0.02:
            time_weight = 1.2
        elif overnight_gap > 0.005:
            time_weight = 1.3
        else:
            time_weight = 1.0
        
        context['time_patterns'] = {
            'intraday_vol': avg_intraday_vol,
            'overnight_gap': overnight_gap,
            'weight': time_weight,
            'category': self._categorize_time_patterns(avg_intraday_vol, overnight_gap)
        }
    
    def _analyze_causal_factors(self, trigger_event: Optional[str]) -> float:
        """Analyze causal factors and return weight"""
        causal_weight = 1.0
        
        if trigger_event and self.causal_engine:
            projections = self.causal_engine.generate_causal_projections(trigger_event)
            if projections:
                impact_sum = sum(p['expected_impact'] for p in projections)
                causal_weight = 1.0 + (impact_sum * 0.2)
        
        return causal_weight
    
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
        """Calculates confidence у режимі ринку"""
        if vix < 10:
            return 0.9  # Дуже низький VIX - висока confidence
        elif vix < 20:
            return 0.7  # Низький VIX - помірна confidence
        elif vix < 30:
            return 0.5  # Середній VIX - низька confidence
        else:
            return 0.3  # Високий VIX - дуже низька confidence
    
    def _calculate_dynamic_feature_importance(self, df: pd.DataFrame, context: Dict) -> Dict[str, float]:
        """Calculates динамічну важливість показників"""
        
        importance = {}
        
        # Technical indicators
        self._calculate_technical_importance(df, context, importance)
        
        # Volume indicators
        self._calculate_volume_importance(df, context, importance)
        
        # Quality indicators
        self._calculate_quality_importance(df, context, importance)
        
        return importance
    
    def _calculate_technical_importance(self, df: pd.DataFrame, context: Dict, importance: Dict) -> None:
        """Calculate importance for technical indicators"""
        technical_indicators = ['rsi', 'macd', 'bb_width', 'volume_ratio']
        
        for indicator in technical_indicators:
            if indicator not in df.columns:
                continue
            
            base_importance = self._get_base_importance(indicator)
            adjusted_importance = base_importance
            
            # Adjust based on volatility
            if context['volatility']['category'] == 'high':
                adjusted_importance *= 1.5
            elif context['volatility']['category'] == 'low':
                adjusted_importance *= 0.7
            
            # Adjust based on trend
            if context.get('trend'):
                if context['trend']['dominant_period'] == 'short_term':
                    adjusted_importance *= 1.3
                elif context['trend']['dominant_period'] == 'long_term':
                    adjusted_importance *= 0.8
            
            importance[indicator] = adjusted_importance
    
    def _calculate_volume_importance(self, df: pd.DataFrame, context: Dict, importance: Dict) -> None:
        """Calculate importance for volume indicators"""
        volume_indicators = ['volume', 'trades_count']
        
        for indicator in volume_indicators:
            if indicator not in df.columns:
                continue
            
            base_importance = self._get_base_importance(indicator)
            
            if context['data_volume']['category'] == 'small':
                importance[indicator] = base_importance * 1.4
            elif context['data_volume']['category'] == 'large':
                importance[indicator] = base_importance * 0.9
            else:
                importance[indicator] = base_importance
    
    def _calculate_quality_importance(self, df: pd.DataFrame, context: Dict, importance: Dict) -> None:
        """Calculate importance for quality indicators"""
        quality_indicators = ['close', 'high', 'low']
        
        for indicator in quality_indicators:
            if indicator not in df.columns:
                continue
            
            base_importance = self._get_base_importance(indicator)
            
            if context['data_quality']['category'] == 'poor':
                importance[indicator] = base_importance * 1.3
            elif context['data_quality']['category'] == 'excellent':
                importance[indicator] = base_importance * 0.8
            else:
                importance[indicator] = base_importance
    
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
        """Calculates загальний контекстний бал"""
        
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