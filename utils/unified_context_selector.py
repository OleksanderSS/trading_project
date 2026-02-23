#!/usr/bin/env python3
"""
Unified Context Selector - Гнучкий вибір показників для економічного контексту
Об'єднує всі доступні показники з різних джерел та дозволяє динамічно обирати
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class IndicatorSource:
    """Джерело показника"""
    name: str
    source_type: str  # 'technical', 'economic', 'sentiment', 'temporal', 'market'
    data_path: Optional[str] = None
    collector_func: Optional[str] = None
    update_frequency: str = 'daily'  # 'realtime', 'daily', 'weekly', 'monthly'
    reliability: float = 1.0  # 0-1


class UnifiedContextSelector:
    """
    Уніфікований селектор контексту - гнучкий вибір показників
    Логіка: збираємо всі доступні показники → фільтруємо → обираємо оптимальні
    """
    
    def __init__(self):
        self.logger = logging.getLogger("UnifiedContextSelector")
        
        # Реєстр всіх доступних показників
        self.all_indicators = self._register_all_indicators()
        
        # Кешування результатів
        self.context_cache = {}
        self.cache_duration = timedelta(hours=1)
        
        # Параметри вибору
        self.selection_params = {
            'max_indicators': 100,        # Максимальна кількість показників
            'min_reliability': 0.7,       # Мінімальна надійність
            'correlation_threshold': 0.8,  # Порог кореляції для видалення
            'importance_threshold': 0.01,  # Порог важливості
            'noise_filter_level': 0.1     # Рівень фільтрації шуму
        }
        
        self.logger.info(f"UnifiedContextSelector initialized with {len(self.all_indicators)} indicators")
    
    def _register_all_indicators(self) -> Dict[str, IndicatorSource]:
        """Зареєструвати всі доступні показники"""
        indicators = {}
        
        # 1. Технічні індикатори (з flexible_feature_selection.py)
        technical_indicators = {
            # Price indicators
            'rsi_14': IndicatorSource('RSI_14', 'technical', update_frequency='daily', reliability=0.9),
            'rsi_30': IndicatorSource('RSI_30', 'technical', update_frequency='daily', reliability=0.9),
            'macd_signal': IndicatorSource('MACD_Signal', 'technical', update_frequency='daily', reliability=0.85),
            'macd_histogram': IndicatorSource('MACD_Histogram', 'technical', update_frequency='daily', reliability=0.85),
            'bb_upper': IndicatorSource('BB_Upper', 'technical', update_frequency='daily', reliability=0.8),
            'bb_lower': IndicatorSource('BB_Lower', 'technical', update_frequency='daily', reliability=0.8),
            'bb_width': IndicatorSource('BB_Width', 'technical', update_frequency='daily', reliability=0.8),
            
            # Volume indicators
            'volume_sma': IndicatorSource('Volume_SMA', 'technical', update_frequency='daily', reliability=0.85),
            'volume_ratio': IndicatorSource('Volume_Ratio', 'technical', update_frequency='daily', reliability=0.85),
            'obv': IndicatorSource('OBV', 'technical', update_frequency='daily', reliability=0.8),
            'vwap': IndicatorSource('VWAP', 'technical', update_frequency='daily', reliability=0.9),
            
            # Momentum indicators
            'momentum_10': IndicatorSource('Momentum_10', 'technical', update_frequency='daily', reliability=0.85),
            'roc_5': IndicatorSource('ROC_5', 'technical', update_frequency='daily', reliability=0.85),
            'stoch_k': IndicatorSource('Stoch_K', 'technical', update_frequency='daily', reliability=0.8),
            'stoch_d': IndicatorSource('Stoch_D', 'technical', update_frequency='daily', reliability=0.8),
            
            # Trend indicators
            'sma_20': IndicatorSource('SMA_20', 'technical', update_frequency='daily', reliability=0.95),
            'sma_50': IndicatorSource('SMA_50', 'technical', update_frequency='daily', reliability=0.95),
            'ema_12': IndicatorSource('EMA_12', 'technical', update_frequency='daily', reliability=0.9),
            'ema_26': IndicatorSource('EMA_26', 'technical', update_frequency='daily', reliability=0.9),
            'adx': IndicatorSource('ADX', 'technical', update_frequency='daily', reliability=0.85),
            
            # Volatility indicators
            'atr': IndicatorSource('ATR', 'technical', update_frequency='daily', reliability=0.9),
            'volatility_20': IndicatorSource('Volatility_20', 'technical', update_frequency='daily', reliability=0.85),
            'historical_volatility': IndicatorSource('Hist_Vol', 'technical', update_frequency='daily', reliability=0.85),
        }
        
        # 2. Економічні показники (з economic_context_mapper.py)
        economic_indicators = {
            'fedfunds': IndicatorSource('Fed_Funds_Rate', 'economic', update_frequency='daily', reliability=0.95),
            't10y2y': IndicatorSource('10Y_2Y_Spread', 'economic', update_frequency='daily', reliability=0.95),
            'vix': IndicatorSource('VIX_Index', 'economic', update_frequency='realtime', reliability=0.98),
            'unrate': IndicatorSource('Unemployment_Rate', 'economic', update_frequency='monthly', reliability=0.9),
            'cpi': IndicatorSource('CPI', 'economic', update_frequency='monthly', reliability=0.9),
            'gdp': IndicatorSource('GDP_Growth', 'economic', update_frequency='quarterly', reliability=0.85),
            'oil': IndicatorSource('Crude_Oil_Price', 'economic', update_frequency='daily', reliability=0.95),
            'gold': IndicatorSource('Gold_Price', 'economic', update_frequency='daily', reliability=0.95),
            'dxy': IndicatorSource('Dollar_Index', 'economic', update_frequency='daily', reliability=0.95),
            'confidence': IndicatorSource('Consumer_Confidence', 'economic', update_frequency='monthly', reliability=0.85),
            'pmi': IndicatorSource('Manufacturing_PMI', 'economic', update_frequency='monthly', reliability=0.85),
            
            # Додаткові макро показники
            'durable_goods': IndicatorSource('Durable_Goods', 'economic', update_frequency='monthly', reliability=0.8),
            'retail_sales': IndicatorSource('Retail_Sales', 'economic', update_frequency='monthly', reliability=0.85),
            'housing_starts': IndicatorSource('Housing_Starts', 'economic', update_frequency='monthly', reliability=0.8),
            'industrial_production': IndicatorSource('Industrial_Production', 'economic', update_frequency='monthly', reliability=0.85),
            'capacity_utilization': IndicatorSource('Capacity_Utilization', 'economic', update_frequency='monthly', reliability=0.8),
        }
        
        # 3. Sentiment показники
        sentiment_indicators = {
            'sentiment_score': IndicatorSource('Sentiment_Score', 'sentiment', update_frequency='daily', reliability=0.75),
            'news_count': IndicatorSource('News_Count', 'sentiment', update_frequency='daily', reliability=0.8),
            'news_impact': IndicatorSource('News_Impact', 'sentiment', update_frequency='daily', reliability=0.7),
            'social_sentiment': IndicatorSource('Social_Sentiment', 'sentiment', update_frequency='realtime', reliability=0.6),
            'fear_greed_index': IndicatorSource('Fear_Greed_Index', 'sentiment', update_frequency='daily', reliability=0.8),
            'put_call_ratio': IndicatorSource('Put_Call_Ratio', 'sentiment', update_frequency='daily', reliability=0.85),
        }
        
        # 4. Часові показники
        temporal_indicators = {
            'weekday': IndicatorSource('Weekday', 'temporal', update_frequency='daily', reliability=1.0),
            'hour_of_day': IndicatorSource('Hour_of_Day', 'temporal', update_frequency='daily', reliability=1.0),
            'is_market_hours': IndicatorSource('Is_Market_Hours', 'temporal', update_frequency='daily', reliability=1.0),
            'month': IndicatorSource('Month', 'temporal', update_frequency='daily', reliability=1.0),
            'quarter': IndicatorSource('Quarter', 'temporal', update_frequency='daily', reliability=1.0),
            'is_month_end': IndicatorSource('Is_Month_End', 'temporal', update_frequency='daily', reliability=1.0),
            'is_quarter_end': IndicatorSource('Is_Quarter_End', 'temporal', update_frequency='daily', reliability=1.0),
            'days_to_expiry': IndicatorSource('Days_To_Expiry', 'temporal', update_frequency='daily', reliability=0.9),
        }
        
        # 5. Ринкові показники
        market_indicators = {
            'spy_return': IndicatorSource('SPY_Return', 'market', update_frequency='daily', reliability=0.98),
            'qqq_return': IndicatorSource('QQQ_Return', 'market', update_frequency='daily', reliability=0.98),
            'market_cap': IndicatorSource('Market_Cap', 'market', update_frequency='daily', reliability=0.95),
            'pe_ratio': IndicatorSource('PE_Ratio', 'market', update_frequency='daily', reliability=0.9),
            'dividend_yield': IndicatorSource('Dividend_Yield', 'market', update_frequency='daily', reliability=0.9),
            'beta': IndicatorSource('Beta', 'market', update_frequency='daily', reliability=0.85),
            'short_interest': IndicatorSource('Short_Interest', 'market', update_frequency='weekly', reliability=0.8),
            'insider_trading': IndicatorSource('Insider_Trading', 'market', update_frequency='weekly', reliability=0.7),
        }
        
        # Об'єднуємо всі показники
        indicators.update(technical_indicators)
        indicators.update(economic_indicators)
        indicators.update(sentiment_indicators)
        indicators.update(temporal_indicators)
        indicators.update(market_indicators)
        
        return indicators
    
    def get_available_indicators(self, source_types: List[str] = None) -> Dict[str, IndicatorSource]:
        """
        Отримати доступні показники за типами джерел
        
        Args:
            source_types: Список типів джерел (напр. ['economic', 'technical'])
            
        Returns:
            Dict: Доступні показники
        """
        if source_types is None:
            return self.all_indicators
        
        return {
            name: indicator for name, indicator in self.all_indicators.items()
            if indicator.source_type in source_types
        }
    
    def select_context_indicators(self, 
                                available_data: pd.DataFrame,
                                target_variable: str = None,
                                max_indicators: int = None,
                                source_types: List[str] = None,
                                importance_method: str = 'correlation') -> Dict[str, any]:
        """
        Гнучкий вибір контекстних показників
        
        Args:
            available_data: Доступні дані
            target_variable: Цільова змінна для важливості
            max_indicators: Максимальна кількість показників
            source_types: Типи джерел для вибору
            importance_method: Метод розрахунку важливості
            
        Returns:
            Dict: Результати вибору показників
        """
        try:
            if max_indicators is None:
                max_indicators = self.selection_params['max_indicators']
            
            # 1. Фільтруємо доступні показники
            candidate_indicators = self._filter_available_indicators(available_data, source_types)
            
            # 2. Розраховуємо важливість
            importance_scores = self._calculate_importance_scores(
                available_data, candidate_indicators, target_variable, importance_method
            )
            
            # 3. Видаляємо корельовані показники
            filtered_indicators = self._remove_correlated_indicators(
                available_data, importance_scores
            )
            
            # 4. Обираємо топ-N показників
            selected_indicators = self._select_top_indicators(
                filtered_indicators, max_indicators
            )
            
            # 5. Створюємо контекстний датасет
            context_data = self._create_context_dataset(available_data, selected_indicators)
            
            # 6. Розраховуємо контекстний скор
            context_scores = self._calculate_context_scores(context_data, selected_indicators)
            
            result = {
                'selected_indicators': selected_indicators,
                'importance_scores': importance_scores,
                'context_data': context_data,
                'context_scores': context_scores,
                'selection_stats': {
                    'total_available': len(candidate_indicators),
                    'after_correlation_filter': len(filtered_indicators),
                    'final_selected': len(selected_indicators),
                    'source_distribution': self._get_source_distribution(selected_indicators)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Selected {len(selected_indicators)} indicators from {len(candidate_indicators)} available")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error selecting context indicators: {e}")
            return {'error': str(e)}
    
    def _filter_available_indicators(self, data: pd.DataFrame, 
                                   source_types: List[str] = None) -> Dict[str, IndicatorSource]:
        """Фільтрувати показники, що доступні в data"""
        available_indicators = {}
        
        candidates = self.get_available_indicators(source_types)
        
        for name, indicator in candidates.items():
            # Перевіряємо надійність
            if indicator.reliability < self.selection_params['min_reliability']:
                continue
            
            # Перевіряємо наявність в data
            if name in data.columns:
                # Перевіряємо якість data
                if self._check_data_quality(data[name]):
                    available_indicators[name] = indicator
        
        return available_indicators
    
    def _check_data_quality(self, series: pd.Series) -> bool:
        """Перевірити якість data"""
        # Перевіряємо кількість пропущених значень
        missing_ratio = series.isna().sum() / len(series)
        if missing_ratio > 0.2:  # Більше 20% пропущених
            return False
        
        # Перевіряємо варіативність
        if series.nunique() < 2:  # Всі значення однакові
            return False
        
        return True
    
    def _calculate_importance_scores(self, data: pd.DataFrame, 
                                   indicators: Dict[str, IndicatorSource],
                                   target_variable: str = None,
                                   method: str = 'correlation') -> Dict[str, float]:
        """Розрахувати важливість показників"""
        importance_scores = {}
        
        for name in indicators.keys():
            if name not in data.columns:
                continue
            
            series = data[name].dropna()
            
            if method == 'correlation' and target_variable and target_variable in data.columns:
                # Кореляція з цільовою змінною
                target_series = data[target_variable].dropna()
                common_index = series.index.intersection(target_series.index)
                
                if len(common_index) > 10:
                    corr = series.loc[common_index].corr(target_series.loc[common_index])
                    importance_scores[name] = abs(corr) if not pd.isna(corr) else 0.0
                else:
                    importance_scores[name] = 0.0
            
            elif method == 'variance':
                # Дисперсія як міра важливості
                if series.dtype in ['float64', 'int64']:
                    variance = series.var()
                    importance_scores[name] = variance if not pd.isna(variance) else 0.0
                else:
                    importance_scores[name] = 0.0
            
            elif method == 'trend':
                # Трендова важливість (лінійний тренд)
                if len(series) > 10 and series.dtype in ['float64', 'int64']:
                    x = np.arange(len(series))
                    slope = np.polyfit(x, series, 1)[0]
                    importance_scores[name] = abs(slope) if not pd.isna(slope) else 0.0
                else:
                    importance_scores[name] = 0.0
            
            else:
                # За замовчуванням - надійність джерела
                importance_scores[name] = indicators[name].reliability
        
        return importance_scores
    
    def _remove_correlated_indicators(self, data: pd.DataFrame, 
                                    importance_scores: Dict[str, float]) -> Dict[str, IndicatorSource]:
        """Видалити корельовані показники"""
        # Сортуємо за важливістю
        sorted_indicators = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        
        filtered_indicators = {}
        correlation_matrix = data.corr()
        
        for name, score in sorted_indicators:
            if score < self.selection_params['importance_threshold']:
                continue
            
            # Перевіряємо кореляцію з вже відібраними
            is_correlated = False
            for selected_name in filtered_indicators.keys():
                if selected_name in correlation_matrix.columns and name in correlation_matrix.columns:
                    corr = abs(correlation_matrix.loc[selected_name, name])
                    if corr > self.selection_params['correlation_threshold']:
                        is_correlated = True
                        break
            
            if not is_correlated:
                filtered_indicators[name] = self.all_indicators[name]
        
        return filtered_indicators
    
    def _select_top_indicators(self, indicators: Dict[str, IndicatorSource], 
                             max_indicators: int) -> List[str]:
        """Обрати топ-N показників"""
        # Пріоритет за надійністю та типом джерела
        prioritized_indicators = []
        
        # Пріоритет: economic > market > technical > sentiment > temporal
        priority_order = ['economic', 'market', 'technical', 'sentiment', 'temporal']
        
        for source_type in priority_order:
            type_indicators = [name for name, indicator in indicators.items() 
                             if indicator.source_type == source_type]
            prioritized_indicators.extend(type_indicators)
        
        # Обираємо топ-N
        return prioritized_indicators[:max_indicators]
    
    def _create_context_dataset(self, data: pd.DataFrame, 
                               selected_indicators: List[str]) -> pd.DataFrame:
        """Створити контекстний датасет"""
        context_data = data[selected_indicators].copy()
        
        # Додаємо лагові значення для порівняння
        for name in selected_indicators:
            if name in context_data.columns:
                context_data[f'{name}_lag1'] = context_data[name].shift(1)
                context_data[f'{name}_change'] = context_data[name] - context_data[f'{name}_lag1']
                context_data[f'{name}_pct_change'] = context_data[name].pct_change()
        
        # Створюємо загальний контекстний скор
        numeric_columns = context_data.select_dtypes(include=[np.number]).columns
        context_data['context_score'] = context_data[numeric_columns].mean(axis=1)
        
        return context_data
    
    def _calculate_context_scores(self, context_data: pd.DataFrame, 
                               selected_indicators: List[str]) -> pd.Series:
        """Розрахувати контекстні скори"""
        scores = pd.Series(index=context_data.index, dtype=float)
        
        for i, row in context_data.iterrows():
            score = 0
            valid_indicators = 0
            
            for name in selected_indicators:
                if name in context_data.columns and not pd.isna(row[name]):
                    # Вага залежно від типу джерела
                    indicator = self.all_indicators[name]
                    weight = indicator.reliability
                    
                    # Нормалізуємо значення
                    if indicator.source_type in ['temporal']:
                        # Часові показники вже нормалізовані
                        normalized_value = row[name]
                    else:
                        # Інші показники - z-score нормалізація
                        mean_val = context_data[name].mean()
                        std_val = context_data[name].std()
                        if std_val > 0:
                            normalized_value = (row[name] - mean_val) / std_val
                        else:
                            normalized_value = 0
                    
                    score += normalized_value * weight
                    valid_indicators += 1
            
            # Середній скор
            scores.iloc[i] = score / valid_indicators if valid_indicators > 0 else 0
        
        return scores
    
    def _get_source_distribution(self, selected_indicators: List[str]) -> Dict[str, int]:
        """Отримати розподіл за типами джерел"""
        distribution = {}
        
        for name in selected_indicators:
            if name in self.all_indicators:
                source_type = self.all_indicators[name].source_type
                distribution[source_type] = distribution.get(source_type, 0) + 1
        
        return distribution
    
    def get_market_regime_signal(self, context_scores: pd.Series) -> pd.Series:
        """Отримати сигнал режиму ринку"""
        # Нормалізуємо скори
        normalized_scores = (context_scores - context_scores.mean()) / context_scores.std()
        
        # Визначаємо режими
        regimes = pd.Series(index=context_scores.index, dtype=str)
        regimes[normalized_scores > 1.0] = 'bullish'
        regimes[normalized_scores < -1.0] = 'bearish'
        regimes[(normalized_scores >= -1.0) & (normalized_scores <= 1.0)] = 'neutral'
        
        return regimes
    
    def generate_context_report(self, selection_result: Dict[str, any]) -> str:
        """Згенерувати звіт по контексту"""
        if 'error' in selection_result:
            return f"Error: {selection_result['error']}"
        
        report = []
        report.append("=" * 60)
        report.append("UNIFIED CONTEXT SELECTION REPORT")
        report.append("=" * 60)
        
        stats = selection_result['selection_stats']
        report.append(f"\n[DATA] Selection Statistics:")
        report.append(f"   Total Available: {stats['total_available']}")
        report.append(f"   After Correlation Filter: {stats['after_correlation_filter']}")
        report.append(f"   Final Selected: {stats['final_selected']}")
        
        report.append(f"\n[UP] Source Distribution:")
        for source_type, count in stats['source_distribution'].items():
            report.append(f"   {source_type}: {count}")
        
        report.append(f"\n[TARGET] Top 10 Selected Indicators:")
        selected = selection_result['selected_indicators'][:10]
        for i, name in enumerate(selected, 1):
            indicator = self.all_indicators[name]
            importance = selection_result['importance_scores'].get(name, 0)
            report.append(f"   {i}. {name} ({indicator.source_type}) - Importance: {importance:.3f}")
        
        if 'context_scores' in selection_result:
            scores = selection_result['context_scores']
            latest_score = scores.iloc[-1] if len(scores) > 0 else 0
            report.append(f"\n[DATA] Latest Context Score: {latest_score:.3f}")
            
            # Режим ринку
            regime = self.get_market_regime_signal(scores)
            if len(regime) > 0:
                latest_regime = regime.iloc[-1]
                report.append(f"[TARGET] Market Regime: {latest_regime}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# Глобальний екземпляр
unified_context_selector = UnifiedContextSelector()


def select_context_indicators(available_data: pd.DataFrame, **kwargs) -> Dict[str, any]:
    """Обрати контекстні показники"""
    return unified_context_selector.select_context_indicators(available_data, **kwargs)


def get_available_indicators(source_types: List[str] = None) -> Dict[str, IndicatorSource]:
    """Отримати доступні показники"""
    return unified_context_selector.get_available_indicators(source_types)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("[TARGET] Unified Context Selector Test")
    print("="*50)
    
    # Симуляція data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', '2024-01-31', freq='D')
    
    # Створюємо датасет з різними показниками
    data = pd.DataFrame(index=dates)
    
    # Технічні індикатори
    data['rsi_14'] = np.random.uniform(20, 80, len(dates))
    data['macd_signal'] = np.random.normal(0, 1, len(dates))
    data['sma_20'] = np.random.normal(100, 10, len(dates))
    data['atr'] = np.random.uniform(1, 5, len(dates))
    
    # Економічні показники
    data['fedfunds'] = np.random.uniform(4.5, 5.5, len(dates))
    data['vix'] = np.random.uniform(15, 35, len(dates))
    data['unrate'] = np.random.uniform(3.5, 4.5, len(dates))
    
    # Sentiment показники
    data['sentiment_score'] = np.random.uniform(-1, 1, len(dates))
    data['news_count'] = np.random.poisson(50, len(dates))
    
    # Часові показники
    data['weekday'] = data.index.weekday
    data['month'] = data.index.month
    
    # Цільова змінна
    data['target_return'] = np.random.normal(0.001, 0.02, len(dates))
    
    print(f"[DATA] Generated data: {len(data)} rows, {len(data.columns)} columns")
    
    # Обираємо контекстні показники
    result = select_context_indicators(
        data,
        target_variable='target_return',
        max_indicators=15,
        source_types=['economic', 'technical', 'sentiment'],
        importance_method='correlation'
    )
    
    if 'selected_indicators' in result:
        print(f"\n[OK] Context selection completed!")
        print(f"[DATA] Selected {len(result['selected_indicators'])} indicators")
        
        # Звіт
        report = unified_context_selector.generate_context_report(result)
        print(f"\n{report}")
        
        print(f"\n[TARGET] Unified Context Selector working correctly!")
    else:
        print(f"[ERROR] Context selection failed")
