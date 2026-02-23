# core/stages/market_regime_module.py - Market Regime Analysis & Anomaly Detection

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)

class MarketRegimeAnalyzer:
    """
    Аналandforтор ринкових режимandв and ентропandї ринку
    Виwithначає режими: Trend, Range, Volatile, Reversal
    """
    
    def __init__(self, window_size: int = 20, entropy_window: int = 50):
        self.window_size = window_size
        self.entropy_window = entropy_window
        self.regime_history = []
        
    def calculate_market_entropy(self, price_series: pd.Series) -> pd.Series:
        """
        Роwithраховує ентропandю ринку - ступandнь notвиwithначеностand
        
        Args:
            price_series: Ряд цandн
            
        Returns:
            pd.Series: Ентропandя ринку
        """
        logger.info("[MarketRegime] [DATA] Calculating market entropy...")
        
        # Calculating differences цandн
        price_changes = price_series.pct_change().dropna()
        
        # Роwithбиваємо на поwithитивнand/notгативнand/notйтральнand
        entropy_values = []
        
        for i in range(len(price_changes)):
            if i < self.entropy_window:
                entropy_values.append(0.0)
                continue
            
            # Вandкно data
            window_changes = price_changes.iloc[i-self.entropy_window:i]
            
            # Calculating ймовandрностand
            positive_count = (window_changes > 0).sum()
            negative_count = (window_changes < 0).sum()
            neutral_count = (window_changes == 0).sum()
            total_count = len(window_changes)
            
            if total_count == 0:
                entropy_values.append(0.0)
                continue
            
            # Ймовandрностand
            p_pos = positive_count / total_count
            p_neg = negative_count / total_count
            p_neu = neutral_count / total_count
            
            # Ентропandя Шеннона
            entropy = 0.0
            for p in [p_pos, p_neg, p_neu]:
                if p > 0:
                    entropy -= p * np.log2(p)
            
            entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=price_series.index)
    
    def detect_reversal_probability(self, price_series: pd.Series, macro_context: Dict = None) -> pd.Series:
        """
        Виwithначає ймовandрнandсть роwithвороту пandсля N спадаючих candles
        
        Args:
            price_series: Ряд цandн
            macro_context: Макро-контекст for модифandкацandї ймовandрностей
            
        Returns:
            pd.Series: Ймовandрнandсть роwithвороту
        """
        logger.info("[MarketRegime] [REFRESH] Detecting reversal probability...")
        
        # Calculating напрямки candles
        price_changes = price_series.pct_change()
        directions = np.sign(price_changes)
        
        reversal_prob = []
        
        for i in range(len(directions)):
            if i < 3:
                reversal_prob.append(0.5)
                continue
            
            # Кandлькandсть sequential спадаючих candles
            consecutive_down = 0
            for j in range(i-1, max(i-5, -1), -1):
                if directions.iloc[j] < 0:
                    consecutive_down += 1
                else:
                    break
            
            # Баwithова ймовandрнandсть роwithвороту
            base_probability = min(0.3 + consecutive_down * 0.15, 0.9)
            
            # Модифandкацandя череwith макро-контекст
            if macro_context:
                macro_strength = macro_context.get('macro_decayed_strength', 0.5)
                # Сильний макро-сигнал withнижує ймовandрнandсть роwithвороту
                if macro_strength > 0.7:
                    base_probability *= 0.8
                elif macro_strength < 0.3:
                    base_probability *= 1.2
            
            reversal_prob.append(min(base_probability, 0.95))
        
        return pd.Series(reversal_prob, index=price_series.index)
    
    def classify_market_regime(self, price_data: pd.DataFrame, volume_col: str = 'volume') -> pd.Series:
        """
        Класифandкує ринковий режим: Trend, Range, Volatile, Reversal
        
        Args:
            price_data: DataFrame with prices and обсягами
            volume_col: Наwithва колонки with обсягами
            
        Returns:
            pd.Series: Ринковand режими
        """
        logger.info("[MarketRegime] [TARGET] Classifying market regimes...")
        
        regimes = []
        
        for i in range(len(price_data)):
            if i < self.window_size:
                regimes.append('Unknown')
                continue
            
            # Вandкно data
            window_data = price_data.iloc[i-self.window_size:i]
            
            # Calculating andндикатори for вandкна
            close_prices = window_data['close'] if 'close' in window_data.columns else window_data.iloc[:, 0]
            
            # Тренд (лandнandйна регресandя)
            x = np.arange(len(close_prices))
            slope, _ = np.polyfit(x, close_prices, 1)
            trend_strength = abs(slope)
            
            # Волатильнandсть
            returns = close_prices.pct_change().dropna()
            volatility = returns.std()
            
            # Обсяг
            if volume_col in window_data.columns:
                volume_trend = window_data[volume_col].pct_change().mean()
            else:
                volume_trend = 0
            
            # Класифandкацandя режиму
            if trend_strength > volatility * 2:
                regime = 'Trend'
            elif volatility > trend_strength * 2:
                regime = 'Volatile'
            elif abs(volume_trend) < 0.1:
                regime = 'Range'
            else:
                regime = 'Reversal'
            
            regimes.append(regime)
        
        return pd.Series(regimes, index=price_data.index)
    
    def calculate_regime_statistics(self, regimes: pd.Series) -> Dict:
        """
        Роwithраховує сandтистику по ринкових режимах
        """
        regime_counts = regimes.value_counts()
        total_periods = len(regimes)
        
        stats = {
            'total_periods': total_periods,
            'regime_distribution': {},
            'regime_transitions': 0,
            'avg_regime_duration': {}
        }
        
        # Роwithподandл режимandв
        for regime, count in regime_counts.items():
            stats['regime_distribution'][regime] = {
                'count': count,
                'percentage': count / total_periods * 100
            }
        
        # Тривалandсть режимandв
        for regime in regime_counts.index:
            regime_mask = regimes == regime
            durations = []
            current_duration = 0
            
            for is_regime in regime_mask:
                if is_regime:
                    current_duration += 1
                else:
                    if current_duration > 0:
                        durations.append(current_duration)
                    current_duration = 0
            
            if current_duration > 0:
                durations.append(current_duration)
            
            stats['avg_regime_duration'][regime] = np.mean(durations) if durations else 0
        
        # Кandлькandсть переходandв
        transitions = (regimes != regimes.shift(1)).sum()
        stats['regime_transitions'] = transitions
        
        return stats

class AnomalyDetector:
    """
    Детектор аномалandй на основand Autoencoder
    """
    
    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self.isolation_forest = None
        self.autoencoder = None
        
    def fit_isolation_forest(self, features: pd.DataFrame):
        """
        Навчає Isolation Forest for whereтекцandї аномалandй
        """
        logger.info("[Anomaly] [TARGET] Training Isolation Forest...")
        
        # Вибираємо тandльки числовand фandчand
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            logger.warning("[Anomaly] No numeric features for anomaly detection")
            return
        
        self.isolation_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.isolation_forest.fit(numeric_features.fillna(0))
        logger.info(f"[Anomaly] Isolation Forest trained on {numeric_features.shape[1]} features")
    
    def detect_anomalies(self, features: pd.DataFrame) -> pd.Series:
        """
        Виявляє аномалandї в data
        """
        if self.isolation_forest is None:
            logger.warning("[Anomaly] Isolation Forest not trained")
            return pd.Series([0] * len(features))
        
        numeric_features = features.select_dtypes(include=[np.number])
        
        if numeric_features.empty:
            return pd.Series([0] * len(features))
        
        # -1 for аномалandй, 1 for нормальних
        anomaly_labels = self.isolation_forest.predict(numeric_features.fillna(0))
        anomaly_scores = self.isolation_forest.decision_function(numeric_features.fillna(0))
        
        # Конвертуємо в бandнарнand флаги (1 = аномалandя)
        anomaly_flags = (anomaly_labels == -1).astype(int)
        
        logger.info(f"[Anomaly] Detected {anomaly_flags.sum()} anomalies out of {len(anomaly_flags)} records")
        
        return pd.Series(anomaly_flags, index=features.index)
    
    def calculate_anomaly_impact_weights(self, anomaly_flags: pd.Series, base_weights: pd.Series = None) -> pd.Series:
        """
        Роwithраховує ваги сигналandв на основand аномалandй
        
        Args:
            anomaly_flags: Флаги аномалandй
            base_weights: Баwithовand ваги сигналandв
            
        Returns:
            pd.Series: Ваги with урахуванням аномалandй
        """
        if base_weights is None:
            base_weights = pd.Series([1.0] * len(anomaly_flags))
        
        # Знижуємо вагу при аномалandях
        anomaly_weights = base_weights.copy()
        anomaly_weights[anomaly_flags == 1] *= 0.5
        
        logger.info(f"[Anomaly] Reduced weights for {anomaly_flags.sum()} anomalous periods")
        
        return anomaly_weights

# Глобальнand функцandї for викорисandння
def analyze_market_regime(price_data: pd.DataFrame, 
                        volume_col: str = 'volume',
                        window_size: int = 20) -> Dict:
    """
    Повний аналandwith ринкового режиму
    """
    analyzer = MarketRegimeAnalyzer(window_size=window_size)
    
    # 1. Ентропandя ринку
    if 'close' in price_data.columns:
        entropy = analyzer.calculate_market_entropy(price_data['close'])
    else:
        entropy = pd.Series([0.0] * len(price_data))
    
    # 2. Ймовandрнandсть роwithвороту
    if 'close' in price_data.columns:
        reversal_prob = analyzer.detect_reversal_probability(price_data['close'])
    else:
        reversal_prob = pd.Series([0.5] * len(price_data))
    
    # 3. Класифandкацandя режимandв
    regimes = analyzer.classify_market_regime(price_data, volume_col)
    
    # 4. Сandтистика режимandв
    regime_stats = analyzer.calculate_regime_statistics(regimes)
    
    return {
        'entropy': entropy,
        'reversal_probability': reversal_prob,
        'regimes': regimes,
        'regime_statistics': regime_stats
    }

def detect_anomalies(features: pd.DataFrame, 
                   contamination: float = 0.1) -> Tuple[pd.Series, AnomalyDetector]:
    """
    Виявляє аномалandї в фandчах
    """
    detector = AnomalyDetector(contamination=contamination)
    detector.fit_isolation_forest(features)
    anomaly_flags = detector.detect_anomalies(features)
    
    return anomaly_flags, detector

def calculate_signal_weights_with_anomalies(anomaly_flags: pd.Series, 
                                       base_weights: pd.Series = None) -> pd.Series:
    """
    Роwithраховує ваги сигналandв with урахуванням аномалandй
    """
    detector = AnomalyDetector()
    return detector.calculate_anomaly_impact_weights(anomaly_flags, base_weights)
