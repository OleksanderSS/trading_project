#!/usr/bin/env python3
"""
Smart Feature Selector - Гнучка система вибору фіч
Інтелектуальний вибір фіч з урахуванням патернів, таргетів та контексту
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SmartFeatureSelector:
    """
    [START] Інтелектуальна система вибору фіч
    Адаптивний вибір під кожен тікер/таймфрейм/таргет
    """
    
    def __init__(self, pattern_metadata: Dict = None):
        self.pattern_metadata = pattern_metadata or {}
        self.pattern_weights = self._get_pattern_weights()
        self.feature_cache = {}
        
    def _get_pattern_weights(self) -> Dict[str, float]:
        """Ваги патернів для вибору фіч"""
        return {
            'anomaly': 0.9,      # Високий пріоритет
            'gap': 0.8,         # Високий пріоритет
            'regime': 0.85,       # Високий пріоритет
            'quality': 0.7,       # Середній пріоритет
            'volatility': 0.75,   # Середній пріоритет
            'trend': 0.8,         # Високий пріоритет
            'momentum': 0.85      # Високий пріоритет
        }
    
    def select_features_for_target(self, 
                                 df: pd.DataFrame, 
                                 ticker: str, 
                                 timeframe: str, 
                                 target: str,
                                 max_features: int = 50) -> List[str]:
        """
        [START] Вибір фіч для конкретного тікера/таймфрейму/таргета
        
        Args:
            df: DataFrame з даними
            ticker: Тікер
            timeframe: Таймфрейм
            target: Цільова змінна
            max_features: Максимальна кількість фіч
            
        Returns:
            List[str]: Вибрані фічі
        """
        logger.info(f"[TARGET] Selecting features for {ticker}_{timeframe} -> {target}")
        
        # [TARGET] Отримуємо фічі для цього тікера/таймфрейму
        ticker_features = self._get_ticker_timeframe_features(df, ticker, timeframe)
        
        if not ticker_features:
            logger.warning(f"No features found for {ticker}_{timeframe}")
            return []
        
        # [TARGET] Отримуємо цільову змінну
        target_col = f"{ticker}_{timeframe}_{target}"
        if target_col not in df.columns:
            logger.warning(f"Target {target_col} not found")
            return []
        
        y = df[target_col].dropna()
        X = df.loc[y.index, ticker_features].fillna(0)
        
        if len(X) < 100:
            logger.warning(f"Insufficient data for {ticker}_{timeframe}")
            return ticker_features[:max_features]
        
        # [TARGET] Pattern-aware ваги
        pattern_weights = self._get_pattern_weights_for_features(ticker_features)
        
        # [TARGET] Комбінований вибір фіч
        selected_features = self._combined_feature_selection(
            X, y, ticker_features, pattern_weights, max_features
        )
        
        logger.info(f"[OK] Selected {len(selected_features)} features for {ticker}_{timeframe}")
        return selected_features
    
    def _get_ticker_timeframe_features(self, df: pd.DataFrame, ticker: str, timeframe: str) -> List[str]:
        """Отримати фічі для конкретного тікера/таймфрейму"""
        prefix = f"{ticker}_{timeframe}_"
        features = [col for col in df.columns if col.startswith(prefix)]
        
        # Видаляємо базові OHLCV
        basic_cols = [f"{prefix}open", f"{prefix}high", f"{prefix}low", f"{prefix}close", f"{prefix}volume"]
        features = [f for f in features if f not in basic_cols]
        
        return features
    
    def _get_pattern_weights_for_features(self, features: List[str]) -> Dict[str, float]:
        """Отримати ваги патернів для фіч"""
        weights = {}
        
        for feature in features:
            feature_weight = 0.0
            
            # [TARGET] Аналіз назви фічі на предмет патернів
            for pattern, weight in self.pattern_weights.items():
                if pattern in feature.lower():
                    feature_weight = max(feature_weight, weight)
            
            # [TARGET] Додаткова вага для технічних індикаторів
            if any(indicator in feature.lower() for indicator in ['rsi', 'macd', 'bollinger', 'stochastic']):
                feature_weight = max(feature_weight, 0.8)
            
            # [TARGET] Вага для макро фіч
            if feature.startswith('macro_'):
                feature_weight = max(feature_weight, 0.7)
            
            # [TARGET] Вага для новинних фіч
            if feature.startswith('news_'):
                feature_weight = max(feature_weight, 0.75)
            
            weights[feature] = feature_weight
        
        return weights
    
    def _combined_feature_selection(self, 
                                   X: pd.DataFrame, 
                                   y: pd.Series, 
                                   features: List[str], 
                                   pattern_weights: Dict[str, float],
                                   max_features: int) -> List[str]:
        """
        [START] Комбінований вибір фіч - поєднує кілька методів
        """
        logger.info("[SEARCH] Using combined feature selection...")
        
        # [TARGET] Метод 1: Кореляція
        correlation_scores = self._calculate_correlation_scores(X, y)
        
        # [TARGET] Метод 2: Mutual Information
        mi_scores = self._calculate_mutual_info_scores(X, y)
        
        # [TARGET] Метод 3: Random Forest importance
        rf_scores = self._calculate_random_forest_scores(X, y)
        
        # [TARGET] Метод 4: Pattern-aware ваги
        pattern_scores = pattern_weights
        
        # [TARGET] Комбінування оцінок
        combined_scores = self._combine_scores(
            correlation_scores, mi_scores, rf_scores, pattern_scores
        )
        
        # [TARGET] Вибір топ фіч
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:max_features]]
        
        return selected_features
    
    def _calculate_correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Розрахунок кореляційних оцінок"""
        scores = {}
        for col in X.columns:
            correlation = abs(X[col].corr(y))
            scores[col] = correlation if not np.isnan(correlation) else 0.0
        return scores
    
    def _calculate_mutual_info_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Розрахунок Mutual Information оцінок"""
        try:
            mi = mutual_info_regression(X, y, random_state=42)
            return dict(zip(X.columns, mi))
        except:
            return {col: 0.0 for col in X.columns}
    
    def _calculate_random_forest_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Розрахунок Random Forest importance"""
        try:
            rf = RandomForestRegressor(n_estimators=50, random_state=42)
            rf.fit(X, y)
            return dict(zip(X.columns, rf.feature_importances_))
        except:
            return {col: 0.0 for col in X.columns}
    
    def _combine_scores(self, 
                       correlation_scores: Dict[str, float],
                       mi_scores: Dict[str, float],
                       rf_scores: Dict[str, float],
                       pattern_scores: Dict[str, float]) -> Dict[str, float]:
        """
        [START] Комбінування оцінок з різних методів
        """
        combined_scores = {}
        
        for feature in correlation_scores.keys():
            # [TARGET] Ваги для кожного методу
            correlation_weight = 0.3
            mi_weight = 0.3
            rf_weight = 0.2
            pattern_weight = 0.2
            
            # [TARGET] Комбінована оцінка
            combined_score = (
                correlation_scores.get(feature, 0.0) * correlation_weight +
                mi_scores.get(feature, 0.0) * mi_weight +
                rf_scores.get(feature, 0.0) * rf_weight +
                pattern_scores.get(feature, 0.0) * pattern_weight
            )
            
            combined_scores[feature] = combined_score
        
        return combined_scores
    
    def select_features_global(self, 
                             df: pd.DataFrame, 
                             tickers: List[str], 
                             timeframes: List[str],
                             max_features_per_target: int = 30) -> Dict[str, List[str]]:
        """
        [START] Глобальний вибір фіч для всіх тікерів/таймфреймів/таргетів
        
        Returns:
            Dict[str, List[str]]: {target: [features]}
        """
        logger.info(f"[TARGET] Global feature selection for {len(tickers)} tickers, {len(timeframes)} timeframes")
        
        # [TARGET] Стандартні таргети
        targets = ['close', 'high', 'low', 'volume', 'returns']
        
        results = {}
        
        for target in targets:
            target_features = []
            
            for ticker in tickers:
                for timeframe in timeframes:
                    features = self.select_features_for_target(
                        df, ticker, timeframe, target, max_features_per_target
                    )
                    
                    # [TARGET] Додаємо префікс для унікальності
                    prefixed_features = [f"{ticker}_{timeframe}_{f.split('_', 1)[-1]}" for f in features]
                    target_features.extend(prefixed_features)
            
            # [TARGET] Видаляємо дублікати
            target_features = list(set(target_features))
            
            # [TARGET] Обмежуємо кількість
            if len(target_features) > max_features_per_target * 2:
                # [TARGET] Вибираємо найкращі фічі
                target_features = self._select_best_global_features(df, target_features, target, max_features_per_target * 2)
            
            results[target] = target_features
            logger.info(f"[OK] Selected {len(target_features)} features for target {target}")
        
        return results
    
    def _select_best_global_features(self, 
                                   df: pd.DataFrame, 
                                   features: List[str], 
                                   target: str, 
                                   max_features: int) -> List[str]:
        """Вибір найкращих глобальних фіч"""
        # [TARGET] Знаходимо всі колонки для цього таргета
        target_cols = [col for col in df.columns if col.endswith(f"_{target}")]
        
        if not target_cols:
            return features[:max_features]
        
        # [TARGET] Використовуємо перший available таргет
        target_col = target_cols[0]
        y = df[target_col].dropna()
        
        # [TARGET] Фільтруємо фічі
        available_features = [f for f in features if f in df.columns]
        X = df.loc[y.index, available_features].fillna(0)
        
        if len(X) < 100:
            return features[:max_features]
        
        # [TARGET] Комбінований вибір
        correlation_scores = self._calculate_correlation_scores(X, y)
        mi_scores = self._calculate_mutual_info_scores(X, y)
        
        # [TARGET] Комбінуємо оцінки
        combined_scores = {}
        for feature in available_features:
            combined_scores[feature] = (
                correlation_scores.get(feature, 0.0) * 0.5 +
                mi_scores.get(feature, 0.0) * 0.5
            )
        
        # [TARGET] Вибір топ фіч
        sorted_features = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [feat for feat, score in sorted_features[:max_features]]
    
    def get_feature_importance_report(self, 
                                    df: pd.DataFrame, 
                                    ticker: str, 
                                    timeframe: str, 
                                    target: str) -> Dict[str, Any]:
        """
        [START] Детальний звіт про важливість фіч
        """
        features = self._get_ticker_timeframe_features(df, ticker, timeframe)
        target_col = f"{ticker}_{timeframe}_{target}"
        
        if target_col not in df.columns or not features:
            return {}
        
        y = df[target_col].dropna()
        X = df.loc[y.index, features].fillna(0)
        
        if len(X) < 100:
            return {}
        
        # [TARGET] Розраховуємо всі оцінки
        correlation_scores = self._calculate_correlation_scores(X, y)
        mi_scores = self._calculate_mutual_info_scores(X, y)
        rf_scores = self._calculate_random_forest_scores(X, y)
        pattern_weights = self._get_pattern_weights_for_features(features)
        
        # [TARGET] Комбіновані оцінки
        combined_scores = self._combine_scores(
            correlation_scores, mi_scores, rf_scores, pattern_weights
        )
        
        # [TARGET] Створюємо звіт
        report = {
            'ticker': ticker,
            'timeframe': timeframe,
            'target': target,
            'total_features': len(features),
            'top_features': dict(sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:10]),
            'method_scores': {
                'correlation': correlation_scores,
                'mutual_info': mi_scores,
                'random_forest': rf_scores,
                'pattern_weights': pattern_weights
            },
            'feature_categories': self._categorize_features(features)
        }
        
        return report
    
    def _categorize_features(self, features: List[str]) -> Dict[str, List[str]]:
        """Категоризація фіч"""
        categories = {
            'technical': [],
            'macro': [],
            'news': [],
            'linguistic': [],
            'other': []
        }
        
        for feature in features:
            if any(indicator in feature.lower() for indicator in ['rsi', 'macd', 'bollinger', 'stochastic', 'momentum', 'sma', 'ema']):
                categories['technical'].append(feature)
            elif feature.startswith('macro_'):
                categories['macro'].append(feature)
            elif feature.startswith('news_'):
                categories['news'].append(feature)
            elif any(ling in feature.lower() for ling in ['historical', 'commitment', 'semantic', 'dist']):
                categories['linguistic'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories


# [TARGET] ГОЛОВНА ФУНКЦІЯ ДЛЯ ІНТЕГРАЦІЇ
def smart_feature_selection(df: pd.DataFrame, 
                           tickers: List[str], 
                           timeframes: List[str],
                           pattern_metadata: Dict = None,
                           max_features_per_target: int = 30) -> Dict[str, List[str]]:
    """
    [START] Основна функція для інтелектуального вибору фіч
    
    Args:
        df: DataFrame з даними
        tickers: Список тікерів
        timeframes: Список таймфреймів
        pattern_metadata: Метадані патернів
        max_features_per_target: Максимальна кількість фіч на таргет
    
    Returns:
        Dict[str, List[str]]: {target: [selected_features]}
    """
    selector = SmartFeatureSelector(pattern_metadata)
    return selector.select_features_global(df, tickers, timeframes, max_features_per_target)


if __name__ == "__main__":
    print("Smart Feature Selector - готовий до використання")
    print("[START] Інтелектуальний вибір фіч з урахуванням патернів")
    print("[DATA] Pattern-aware, target-specific, adaptive selection!")
