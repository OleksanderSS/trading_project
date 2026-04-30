"""
Вибір ознак для моделей
"""
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
import numpy as np


class FeatureSelectorConfig:
    """Конфігурація для вибору ознак"""
    def __init__(self, project_path: str):
        self.project_path = project_path
        self.selector = None


class SimpleFeatureSelector:
    """Простий вибір ознак"""

    def select(self, X: np.ndarray, max_features: Optional[int] = None) -> np.ndarray:
        """Вибрати ознаки"""
        if max_features is None or max_features >= X.shape[1]:
            return X
        
        # Вибрати перші max_features ознак
        return X[:, :max_features]


class ColabFeatureSelector:
    """Вибір ознак для Colab"""

    def __init__(self, config: FeatureSelectorConfig):
        self.config = config
        self.selector = None
        self._init_selector()

    def _init_selector(self) -> None:
        """Ініціалізувати вибір ознак"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            self.selector = SelectKBest(f_regression)
        except ImportError:
            self.selector = SimpleFeatureSelector()

    def select_features(self, features_df: object, targets_df: object, 
                       ticker: str, model_type: str) -> Tuple[np.ndarray, List[str]]:
        """Вибрати ознаки для тікера"""
        ticker_features = self._filter_features_for_ticker(features_df, ticker)
        ticker_targets = self._filter_targets_for_ticker(targets_df, ticker)
        
        self._validate_data_sufficiency(ticker_features, ticker)
        
        selected_names = self._perform_feature_selection(
            ticker_features, ticker_targets, ticker, model_type
        )
        
        feature_indices = self._convert_feature_names_to_indices(
            selected_names, ticker_features
        )
        
        filtered_features = self._create_filtered_features(
            ticker_features, feature_indices
        )
        
        return filtered_features, selected_names

    def _filter_features_for_ticker(self, features_df: object, ticker: str) -> object:
        """Фільтрувати ознаки для тікера"""
        if hasattr(features_df, 'xs'):
            try:
                return features_df.xs(ticker, level='ticker')
            except KeyError:
                pass
        
        return features_df

    def _filter_targets_for_ticker(self, targets_df: object, ticker: str) -> object:
        """Фільтрувати цілі для тікера"""
        if hasattr(targets_df, 'xs'):
            try:
                return targets_df.xs(ticker, level='ticker')
            except KeyError:
                pass
        
        return targets_df

    def _validate_data_sufficiency(self, ticker_features: object, ticker: str) -> None:
        """Перевірити достатність даних"""
        if len(ticker_features) < 10:
            raise ValueError(f"Недостатньо даних для {ticker}: {len(ticker_features)} рядків")

    def _perform_feature_selection(self, ticker_features: object, ticker_targets: object,
                                  ticker: str, model_type: str) -> List[str]:
        """Виконати вибір ознак"""
        if self.selector is None:
            return list(ticker_features.columns)
        
        try:
            max_features = self._get_model_max_features(model_type)
            
            if hasattr(self.selector, 'fit'):
                self.selector.fit(ticker_features, ticker_targets)
                selected_indices = self.selector.get_support(indices=True)
                return [ticker_features.columns[i] for i in selected_indices[:max_features]]
        except Exception as e:
            print(f"Помилка при виборі ознак для {ticker}: {e}")
        
        return list(ticker_features.columns[:max_features])

    def _convert_feature_names_to_indices(self, selected_names: List[str], 
                                         ticker_features: object) -> List[int]:
        """Конвертувати імена ознак в індекси"""
        return [list(ticker_features.columns).index(name) for name in selected_names]

    def _create_filtered_features(self, ticker_features: object, 
                                 feature_indices: List[int]) -> np.ndarray:
        """Створити відфільтровані ознаки"""
        import numpy as np
        return np.array(ticker_features.iloc[:, feature_indices])

    def _get_model_max_features(self, model_type: str) -> int:
        """Отримати максимальну кількість ознак для моделі"""
        max_features_map = {
            'mlp': 256,
            'lstm': 128,
            'gru': 128,
            'cnn': 64,
            'transformer': 128,
            'tabnet': 256,
            'autoencoder': 128,
            'random_forest': 256
        }
        return max_features_map.get(model_type.lower(), 128)
