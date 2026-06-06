"""
Вибір ознак для моделей
"""
from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('FeatureSelector')


class FeatureSelectorConfig:
    """Конфігурація для вибору ознак"""

    def __init__(self, project_path: str):
        self.project_path = project_path
        self.selector = None


class SimpleFeatureSelector:
    """Простий вибір ознак"""

    def select(self, X: np.ndarray, max_features: int | None=None
        ) ->np.ndarray:
        """Вибрати ознаки"""
        if max_features is None or max_features >= X.shape[1]:
            return X
        return X[:, :max_features]


class ColabFeatureSelector:
    """Вибір ознак для Colab"""

    def __init__(self, config: FeatureSelectorConfig):
        self.config = config
        self.selector = None
        self._init_selector()

    def _init_selector(self) ->None:
        """Ініціалізувати вибір ознак"""
        try:
            from sklearn.feature_selection import SelectKBest, f_regression
            self.selector = SelectKBest(f_regression)
        except ImportError:
            self.selector = SimpleFeatureSelector()

    def select_features(self, features_df: object, targets_df: object,
        ticker: str, model_type: str) ->tuple[np.ndarray, list[str]]:
        """Вибрати ознаки для тікера"""
        ticker_features = self._filter_features_for_ticker(features_df, ticker)
        ticker_targets = self._filter_targets_for_ticker(targets_df, ticker)
        self._validate_data_sufficiency(ticker_features, ticker)
        selected_names = self._perform_feature_selection(ticker_features,
            ticker_targets, ticker, model_type)
        feature_indices = self._convert_feature_names_to_indices(selected_names
            , ticker_features)
        filtered_features = self._create_filtered_features(ticker_features,
            feature_indices)
        return filtered_features, selected_names

    def _filter_features_for_ticker(self, features_df: object, ticker: str
        ) ->object:
        """Фільтрувати ознаки для тікера"""
        if hasattr(features_df, 'xs'):
            try:
                return features_df.xs(ticker, level='ticker')
            except KeyError as e:
                logger.debug(f"Ticker {ticker} not found in features index: {e}")
        return features_df

    def _filter_targets_for_ticker(self, targets_df: object, ticker: str
        ) ->object:
        """Фільтрувати цілі для тікера"""
        if hasattr(targets_df, 'xs'):
            try:
                return targets_df.xs(ticker, level='ticker')
            except KeyError as e:
                logger.debug(f"Ticker {ticker} not found in features index: {e}")
        return targets_df

    def _validate_data_sufficiency(self, ticker_features: object, ticker: str
        ) ->None:
        """Перевірити достатність даних"""
        if len(ticker_features) < 10:
            raise ValueError(
                f'Недостатньо даних для {ticker}: {len(ticker_features)} рядків'
                )

    def _perform_feature_selection(self, ticker_features: object,
        ticker_targets: object, ticker: str, model_type: str) ->list[str]:
        """Виконати вибір ознак"""
        if self.selector is None:
            return list(ticker_features.columns)
        try:
            max_features = self._get_model_max_features(model_type)
            if hasattr(self.selector, 'fit'):
                self.selector.fit(ticker_features, ticker_targets)
                selected_indices = self.selector.get_support(indices=True)
                return [ticker_features.columns[i] for i in
                    selected_indices[:max_features]]
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка: {e}', exc_info=True)
            print(f'Помилка при виборі ознак для {ticker}: {e}')
            raise
        return list(ticker_features.columns[:max_features])

    def _convert_feature_names_to_indices(self, selected_names: list[str],
        ticker_features: object) ->list[int]:
        """Конвертувати імена ознак в індекси"""
        return [list(ticker_features.columns).index(name) for name in
            selected_names]

    def _create_filtered_features(self, ticker_features: object,
        feature_indices: list[int]) ->np.ndarray:
        """Створити відфільтровані ознаки"""
        import numpy as np
        return np.array(ticker_features.iloc[:, feature_indices])

    def _get_model_max_features(self, model_type: str) ->int:
        """Отримати максимальну кількість ознак для моделі"""
        max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
            'transformer': 128, 'tabnet': 256, 'autoencoder': 128,
            'random_forest': 256}
        return max_features_map.get(model_type.lower(), 128)


class FeatureSelector:
    """
    Unified entry point for feature selection.
    Delegates implementation to EnhancedSmartFeatureSelector.
    """

    def __init__(self, config: Any=None):
        """
        Initialize the unified feature selector.
        """
        from src.core.logging.logger import ProjectLogger
        from src.features.selection.enhanced_smart_selector import EnhancedSmartFeatureSelector
        self.selector = EnhancedSmartFeatureSelector()
        self.logger = ProjectLogger.get_logger('FeatureSelector')
        self.logger.info(
            '✅ Unified FeatureSelector initialized with EnhancedSmartFeatureSelector'
            )

    def get_model_max_features(self, model_type: str) ->int:
        max_features_map = {'mlp': 256, 'lstm': 128, 'gru': 128, 'cnn': 64,
            'transformer': 128, 'tabnet': 256, 'autoencoder': 128,
            'random_forest': 256}
        return max_features_map.get(model_type.lower(), 128)

    def select(self, features_df: Any, targets_df: Any, ticker: str,
        target_col: str, model_type: str='mlp', market_regime: str='normal'
        ) ->tuple[np.ndarray, list[str]]:
        """
        Unifies feature selection across assets and models.
        """
        self.logger.info(
            f'🔍 Selecting features for {ticker} (Target: {target_col}, Model: {model_type})'
            )
        ticker_features = self._filter_for_ticker(features_df, ticker)
        ticker_targets = self._filter_for_ticker(targets_df, ticker)
        if ticker_features.empty or ticker_targets.empty:
            self.logger.warning(
                f'⚠️ No data for {ticker}. Returning empty selection.')
            return np.array([]), []
        if target_col not in ticker_targets.columns:
            self.logger.error(f"❌ Target '{target_col}' not found for {ticker}"
                )
            return np.array([]), []
        target_series = ticker_targets[target_col]
        max_features = self.get_model_max_features(model_type)
        context_id = f'{ticker}_{target_col}_{model_type}'
        selected_names = self.selector.select(features_df=ticker_features,
            target_series=target_series, context_id=context_id,
            market_regime=market_regime, max_features=max_features)
        if not selected_names:
            self.logger.warning(
                f'⚠️ Selection returned no features for {ticker}. Using fallback.'
                )
            numeric_cols = ticker_features.select_dtypes(include=[np.number]
                ).columns.tolist()
            selected_names = numeric_cols[:min(len(numeric_cols), 50)]
        selected_array = np.array(ticker_features[selected_names])
        self.logger.info(
            f'✅ Selected {len(selected_names)} features for {ticker}')
        return selected_array, selected_names

    def _filter_for_ticker(self, df: Any, ticker: str) ->Any:
        """Helper to filter DataFrame by ticker."""
        if 'ticker' in df.columns:
            return df[df['ticker'] == ticker].copy()
        elif hasattr(df.index, 'levels') and 'ticker' in df.index.names:
            try:
                return df.xs(ticker, level='ticker')
            except KeyError as e:
                logger.debug(f"Ticker {ticker} not found in features index: {e}")
        return df
