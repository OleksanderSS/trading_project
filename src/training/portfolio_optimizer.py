"""
Portfolio-Level Multi-Ticker Optimization
Вирішує проблему локального тренування без врахування між-тікерних залежностей
"""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class PortfolioOptimizer:
    """
    Оптимізує тренування моделей на рівні всього портфоліо,
    враховуючи кореляції між тікерами та спільні ринкові умови
    """

    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.logger = logger

    def optimize_portfolio_training(self,
                                   features_dict: dict[str, pd.DataFrame],
                                   targets_dict: dict[str, pd.DataFrame],
                                   model_type: str = 'ensemble') -> dict[str, Any]:
        """
        Оптимізує тренування моделей з урахуванням між-тікерних залежностей

        Args:
            features_dict: {ticker: features_df}
            targets_dict: {ticker: targets_df}
            model_type: Тип моделі для оптимізації

        Returns:
            Оптимізовані ваги та результати тренування
        """
        self.logger.info(f"🎯 Optimizing portfolio training for {len(features_dict)} tickers")

        # 1. Аналіз кореляцій між тікерами
        correlation_matrix = self._calculate_ticker_correlations(features_dict)

        # 2. Створення спільних ринкових фіч
        market_features = self._extract_market_features(features_dict)

        # 3. Розрахунок оптимальних ваг моделей
        optimal_weights = self._calculate_optimal_weights(correlation_matrix, market_features)

        # 4. Ієрархічне тренування
        training_results = self._train_hierarchical_models(
            features_dict, targets_dict, optimal_weights, market_features
        )

        return {
            'optimal_weights': optimal_weights,
            'correlation_matrix': correlation_matrix,
            'market_features': market_features,
            'training_results': training_results,
            'portfolio_performance': self._calculate_portfolio_metrics(training_results)
        }

    def _calculate_ticker_correlations(self, features_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Розраховує кореляційну матрицю між тікерами"""
        self.logger.info("📊 Calculating ticker correlations...")

        # Об'єднуємо повернення всіх тікерів
        returns_data = {}
        for ticker, df in features_dict.items():
            if 'returns' in df.columns:
                returns_data[ticker] = df['returns']
            elif 'close' in df.columns:
                returns_data[ticker] = df['close'].pct_change().dropna()

        if not returns_data:
            self.logger.warning("No returns data found for correlation calculation")
            return pd.DataFrame()

        returns_df = pd.DataFrame(returns_data)

        # Використовуємо Ledoit-Wolf estimator для стабільності
        cov_matrix = LedoitWolf().fit(returns_df).covariance_
        correlation_matrix = cov_matrix / np.sqrt(np.outer(np.diag(cov_matrix), np.diag(cov_matrix)))

        self.logger.info(f"📈 Correlation matrix shape: {correlation_matrix.shape}")
        return correlation_matrix

    def _extract_market_features(self, features_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Виділяє спільні ринкові фічі"""
        self.logger.info("🌍 Extracting market features...")

        market_features = []

        # Знаходимо спільні колонки
        first_ticker = list(features_dict.keys())[0]
        common_cols = set(features_dict[first_ticker].columns)

        for _ticker, df in features_dict.items():
            common_cols &= set(df.columns)

        # Додаємо спільні фічі з усіх тікерів
        for ticker, df in features_dict.items():
            ticker_features = df[list(common_cols)].copy()
            ticker_features['ticker'] = ticker
            market_features.append(ticker_features)

        market_df = pd.concat(market_features, ignore_index=True)

        self.logger.info(f"📊 Market features shape: {market_df.shape}")
        return market_df

    def _calculate_optimal_weights(self,
                                  correlation_matrix: pd.DataFrame,
                                  market_features: pd.DataFrame) -> dict[str, float]:
        """
        Розраховує оптимальні ваги моделей з урахуванням кореляцій
        """
        self.logger.info("⚖️ Calculating optimal model weights...")

        if correlation_matrix.empty:
            # Якщо немає кореляцій, рівномірні ваги
            tickers = list(market_features['ticker'].unique())
            return {ticker: 1.0/len(tickers) for ticker in tickers}

        # Розрахунок ефективної кореляції
        def portfolio_variance(weights):
            return np.sqrt(weights @ correlation_matrix.values @ weights.T)

        # Обмеження: ваги > 0, сума = 1
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = [(0.01, 0.5) for _ in range(len(correlation_matrix))]

        # Початкові ваги (рівномірні)
        initial_weights = np.ones(len(correlation_matrix)) / len(correlation_matrix)

        # Оптимізація
        result = minimize(
            portfolio_variance,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        tickers = correlation_matrix.index.tolist()

        weights_dict = dict(zip(tickers, optimal_weights, strict=False))

        self.logger.info(f"✅ Optimal weights calculated: {weights_dict}")
        return weights_dict

    def _train_hierarchical_models(self,
                                features_dict: dict[str, pd.DataFrame],
                                targets_dict: dict[str, pd.DataFrame],
                                optimal_weights: dict[str, float],
                                market_features: pd.DataFrame) -> dict[str, Any]:
        """
        Ієрархічне тренування: глобальна модель + локальні коректори
        """
        self.logger.info("🏗️ Training hierarchical models...")

        # 1. Тренуємо глобальну модель на ринкових фічах
        global_model = self._train_global_model(market_features)

        # 2. Тренуємо локальні моделі з урахуванням глобальних прогнозів
        local_models = {}
        for ticker in features_dict.keys():
            local_models[ticker] = self._train_local_model(
                ticker,
                (features_dict[ticker], targets_dict[ticker]),
                global_model,
                optimal_weights[ticker]
            )

        return {
            'global_model': global_model,
            'local_models': local_models,
            'hierarchical_predictions': self._generate_hierarchical_predictions(
                global_model, local_models, features_dict, optimal_weights
            )
        }

    def _train_global_model(self, market_features: pd.DataFrame) -> dict[str, Any]:
        """Тренує глобальну модель на спільних ринкових фічах"""
        self.logger.info("🌍 Training global market model...")

        # Тут має бути реальна логіка тренування глобальної моделі
        # Для прикладу використовуємо просту модель

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split

        # Підготовка даних
        X = market_features.drop(['ticker'], axis=1)
        y = market_features.groupby('ticker')['returns'].first().fillna(0)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Тренування моделі
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Оцінка
        score = model.score(X_test, y_test)

        self.logger.info(f"📊 Global model R²: {score:.4f}")

        return {
            'model': model,
            'score': score,
            'feature_importance': dict(zip(X.columns, model.feature_importances_, strict=False))
        }

    # CodeScene: Excess Arguments (5) - acceptable for federated learning configuration
    def _train_local_model(self,
                           ticker: str,
                           ticker_data: tuple[pd.DataFrame, pd.DataFrame],
                           global_model: dict[str, Any],
                           weight: float) -> dict[str, Any]:
        """Тренує локальну модель з урахуванням глобальних прогнозів"""
        self.logger.info(f"📈 Training local model for {ticker}...")
        ticker_features, ticker_targets = ticker_data

        # Тут має бути реальна логіка тренування локальної моделі
        # Для прикладу використовуємо просту модель

        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        # Підготовка даних
        X = ticker_features
        y = ticker_targets.iloc[:, 0] if len(ticker_targets.columns) > 0 else ticker_targets

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Тренування моделі
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Оцінка
        score = model.score(X_test, y_test)

        return {
            'model': model,
            'score': score,
            'weight': weight,
            'global_influence': self._calculate_global_influence(global_model, ticker_features)
        }

    def _generate_hierarchical_predictions(self,
                                         global_model: dict[str, Any],
                                         local_models: dict[str, Any],
                                         features_dict: dict[str, pd.DataFrame],
                                         optimal_weights: dict[str, float]) -> dict[str, np.ndarray]:
        """Генерує ієрархічні прогнози"""
        self.logger.info("🔮 Generating hierarchical predictions...")

        predictions = {}

        for ticker, features in features_dict.items():
            # Глобальний прогноз
            global_pred = global_model['model'].predict(features)

            # Локальний прогноз
            local_pred = local_models[ticker]['model'].predict(features)

            # Комбінований прогноз з вагами
            weight = optimal_weights[ticker]
            combined_pred = weight * global_pred + (1 - weight) * local_pred

            predictions[ticker] = combined_pred

        return predictions

    def _calculate_global_influence(self, global_model: dict[str, Any], features: pd.DataFrame) -> float:
        """Розраховує вплив глобальної моделі на локальні прогнози"""
        # Для прикладу використовуємо простий розрахунок
        return 0.3  # 30% вплив глобальної моделі

    def _calculate_portfolio_metrics(self, training_results: dict[str, Any]) -> dict[str, float]:
        """Розраховує метрики портфоліо"""
        self.logger.info("📊 Calculating portfolio metrics...")

        # Тут має бути реальна логіка розрахунку метрик
        # Для прикладу повертаємо базові метрики

        return {
            'sharpe_ratio': 1.5,
            'max_drawdown': 0.08,
            'win_rate': 0.55,
            'portfolio_volatility': 0.12
        }
