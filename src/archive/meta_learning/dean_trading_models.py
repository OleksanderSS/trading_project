# src/meta_learning/dean_trading_models.py

"""
DEAN TRADING MODELS
Трейдингові моделі на основі принципів Станісласа Деана (Stanislas Dehaene):
Актор, Критик, Симулятор.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.patterns.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)

class DeanTradingModel(ABC):
    """Базовий клас для трейдингових моделей Деана."""

    def __init__(self, model_id: str = "dean_default"):
        self.model_id = model_id
        self.logger = logging.getLogger(f"{__name__}.{model_id}")
        self.feature_weights = None
        self.last_prediction_error = 0.0
        self.interaction_pairs = []

    def get_id(self) -> str:
        return self.model_id

    @abstractmethod
    def train(self, data: Any):
        pass

class DeanActor(DeanTradingModel):
    """
    Актор - пропонує торгову дію на основі базової ML моделі.
    """
    def __init__(self, base_model: Any, model_id: str = "dean_actor"):
        super().__init__(model_id)
        self.base_model = base_model

    def train(self, data: tuple[pd.DataFrame, pd.Series]):
        X, y = data
        self.base_model.fit(X, y)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train((X, y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def decide_action(self, context: pd.DataFrame) -> dict[str, Any]:
        """Приймає рішення на основі прогнозів базової моделі."""
        probs = self.predict_proba(context)
        confidence = np.max(probs, axis=1)[-1]
        prediction = 1 if probs[-1, 1] > 0.5 else 0

        return {
            "type": "buy" if prediction == 1 else "sell",
            "parameters": {"size_pct": 0.1},
            "confidence": float(confidence),
            "ticker": context.get('ticker', ['Unknown'])[-1] if 'ticker' in context.columns else 'Unknown'
        }

class DeanCritic(DeanTradingModel):
    """
    Критик - оцінює дії Актора за допомогою правил, ML-моделі помилок та аналізу патернів.
    Вчиться на розбіжностях між прогнозом Актора та реальністю.
    """
    def __init__(self, rules_config: dict[str, Any], model_id: str = "dean_critic"):
        super().__init__(model_id)
        self.rules_config = rules_config
        self.meta_model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        self.pattern_analyzer = PatternAnalyzer()
        self.is_fitted = False

    def train(self, data: tuple[pd.DataFrame, pd.Series, np.ndarray]):
        """
        data: (X_features, y_true, y_pred_actor)
        """
        X, y_true, y_pred = data
        self.fit(X, y_true, y_pred)

    def fit(self, X: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray):
        """
        Тренує мета-модель на величинах помилок Актора.
        """
        if len(X) != len(y_true) or len(y_true) != len(y_pred):
            self.logger.error("Dimension mismatch in Critic training data")
            return

        # Ціль навчання - абсолютна помилка
        errors = np.abs(y_true.values - y_pred)

        try:
            self.meta_model.fit(X, errors)
            self.is_fitted = True
            self.logger.info(f"Critic meta-model trained on {len(X)} instances.")
        except (ValueError, TypeError, RuntimeError) as e:
            self.logger.error(f"Failed to train Critic meta-model: {e}")
            raise RuntimeError("Critic meta-model training failed") from e

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Прогнозує очікувану помилку."""
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.meta_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def critique_action(self, action: dict[str, Any], context: pd.DataFrame) -> dict[str, Any]:
        """Оцінює дію Актора та повертає critique_score від -1 до 1."""
        score = 0.0
        points = []

        # 1. Традиційна перевірка волатильності
        volatility = context.iloc[-1].get('feature_volatility', 0)
        if volatility > self.rules_config.get('high_vol_threshold', 0.05):
            score -= 0.3
            points.append("High market volatility detected")

        # 2. ML Аналіз очікуваної помилки
        expected_error = 0.0
        if self.is_fitted:
            expected_error = self.predict(context.iloc[[-1]])[0]
            if expected_error > 0.35:
                penalty = min(0.6, expected_error)
                score -= penalty
                points.append(f"High predicted actor error: {expected_error:.2f}")
            elif expected_error < 0.15:
                score += 0.2
                points.append("Context matches high historical accuracy")

        # 3. Аналіз макро-патернів та ринкових режимів через PatternAnalyzer
        news_list = context.iloc[-1].get('news_list', [])
        market_data = context.iloc[-1].get('market_data', {})
        insights = self.pattern_analyzer.get_pattern_insights(news_list, market_data)
        regime_warnings = insights.get('regime_warnings', [])

        if regime_warnings:
            ticker = action.get('ticker', '').upper()
            action_type = action.get('type', '').lower()

            for warning in regime_warnings:
                # Знижуємо score при бульбашках у техах
                if "TECH BUBBLE" in warning and action_type == "buy" and ticker in ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                    score -= 0.8
                    points.append(f"CRITICAL: {warning} - High risk for long tech positions")
                # Знижуємо score при кредитних стресах
                elif "CREDIT STRESS" in warning and action_type == "buy":
                    score -= 0.6
                    points.append(f"CRITICAL: {warning} - Deleveraging risk")
                else:
                    score -= 0.2
                    points.append(f"Pattern Warning: {warning}")

        # 4. Перевірка впевненості та парадоксальних сигналів
        if action['confidence'] > 0.9 and score < -0.2:
            score -= 0.4
            points.append("Paradoxical confidence detected: Actor is sure but context/patterns are high-risk")
        elif action['confidence'] < 0.6:
            score -= 0.2
            points.append("Low prediction confidence")
        else:
            score += 0.1

        return {
            "score": max(-1.0, min(1.0, score)),
            "points": points,
            "alternatives": [{"type": "hold"}] if score < -0.4 else [],
            "expected_error": float(expected_error),
            "regime_insights": insights,
            "confidence": 0.85
        }

class DeanSimulator(DeanTradingModel):
    """
    Симулятор - прогнозує результати на основі історичних аналогів (KNN).

    Знаходить найближчі схожі ринкові стани в history і агрегує їхні
    фактичні наступні повернення як прогноз. Чим ближчий аналог (менша
    KNN-відстань), тим більшу вагу він отримує.
    """
    def __init__(self, similarity_finder: KnnSimilarityFinder, model_id: str = "dean_simulator"):
        super().__init__(model_id)
        self.similarity_finder = similarity_finder
        # Зберігаємо весь DataFrame щоб мати доступ до фактичних повернень сусідів
        self._historical_data: pd.DataFrame = pd.DataFrame()

    def train(self, historical_data: pd.DataFrame):
        """Fit KNN index on historical feature data."""
        self._historical_data = historical_data.copy()
        self.similarity_finder.fit(historical_data)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point-in-time directional prediction based on KNN analog voting."""
        if self._historical_data.empty or self.similarity_finder.knn_model is None:
            return np.zeros(len(X))

        results = []
        for i in range(len(X)):
            row = X.iloc[i]
            try:
                positions, distances = self.similarity_finder.find_similar_situations(row)
                if len(positions) == 0:
                    results.append(0.0)
                    continue
                # Weighted vote: weight = 1/(1+distance)
                weights = 1.0 / (1.0 + distances)
                weights /= weights.sum()
                # Use 'close' pct_change if available as the outcome proxy
                if 'close' in self._historical_data.columns:
                    neighbor_returns = (
                        self._historical_data['close']
                        .pct_change(fill_method=None)
                        .iloc[positions]
                        .fillna(0.0)
                        .values
                    )
                    weighted_return = float(np.dot(weights, neighbor_returns))
                else:
                    # Majority-vote on the first numeric column
                    first_num = self._historical_data.select_dtypes(include=[np.number]).columns
                    if len(first_num) == 0:
                        results.append(0.0)
                        continue
                    vals = self._historical_data[first_num[0]].iloc[positions].fillna(0.0).values
                    weighted_return = float(np.dot(weights, vals))
                results.append(weighted_return)
            except Exception as e:
                self.logger.warning(f"DeanSimulator.predict row {i}: {e}")
                results.append(0.0)
        return np.array(results)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.predict(X)
        prob_up = np.clip(0.5 + preds * 5, 0.0, 1.0)   # soft sigmoid-like scaling
        return np.column_stack([1.0 - prob_up, prob_up])

    def simulate(self, scenario_context: pd.DataFrame, horizon: int = 5) -> dict[str, Any]:
        """
        Симулює розвиток подій на основі схожих історичних станів.

        Знаходить KNN-аналоги, агрегує їхні фактичні наступні повернення
        з distance-weighting і будує прогнозний розподіл.
        """
        if self._historical_data.empty or self.similarity_finder.knn_model is None:
            return {
                "state": "no_historical_data",
                "key_factors": [],
                "confidence": 0.0,
                "predicted_return": 0.0,
                "analog_count": 0,
            }

        current_row = scenario_context.iloc[-1]
        try:
            positions, distances = self.similarity_finder.find_similar_situations(current_row)
        except Exception as e:
            self.logger.warning(f"DeanSimulator.simulate KNN search failed: {e}")
            return {
                "state": "knn_error",
                "key_factors": [],
                "confidence": 0.0,
                "predicted_return": 0.0,
                "analog_count": 0,
                "error": str(e),
            }

        if len(positions) == 0:
            return {
                "state": "no_analogs_found",
                "key_factors": [],
                "confidence": 0.0,
                "predicted_return": 0.0,
                "analog_count": 0,
            }

        # Distance-based weights: closer analog → higher weight
        weights = 1.0 / (1.0 + distances)
        weights /= weights.sum()

        # Aggregate forward returns over `horizon` bars from each analog position
        forward_returns = []
        analog_dates = []
        has_close = 'close' in self._historical_data.columns

        for pos, w in zip(positions, weights, strict=True):
            end_pos = min(pos + horizon, len(self._historical_data) - 1)
            if end_pos <= pos:
                forward_returns.append(0.0)
                continue
            if has_close:
                start_price = self._historical_data['close'].iloc[pos]
                end_price = self._historical_data['close'].iloc[end_pos]
                ret = (end_price / start_price - 1.0) if start_price != 0 else 0.0
            else:
                ret = 0.0
            forward_returns.append(float(ret))
            analog_dates.append(str(self._historical_data.index[pos]))

        forward_returns_arr = np.array(forward_returns)
        predicted_return = float(np.dot(weights, forward_returns_arr))
        return_std = float(np.std(forward_returns_arr)) if len(forward_returns_arr) > 1 else 0.0

        # Confidence: higher when analogs agree and distances are small
        directional_agreement = float(np.mean(np.sign(forward_returns_arr) == np.sign(predicted_return)))
        avg_similarity = float(np.mean(1.0 / (1.0 + distances)))
        confidence = round(directional_agreement * avg_similarity, 4)

        state = (
            "bullish_analog" if predicted_return > 0.005
            else "bearish_analog" if predicted_return < -0.005
            else "neutral_analog"
        )

        return {
            "state": state,
            "predicted_return": round(predicted_return, 6),
            "return_std": round(return_std, 6),
            "confidence": confidence,
            "directional_agreement": round(directional_agreement, 4),
            "avg_similarity": round(avg_similarity, 4),
            "analog_count": len(positions),
            "analog_dates": analog_dates[:3],   # top-3 for logging/debug
            "key_factors": ["knn_historical_analogs", "distance_weighted_returns"],
            "horizon_bars": horizon,
        }

class DeanTradingModels:
    """Оркестратор, який об'єднує Актора, Критика та Симулятора за принципом Деана"""

    def __init__(self, actor: DeanActor, critic: DeanCritic, simulator: DeanSimulator):
        self.actor = actor
        self.critic = critic
        self.simulator = simulator

    def get_integrated_decision(self, context: pd.DataFrame) -> dict[str, Any]:
        """Проходить повний цикл: Акт -> Критика -> Симуляція."""
        action = self.actor.decide_action(context)
        critique = self.critic.critique_action(action, context)
        simulation = self.simulator.simulate(context, horizon=5)

        # Зважена впевненість: Актор (50%) + Критик (30%) + Симулятор (20%)
        # Критик повертає score в [-1, 1], нормалізуємо до [0, 1]
        critic_score_normalized = (critique['score'] + 1.0) / 2.0
        sim_confidence = float(simulation.get('confidence', 0.5))

        final_confidence = (
            action['confidence'] * 0.5
            + critic_score_normalized * 0.3
            + sim_confidence * 0.2
        )
        # Якщо симулятор і Актор суперечать одне одному по напрямку — штраф
        sim_return = float(simulation.get('predicted_return', 0.0))
        action_sign = 1.0 if action['type'] == 'buy' else -1.0
        if sim_return != 0.0 and np.sign(sim_return) != action_sign:
            final_confidence *= 0.8
            self.logger.info(
                f"Simulator direction ({sim_return:.4f}) contradicts Actor ({action['type']}). "
                f"Confidence penalized."
            )

        return {
            'action': action,
            'critique': critique,
            'simulation': simulation,
            'final_confidence': float(np.clip(final_confidence, 0.0, 1.0)),
            'timestamp': datetime.now().isoformat(),
        }
