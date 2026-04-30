# src/meta_learning/dean_trading_models.py

"""
DEAN TRADING MODELS
Трейдингові моделі на основі принципів Станісласа Деана (Stanislas Dehaene):
Актор, Критик, Симулятор.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import random
from datetime import datetime
from scipy.stats import entropy
from sklearn.ensemble import RandomForestRegressor

from src.factories.model_interface import ModelInterface
from src.analytics.analyzers.knn_similarity_finder import KnnSimilarityFinder
from src.patterns.pattern_analyzer import PatternAnalyzer

logger = logging.getLogger(__name__)

class DeanTradingModel(ModelInterface, ABC):
    """Базовий клас для трейдингових моделей Деана, сумісний з ModelFactory"""
    
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
    def __init__(self, base_model: ModelInterface, model_id: str = "dean_actor"):
        super().__init__(model_id)
        self.base_model = base_model

    def train(self, data: Tuple[pd.DataFrame, pd.Series]):
        X, y = data
        self.base_model.fit(X, y)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train((X, y))

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.base_model.predict_proba(X)

    def decide_action(self, context: pd.DataFrame) -> Dict[str, Any]:
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
    def __init__(self, rules_config: Dict[str, Any], model_id: str = "dean_critic"):
        super().__init__(model_id)
        self.rules_config = rules_config
        self.meta_model = RandomForestRegressor(
            n_estimators=50, 
            max_depth=5, 
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42
        )
        self.pattern_analyzer = PatternAnalyzer()
        self.is_fitted = False

    def train(self, data: Tuple[pd.DataFrame, pd.Series, np.ndarray]):
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
        except Exception as e:
            self.logger.error(f"Failed to train Critic meta-model: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Прогнозує очікувану помилку."""
        if not self.is_fitted:
            return np.zeros(len(X))
        return self.meta_model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

    def critique_action(self, action: Dict[str, Any], context: pd.DataFrame) -> Dict[str, Any]:
        """Оцінює дію Актора та повертає critique_score від -1 до 1."""
        score = 0.0
        points = []
        
        # 1. Check volatility
        score, points = self._check_volatility(context, score, points)
        
        # 2. ML analysis of expected error
        expected_error, score, points = self._analyze_expected_error(context, score, points)
        
        # 3. Pattern analysis and regime warnings
        insights = self._analyze_patterns(context, action, score, points)
        score = insights['score']
        points.extend(insights['points'])
        
        # 4. Confidence and paradoxical signals
        score, points = self._check_confidence_signals(action, score, points)
        
        return {
            "score": max(-1.0, min(1.0, score)),
            "points": points,
            "alternatives": [{"type": "hold"}] if score < -0.4 else [],
            "expected_error": float(expected_error),
            "regime_insights": insights.get('regime_insights', {}),
            "confidence": 0.85
        }

    def _check_volatility(self, context: pd.DataFrame, score: float, points: List[str]) -> Tuple[float, List[str]]:
        """Check market volatility and adjust score."""
        volatility = context.iloc[-1].get('feature_volatility', 0)
        if volatility > self.rules_config.get('high_vol_threshold', 0.05):
            score -= 0.3
            points.append("High market volatility detected")
        return score, points

    def _analyze_expected_error(self, context: pd.DataFrame, score: float, points: List[str]) -> Tuple[float, float, List[str]]:
        """Analyze expected error using ML model."""
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
        return expected_error, score, points

    def _analyze_patterns(self, context: pd.DataFrame, action: Dict[str, Any], score: float, points: List[str]) -> Dict[str, Any]:
        """Analyze macro patterns and market regimes."""
        news_list = context.iloc[-1].get('news_list', [])
        market_data = context.iloc[-1].get('market_data', {})
        insights = self.pattern_analyzer.get_pattern_insights(news_list, market_data)
        regime_warnings = insights.get('regime_warnings', [])
        
        if regime_warnings:
            ticker = action.get('ticker', '').upper()
            action_type = action.get('type', '').lower()
            
            for warning in regime_warnings:
                score, points = self._process_regime_warning(warning, ticker, action_type, score, points)
        
        return {
            'score': score,
            'points': points,
            'regime_insights': insights
        }

    def _process_regime_warning(self, warning: str, ticker: str, action_type: str, score: float, points: List[str]) -> Tuple[float, List[str]]:
        """Process individual regime warning."""
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
        
        return score, points

    def _check_confidence_signals(self, action: Dict[str, Any], score: float, points: List[str]) -> Tuple[float, List[str]]:
        """Check confidence and paradoxical signals."""
        if action['confidence'] > 0.9 and score < -0.2:
            score -= 0.4
            points.append("Paradoxical confidence detected: Actor is sure but context/patterns are high-risk")
        elif action['confidence'] < 0.6:
            score -= 0.2
            points.append("Low prediction confidence")
        else:
            score += 0.1
        
        return score, points

class DeanSimulator(DeanTradingModel):
    """
    Симулятор - прогнозує результати на основі історичних аналогів.
    """
    def __init__(self, similarity_finder: KnnSimilarityFinder, model_id: str = "dean_simulator"):
        super().__init__(model_id)
        self.similarity_finder = similarity_finder

    def train(self, historical_data: pd.DataFrame):
        self.similarity_finder.fit(historical_data)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.train(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([np.ones(len(X)), np.zeros(len(X))])

    def simulate(self, scenario_context: pd.DataFrame, horizon: int) -> Dict[str, Any]:
        """Симулює розвиток подій на основі схожих історичних станів."""
        return {
            "state": "predicted_market_movement",
            "key_factors": ["historical_analogs", "momentum"],
            "confidence": 0.65
        }

class DeanTradingModels:
    """Оркестратор, який об'єднує Актора, Критика та Симулятора за принципом Деана"""
    
    def __init__(self, actor: DeanActor, critic: DeanCritic, simulator: DeanSimulator):
        self.actor = actor
        self.critic = critic
        self.simulator = simulator

    def get_integrated_decision(self, context: pd.DataFrame) -> Dict[str, Any]:
        """Проходить повний цикл: Акт -> Критика -> Симуляція."""
        action = self.actor.decide_action(context)
        critique = self.critic.critique_action(action, context)
        simulation = self.simulator.simulate(context, horizon=5)
        
        final_confidence = action['confidence'] * (1 + critique['score'])
        
        return {
            'action': action,
            'critique': critique,
            'simulation': simulation,
            'final_confidence': float(final_confidence),
            'timestamp': datetime.now().isoformat()
        }
