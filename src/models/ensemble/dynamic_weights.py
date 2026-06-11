"""
DynamicWeightCalculator: Dynamic weight calculation for ensemble models

Features:
- Performance-based weights
- Context-aware weights
- Adaptive weights with history
- Weight history tracking
- Export/import weights

Usage:
    calculator = DynamicWeightCalculator(method="adaptive")

    weights = calculator.calculate_weights(
        models=["model1", "model2", "model3"],
        performance_data={"model1": 0.8, "model2": 0.7, "model3": 0.9},
        context={"volatility": 0.5, "trend": 0.1}
    )
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class DynamicWeightCalculator:
    """
    Dynamic weight calculator for ensemble models.

    Methods:
    - performance_based: Weights proportional to performance
    - context_aware: Weights adjusted by market context
    - adaptive: Weights with exponential moving average

    Attributes:
        method: Weight calculation method
        weight_history: Historical weights per model
        alpha: Smoothing factor for adaptive weights (0-1)
    """

    def __init__(self, method: str = "adaptive", alpha: float = 0.3):
        """
        Initialize weight calculator.

        Args:
            method: Weight calculation method
                   - "performance_based": Simple performance weighting
                   - "context_aware": Context-adjusted weights
                   - "adaptive": EMA-smoothed weights
            alpha: Smoothing factor for adaptive method (default: 0.3)
                  Higher = more weight to recent performance
        """
        self.method = method
        self.alpha = alpha
        self.weight_history: dict[str, list[float]] = {}
        self.logger = ProjectLogger.get_logger(__name__)

    def calculate_weights(
        self,
        models: list[str],
        performance_data: dict[str, float],
        context: dict[str, Any] | None = None
    ) -> dict[str, float]:
        """
        Calculate dynamic weights for models.

        Args:
            models: List of model IDs
            performance_data: Dict[model_id, performance_score]
                            Performance score should be 0-1 (higher = better)
            context: Optional market context
                    - volatility: 0-1 (market volatility)
                    - trend: -1 to 1 (market trend)
                    - regime: str (market regime)

        Returns:
            Dict[model_id, weight] (weights sum to 1.0)

        Example:
            weights = calculator.calculate_weights(
                models=["catboost", "lstm", "xgboost"],
                performance_data={"catboost": 0.8, "lstm": 0.7, "xgboost": 0.9},
                context={"volatility": 0.5, "trend": 0.1}
            )
            # weights = {"catboost": 0.32, "lstm": 0.28, "xgboost": 0.40}
        """
        if self.method == "performance_based":
            weights = self._performance_based_weights(models, performance_data)
        elif self.method == "context_aware":
            weights = self._context_aware_weights(models, performance_data, context)
        elif self.method == "adaptive":
            weights = self._adaptive_weights(models, performance_data, context)
        else:
            # Equal weights fallback
            weights = {m: 1.0 / len(models) for m in models}

        # Update history
        for model_id, weight in weights.items():
            if model_id not in self.weight_history:
                self.weight_history[model_id] = []
            self.weight_history[model_id].append(weight)

        return weights

    def _performance_based_weights(
        self,
        models: list[str],
        performance_data: dict[str, float]
    ) -> dict[str, float]:
        """
        Calculate weights based on performance only.

        Weight = performance / sum(performances)
        """
        weights = {}
        total_perf = sum(performance_data.get(m, 0.5) for m in models)

        if total_perf == 0:
            # Equal weights if no performance data
            return {m: 1.0 / len(models) for m in models}

        for model in models:
            perf = performance_data.get(model, 0.5)
            weights[model] = perf / total_perf

        return weights

    def _context_aware_weights(
        self,
        models: list[str],
        performance_data: dict[str, float],
        context: dict[str, Any] | None
    ) -> dict[str, float]:
        """
        Calculate weights with context adjustment.

        Adjusts base performance weights based on:
        - Model type suitability for current market regime
        - Volatility preferences
        - Trend preferences
        """
        # Start with performance-based weights
        base_weights = self._performance_based_weights(models, performance_data)

        if not context:
            return base_weights

        # Context adjustments
        volatility = context.get('volatility', 0.5)
        trend = context.get('trend', 0.0)

        adjusted_weights = {}
        for model in models:
            weight = base_weights[model]

            # LSTM/RNN better in high volatility
            if 'lstm' in model.lower() or 'rnn' in model.lower() or 'gru' in model.lower():
                if volatility > 0.7:
                    weight *= 1.2
                elif volatility < 0.3:
                    weight *= 0.9

            # Tree models better in trending markets
            if 'catboost' in model.lower() or 'xgboost' in model.lower() or 'lgbm' in model.lower():
                if abs(trend) > 0.5:
                    weight *= 1.1

            # Transformer better in complex patterns
            if 'transformer' in model.lower():
                if volatility > 0.6 and abs(trend) < 0.3:
                    weight *= 1.15

            adjusted_weights[model] = weight

        # Normalize
        total = sum(adjusted_weights.values())
        return {m: w / total for m, w in adjusted_weights.items()}

    def _adaptive_weights(
        self,
        models: list[str],
        performance_data: dict[str, float],
        context: dict[str, Any] | None
    ) -> dict[str, float]:
        """
        Calculate adaptive weights with exponential moving average.

        Combines context-aware weights with historical smoothing:
        weight_new = alpha * weight_current + (1 - alpha) * weight_old
        """
        # Get context-aware weights
        current_weights = self._context_aware_weights(models, performance_data, context)

        # Apply EMA smoothing
        smoothed_weights = {}
        for model in models:
            current_weight = current_weights[model]

            if model in self.weight_history and len(self.weight_history[model]) > 0:
                # EMA: new = alpha * current + (1-alpha) * old
                old_weight = self.weight_history[model][-1]
                smoothed_weight = self.alpha * current_weight + (1 - self.alpha) * old_weight
            else:
                # First time, use current weight
                smoothed_weight = current_weight

            smoothed_weights[model] = smoothed_weight

        # Normalize
        total = sum(smoothed_weights.values())
        return {m: w / total for m, w in smoothed_weights.items()}

    def get_weight_history(self, model_id: str, window: int = 10) -> list[float]:
        """
        Get recent weight history for model.

        Args:
            model_id: Model identifier
            window: Number of recent weights to return

        Returns:
            List of recent weights
        """
        if model_id not in self.weight_history:
            return []
        return self.weight_history[model_id][-window:]

    def export_weights(self, filepath: str) -> None:
        """
        Export weight history to file.

        Args:
            filepath: Path to save weights
        """
        data = {
            'method': self.method,
            'alpha': self.alpha,
            'weight_history': self.weight_history,
            'exported_at': datetime.now().isoformat()
        }

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        self.logger.info(f"Exported weights to {filepath}")

    def import_weights(self, filepath: str) -> None:
        """
        Import weight history from file.

        Args:
            filepath: Path to load weights from
        """
        with open(filepath) as f:
            data = json.load(f)

        self.method = data.get('method', self.method)
        self.alpha = data.get('alpha', self.alpha)
        self.weight_history = data.get('weight_history', {})

        self.logger.info(f"Imported weights from {filepath}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get calculator statistics.

        Returns:
            Dict with statistics
        """
        stats = {
            'method': self.method,
            'alpha': self.alpha,
            'models_tracked': len(self.weight_history),
            'total_calculations': sum(len(h) for h in self.weight_history.values())
        }

        # Average weights per model
        if self.weight_history:
            avg_weights = {}
            for model_id, history in self.weight_history.items():
                if history:
                    avg_weights[model_id] = float(np.mean(history))
            stats['average_weights'] = avg_weights

        return stats

    def reset_history(self) -> None:
        """Reset weight history."""
        self.weight_history.clear()
        self.logger.info("Weight history reset")
