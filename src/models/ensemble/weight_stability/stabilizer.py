from typing import Dict, Any, Optional
import logging
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilizer")

class WeightStabilizer:
    """Provides methods to stabilize weight changes."""
    
    def __init__(self, config: Any):
        self.logger = logger
        self.config = config

    def apply_constrained_stabilization(self, 
                                      proposed_weights: Dict[str, float], 
                                      last_weights: Dict[str, float]) -> Dict[str, float]:
        """Limit weight changes to a maximum allowed value per update."""
        stabilized = {}
        for model_name, weight in proposed_weights.items():
            last_weight = last_weights.get(model_name, weight)
            change = weight - last_weight
            if abs(change) > self.config.max_change_per_update:
                stabilized[model_name] = last_weight + (self.config.max_change_per_update if change > 0 else -self.config.max_change_per_update)
            else:
                stabilized[model_name] = weight
        return self._normalize(stabilized)

    def apply_exponential_smoothing(self, 
                                  proposed_weights: Dict[str, float], 
                                  last_weights: Dict[str, float], 
                                  alpha: float = 0.7) -> Dict[str, float]:
        """Apply EMA to weights."""
        smoothed = {}
        for model_name, weight in proposed_weights.items():
            last_weight = last_weights.get(model_name, weight)
            smoothed[model_name] = alpha * weight + (1 - alpha) * last_weight
        return self._normalize(smoothed)

    def apply_volatility_based_stabilization(self, 
                                           proposed_weights: Dict[str, float], 
                                           last_weights: Dict[str, float],
                                           model_volatilities: Dict[str, float]) -> Dict[str, float]:
        """Adjust constraints based on historical volatility."""
        stabilized = {}
        for model_name, weight in proposed_weights.items():
            last_weight = last_weights.get(model_name, weight)
            vol = model_volatilities.get(model_name, 0.0)
            adj_threshold = self.config.max_change_per_update * (1 + vol)
            change = weight - last_weight
            if abs(change) > adj_threshold:
                stabilized[model_name] = last_weight + (adj_threshold if change > 0 else -adj_threshold)
            else:
                stabilized[model_name] = weight
        return self._normalize(stabilized)

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        total = sum(weights.values())
        return {m: w/total for m, w in weights.items()} if total > 0 else weights
