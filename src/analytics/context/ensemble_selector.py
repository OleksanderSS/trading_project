"""
EnsembleSelector: Intelligent selection of the best ensemble method based on context.
Chooses between LiveAdaptiveEnsemble, StackedEnsemble, ConsensusEngine, and simple methods.

Moved from src/integration/ and cleaned of mock/placeholder logic.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.core.logging.logger import ProjectLogger


@dataclass
class EnsembleContext:
    """Context information for ensemble selection."""
    data_size: int
    has_real_time_data: bool
    model_count: int
    market_regime: str
    volatility_level: float
    prediction_frequency: str
    computational_resources: str  # 'low', 'medium', 'high'
    latency_requirement: str      # 'low', 'medium', 'high'


class EnsembleSelector:
    """
    Selects the best ensemble method based on data characteristics,
    market conditions, and resource constraints.

    Complements ContextualModelSelector (which picks the best single model via kNN)
    by operating at a higher level — choosing which ensemble *strategy* to use.
    """

    _METHODS: dict[str, dict[str, Any]] = {
        'live_adaptive': {
            'class_path': 'src.trading.live_adaptive_ensemble.LiveAdaptiveEnsemble',
            'strengths': {'real_time_adaptation', 'performance_tracking', 'regime_aware'},
            'requirements': {'min_models': 3, 'real_time_data': True},
            'best_for': {'live_trading', 'adaptive_strategies', 'multi_model_systems'},
        },
        'stacked_ensemble': {
            'class_path': 'src.ensembling.stacked_ensemble.StackedEnsemble',
            'strengths': {'meta_learning', 'live_efficiency_weighting'},
            'requirements': {'min_models': 2, 'training_data': True},
            'best_for': {'meta_learning', 'weighted_combinations'},
        },
        'consensus_engine': {
            'class_path': 'src.trading.consensus_engine.ConsensusEngine',
            'strengths': {'decision_core', 'regime_aware', 'critic_filters'},
            'requirements': {'min_models': 3, 'experience_diary': True},
            'best_for': {'final_decisions', 'risk_aware', 'quality_signals'},
        },
        'weighted_average': {
            'class_path': 'built_in',
            'strengths': {'performance_based', 'simple'},
            'requirements': {'min_models': 1, 'performance_data': True},
            'best_for': {'performance_weighted', 'moderate_complexity'},
        },
    }

    def __init__(self) -> None:
        self.logger = ProjectLogger.get_logger(__name__)
        self.logger.info("EnsembleSelector initialized")

    def select_best_ensemble(
        self,
        context: EnsembleContext,
        available_models: list[str],
        performance_data: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Select the best ensemble method for the given context.

        Returns a dict with keys: selected_ensemble, score, reasoning, confidence.
        """
        scores: dict[str, float] = {}
        for name, cfg in self._METHODS.items():
            scores[name] = self._score(name, cfg, context, available_models, performance_data)

        best = max(scores, key=scores.__getitem__)
        best_score = scores[best]

        result = {
            'selected_ensemble': best,
            'score': best_score,
            'reasoning': self._reasoning(best, best_score, context),
            'all_scores': scores,
            'confidence': self._confidence(best_score),
            'selection_time': datetime.now(),
        }
        self.logger.info(f"Selected '{best}' ensemble (score={best_score:.2f})")
        return result

    def create_ensemble_instance(self, method_name: str, **kwargs) -> Any | None:
        """Instantiate the selected ensemble class."""
        try:
            if method_name == 'live_adaptive':
                from src.trading.live_adaptive_ensemble import LiveAdaptiveEnsemble
                return LiveAdaptiveEnsemble(**kwargs)
            if method_name == 'stacked_ensemble':
                from src.ensembling.stacked_ensemble import StackedEnsemble
                return StackedEnsemble(**kwargs)
            if method_name == 'consensus_engine':
                from src.trading.consensus_engine import ConsensusEngine
                return ConsensusEngine(**kwargs)
            if method_name == 'weighted_average':
                from src.models.ensemble.ensemble_model import EnsembleModel
                return EnsembleModel(**kwargs)
            self.logger.error(f"Unknown ensemble method: {method_name}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to create '{method_name}' instance: {e}")
            return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _score(
        self,
        name: str,
        cfg: dict[str, Any],
        ctx: EnsembleContext,
        models: list[str],
        perf: dict[str, Any] | None,
    ) -> float:
        if not self._meets_requirements(cfg['requirements'], ctx, models, perf):
            return 0.0

        score = 0.0

        # Real-time data bonus
        if ctx.has_real_time_data and 'real_time_adaptation' in cfg['strengths']:
            score += 0.25

        # Regime-aware bonus in volatile markets
        if ctx.market_regime == 'volatile' and 'regime_aware' in cfg['strengths']:
            score += 0.20

        # Multi-model bonus
        if ctx.model_count > 5 and 'multi_model_systems' in cfg.get('best_for', set()):
            score += 0.20

        # Performance data bonus
        if perf and 'performance_based' in cfg['strengths']:
            score += 0.15

        # Low-resource penalty for heavy methods
        if ctx.computational_resources == 'low' and name in ('live_adaptive', 'consensus_engine'):
            score -= 0.15

        return min(score, 1.0)

    def _meets_requirements(
        self,
        reqs: dict[str, Any],
        ctx: EnsembleContext,
        models: list[str],
        perf: dict[str, Any] | None,
    ) -> bool:
        if len(models) < reqs.get('min_models', 1):
            return False
        if reqs.get('real_time_data') and not ctx.has_real_time_data:
            return False
        if reqs.get('performance_data') and not perf:
            return False
        return True

    def _reasoning(self, method: str, score: float, ctx: EnsembleContext) -> str:
        quality = "Excellent" if score > 0.8 else "Good" if score > 0.5 else "Best available"
        parts = [f"{quality} fit: {method}"]
        if ctx.has_real_time_data and method == 'live_adaptive':
            parts.append("real-time adaptation available")
        if ctx.market_regime == 'volatile':
            parts.append("volatile market regime")
        return " — ".join(parts)

    @staticmethod
    def _confidence(score: float) -> float:
        if score > 0.8:
            return 0.9
        if score > 0.6:
            return 0.7
        if score > 0.4:
            return 0.5
        return 0.3
