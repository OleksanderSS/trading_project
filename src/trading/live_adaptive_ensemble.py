"""
Live-Adaptive Ensemble Engine
- Dynamic weighting based on recent performance
- Model performance tracking (Sharpe, hit rate, precision)
- Quarterly optimization with historical backtesting
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np

from src.core.logging.logger import ProjectLogger


@dataclass
class ModelPerformanceMetric:
    """Metric for a single model"""
    timestamp: datetime
    model_id: str
    model_type: str
    sharpe_ratio: float
    hit_rate: float  # % correct signals
    precision: float  # % BUY signals that were correct
    recall: float  # % actual BUYs that were caught
    avg_return_per_trade: float
    max_consecutive_losses: int
    predictions_count: int

@dataclass
class EnsembleWeights:
    """Current weights for ensemble"""
    timestamp: datetime
    regime: str
    weights: dict[str, float] = field(default_factory=dict)
    model_scores: dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

class LiveAdaptiveEnsemble:
    """
    Adaptive ensemble that recalibrates weights based on live performance
    """

    def __init__(self, logger=None, reweight_interval_days=7):
        """
        Args:
            reweight_interval_days: How often to reweight (7 days by default)
        """
        self.logger = logger or ProjectLogger.get_logger(__name__)
        self.reweight_interval_days = reweight_interval_days

        # Performance tracking
        self.model_metrics_history = []  # List[ModelPerformanceMetric]
        self.ensemble_weights_history = []  # List[EnsembleWeights]
        self.last_reweight_time = None

        # Regime-based baseline weights (fallback)
        self.baseline_weights = {
            'trending_up': {
                'transformer': 0.28,
                'lstm': 0.22,
                'cnn': 0.18,
                'catboost': 0.16,
                'linear': 0.08,
                'knn': 0.08
            },
            'ranging': {
                'linear': 0.25,
                'catboost': 0.22,
                'knn': 0.18,
                'transformer': 0.15,
                'lstm': 0.12,
                'cnn': 0.08
            },
            'volatile': {
                'cnn': 0.26,
                'transformer': 0.24,
                'lstm': 0.20,
                'linear': 0.16,
                'catboost': 0.14
            }
        }

        self.current_weights = {}

    def record_model_performance(self,
                                model_id: str,
                                model_type: str,
                                sharpe_ratio: float,
                                hit_rate: float,
                                precision: float,
                                recall: float,
                                avg_return: float,
                                max_consecutive_losses: int,
                                predictions_count: int):
        """Record performance metric for model"""
        metric = ModelPerformanceMetric(
            timestamp=datetime.now(),
            model_id=model_id,
            model_type=model_type,
            sharpe_ratio=sharpe_ratio,
            hit_rate=hit_rate,
            precision=precision,
            recall=recall,
            avg_return_per_trade=avg_return,
            max_consecutive_losses=max_consecutive_losses,
            predictions_count=predictions_count
        )

        self.model_metrics_history.append(metric)

        # Keep only last 500 metrics
        if len(self.model_metrics_history) > 500:
            self.model_metrics_history = self.model_metrics_history[-500:]

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"📊 Recorded {model_type} performance: Sharpe={sharpe_ratio:.2f}, Hit Rate={hit_rate:.1%}")

    def _should_reweight(self) -> bool:
        """Check if reweighting is needed based on time interval."""
        return (
            self.last_reweight_time is None or
            (datetime.now() - self.last_reweight_time).days >= self.reweight_interval_days
        )

    def _collect_recent_metrics(self, lookback_days: int = 7) -> list:
        """Collect recent metrics within lookback period."""
        cutoff_time = datetime.now() - timedelta(days=lookback_days)
        return [m for m in self.model_metrics_history if m.timestamp >= cutoff_time]

    def _group_metrics_by_type(self, metrics: list) -> dict:
        """Group metrics by model type."""
        metrics_by_type = {}
        for metric in metrics:
            if metric.model_type not in metrics_by_type:
                metrics_by_type[metric.model_type] = []
            metrics_by_type[metric.model_type].append(metric)
        return metrics_by_type

    def _compute_model_score(self, model_type: str, metrics: list) -> dict:
        """Compute composite score for a model type."""
        avg_sharpe = np.mean([m.sharpe_ratio for m in metrics])
        avg_hit_rate = np.mean([m.hit_rate for m in metrics])
        avg_precision = np.mean([m.precision for m in metrics])
        avg_recall = np.mean([m.recall for m in metrics])
        total_predictions = sum([m.predictions_count for m in metrics])

        # Composite score: weighted average of metrics
        # Sharpe (40%) + Hit Rate (25%) + Precision (20%) + Recall (15%)
        score = (
            0.40 * self._normalize_sharpe(avg_sharpe) +
            0.25 * avg_hit_rate +
            0.20 * avg_precision +
            0.15 * avg_recall
        )

        self.logger.info(f"📊 {model_type}: score={score:.3f}, Sharpe={avg_sharpe:.2f}, Hit Rate={avg_hit_rate:.1%}")

        return {
            'score': score,
            'sharpe': avg_sharpe,
            'hit_rate': avg_hit_rate,
            'precision': avg_precision,
            'recall': avg_recall,
            'n_metrics': len(metrics),
            'n_predictions': total_predictions
        }

    def _scores_to_weights(self, model_scores: dict) -> dict:
        """Convert model scores to weights using softmax."""
        scores = np.array([s['score'] for s in model_scores.values()])
        scores = np.clip(scores, -5, 5)  # Avoid extreme values

        # Softmax: e^score / sum(e^score)
        exp_scores = np.exp(scores - np.max(scores))  # Numerical stability
        raw_weights = exp_scores / exp_scores.sum()

        # Create weights dict
        computed_weights = {}
        for (model_type, _score_data), weight in zip(model_scores.items(), raw_weights, strict=False):
            computed_weights[model_type] = float(weight)

        return computed_weights

    def _smooth_weights_against_baseline(self, computed_weights: dict, regime: str) -> dict:
        """Smooth computed weights against baseline (70% live, 30% baseline)."""
        baseline = self.baseline_weights.get(regime, {})
        smoothed_weights = {}

        all_models = set(computed_weights.keys()) | set(baseline.keys())
        for model_type in all_models:
            live_weight = computed_weights.get(model_type, 0.0)
            baseline_weight = baseline.get(model_type, 1.0 / len(baseline) if baseline else 0.0)

            # 70% live, 30% baseline
            smoothed_weight = 0.7 * live_weight + 0.3 * baseline_weight
            smoothed_weights[model_type] = max(0.01, smoothed_weight)  # Min 1%

        # Renormalize
        total = sum(smoothed_weights.values())
        return {k: v / total for k, v in smoothed_weights.items()}

    def _store_weights_history(self, regime: str, smoothed_weights: dict, model_scores: dict, recent_metrics: list) -> None:
        """Store weights in ensemble history."""
        ew = EnsembleWeights(
            timestamp=datetime.now(),
            regime=regime,
            weights=smoothed_weights,
            model_scores=model_scores,
            reasoning=f"7-day performance: {len(recent_metrics)} metrics, {len(model_scores)} models"
        )
        self.ensemble_weights_history.append(ew)

    def compute_ensemble_weights(self, regime: str) -> dict[str, float]:
        """
        Compute ensemble weights based on live performance

        Algorithm:
        1. Extract latest metrics for each model (for the last week)
        2. Compute composite score for each model
        3. Normalize to weights
        4. Smoothing relative to baseline (to avoid overfitting on noise)
        """
        # 1. Determine if reweighting is needed
        if not self._should_reweight() and self.current_weights:
            return self.current_weights

        # 2. Collect recent metrics by model type
        recent_metrics = self._collect_recent_metrics(lookback_days=7)

        if not recent_metrics:
            self.logger.warning(f"No recent metrics. Using baseline weights for {regime}")
            self.current_weights = self.baseline_weights.get(regime, {})
            return self.current_weights

        # 3. Group by model type
        metrics_by_type = self._group_metrics_by_type(recent_metrics)

        # 4. Compute composite score for each model type
        model_scores = {}
        for model_type, metrics in metrics_by_type.items():
            model_scores[model_type] = self._compute_model_score(model_type, metrics)

        # 5. Convert scores to weights (softmax)
        computed_weights = self._scores_to_weights(model_scores)

        # 6. Smooth against baseline (70% live, 30% baseline)
        smoothed_weights = self._smooth_weights_against_baseline(computed_weights, regime)

        # 7. Record and return
        self.current_weights = smoothed_weights
        self.last_reweight_time = datetime.now()

        # Store in history
        self._store_weights_history(regime, smoothed_weights, model_scores, recent_metrics)

        self.logger.info(f"🔄 Ensemble weights recomputed for {regime}:")
        for model_type, weight in sorted(smoothed_weights.items(), key=lambda x: x[1], reverse=True):
            self.logger.info(f"  {model_type}: {weight:.1%}")

        return smoothed_weights

    def _normalize_sharpe(self, sharpe: float) -> float:
        """
        Normalize Sharpe ratio to [0, 1] range

        Sharpe < 0 -> 0
        Sharpe = 0.5 -> 0.3
        Sharpe = 1.0 -> 0.6
        Sharpe = 2.0 -> 1.0
        """
        if sharpe < 0:
            return 0.0
        # Sigmoid-like transformation
        return min(1.0, sharpe / 2.0)

    def get_weighted_ensemble_prediction(self,
                                        predictions_dict: dict[str, float],
                                        regime: str) -> tuple[float, dict[str, float]]:
        """
        Generate weighted ensemble prediction

        Returns:
            (ensemble_prediction, active_weights)
        """
        weights = self.compute_ensemble_weights(regime)

        ensemble_pred = 0.0
        used_weights = {}

        for model_type, prediction in predictions_dict.items():
            weight = weights.get(model_type, 0.0)
            if weight > 0.001:
                try:
                    pred_value = float(prediction)
                    ensemble_pred += weight * pred_value
                    used_weights[model_type] = weight
                except (TypeError, ValueError):
                    self.logger.warning(f"Invalid prediction for {model_type}: {prediction}")

        # Renormalise the weights AND the prediction they produced.
        #
        # Only the weights were renormalised here, and the prediction was left
        # as the raw sum of `weight * value`. When a model drops out -- no
        # prediction this bar, or a weight below the 0.001 floor -- the
        # surviving weights no longer sum to 1, so the returned number is that
        # fraction of what it should be. A 25% model missing turns every
        # forecast into 0.75x its own magnitude.
        #
        # For a regression signal that is a shrinkage toward zero, and
        # KellyCriterion sizes on magnitude, so positions got smaller whenever
        # a model was unavailable -- an availability accident expressed as
        # reduced conviction.
        if used_weights:
            total = sum(used_weights.values())
            if total > 0:
                ensemble_pred /= total
                used_weights = {k: v / total for k, v in used_weights.items()}
        else:
            # Nothing participated. The number below is 0.0, which downstream
            # cannot tell from "the models agree on no move" -- the same
            # confusion that made the regime ensemble return a permanent HOLD.
            # The empty weights dict IS the signal, and the caller is now
            # required to check it; this makes the reason findable.
            self.logger.error(
                "Live ensemble recognised none of %d prediction(s) for regime "
                "'%s'. Returning 0.0 with EMPTY weights: callers must treat an "
                "empty contribution map as absence, not as agreement on zero.",
                len(predictions_dict), regime,
            )

        return ensemble_pred, used_weights

    def get_ensemble_report(self) -> dict:
        """Generate detailed ensemble report"""
        if not self.ensemble_weights_history:
            return {'status': 'no_data'}

        last_ew = self.ensemble_weights_history[-1]

        return {
            'current_regime': last_ew.regime,
            'last_reweight': last_ew.timestamp.isoformat(),
            'weights': last_ew.weights,
            'model_scores': last_ew.model_scores,
            'reasoning': last_ew.reasoning,
            'weights_history_length': len(self.ensemble_weights_history),
            'metrics_history_length': len(self.model_metrics_history)
        }
