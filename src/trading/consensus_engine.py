"""
Consensus Engine - The decision core of the DEAN trading system.
Aggregates predictions from multiple heterogeneous models using an ensemble meta-model
or regime-aware weighted averaging.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger
from src.core.utils.prediction_utils import normalize_prediction
from src.ensembling.stacked_ensemble import StackedEnsemble
from src.models.dean.dean_bootstrap_system import get_dean_system


@dataclass
class ConsensusReport:
    """Detailed breakdown of the decision-making process for transparency and auditing."""
    final_signal: str  # BUY, SELL, HOLD
    raw_score: float
    confidence: float
    market_regime: str
    context_fingerprint: str
    model_contributions: dict[str, float]
    knn_adjustment: float
    critic_score: float
    blocked_by_critic: bool
    timestamp: datetime = field(default_factory=datetime.now)

class ConsensusEngine:
    """
    The central decision node of the architecture.
    Aggregates predictions using a trained meta-model,
    cross-references with historical KNN patterns, and applies Critic risk filters.
    """

    def __init__(self,
                 experience_diary: Any,
                 threshold_analyzer: Any,
                 config_manager: Any | None = None,
                 meta_model_path: str | None = None,
                 live_ensemble: Any | None = None):
        """Initializes the ConsensusEngine with its required dependencies."""
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.diary = experience_diary
        self.threshold_analyzer = threshold_analyzer
        self.dean_system = get_dean_system()
        self.live_ensemble = live_ensemble
        self._lock = RLock()

        # Resolve meta_model_path from configuration
        if meta_model_path is None:
            meta_model_path = "data/trained_models/consensus_meta_model.pkl"

        # Load the trained meta-model
        self.meta_model = None
        if Path(meta_model_path).exists():
            try:
                self.meta_model = StackedEnsemble.load(meta_model_path)
                self.logger.info(f"Meta-model successfully synchronized from {meta_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load Meta-model at {meta_model_path}: {e}")
        else:
            self.logger.warning(f"Meta-model not found at {meta_model_path}. Falling back to variance-weighted ensembling.")

    def generate_consensus(self,
                           model_predictions: dict[str, float],
                           context_data: dict[str, Any],
                           knn_results: dict[str, Any] | None = None) -> ConsensusReport:
        """
        Processes predictions from all architectures to reach a single unified trade decision.
        """
        fingerprint = context_data.get('fingerprint', '0|0|0')
        regime = context_data.get('regime', 'neutral')

        with self._lock:
            # 🛡️ UNIFIED ENSEMBLE LOGIC
            if self.meta_model and self.meta_model.is_trained:
                raw_score, contributions = self._predict_with_meta_model(model_predictions, context_data, regime)
            elif self.live_ensemble:
                # Preferred fallback: Live-Adaptive weighting based on last 7 days
                raw_score, contributions = self.live_ensemble.get_weighted_ensemble_prediction(
                    model_predictions, regime
                )
                self.logger.info(f"[CONSENSUS] Using Live-Adaptive performance-based weights for {regime}")
            else:
                # Emergency fallback: Simple variance-weighted
                self.logger.warning("[CONSENSUS] No Meta-model or LiveEnsemble. Using variance fallback.")
                raw_score, contributions = self._predict_with_variance_weighted_aggregation(model_predictions, fingerprint)

        raw_score, knn_adjustment = self._apply_knn_adjustment(raw_score, knn_results)

        min_confidence = self._get_min_confidence_threshold(context_data)
        normalized_score = self._normalize_score(raw_score)
        signal_threshold = self._calculate_signal_threshold(min_confidence)

        self._log_consensus_inference(raw_score, normalized_score, signal_threshold, context_data)

        initial_signal = self._determine_initial_signal(normalized_score, signal_threshold)
        final_signal, critic_score, blocked_by_critic = self._apply_critic_filter(initial_signal, context_data)

        return ConsensusReport(
            final_signal=final_signal,
            raw_score=raw_score,
            confidence=abs(normalized_score),
            market_regime=regime,
            context_fingerprint=fingerprint,
            model_contributions=contributions,
            knn_adjustment=knn_adjustment,
            critic_score=critic_score,
            blocked_by_critic=blocked_by_critic,
        )

    def _predict_with_meta_model(self, model_predictions, context_data, regime):
        """Predict consensus score using the trained meta-model."""
        if self.meta_model is None:
            raise RuntimeError("Meta-model must be loaded")

        predictions_df = pd.DataFrame([model_predictions])
        predictions_df = predictions_df.reindex(columns=self.meta_model.feature_names, fill_value=0.0)

        ensemble_result = self.meta_model.predict(
            predictions_df,
            context_params={'ticker': context_data.get('ticker', 'any'), 'tf': context_data.get('tf', 'any'), 'regime': regime},
        )

        return ensemble_result.final_signal[0], ensemble_result.active_weights

    def _predict_with_variance_weighted_aggregation(self, model_predictions, fingerprint):
        """Predict consensus score using Variance-Weighted Aggregation."""
        weights = self.diary.get_contextual_model_weights(fingerprint)

        weighted_sum = 0.0
        total_weight = 0.0
        contributions: dict[str, float] = {}

        for model_id, pred in model_predictions.items():
            pred_value = self._safe_normalize_prediction(model_id, pred)

            # ✅ Regularized variance shrinkage (Ridge weights) to prevent numerical explosions during low-volatility/consolidating regimes
            variance_shrinkage = 0.02  # baseline regularizer (2% variance cushion)
            var = weights.get(model_id, -1.0) if isinstance(weights, dict) else -1.0
            w = 1.0 / (var + variance_shrinkage) if var >= 0 else 1.0

            weighted_sum += pred_value * w
            total_weight += w
            contributions[model_id] = pred_value * w

        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return raw_score, contributions

    def _safe_normalize_prediction(self, model_id: str, pred: float) -> float:
        """Normalize a prediction with safe fallback."""
        try:
            return normalize_prediction(pred)
        except Exception:
            return 0.0

    def _apply_knn_adjustment(self, raw_score: float, knn_results: dict[str, Any] | None) -> tuple[float, float]:
        """Adjust raw score using KNN reversal probability if available."""
        knn_adjustment = 1.0
        if knn_results and 'reversal_probability' in knn_results:
            knn_adjustment = 1.0 - knn_results['reversal_probability']
        return raw_score * knn_adjustment, knn_adjustment

    def _get_min_confidence_threshold(self, context_data: dict[str, Any]) -> float:
        """Get minimum confidence threshold."""
        threshold_report = self.threshold_analyzer.analyze(context_data)
        return float(threshold_report.get('adaptive_confidence_threshold', 0.5)) if threshold_report else 0.5

    def _normalize_score(self, raw_score: float) -> float:
        """Standardize raw score into [-1, 1] range."""
        if abs(raw_score) < 1e-9: return 0.0
        if 0.0 <= raw_score <= 1.0: return (raw_score - 0.5) * 2.0
        return max(-1.0, min(1.0, raw_score))

    def _calculate_signal_threshold(self, min_confidence: float) -> float:
        """Convert confidence threshold into normalized score space."""
        signal_threshold = (min_confidence - 0.5) * 2.0 if 0.0 < min_confidence < 1.0 else min_confidence
        return max(0.01, abs(signal_threshold))

    def _log_consensus_inference(self, raw_score, normalized_score, signal_threshold, context_data):
        self.logger.info(f"[CONSENSUS] raw={raw_score:.4f} → normalized={normalized_score:.4f}, threshold={signal_threshold:.4f}, asset={context_data.get('ticker')}")

    def _determine_initial_signal(self, normalized_score: float, signal_threshold: float) -> str:
        """Determine BUY/SELL/HOLD."""
        if normalized_score > signal_threshold: return "BUY"
        if normalized_score < -signal_threshold: return "SELL"
        return "HOLD"

    def _apply_critic_filter(self, initial_signal: str, context_data: dict[str, Any]) -> tuple[str, float, bool]:
        """
        Applies Stanislav Dean's Critic model filter. If the Critic is highly confident and
        opposes the proposed action, the signal is downgraded to HOLD to protect capital.
        """
        import time
        from src.models.dean.dean_bootstrap_system import ModelRole, DeanAction

        critic_score = 1.0
        blocked_by_critic = False
        final_signal = initial_signal

        # Check if the global DEAN system has a registered critic model
        critic_models = [m for m in self.dean_system.models.values() if m['role'] == ModelRole.CRITIC]
        if critic_models:
            try:
                critic = critic_models[0]['instance']
                # Critique the proposed initial_signal
                action_param = {'signal': initial_signal}
                action = DeanAction(
                    action_id=f"act_eval_{int(time.time() * 1000)}",
                    action_type="trade_signal",
                    parameters=action_param,
                    confidence=context_data.get('confidence', 0.8),
                    timestamp=datetime.now(),
                    context=context_data
                )
                critique = self.dean_system._generate_critique(critic, action, context_data)
                critic_score = critique.critique_score
                
                # If critique score is below a risk threshold (e.g. -0.5), we block the trade!
                if critique.critique_score < -0.5:
                    blocked_by_critic = True
                    final_signal = "HOLD"
                    self.logger.warning(f"🛡️ [CRITIC] proposed {initial_signal} blocked by Critic (score={critic_score:.2f})!")
            except Exception as e:
                self.logger.error(f"Error executing Critic filter: {e}. Bypassing.")

        return final_signal, critic_score, blocked_by_critic

class EnhancedConsensusEngine(ConsensusEngine):
    """
    Alias for ConsensusEngine to maintain backward compatibility.
    """
    pass
