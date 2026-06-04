"""
Consensus Engine - The decision core of the DEAN trading system.
Aggregates predictions from multiple heterogeneous models using an ensemble meta-model
or regime-aware weighted averaging.
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
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
    final_signal: str
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

    def __init__(self, experience_diary: Any, threshold_analyzer: Any,
        config_manager: Any | None=None, meta_model_path: str | None=
        None, live_ensemble: Any | None=None):
        """Initializes the ConsensusEngine with its required dependencies."""
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.diary = experience_diary
        self.threshold_analyzer = threshold_analyzer
        self.dean_system = get_dean_system()
        self.live_ensemble = live_ensemble
        if meta_model_path is None:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    'Meta-model path not provided. Checking configuration defaults.'
                    )
            meta_model_path = 'data/trained_models/consensus_meta_model.pkl'
        self.meta_model = None
        if Path(meta_model_path).exists():
            try:
                self.meta_model = StackedEnsemble.load(meta_model_path)
                self.logger.info(
                    f'Meta-model successfully synchronized from {meta_model_path}'
                    )
            except Exception as e:
                self.logger.error(
                    f'Failed to load Meta-model at {meta_model_path}: {e}')
        else:
            self.logger.warning(
                f'Meta-model not found at {meta_model_path}. Falling back to live-adaptive ensembling.'
                )

    def generate_consensus(self, model_predictions: dict[str, float],
        context_data: dict[str, Any], knn_results: dict[str, Any] | None
        =None) ->ConsensusReport:
        """
        Processes predictions from all architectures to reach a single unified trade decision.
        """
        fingerprint = context_data.get('fingerprint', '0|0|0')
        regime = context_data.get('regime', 'neutral')
        if self.meta_model and self.meta_model.is_trained:
            raw_score, contributions = self._predict_with_meta_model(
                model_predictions, context_data, regime)
        elif self.live_ensemble:
            raw_score, contributions = (self.live_ensemble.
                get_weighted_ensemble_prediction(model_predictions, regime))
            self.logger.info(
                f'[CONSENSUS] Using Live-Adaptive weights for {regime}')
        else:
            raw_score, contributions = self._predict_with_weighted_aggregation(
                model_predictions, fingerprint)
        raw_score, knn_adjustment = self._apply_knn_adjustment(raw_score,
            knn_results)
        min_confidence = self._get_min_confidence_threshold(context_data)
        normalized_score = self._normalize_score(raw_score)
        signal_threshold = self._calculate_signal_threshold(min_confidence)
        self._log_consensus_inference(raw_score, normalized_score,
            signal_threshold, context_data)
        initial_signal = self._determine_initial_signal(normalized_score,
            signal_threshold)
        final_signal, critic_score, blocked_by_critic = (self.
            _apply_critic_filter(initial_signal, context_data))
        report = ConsensusReport(final_signal=final_signal, raw_score=
            raw_score, confidence=abs(normalized_score), market_regime=
            regime, context_fingerprint=fingerprint, model_contributions=
            contributions, knn_adjustment=knn_adjustment, critic_score=
            critic_score, blocked_by_critic=blocked_by_critic)
        return report

    def _predict_with_meta_model(self, model_predictions: dict[str, float],
        context_data: dict[str, Any], regime: str) ->tuple[float, dict[str,
        float]]:
        """Predict consensus score using the trained meta-model."""
        if self.meta_model is None:
            raise RuntimeError(
                'Meta-model must be loaded before calling _predict_with_meta_model'
                )
        meta_model = self.meta_model
        predictions_df = pd.DataFrame([model_predictions])
        predictions_df = predictions_df.reindex(columns=meta_model.
            feature_names, fill_value=0.0)
        ensemble_result = meta_model.predict(predictions_df, context_params
            ={'ticker': context_data.get('ticker', 'any'), 'tf':
            context_data.get('tf', 'any'), 'regime': regime})
        raw_score = ensemble_result.final_signal[0]
        contributions = ensemble_result.active_weights
        return raw_score, contributions

    def _predict_with_weighted_aggregation(self, model_predictions: dict[
        str, float], fingerprint: str) ->tuple[float, dict[str, float]]:
        """Predict consensus score using contextual weighted averaging."""
        weights = self.diary.get_contextual_model_weights(fingerprint)
        weighted_sum = 0.0
        total_weight = 0.0
        contributions: dict[str, float] = {}
        for model_id, pred in model_predictions.items():
            pred_value = self._safe_normalize_prediction(model_id, pred)
            w = weights.get(model_id, 1.0)
            weighted_sum += pred_value * w
            total_weight += w
            contributions[model_id] = pred_value * w
        raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        return raw_score, contributions

    def _safe_normalize_prediction(self, model_id: str, pred: float) ->float:
        """Normalize a prediction with safe fallback."""
        try:
            return normalize_prediction(pred)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'Normalization failed for {model_id}: {e}. Defaulting to 0.0')
            return 0.0

    def _apply_knn_adjustment(self, raw_score: float, knn_results: dict[str, Any] | None) ->tuple[float, float]:
        """Adjust raw score using KNN reversal probability if available."""
        knn_adjustment = 1.0
        if not knn_results or 'reversal_probability' not in knn_results:
            return raw_score, knn_adjustment
        reversal_prob = knn_results['reversal_probability']
        knn_adjustment = 1.0 - reversal_prob
        return raw_score * knn_adjustment, knn_adjustment

    def _get_min_confidence_threshold(self, context_data: dict[str, Any]
        ) ->float:
        """Get minimum confidence threshold from AdaptiveConfidenceAnalyzer."""
        threshold_report = self.threshold_analyzer.analyze(context_data)
        if not threshold_report:
            return 0.5
        return float(threshold_report.get('adaptive_confidence_threshold', 0.5)
            )

    def _normalize_score(self, raw_score: float) ->float:
        """Standardize raw score into [-1, 1] range."""
        if abs(raw_score) < 1e-09:
            return 0.0
        if 0.0 <= raw_score <= 1.0:
            return (raw_score - 0.5) * 2.0
        return max(-1.0, min(1.0, raw_score))

    def _calculate_signal_threshold(self, min_confidence: float) ->float:
        """Convert confidence threshold into normalized score space."""
        signal_threshold = (min_confidence - 0.5
            ) * 2.0 if 0.0 < min_confidence < 1.0 else min_confidence
        return max(0.01, abs(signal_threshold))

    def _log_consensus_inference(self, raw_score: float, normalized_score:
        float, signal_threshold: float, context_data: dict[str, Any]) ->None:
        """Log decision diagnostics."""
        self.logger.info(
            f"[CONSENSUS] Inference: raw={raw_score:.4f} → normalized={normalized_score:.4f}, threshold={signal_threshold:.4f}, asset={context_data.get('ticker')}"
            )

    def _determine_initial_signal(self, normalized_score: float,
        signal_threshold: float) ->str:
        """Determine BUY/SELL/HOLD based on normalized score and threshold."""
        if normalized_score > signal_threshold:
            return 'BUY'
        if normalized_score < -signal_threshold:
            return 'SELL'
        return 'HOLD'

    def _apply_critic_filter(self, initial_signal: str, context_data: dict[
        str, Any]) ->tuple[str, float, bool]:
        """Apply DEAN critic and Anomaly hard-block to potentially block risky decisions."""
        final_signal = initial_signal
        blocked_by_critic = False
        critic_score = 0.0
        try:
            _, critique = self.dean_system.bootstrap_action_critique(
                context_data)
            critic_score = critique.critique_score
            if critique.critique_score < 0 and initial_signal != 'HOLD':
                self.logger.warning(
                    f'[CONSENSUS] DEAN Critic blocked {initial_signal}. Critique Score: {critique.critique_score}'
                    )
                final_signal = 'HOLD'
                blocked_by_critic = True
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            critic_score = 0.0
            raise
        anomaly_score = context_data.get('anomaly_score', 0.0)
        anomaly_threshold = self.config_manager.get(
            'strategy.risk_management.anomaly_threshold', 0.8)
        if anomaly_score >= anomaly_threshold and initial_signal != 'HOLD':
            self.logger.warning(
                f'[CONSENSUS] ANOMALY BLOCK: score {anomaly_score:.2f} >= {anomaly_threshold}. Blocking {initial_signal}.'
                )
            final_signal = 'HOLD'
            blocked_by_critic = True
        return final_signal, critic_score, blocked_by_critic

    def get_ensemble_summary(self, reports: list[ConsensusReport]) ->dict[
        str, Any]:
        """Analyzes historical reports to determine architectural leaders in the current regime."""
        if not reports:
            return {}
        leaderboard: dict[str, float] = {}
        for r in reports:
            for arch, contrib in r.model_contributions.items():
                leaderboard[arch] = leaderboard.get(arch, 0.0) + abs(contrib)
        return dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse
            =True))


class EnhancedConsensusEngine(ConsensusEngine):
    """Refined ensembling logic focusing on regime-dependent sensitivity."""

    def __init__(self):
        """Initializes EnhancedConsensusEngine with regime detection capabilities."""
        from src.algorithms.regime_detector import MarketRegimeDetector
        self.regime_detector = MarketRegimeDetector()
        self.logger = ProjectLogger.get_logger('EnhancedConsensusEngine')
        self.regime_weights = {'trending_up': {'transformer': 0.35, 'lstm':
            0.25, 'cnn': 0.2, 'linear': 0.1, 'catboost': 0.1}, 'ranging': {
            'linear': 0.3, 'catboost': 0.25, 'knn': 0.2, 'transformer':
            0.15, 'lstm': 0.1}, 'volatile': {'cnn': 0.3, 'transformer':
            0.25, 'lstm': 0.2, 'linear': 0.15, 'catboost': 0.1}}

    def _determine_regime(self, market_context: dict[str, Any]) ->str:
        """Identifies current market regime using provided technical context."""
        try:
            volatility = market_context.get('volatility', 0.01)
            trend = market_context.get('trend', 0.0)
            if volatility > 0.03:
                return 'volatile'
            elif abs(trend) > 0.5:
                return 'trending_up' if trend > 0 else 'ranging'
            else:
                return 'ranging'
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'Market regime determination failed: {e}. Defaulting to neutral state.'
                )
            return 'ranging'

    def generate_weighted_ensemble(self, predictions_dict: dict[str, float],
        market_context: dict[str, Any]) ->dict[str, Any]:
        """Generates a weighted ensemble score based on active market regime."""
        regime = self._determine_regime(market_context)
        weights = self.regime_weights.get(regime, self.regime_weights[
            'ranging'])
        ensemble_score = 0.0
        total_weight = 0.0
        for arch_type, arch_pred in predictions_dict.items():
            weight = weights.get(arch_type, 0.0)
            if weight > 0:
                try:
                    score_val = float(arch_pred)
                    ensemble_score += weight * score_val
                    total_weight += weight
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f'Prediction from architecture {arch_type} is non-numeric: {e}'
                            )
                    raise
        if total_weight > 0:
            ensemble_score = ensemble_score / total_weight
        return {'ensemble_prediction': ensemble_score, 'regime': regime,
            'active_weights': weights, 'participating_architectures': [arch for
            arch, w in weights.items() if w > 0]}
