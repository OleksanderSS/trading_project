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
    #: Id of the DeanAction the critic scored, so the eventual realised PnL can
    #: be fed back to `DeanBootstrapSystem.calculate_reward` for THIS decision.
    #: None when no critic ran.
    critic_action_id: str | None = None
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
        self._ensure_critic_registered()
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
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                self.logger.error(
                    f'Failed to load Meta-model at {meta_model_path}: {e}')
        else:
            self.logger.warning(
                f'Meta-model not found at {meta_model_path}. Falling back to live-adaptive ensembling.'
                )

    def _ensure_critic_registered(self) ->None:
        """Register a DeanCritic so the critic filter actually has a critic.

        `register_model()` was never called anywhere in this codebase, so
        `_apply_critic_filter` always hit the "no critic" path and silently
        no-opped on every consensus decision since the day it was written.

        The critic is registered UNFITTED on purpose: its rule-based terms
        (volatility, anomaly score, regime-vs-direction, paradoxical
        confidence) work immediately, while `DeanCritic.predict` returns zeros
        until someone trains its meta-model on historical
        (features, actual, actor-prediction) triples. That term simply
        contributes nothing until then -- no silent failure either way.
        """
        try:
            from src.meta_learning.dean_trading_models import DeanCritic
            from src.models.dean.dean_bootstrap_system import ModelRole

            already = any(
                m.get('role') == ModelRole.CRITIC
                for m in self.dean_system.models.values()
            )
            if already:
                return

            rules = {
                'high_vol_threshold': self.config_manager.get(
                    'strategy.risk_management.high_volatility_threshold', 0.05),
                'anomaly_threshold': self.config_manager.get(
                    'strategy.risk_management.anomaly_threshold', 0.8),
            }
            self.dean_system.register_model(
                'dean_critic', ModelRole.CRITIC, DeanCritic(rules_config=rules)
            )
            self.logger.info(
                '[CONSENSUS] DEAN Critic registered (rule-based; meta-model '
                'inactive until trained).'
            )
        except (ImportError, ValueError, TypeError, AttributeError, KeyError) as e:
            self.logger.warning(f'Could not register DEAN Critic: {e}')

    def generate_consensus(self, model_predictions: dict[str, float],
        context_data: dict[str, Any], knn_results: dict[str, Any] | None
        =None, features: pd.DataFrame | None=None) ->ConsensusReport:
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
            if model_predictions and not contributions:
                # An empty contribution map means NO model was recognised, and
                # the score beside it is 0.0 by construction. Reading that as
                # "the models agree on no move" is what let the regime
                # ensemble issue a permanent HOLD; falling back to the
                # weighted aggregation at least uses the predictions that
                # exist rather than acting on an absence.
                self.logger.error(
                    '[CONSENSUS] Live-Adaptive ensemble recognised none of %d '
                    'prediction(s) for regime %s. Falling back to weighted '
                    'aggregation rather than treating 0.0 as agreement.',
                    len(model_predictions), regime)
                raw_score, contributions = self._predict_with_weighted_aggregation(
                    model_predictions, fingerprint)
            else:
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
        critic_context = dict(context_data)
        critic_context.setdefault('confidence', abs(normalized_score))
        final_signal, critic_score, blocked_by_critic, critic_action_id = (
            self._apply_critic_filter(initial_signal, critic_context, features))
        report = ConsensusReport(final_signal=final_signal, raw_score=
            raw_score, confidence=abs(normalized_score), market_regime=
            regime, context_fingerprint=fingerprint, model_contributions=
            contributions, knn_adjustment=knn_adjustment, critic_score=
            critic_score, blocked_by_critic=blocked_by_critic,
            critic_action_id=critic_action_id)
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
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
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
        str, Any], features: pd.DataFrame | None=None
        ) ->tuple[str, float, bool, str | None]:
        """Apply DEAN critic and Anomaly hard-block to potentially block risky decisions.

        Calls `critique_existing_action`, not `bootstrap_action_critique`: the
        signal is already decided by the time we get here, so the consensus is
        the actor. The old call passed `context_data` (a flat 7-key dict) into
        a path that generated an action from a registered DeanActor expecting a
        feature DataFrame, which raised TypeError on every invocation and was
        swallowed below as "critic unavailable" — the filter had never once run.
        """
        final_signal = initial_signal
        blocked_by_critic = False
        critic_score = 0.0
        action_id: str | None = None
        try:
            confidence = float(context_data.get('confidence', 0.0) or 0.0)
            action, critique = self.dean_system.critique_existing_action(
                action_type='buy' if initial_signal == 'BUY' else
                'sell' if initial_signal == 'SELL' else 'hold',
                confidence=confidence,
                context=context_data,
                features=features,
            )
            critic_score = critique.critique_score
            action_id = action.action_id
            if critique.critique_score < 0 and initial_signal != 'HOLD':
                self.logger.warning(
                    f'[CONSENSUS] DEAN Critic blocked {initial_signal}. '
                    f'Score: {critique.critique_score:.3f}. '
                    f'Reasons: {"; ".join(critique.critique_points) or "n/a"}'
                    )
                final_signal = 'HOLD'
                blocked_by_critic = True
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(f'DEAN Critic unavailable, skipping filter: {e}')
            critic_score = 0.0
        anomaly_score = context_data.get('anomaly_score', 0.0)
        anomaly_threshold = self.config_manager.get(
            'strategy.risk_management.anomaly_threshold', 0.8)
        if anomaly_score >= anomaly_threshold and initial_signal != 'HOLD':
            self.logger.warning(
                f'[CONSENSUS] ANOMALY BLOCK: score {anomaly_score:.2f} >= {anomaly_threshold}. Blocking {initial_signal}.'
                )
            final_signal = 'HOLD'
            blocked_by_critic = True
        return final_signal, critic_score, blocked_by_critic, action_id

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
    """Refined ensembling logic focusing on regime-dependent sensitivity.

    Extends ConsensusEngine by inserting a regime-aware weighted ensemble
    step *before* the standard KNN-adjustment + critic-filter pipeline.
    When the market context carries predictions_by_model data, the ensemble
    score replaces the raw weighted aggregation of the base class.
    """

    def __init__(self):
        """Initializes EnhancedConsensusEngine with regime detection capabilities."""
        from src.analytics.detectors.regime_detector import MarketRegimeDetector
        # Provide minimal stubs so the parent ConsensusEngine initializes safely.
        class _NoopDiary:
            def get_contextual_model_weights(self, *a, **kw):
                return {}
        class _NoopThreshold:
            def analyze(self, *a, **kw):
                return {}
        super().__init__(
            experience_diary=_NoopDiary(),
            threshold_analyzer=_NoopThreshold(),
        )
        self.regime_detector = MarketRegimeDetector()
        self.logger = ProjectLogger.get_logger('EnhancedConsensusEngine')
        # regime_weights now covers all four live regimes including trending_down.
        # A strong downtrend favours models that capture momentum (lstm/cnn)
        # and penalises trend-following ones (transformer).
        self.regime_weights = {
            'trending_up': {
                'transformer': 0.35, 'lstm': 0.25, 'cnn': 0.20,
                'linear': 0.10, 'catboost': 0.10,
            },
            'trending_down': {
                'lstm': 0.30, 'cnn': 0.25, 'catboost': 0.20,
                'linear': 0.15, 'transformer': 0.10,
            },
            'ranging': {
                'linear': 0.30, 'catboost': 0.25, 'knn': 0.20,
                'transformer': 0.15, 'lstm': 0.10,
            },
            'volatile': {
                'cnn': 0.30, 'transformer': 0.25, 'lstm': 0.20,
                'linear': 0.15, 'catboost': 0.10,
            },
        }

    def _determine_regime(self, market_context: dict[str, Any]) -> str:
        """Identify current market regime from context signals.

        Returns one of: 'trending_up', 'trending_down', 'volatile', 'ranging'.
        """
        try:
            volatility = market_context.get('volatility', 0.01)
            trend = market_context.get('trend', 0.0)
            if volatility > 0.03:
                return 'volatile'
            if abs(trend) > 0.5:
                # Differentiate bullish vs bearish trends
                return 'trending_up' if trend > 0 else 'trending_down'
            return 'ranging'
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            self.logger.warning(
                f'Market regime determination failed: {e}. Defaulting to ranging.')
            return 'ranging'

    def generate_consensus(
        self,
        model_predictions: dict[str, float],
        context_data: dict[str, Any],
        knn_results: dict[str, Any] | None = None,
    ) -> ConsensusReport:
        """Override: inject regime-aware weighted ensemble before the
        standard KNN + critic pipeline.

        If predictions_by_model is available in context_data (as populated
        by Stage 5), use generate_weighted_ensemble to compute a more
        accurate raw_score that reflects the current regime.  Fall back to
        the base class implementation when no model-level predictions exist.
        """
        predictions_by_model: dict[str, float] = (
            context_data.get('predictions_by_model') or model_predictions or {}
        )

        if predictions_by_model:
            ensemble_result = self.generate_weighted_ensemble(
                predictions_by_model, context_data
            )
            regime_score = ensemble_result['ensemble_prediction']
            regime = ensemble_result['regime']
            # Build a single-key dict so the parent aggregation logic still
            # accounts for contributions correctly.
            model_predictions_for_parent = {'enhanced_ensemble': regime_score}
            context_data = dict(context_data)
            context_data['regime'] = regime
            self.logger.info(
                f"[ENHANCED] regime={regime}, "
                f"ensemble_score={regime_score:.4f}, "
                f"architectures={ensemble_result['participating_architectures']}"
            )
        else:
            model_predictions_for_parent = model_predictions

        return super().generate_consensus(
            model_predictions=model_predictions_for_parent,
            context_data=context_data,
            knn_results=knn_results,
        )

    def generate_weighted_ensemble(self, predictions_dict: dict[str, float],
        market_context: dict[str, Any]) ->dict[str, Any]:
        """Generates a weighted ensemble score based on active market regime."""
        regime = self._determine_regime(market_context)
        weights = self.regime_weights.get(regime, self.regime_weights['ranging'])
        ensemble_score = 0.0
        total_weight = 0.0
        unmatched: list[str] = []
        for raw_name, arch_pred in predictions_dict.items():
            # `predictions_dict` is keyed by the model IDs the system actually
            # produces -- 'LGBM_5m', 'Transformer_v1' -- while `regime_weights`
            # lists bare architecture names. A direct lookup therefore misses
            # everything, every weight is 0.0, and the engine returns an
            # ensemble score of exactly 0.0 for every regime-aware call.
            #
            # Worse, three of the five names it does list (transformer, lstm,
            # cnn) were moved to the archive and are not produced at all any
            # more, so even an exact match could not save it.
            arch_type = self._architecture_of(raw_name, weights)
            weight = weights.get(arch_type, 0.0) if arch_type else 0.0
            if not weight:
                unmatched.append(str(raw_name))
            if weight > 0:
                try:
                    score_val = float(arch_pred)
                    ensemble_score += weight * score_val
                    total_weight += weight
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self.logger.debug(
                            f'Prediction from architecture {arch_type} is non-numeric: {e}'
                        )
                    raise
        if total_weight > 0:
            ensemble_score = ensemble_score / total_weight
        elif predictions_dict:
            # Nothing participated. Returning 0.0 here makes "the models agree
            # on no move" and "no model was recognised" the same number, and
            # downstream reads both as HOLD -- so a silenced ensemble looks
            # exactly like a calm one, forever.
            self.logger.error(
                "Regime ensemble recognised NONE of %d predictions (%s) against "
                "the weights for regime '%s' (%s). Returning no prediction "
                "rather than 0.0, which downstream cannot tell from agreement.",
                len(predictions_dict), ', '.join(unmatched[:6]), regime,
                ', '.join(sorted(weights)),
            )
            return {
                'ensemble_prediction': None,
                'regime': regime,
                'active_weights': weights,
                'participating_architectures': [],
                'unmatched_models': unmatched,
                'status': 'no_recognised_architecture',
            }
        if unmatched:
            self.logger.warning(
                "Regime ensemble ignored %d prediction(s) with no known "
                "architecture: %s. A new model type is silenced rather than "
                "weighted until it is added to regime_weights.",
                len(unmatched), ', '.join(unmatched[:6]),
            )
        return {
            'ensemble_prediction': ensemble_score,
            'regime': regime,
            'active_weights': weights,
            'participating_architectures': [
                arch for arch, w in weights.items() if w > 0
            ],
            'unmatched_models': unmatched,
        }

    @staticmethod
    def _architecture_of(model_name: str, weights: dict) -> str | None:
        """Infer the architecture category from a real model id.

        Models arrive named as the system builds them -- 'LGBM_5m',
        'CatBoost_AAPL_1d', 'Transformer_v1' -- and the regime weights are
        keyed by bare architecture. Matching them by equality silently drops
        every model, which is how this engine came to return 0.0 for every
        regime-aware ensemble it has ever computed.

        Longest match first, so 'lightgbm' is not shadowed by a shorter key
        that happens to be a substring of it.
        """
        if not model_name:
            return None
        name = str(model_name).lower()
        aliases = {
            'lgbm': 'lightgbm', 'lgb': 'lightgbm',
            'xgb': 'xgboost', 'rf': 'random_forest',
            'gru': 'lstm',
        }
        for key in sorted(weights, key=len, reverse=True):
            if key.lower() in name:
                return key
        for alias, canonical in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
            if alias in name and canonical in weights:
                return canonical
        return None
