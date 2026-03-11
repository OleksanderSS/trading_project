import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.core.logging.logger import ProjectLogger
from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine
from src.analytics.analyzers.adaptive_confidence_analyzer import AdaptiveThresholdsAnalyzer
from src.models.dean.dean_bootstrap_system import get_dean_system
from src.ensembling.ensemble import StackedEnsemble

@dataclass
class ConsensusReport:
    """Detailed breakdown of the decision-making process for transparency."""
    final_signal: str  # BUY, SELL, HOLD
    raw_score: float
    confidence: float
    market_regime: str
    context_fingerprint: str
    model_contributions: Dict[str, float]
    knn_adjustment: float
    critic_score: float
    blocked_by_critic: bool
    timestamp: datetime = field(default_factory=datetime.now)

class ConsensusEngine:
    """
    The central decision node of DEAN. Aggregates predictions using a trained 
    meta-model, cross-references with historical KNN patterns, and applies Critic risk filters.
    """

    def __init__(self, 
                 experience_diary: ExperienceDiaryEngine,
                 threshold_analyzer: AdaptiveThresholdsAnalyzer,
                 config_manager: Optional[Any] = None,
                 meta_model_path: Optional[str] = None):
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.diary = experience_diary
        self.threshold_analyzer = threshold_analyzer
        self.dean_system = get_dean_system()
        
        # Resolve meta_model_path from config if not provided
        if meta_model_path is None and config_manager is not None:
            meta_model_path = config_manager.get('paths.meta_model', "src/trained_models/consensus_meta_model.pkl")
        elif meta_model_path is None:
            meta_model_path = "src/trained_models/consensus_meta_model.pkl"

        # --- 2. LOAD THE TRAINED META-MODEL ---
        self.meta_model = None
        if Path(meta_model_path).exists():
            try:
                self.meta_model = StackedEnsemble.load(meta_model_path)
                self.logger.info(f"Successfully loaded trained meta-model from {meta_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load meta-model from {meta_model_path}: {e}", exc_info=True)
        else:
            self.logger.warning(f"Meta-model not found at {meta_model_path}. "
                                f"ConsensusEngine will fall back to simple weighted averaging.")
        # -----------------------------------------

    def generate_consensus(self, 
                           model_predictions: Dict[str, float], 
                           context_data: Dict[str, Any],
                           knn_results: Optional[Dict[str, Any]] = None) -> ConsensusReport:
        """
        Processes predictions from all architectures to reach a single unified decision.
        """
        fingerprint = context_data.get('fingerprint', '0|0|0')
        regime = context_data.get('regime', 'neutral')
        
        raw_score = 0.0
        contributions = {}

        # --- 3. USE META-MODEL FOR PREDICTION ---
        if self.meta_model and self.meta_model.is_trained:
            # Convert single-step predictions to a DataFrame row
            predictions_df = pd.DataFrame([model_predictions])
            
            # Ensure columns match the order the model was trained on
            # Missing predictions will be filled with NaN, which the model should handle or we can fill with 0
            predictions_df = predictions_df.reindex(columns=self.meta_model.feature_names, fill_value=0.0)

            # The predict method from StackedEnsemble now includes live efficiency weighting
            ensemble_result = self.meta_model.predict(predictions_df, context_params={
                'ticker': context_data.get('ticker', 'any'),
                'tf': context_data.get('tf', 'any'),
                'regime': regime
            })
            
            raw_score = ensemble_result.final_signal[0]
            contributions = ensemble_result.active_weights

        else:
            # --- 4. FALLBACK TO MANUAL AGGREGATION ---
            self.logger.debug("Using fallback manual aggregation.")
            weights = self.diary.get_contextual_model_weights(fingerprint)
            weighted_sum = 0.0
            total_weight = 0.0

            for model_id, pred in model_predictions.items():
                w = weights.get(model_id, 1.0)
                weighted_sum += pred * w
                total_weight += w
                contributions[model_id] = pred * w

            raw_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        # -----------------------------------------

        # --- 5. ALL SUBSEQUENT STEPS REMAIN UNCHANGED ---
        # They now operate on a more robust `raw_score`
        
        # KNN Pattern Adjustment
        knn_adjustment = 1.0
        if knn_results and 'reversal_probability' in knn_results:
            rev_prob = knn_results['reversal_probability']
            knn_adjustment = 1.0 - rev_prob
            raw_score *= knn_adjustment

        # Adaptive Thresholding
        thresholds = self.threshold_analyzer.analyze(pd.DataFrame([context_data]))
        min_conf = thresholds.get('min_prediction_prob', 0.5)
        
        initial_signal = "HOLD"
        if raw_score > min_conf:
            initial_signal = "BUY"
        elif raw_score < -min_conf:
            initial_signal = "SELL"

        # DEAN Critic Integration
        _, critique = self.dean_system.bootstrap_action_critique(context_data)
        
        final_signal = initial_signal
        blocked_by_critic = False
        
        if critique.critique_score < 0 and initial_signal != "HOLD":
            self.logger.warning(f"[CONSENSUS] Critic blocked {initial_signal}. Score: {critique.critique_score}")
            final_signal = "HOLD"
            blocked_by_critic = True

        # Create Report
        report = ConsensusReport(
            final_signal=final_signal,
            raw_score=raw_score,
            confidence=abs(raw_score),
            market_regime=regime,
            context_fingerprint=fingerprint,
            model_contributions=contributions,
            knn_adjustment=knn_adjustment,
            critic_score=critique.critique_score,
            blocked_by_critic=blocked_by_critic
        )

        self._log_consensus_to_diary(report, context_data)

        return report

    def _log_consensus_to_diary(self, report: ConsensusReport, context: Dict[str, Any]):
        """Records the entire decision matrix for 'The Critic' to study later."""
        try:
            self.diary.record_decision_metadata({
                'timestamp': report.timestamp.isoformat(),
                'fingerprint': report.context_fingerprint,
                'regime': report.market_regime,
                'final_signal': report.final_signal,
                'raw_score': report.raw_score,
                'critic_score': report.critic_score,
                'model_weights': list(report.model_contributions.keys()),
                'blocked': report.blocked_by_critic
            })
        except Exception as e:
            self.logger.error(f"Failed to log consensus metadata: {e}")

    def get_ensemble_summary(self, reports: List[ConsensusReport]) -> Dict[str, Any]:
        """Analyzes a series of reports to find which model architectures are currently 'Leading'."""
        if not reports:
            return {}
            
        leaderboard = {}
        for r in reports:
            for model, contrib in r.model_contributions.items():
                leaderboard[model] = leaderboard.get(model, 0) + abs(contrib)
        
        return dict(sorted(leaderboard.items(), key=lambda x: x[1], reverse=True))