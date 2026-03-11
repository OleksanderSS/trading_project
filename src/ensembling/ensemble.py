import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Union, Any, NamedTuple
from logging import getLogger
from sklearn.linear_model import Ridge
import pickle
from pathlib import Path

from src.meta_learning.memory.diary_engine import ExperienceDiaryEngine

logger = getLogger(__name__)

class EnsembleResult(NamedTuple):
    final_signal: np.ndarray
    confidence: np.ndarray
    divergence: np.ndarray
    active_weights: Dict[str, float]
    stats: Optional[Dict[str, Any]]

class StackedEnsemble:
    """
    A meta-model that learns the optimal way to combine base model predictions.
    Uses Ridge regression as the default meta-learner to prevent overfitting.
    Now integrates 'Live Efficiency Weighting' via Meta-Learning.
    """
    def __init__(self, meta_model=None, config_manager=None):
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.config_manager = config_manager
        self.diary_engine = ExperienceDiaryEngine()
        self.is_trained = False
        self.feature_names = []

    def train(self, X: pd.DataFrame, y: pd.Series):
        """Trains the meta-model on base model predictions."""
        self.feature_names = X.columns.tolist()
        self.meta_model.fit(X, y)
        self.is_trained = True
        logger.info(f"[StackedEnsemble] Trained on {len(X)} samples with {len(self.feature_names)} base models.")

    def predict(self, X: pd.DataFrame, context_params: Optional[Dict[str, str]] = None) -> EnsembleResult:
        """
        Generates ensemble predictions using dynamic live efficiency weighting from Experience Diary.
        """
        if not self.is_trained:
            logger.warning("[StackedEnsemble] Model not trained. Returning simple average.")
            simple_avg = X.mean(axis=1).to_numpy()
            return EnsembleResult(
                final_signal=simple_avg,
                confidence=np.ones(len(X)) * 0.5,
                divergence=X.std(axis=1).to_numpy(),
                active_weights={m: 1.0/len(X.columns) for m in X.columns},
                stats={"trained": False}
            )

        context_fingerprint = "unknown"
        if context_params:
            # Construct context lookup: e.g., "AAPL_15m_bull_market"
            context_fingerprint = f"{context_params.get('ticker', 'any')}_{context_params.get('tf', 'any')}_{context_params.get('regime', 'any')}"

        # 1. Retrieve Recent Live Performance from Experience Diary
        live_stats = self.diary_engine.get_recent_performance(
            models=self.feature_names, 
            context=context_fingerprint,
            window=20
        )

        # 2. Dynamically Adjust Weights
        # Get base weights from meta-model (Ridge coefficients)
        base_weights = self.meta_model.coef_
        adjusted_weights = np.array(base_weights, copy=True)
        
        active_weights_map = {}

        for i, model_name in enumerate(self.feature_names):
            perf = live_stats.get(model_name, {})
            accuracy = perf.get('accuracy', 0.5)
            is_champion = perf.get('is_champion', False)
            
            # Logic 3: Weight Penalties and Bonuses
            if accuracy < 0.5:
                adjusted_weights[i] *= 0.5
                logger.debug(f"[StackedEnsemble] Penalizing {model_name}: Accuracy {accuracy:.2%}")
            
            if is_champion:
                adjusted_weights[i] *= 1.5
                logger.debug(f"[StackedEnsemble] Boosting Champion {model_name}")
            
            active_weights_map[model_name] = float(adjusted_weights[i])

        # Normalize adjusted weights
        weight_sum = np.sum(np.abs(adjusted_weights))
        if weight_sum > 0:
            adjusted_weights /= weight_sum

        # 3. Generate Prediction
        preds_matrix = X[self.feature_names].to_numpy()
        final_preds = np.dot(preds_matrix, adjusted_weights)

        # 4. Calculate Divergence and Adjust Confidence
        # Divergence is the standard deviation across model predictions at each step
        divergence = np.std(preds_matrix, axis=1)
        
        # Logic 4: Handle opposite directions with high confidence
        # Simplified: if models disagree strongly (high divergence), reduce overall confidence
        base_confidence = 0.8 # Assume high if models agree
        final_confidence = np.full(len(X), base_confidence)
        
        # Identify extreme disagreement (e.g., some models say +1, others say -1)
        extreme_mask = divergence > 0.7 
        final_confidence[extreme_mask] *= 0.3
        
        if np.any(extreme_mask):
            logger.info(f"[StackedEnsemble] High divergence detected in {np.sum(extreme_mask)} samples. Lowering confidence.")

        return EnsembleResult(
            final_signal=final_preds,
            confidence=final_confidence,
            divergence=divergence,
            active_weights=active_weights_map,
            stats={
                "context": context_fingerprint,
                "n_models": len(self.feature_names),
                "accuracy_penalties": sum(1 for m in self.feature_names if live_stats.get(m, {}).get('accuracy', 0.5) < 0.5)
            }
        )

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

def ensemble_forecast(
    model_predictions: Dict[str, Union[List[float], np.ndarray]],
    model_confidences: Optional[Dict[str, Union[List[float], np.ndarray]]] = None,
    weights: Optional[Dict[str, float]] = None,
    market_regime: Optional[str] = None,
    regime_configs: Optional[Dict[str, Dict[str, float]]] = None,
    normalize_weights: bool = True,
    max_weight: float = 0.8,
    min_weight: float = 0.0,
    divergence_shrinkage: bool = True,
    rolling_window: Optional[int] = None,
    fill_na: float = 0.0,
    method: str = "weighted"
) -> EnsembleResult:
    """
    Generates an advanced ensemble forecast with dynamic regime-based weights
    and confidence-aware averaging.
    """
    if not model_predictions:
        logger.warning("[Ensemble] No model predictions provided.")
        return EnsembleResult(np.array([]), np.array([]), np.array([]), {}, None)

    # 1. Dynamic Regime-based Weight Selection
    active_base_weights = {}
    if market_regime and regime_configs and market_regime in regime_configs:
        active_base_weights = regime_configs[market_regime]
        logger.info(f"[Ensemble] Applying weights for regime: {market_regime}")
    else:
        active_base_weights = weights or {m: 1.0 for m in model_predictions.keys()}
        if market_regime:
             logger.warning(f"[Ensemble] Regime '{market_regime}' not found in config. Using default weights.")

    # Apply constraints
    active_base_weights = {m: max(min_weight, min(w, max_weight)) for m, w in active_base_weights.items()}

    max_len = max((len(v) for v in model_predictions.values()), default=0)
    
    # Align and handle confidence
    aligned_preds = {}
    aligned_conf = {}
    for m in model_predictions.keys():
        p = np.array(model_predictions[m], dtype=float)
        c = np.array(model_confidences[m], dtype=float) if (model_confidences and m in model_confidences) else np.ones(len(p))
        
        if len(p) < max_len:
            p = np.pad(p, (max_len - len(p), 0), 'constant', constant_values=np.nan)
            c = np.pad(c, (max_len - len(c), 0), 'constant', constant_values=0.0)
            
        aligned_preds[m] = p
        aligned_conf[m] = c

    # 2. Confidence-weighted Averaging
    stacked_preds = np.stack(list(aligned_preds.values()))
    stacked_conf = np.stack(list(aligned_conf.values()))
    
    # Calculate effective weights per time step: base_weight * confidence
    base_weight_vec = np.array([active_base_weights.get(m, 1.0) for m in aligned_preds.keys()]).reshape(-1, 1)
    effective_weights = base_weight_vec * stacked_conf
    
    # Normalize effective weights across models at each time step
    weight_sums = np.nansum(effective_weights, axis=0)
    normalized_weights = np.divide(effective_weights, weight_sums, out=np.zeros_like(effective_weights), where=weight_sums != 0)

    # 3. Generate Final Signal
    if method == "median":
        final_signal = np.nanmedian(stacked_preds, axis=0)
    elif method == "mean":
        final_signal = np.nanmean(stacked_preds, axis=0)
    else: # weighted
        final_signal = np.nansum(stacked_preds * normalized_weights, axis=0)

    # 4. Divergence Penalty (Shrinkage)
    divergence = np.nanstd(stacked_preds, axis=0)
    if divergence_shrinkage:
        # Reduce signal strength where models disagree heavily
        penalty = 1.0 / (1.0 + divergence)
        final_signal = final_signal * penalty
        logger.debug("[Ensemble] Applied divergence penalty (shrinkage).")

    # Smoothing
    if rolling_window and rolling_window > 1:
        final_signal = pd.Series(final_signal).ffill().fillna(fill_na).rolling(rolling_window, min_periods=1).mean().to_numpy()

    # Confidence is average weighted confidence
    final_confidence = np.nansum(stacked_conf * normalized_weights, axis=0)

    stats = {
        "n_models": len(model_predictions),
        "avg_divergence": float(np.nanmean(divergence)),
        "regime_applied": market_regime
    }

    return EnsembleResult(
        final_signal=final_signal,
        confidence=final_confidence,
        divergence=divergence,
        active_weights=active_base_weights,
        stats=stats
    )