import logging
from logging import getLogger
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.meta_learning.memory.diary_engine import DiaryEngine
from src.utils.artifact_security import resolve_trusted_artifact_path

logger = getLogger(__name__)

class EnsembleResult(NamedTuple):
    final_signal: np.ndarray
    confidence: np.ndarray
    divergence: np.ndarray
    active_weights: dict[str, float]
    stats: dict[str, Any] | None

class StackedEnsemble:
    """
    A meta-model that learns the optimal way to combine base model predictions.
    Uses Ridge regression as the default meta-learner to prevent overfitting.
    Now integrates 'Live Efficiency Weighting' via Meta-Learning.

    Supports multiple ensemble methods:
    - stacked: Ridge meta-model with live performance weighting
    - weighted_average: Weighted by model metrics (R², RMSE, MAPE)
    - median: Robust to outliers
    - voting: For directional predictions
    """
    def __init__(self, meta_model=None, config_manager=None, method='stacked', weighting_metric='r2'):
        """
        Args:
            meta_model: Meta-learner for stacked method (default: Ridge)
            config_manager: Configuration manager
            method: Ensemble method ('stacked', 'weighted_average', 'median', 'voting')
            weighting_metric: Metric for weighted_average ('r2', 'rmse', 'mape', 'equal')
        """
        self.meta_model = meta_model or Ridge(alpha=1.0)
        self.config_manager = config_manager
        self.method = method
        self.weighting_metric = weighting_metric
        self.diary_engine = DiaryEngine()
        self.is_trained = False
        self.feature_names = []
        self.model_metrics = {}  # Store model metrics for weighted_average method
        logger.info(f"[StackedEnsemble] Initialized with method='{method}', weighting_metric='{weighting_metric}'")

    def train(self, X: pd.DataFrame, y: pd.Series, model_metrics: dict[str, dict[str, float]] | None = None):
        """
        Trains the meta-model on base model predictions.

        Args:
            X: DataFrame with base model predictions as columns
            y: Target values
            model_metrics: Optional dict of model metrics for weighted_average method
                          Format: {'model_name': {'r2': 0.85, 'rmse': 0.02, 'mape': 5.0}}
        """
        self.feature_names = X.columns.tolist()

        if model_metrics:
            self.model_metrics = model_metrics
            logger.info(f"[StackedEnsemble] Stored metrics for {len(model_metrics)} models")

        if self.method == 'stacked':
            self.meta_model.fit(X, y)
            logger.info(f"[StackedEnsemble] Trained meta-model on {len(X)} samples with {len(self.feature_names)} base models.")
        else:
            logger.info(f"[StackedEnsemble] Method '{self.method}' doesn't require meta-model training.")

        self.is_trained = True

    def predict(self, X: pd.DataFrame, context_params: dict[str, str] | None = None) -> EnsembleResult:
        """
        Generates ensemble predictions using selected method.

        Args:
            X: DataFrame with base model predictions
            context_params: Optional context for live performance weighting (stacked method only)

        Returns:
            EnsembleResult with predictions, confidence, divergence, weights, and stats
        """
        if not self.is_trained:
            logger.warning("[StackedEnsemble] Model not trained. Returning simple average.")
            simple_avg = X.mean(axis=1).to_numpy()
            return EnsembleResult(
                final_signal=simple_avg,
                confidence=np.ones(len(X)) * 0.5,
                divergence=X.std(axis=1).to_numpy(),
                active_weights={m: 1.0/len(X.columns) for m in X.columns},
                stats={"trained": False, "method": "fallback"}
            )

        # Route to appropriate method
        if self.method == 'stacked':
            return self._predict_stacked(X, context_params)
        elif self.method == 'weighted_average':
            return self._predict_weighted_average(X)
        elif self.method == 'median':
            return self._predict_median(X)
        elif self.method == 'voting':
            return self._predict_voting(X)
        else:
            logger.warning(f"[StackedEnsemble] Unknown method '{self.method}', using stacked")
            return self._predict_stacked(X, context_params)

    def _predict_stacked(self, X: pd.DataFrame, context_params: dict[str, str] | None = None) -> EnsembleResult:
        """Original stacked ensemble with live performance weighting."""

        context_fingerprint = "unknown"
        if context_params:
            # Construct context lookup: e.g., "AAPL_15m_bull_market"
            context_fingerprint = f"{context_params.get('ticker', 'any')}_{context_params.get('tf', 'any')}_{context_params.get('regime', 'any')}"

        # 1. Retrieve Recent Live Performance from Experience Diary
        contextual_weights = self.diary_engine.get_contextual_model_weights(context_fingerprint)

        # 2. Dynamically Adjust Weights
        # Get base weights from meta-model (Ridge coefficients)
        base_weights = self.meta_model.coef_
        adjusted_weights = np.array(base_weights, copy=True)

        active_weights_map = {}

        for i, model_name in enumerate(self.feature_names):
            # Use contextual weight if available
            contextual_weight = contextual_weights.get(model_name, 1.0)

            # Logic 3: Apply contextual weighting
            adjusted_weights[i] *= contextual_weight

            if contextual_weight < 0.5:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[StackedEnsemble] Penalizing {model_name}: Contextual weight {contextual_weight:.2f}")
            elif contextual_weight > 1.5:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"[StackedEnsemble] Boosting {model_name}: Contextual weight {contextual_weight:.2f}")

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
        # Adaptive Threshold: instead of hardcoded 0.7, use dynamic threshold based on data dispersion
        dynamic_threshold = max(0.5, min(1.5, np.mean(divergence) + 1.5 * np.std(divergence)))
        extreme_mask = divergence > dynamic_threshold
        final_confidence[extreme_mask] *= 0.3

        if np.any(extreme_mask):
            logger.info(f"[StackedEnsemble] High divergence detected (threshold > {dynamic_threshold:.2f}) in {np.sum(extreme_mask)} samples. Lowering confidence.")

        return EnsembleResult(
            final_signal=final_preds,
            confidence=final_confidence,
            divergence=divergence,
            active_weights=active_weights_map,
            stats={
                "method": "stacked",
                "context": context_fingerprint,
                "n_models": len(self.feature_names),
                "contextual_models": len(contextual_weights)
            }
        )

    def _predict_weighted_average(self, X: pd.DataFrame) -> EnsembleResult:
        """Weighted average based on model metrics (R², RMSE, MAPE)."""
        preds_matrix = X[self.feature_names].to_numpy()

        # Calculate weights based on metrics
        if self.weighting_metric == 'r2':
            r2_scores = [max(0, self.model_metrics.get(m, {}).get('r2', 0.5)) for m in self.feature_names]
            total_r2 = sum(r2_scores)
            weights = np.array([r2 / total_r2 if total_r2 > 0 else 1/len(self.feature_names) for r2 in r2_scores])

        elif self.weighting_metric == 'rmse':
            rmse_scores = [self.model_metrics.get(m, {}).get('rmse', 0.1) for m in self.feature_names]
            inverse_rmse = [1 / (rmse + 1e-6) for rmse in rmse_scores]
            total = sum(inverse_rmse)
            weights = np.array([w / total for w in inverse_rmse])

        elif self.weighting_metric == 'mape':
            mape_scores = [self.model_metrics.get(m, {}).get('mape', 10.0) for m in self.feature_names]
            inverse_mape = [1 / (mape + 1e-6) for mape in mape_scores]
            total = sum(inverse_mape)
            weights = np.array([w / total for w in inverse_mape])

        else:
            # equal weights
            weights = np.ones(len(self.feature_names)) / len(self.feature_names)

        # Generate prediction
        final_preds = np.dot(preds_matrix, weights)
        divergence = np.std(preds_matrix, axis=1)

        # Confidence based on agreement
        base_confidence = 0.8
        final_confidence = np.full(len(X), base_confidence)
        dynamic_threshold = max(0.5, min(1.5, np.mean(divergence) + 1.5 * np.std(divergence)))
        extreme_mask = divergence > dynamic_threshold
        final_confidence[extreme_mask] *= 0.3

        active_weights_map = {m: float(w) for m, w in zip(self.feature_names, weights, strict=False)}

        logger.info(f"[StackedEnsemble] Weighted average: metric={self.weighting_metric}")

        return EnsembleResult(
            final_signal=final_preds,
            confidence=final_confidence,
            divergence=divergence,
            active_weights=active_weights_map,
            stats={"method": "weighted_average", "weighting_metric": self.weighting_metric, "n_models": len(self.feature_names)}
        )

    def _predict_median(self, X: pd.DataFrame) -> EnsembleResult:
        """Median ensemble - robust to outliers."""
        preds_matrix = X[self.feature_names].to_numpy()

        # Median prediction
        final_preds = np.median(preds_matrix, axis=1)
        divergence = np.std(preds_matrix, axis=1)

        # Equal weights for median
        weights = np.ones(len(self.feature_names)) / len(self.feature_names)
        active_weights_map = {m: float(w) for m, w in zip(self.feature_names, weights, strict=False)}

        # Confidence based on agreement
        base_confidence = 0.75
        final_confidence = np.full(len(X), base_confidence)
        dynamic_threshold = max(0.5, min(1.5, np.mean(divergence) + 1.5 * np.std(divergence)))
        extreme_mask = divergence > dynamic_threshold
        final_confidence[extreme_mask] *= 0.4

        logger.info(f"[StackedEnsemble] Median ensemble with {len(self.feature_names)} models")

        return EnsembleResult(
            final_signal=final_preds,
            confidence=final_confidence,
            divergence=divergence,
            active_weights=active_weights_map,
            stats={"method": "median", "n_models": len(self.feature_names)}
        )

    def _predict_voting(self, X: pd.DataFrame) -> EnsembleResult:
        """Voting ensemble - for directional predictions."""
        preds_matrix = X[self.feature_names].to_numpy()

        # Voting: sign of sum
        signs = np.sign(preds_matrix)
        votes = np.sum(signs, axis=1)
        final_preds = np.sign(votes)

        divergence = np.std(preds_matrix, axis=1)

        # Equal weights for voting
        weights = np.ones(len(self.feature_names)) / len(self.feature_names)
        active_weights_map = {m: float(w) for m, w in zip(self.feature_names, weights, strict=False)}

        # Confidence based on vote unanimity
        vote_ratio = np.abs(votes) / len(self.feature_names)
        final_confidence = vote_ratio  # 1.0 = unanimous, 0.0 = split

        logger.info(f"[StackedEnsemble] Voting ensemble with {len(self.feature_names)} models")

        return EnsembleResult(
            final_signal=final_preds,
            confidence=final_confidence,
            divergence=divergence,
            active_weights=active_weights_map,
            stats={"method": "voting", "n_models": len(self.feature_names), "avg_vote_ratio": float(np.mean(vote_ratio))}
        )

    def save(self, path: str):
        """Save the ensemble state safely using joblib instead of pickle."""
        import joblib

        state = {
            'meta_model': self.meta_model,
            'is_trained': self.is_trained,
            'feature_names': self.feature_names,
            'config_manager': self.config_manager  # May need special handling if complex
        }

        with open(path, 'wb') as f:
            joblib.dump(state, f)
        logger.info(f"[StackedEnsemble] Saved to {path}")

    @classmethod
    def load(cls, path: str):
        """Load the ensemble state safely with security validation."""
        import joblib

        from src.config.unified_config_manager import get_current_config

        # Security validation: Ensure path is within expected data or models directories
        trusted_path = resolve_trusted_artifact_path(
            path,
            allowed_suffixes={'.joblib', '.pkl', '.pickle'},
            must_exist=True,
        )

        # Validate against configured model storage paths
        config = get_current_config()
        base_model_path = config.get('models.dual_model_manager.base_path', 'data/models')

        if not trusted_path.resolve().is_relative_to(Path(base_model_path).resolve()):
            logger.warning(f"🚫 Blocking unsafe ensemble load attempt from: {path}")
            raise ValueError(f"Unsafe path for loading: {path}")

        with open(trusted_path, 'rb') as f:
            state = joblib.load(f)  # audit-ignore: UNSAFE_MODEL_OR_PICKLE_LOAD

        instance = cls(
            meta_model=state['meta_model'],
            config_manager=state.get('config_manager')
        )
        instance.is_trained = state['is_trained']
        instance.feature_names = state['feature_names']

        return instance

def ensemble_forecast(
    model_predictions: dict[str, list[float] | np.ndarray],
    model_confidences: dict[str, list[float] | np.ndarray] | None = None,
    weights: dict[str, float] | None = None,
    market_regime: str | None = None,
    regime_configs: dict[str, dict[str, float]] | None = None,
    max_weight: float = 0.8,
    min_weight: float = 0.0,
    divergence_shrinkage: bool = True,
    rolling_window: int | None = None,
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

    active_base_weights = _determine_regime_weights(market_regime, regime_configs, weights, model_predictions)
    constrained_weights = _apply_weight_constraints(active_base_weights, min_weight, max_weight)

    aligned_data = _align_predictions_and_confidences(model_predictions, model_confidences)
    stacked_preds, stacked_conf = aligned_data['predictions'], aligned_data['confidences']

    effective_weights = _calculate_effective_weights(constrained_weights, stacked_conf, aligned_data['model_order'])
    final_signal = _generate_ensemble_signal(stacked_preds, effective_weights, method)

    divergence = np.nanstd(stacked_preds, axis=0)
    final_signal = _apply_divergence_penalty(final_signal, divergence, divergence_shrinkage)
    final_signal = _apply_smoothing(final_signal, rolling_window, fill_na)

    final_confidence = _calculate_final_confidence(stacked_conf, effective_weights)
    stats = _create_ensemble_stats(len(model_predictions), divergence, market_regime)

    return EnsembleResult(
        final_signal=final_signal,
        confidence=final_confidence,
        divergence=divergence,
        active_weights=constrained_weights,
        stats=stats
    )

def _determine_regime_weights(
    market_regime: str | None,
    regime_configs: dict[str, dict[str, float]] | None,
    weights: dict[str, float] | None,
    model_predictions: dict[str, list[float] | np.ndarray]
) -> dict[str, float]:
    """Determine weights based on market regime."""
    if market_regime and regime_configs and market_regime in regime_configs:
        active_base_weights = regime_configs[market_regime]
        logger.info(f"[Ensemble] Applying weights for regime: {market_regime}")
    else:
        active_base_weights = weights or dict.fromkeys(model_predictions.keys(), 1.0)
        if market_regime:
            logger.warning(f"[Ensemble] Regime '{market_regime}' not found in config. Using default weights.")

    return active_base_weights

def _apply_weight_constraints(weights: dict[str, float], min_weight: float, max_weight: float) -> dict[str, float]:
    """Apply min/max weight constraints."""
    return {m: max(min_weight, min(w, max_weight)) for m, w in weights.items()}

def _align_predictions_and_confidences(
    model_predictions: dict[str, list[float] | np.ndarray],
    model_confidences: dict[str, list[float] | np.ndarray] | None
) -> dict[str, Any]:
    """Align predictions and confidences to same length and return as stacked 2D arrays."""
    max_len = max((len(v) for v in model_predictions.values()), default=0)

    aligned_preds = {}
    aligned_conf = {}
    model_order = []

    for m in model_predictions.keys():
        p = np.array(model_predictions[m], dtype=float)
        c = np.array(model_confidences[m], dtype=float) if (model_confidences and m in model_confidences) else np.ones(len(p))

        if len(p) < max_len:
            p = np.pad(p, (max_len - len(p), 0), 'constant', constant_values=np.nan)
            c = np.pad(c, (max_len - len(c), 0), 'constant', constant_values=0.0)

        aligned_preds[m] = p
        aligned_conf[m] = c
        model_order.append(m)

    # Convert dictionaries to stacked 2D numpy arrays of shape (n_models, n_samples)
    preds_array = np.array([aligned_preds[m] for m in model_order])
    conf_array = np.array([aligned_conf[m] for m in model_order])

    return {
        'predictions': preds_array,
        'confidences': conf_array,
        'model_order': model_order
    }

def _calculate_effective_weights(
    base_weights: dict[str, float],
    stacked_conf: np.ndarray,
    model_order: list[str]
) -> np.ndarray:
    """Calculate effective weights combining base weights and confidence."""
    base_weight_vec = np.array([base_weights.get(m, 1.0) for m in model_order]).reshape(-1, 1)
    effective_weights = base_weight_vec * stacked_conf

    # Normalize effective weights across models at each time step
    weight_sums = np.nansum(effective_weights, axis=0)
    normalized_weights = np.divide(effective_weights, weight_sums, out=np.zeros_like(effective_weights), where=weight_sums != 0)

    return normalized_weights

def _generate_ensemble_signal(stacked_preds: np.ndarray, effective_weights: np.ndarray, method: str) -> np.ndarray:
    """Generate final ensemble signal using specified method."""
    if method == "median":
        return np.nanmedian(stacked_preds, axis=0)
    elif method == "mean":
        return np.nanmean(stacked_preds, axis=0)
    else:
        # weighted
        return np.nansum(stacked_preds * effective_weights, axis=0)

def _apply_divergence_penalty(signal: np.ndarray, divergence: np.ndarray, divergence_shrinkage: bool) -> np.ndarray:
    """Apply divergence penalty if enabled."""
    if divergence_shrinkage:
        penalty = 1.0 / (1.0 + divergence)
        signal = signal * penalty
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("[Ensemble] Applied divergence penalty (shrinkage).")
    return signal

def _apply_smoothing(signal: np.ndarray, rolling_window: int | None, fill_na: float) -> np.ndarray:
    """Apply exponential moving average (EMA) smoothing if specified."""
    if rolling_window and rolling_window > 1:
        return pd.Series(signal).ffill().fillna(fill_na).ewm(span=rolling_window, min_periods=1, adjust=False).mean().to_numpy()
    return signal

def _calculate_final_confidence(stacked_conf: np.ndarray, effective_weights: np.ndarray) -> np.ndarray:
    """Calculate final confidence as weighted average."""
    return np.nansum(stacked_conf * effective_weights, axis=0)

def _create_ensemble_stats(n_models: int, divergence: np.ndarray, market_regime: str | None) -> dict[str, Any]:
    """Create ensemble statistics dictionary."""
    return {
        "n_models": n_models,
        "avg_divergence": float(np.nanmean(divergence)),
        "regime_applied": market_regime
    }
