from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import psutil

# Updated import for ProjectLogger
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("HeavyLightModelComparator")

@dataclass
class ModelPerformanceMetrics:
    model_name: str
    model_type: str  # 'heavy' or 'light'
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_latency_ms: float
    memory_usage_mb: float
    cpu_usage_pct: float
    timestamp: datetime

class HeavyLightModelComparator:
    """
    Implements the Heavy vs. Light model comparison logic from ideas.md.
    Analyzes model performance, latency, and resource consumption to
    recommend the optimal architecture for the current market regime.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.performance_history: list[ModelPerformanceMetrics] = []

        # Thresholds for classification
        self.heavy_models = self.config.get('heavy_models', ['cnn', 'transformer', 'tabnet', 'lstm', 'gru'])
        self.light_models = self.config.get('light_models', ['lightgbm', 'catboost', 'xgboost', 'random_forest', 'linear'])

        # Resource constraints
        self.max_mem_pct = self.config.get('max_memory_percent', 85.0)
        self.max_cpu_pct = self.config.get('max_cpu_percent', 80.0)

        logger.info("HeavyLightModelComparator initialized.")

    def record_inference_stats(self,
                               model_name: str,
                               model_type: str,
                               y_true: np.ndarray,
                               y_pred: np.ndarray,
                               start_time: float,
                               end_time: float) -> ModelPerformanceMetrics:
        """
        Captures metrics during a live or backtest inference run.
        """
        process = psutil.Process()

        # Calculate ML metrics
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        # Handle classification (assuming binary for simplicity in this logic)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        latency = (end_time - start_time) * 1000 # ms
        mem_info = process.memory_info().rss / (1024 * 1024) # MB
        cpu_usage = psutil.cpu_percent(interval=None)

        metrics = ModelPerformanceMetrics(
            model_name=model_name,
            model_type='heavy' if model_name.lower() in self.heavy_models else 'light',
            accuracy=acc,
            precision=prec,
            recall=rec,
            f1_score=f1,
            inference_latency_ms=latency,
            memory_usage_mb=mem_info,
            cpu_usage_pct=cpu_usage,
            timestamp=datetime.now()
        )

        self.performance_history.append(metrics)
        return metrics

    def get_consensus_prediction(self, heavy_pred: float, light_pred: float, threshold: float = 0.01) -> dict[str, Any]:
        """
        Compares predictions from Heavy and Light models to find consensus.

        Args:
            heavy_pred: Percentage price change predicted by heavy model.
            light_pred: Percentage price change predicted by light model.
            threshold: Minimum change to consider as a directional signal.

        Returns:
            A dictionary containing the consensus signal and confidence.
        """
        # Determine direction based on sign and threshold
        h_dir = 1 if heavy_pred > threshold else (-1 if heavy_pred < -threshold else 0)
        l_dir = 1 if light_pred > threshold else (-1 if light_pred < -threshold else 0)

        if h_dir == 1 and l_dir == 1:
            signal = 'STRONGLY_CONFIRMED_UP'
        elif h_dir == -1 and l_dir == -1:
            signal = 'STRONGLY_CONFIRMED_DOWN'
        elif h_dir == 0 and l_dir == 0:
            signal = 'NEUTRAL_CONSENSUS'
        else:
            signal = 'DIVERGENCE_WARNING'
            logger.warning(f"Model divergence detected: Heavy({heavy_pred:.4f}) vs Light({light_pred:.4f}).")

        # Calculate consensus confidence: 1.0 if identical, 0.0 if opposite signs
        # Using normalized difference of predictions
        diff = abs(heavy_pred - light_pred)
        avg_mag = (abs(heavy_pred) + abs(light_pred)) / 2

        if avg_mag == 0:
            confidence = 1.0
        else:
            # Scale difference relative to magnitude, capped at 1.0
            confidence = max(0.0, 1.0 - (diff / (avg_mag + 1e-6)))

        # If directions are opposite, slash confidence
        if np.sign(heavy_pred) != np.sign(light_pred) and h_dir != 0 and l_dir != 0:
            confidence *= 0.2

        return {
            "signal": signal,
            "consensus_confidence": round(float(confidence), 4),
            "heavy_prediction": heavy_pred,
            "light_prediction": light_pred,
            "action_recommendation": "REDUCE_SIZE" if signal == 'DIVERGENCE_WARNING' else "NORMAL"
        }

    def get_recommendation(self, market_context: dict[str, Any]) -> dict[str, Any]:
        """
        Determines whether to use a 'Heavy' or 'Light' model based on:
        1. System Load (Safety first)
        2. Market Volatility (Complexity)
        3. Historical Accuracy in similar regimes
        """
        current_mem = psutil.virtual_memory().percent
        current_cpu = psutil.cpu_percent(interval=0.1)

        volatility = market_context.get('volatility_regime', 'normal')
        regime_change = market_context.get('regime_change_detected', False)

        # 1. Hardware Constraint Check
        if current_mem > self.max_mem_pct or current_cpu > self.max_cpu_pct:
            logger.warning(f"High system load detected (CPU: {current_cpu}%, MEM: {current_mem}%). Falling back to LIGHT models.")
            return {
                "recommended_type": "light",
                "reason": "System resource preservation",
                "constraints": {"cpu": current_cpu, "mem": current_mem}
            }

        # 2. Logic for "Heavy" usage (Complex Regimes)
        # Based on ideas.md: Use Heavy during high volatility or complex regime shifts
        if volatility == 'high' or regime_change:
            # Check if Heavy models actually performed better historically in high vol
            heavy_perf = self._get_avg_f1_by_type('heavy')
            light_perf = self._get_avg_f1_by_type('light')

            if heavy_perf >= (light_perf * 0.95): # Use heavy if it's at least competitive
                return {
                    "recommended_type": "heavy",
                    "reason": "High market complexity/volatility requires deep feature extraction",
                    "expected_improvement": heavy_perf - light_perf
                }

        # 3. Default to Light (Efficiency)
        return {
            "recommended_type": "light",
            "reason": "Low/Normal volatility: prioritize speed and prevent overfitting",
            "latency_gain_estimate_ms": self._calculate_latency_gap()
        }

    def _get_avg_f1_by_type(self, model_type: str) -> float:
        relevant = [m.f1_score for m in self.performance_history if m.model_type == model_type]
        return np.mean(relevant) if relevant else 0.0

    def _calculate_latency_gap(self) -> float:
        heavy_lat = [m.inference_latency_ms for m in self.performance_history if m.model_type == 'heavy']
        light_lat = [m.inference_latency_ms for m in self.performance_history if m.model_type == 'light']
        if heavy_lat and light_lat:
            return np.mean(heavy_lat) - np.mean(light_lat)
        return 0.0

    def generate_comparison_report(self) -> pd.DataFrame:
        """Returns a formatted DataFrame comparing current model performances."""
        if not self.performance_history:
            return pd.DataFrame()

        data = [m.__dict__ for m in self.performance_history]
        df = pd.DataFrame(data)

        # Group by type to see the 'Heavy' vs 'Light' summary
        summary = df.groupby('model_type').agg({
            'accuracy': 'mean',
            'f1_score': 'mean',
            'inference_latency_ms': 'mean',
            'memory_usage_mb': 'max'
        }).reset_index()

        logger.info("\n--- Model Architecture Comparison Report ---")
        logger.info(summary.to_string())

        return summary
