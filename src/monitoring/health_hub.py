# src/monitoring/health_hub.py
"""
System health monitoring using Machine Learning to predict failures and financial drift.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.config.unified_config_manager import UnifiedConfigManager, get_current_config
from src.core.cache.cache_manager import CacheManager
from src.core.logging.logger import ProjectLogger
from src.core.logging.notifier import UniversalNotifier as Notifier
from src.data.management.data_manager import DataManager
from src.monitoring.infrastructure.resource_monitor import ResourceMonitor


class HealthHub:
    """
    Monitors system state, predicts hardware issues, detects financial drift,
    and generates corrective recommendations.
    """

    def __init__(self, config_manager: UnifiedConfigManager | None = None,
                 data_manager: DataManager | None = None,
                 results_manager: ModelResultsManager | None = None,
                 notifier: Notifier | None = None):
        """Initializes HealthHub with necessary dependencies."""
        self._initialize_core_components(config_manager, data_manager, results_manager, notifier)
        self._setup_cache_manager()
        self._initialize_monitoring_components()
        self._setup_model_directory()
        self.load_ml_models()
        self.logger.info("HealthHub initialized successfully")

    def _initialize_core_components(self, config_manager: UnifiedConfigManager | None,
                                data_manager: DataManager | None,
                                results_manager: ModelResultsManager | None,
                                notifier: Notifier | None) -> None:
        """Initialize core components and dependencies."""
        self.config_manager = config_manager or get_current_config()
        self.logger = ProjectLogger.get_logger("HealthHub")
        self.data_manager = data_manager
        self.results_manager = results_manager
        self.notifier = notifier

    def _setup_cache_manager(self) -> None:
        """Setup cache manager based on data manager availability."""
        if self.data_manager:
            self.cache_manager = CacheManager(data_manager=self.data_manager, config_manager=self.config_manager)
        else:
            self.cache_manager = None

    def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring components and data structures."""
        self.resource_monitor = ResourceMonitor()
        self.models = {}
        self.scalers = {}

    def _setup_model_directory(self) -> None:
        """Setup directory path for monitoring models."""
        paths_config = self.config_manager.get_config('paths') or {}
        models_path = paths_config.get('models', 'trained_models')

        self.model_dir = Path(models_path) / "system_health_monitor"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_ml_models(self):
        """Loads ML models used for health monitoring and anomaly detection."""
        try:
            self._load_prediction_models()
            self._load_scaler_models()
        except Exception as e:
            self.logger.error(f"Failed to load internal health monitoring ML models: {e}")

    def _get_model_file_mapping(self) -> dict[str, str]:
        """Get mapping of model names to their file paths."""
        return {
            "performance_predictor": "performance_predictor.pkl",
            "memory_predictor": "memory_predictor.pkl",
            "disk_predictor": "disk_predictor.pkl",
            "network_predictor": "network_predictor.pkl",
            "anomaly_detector": "anomaly_detector.pkl"
        }

    def _load_prediction_models(self) -> None:
        """Load prediction models from disk."""
        model_files = self._get_model_file_mapping()

        for model_name, filename in model_files.items():
            model_path = self.model_dir / filename
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)

    def _load_scaler_models(self) -> None:
        """Load scaler models from disk."""
        scaler_path = self.model_dir / "resource_scaler.pkl"
        if scaler_path.exists():
            self.scalers["resource_scaler"] = joblib.load(scaler_path)

    def check_system_health(self) -> dict[str, Any]:
        """Retrieves hardware metrics and runs ML diagnostics/projections."""
        try:
            current_metrics = self._get_current_metrics()
            if not current_metrics:
                return {"status": "failed", "error": "Unable to retrieve real-time resource metrics"}

            self._handle_memory_management(current_metrics)
            features = self.extract_features_from_metrics(current_metrics)
            predictions = self._predict_resource_risks(features)
            anomaly_result = self.detect_anomalies(features)

            return self._build_health_report(current_metrics, predictions, anomaly_result)
        except Exception as e:
            self.logger.error(f"HealthHub diagnostic loop failure: {e}")
            return {"status": "failed", "error": str(e)}

    def _get_current_metrics(self) -> dict[str, Any] | None:
        """Get current system metrics with validation."""
        current_metrics = self.resource_monitor.get_health_status()
        if not current_metrics or current_metrics.get('overall_status') == 'error':
            return None
        return current_metrics

    def _handle_memory_management(self, current_metrics: dict[str, Any]) -> None:
        """Handle autonomous memory management."""
        if not self.cache_manager:
            return

        mem_usage = current_metrics.get('system', {}).get('memory', {}).get('percent', 0)
        if mem_usage > 90.0:
            self.logger.warning("System memory > 90%. Automatic cache purge triggered.")
            self.cache_manager.clear()

    def _predict_resource_risks(self, features: list[float]) -> dict[str, Any]:
        """Predict risks for all resource types."""
        predictions = {}
        problem_types = ["performance", "memory", "disk", "network"]

        for pt in problem_types:
            model_name = f"{pt}_predictor"
            if model_name in self.models:
                predictions[pt] = self._predict_single_risk(model_name, features)

        return predictions

    def _predict_single_risk(self, model_name: str, features: list[float]) -> dict[str, Any]:
        """Predict risk for a single resource type."""
        model = self.models[model_name]
        scaler = self.scalers.get("resource_scaler")
        features_scaled = scaler.transform([features]) if scaler else [features]

        prob = model.predict_proba(features_scaled)[0][1]
        risk = self.calculate_risk_level(prob)
        return {"probability": float(prob), "risk_level": risk}

    def _build_health_report(self, current_metrics: dict[str, Any],
                           predictions: dict[str, Any], anomaly_result: dict) -> dict[str, Any]:
        """Build comprehensive health report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "predictions": predictions,
            "anomalies": anomaly_result,
            "overall_risk": self.calculate_overall_risk(predictions),
            "recommendations": self.generate_ml_recommendations(predictions, anomaly_result)
        }

    def extract_features_from_metrics(self, metrics: dict) -> list[float]:
        """Exctracts stabilized feature vector for ML diagnostic analysis."""
        try:
            sys_m = metrics.get('system', {})
            cpu = sys_m.get('cpu', {}).get('percent', 0) / 100.0
            mem = sys_m.get('memory', {}).get('percent', 0) / 100.0
            disk = sys_m.get('disk', {}).get('percent', 0) / 100.0

            # Pipeline and behavioral metrics
            pipe_perf = metrics.get('pipeline', {}).get('efficiency', 1.0)
            drift = metrics.get('analytics', {}).get('drift_score', 0.0)

            return [cpu, mem, disk, pipe_perf, drift]
        except Exception:
            return [0.5, 0.5, 0.1, 1.0, 0.0] # Conservative fallback

    def calculate_risk_level(self, prob: float) -> str:
        """Maps probability score to human-readable risk level."""
        if prob >= 0.8: return "critical"
        if prob >= 0.6: return "high"
        if prob >= 0.4: return "medium"
        return "low"

    def calculate_overall_risk(self, predictions: dict) -> str:
        """Determines worst-case risk across all monitored subsystems."""
        probs = [p["probability"] for p in predictions.values()]
        return self.calculate_risk_level(max(probs)) if probs else "low"

    def detect_anomalies(self, features: list[float]) -> dict:
        """Runs isolation forest to detect deviations from normal baseline behavior."""
        if "anomaly_detector" not in self.models:
            return {"is_anomaly": False, "score": 0.0}
        try:
            score = self.models["anomaly_detector"].decision_function([features])[0]
            is_anomaly = self.models["anomaly_detector"].predict([features])[0] == -1
            return {"is_anomaly": bool(is_anomaly), "score": float(score)}
        except Exception:
            return {"is_anomaly": False, "score": 0.0}

    def generate_ml_recommendations(self, predictions: dict, anomaly: dict) -> list[str]:
        """Generates actionable recommendations based on prediction outcomes."""
        recs = []
        for subsystem, data in predictions.items():
            if data["probability"] > 0.6:
                recs.append(f"Action Required: Optimize {subsystem} resources (risk: {data['risk_level']})")
        if anomaly.get("is_anomaly"):
            recs.append("Action Required: Investigate system logs for anomalous behavioral patterns.")
        return recs

    def check_model_drift(self, model_name: str, window_days: int = 7) -> dict:
        """Detects financial and performance drift by comparing against historical baseline."""
        if not self.data_manager:
            return {"status": "error", "message": "DataManager not available for historical audit"}

        try:
            perf_df = self._load_performance_data(model_name)
            if not isinstance(perf_df, pd.DataFrame):
                return perf_df  # Return error status

            recent_data, historical_data = self._split_performance_data(perf_df, window_days)
            if recent_data is None:
                return historical_data  # Return error status

            drift_detected = self._calculate_drift_metrics(recent_data, historical_data)

            return {
                "model_name": model_name,
                "drift_detected": drift_detected,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Financial drift analysis failure: {e}")
            return {"status": "error", "message": str(e)}

    def _load_performance_data(self, model_name: str) -> pd.DataFrame | dict[str, str]:
        """Load and validate performance data for drift analysis."""
        query = f"SELECT win_rate, sharpe_ratio, timestamp FROM model_performance WHERE model_name = '{model_name}' ORDER BY timestamp DESC"
        perf_df = self.data_manager.load_data(query)

        if len(perf_df) < 10:
            return {"status": "insufficient_data", "message": "Threshold for historical comparison not met"}

        return perf_df

    def _split_performance_data(self, perf_df: pd.DataFrame, window_days: int) -> tuple[pd.DataFrame | None, pd.DataFrame | dict[str, str]]:
        """Split performance data into recent and historical windows."""
        perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
        cutoff = datetime.now() - timedelta(days=window_days)
        recent = perf_df[perf_df['timestamp'] > cutoff]
        historical = perf_df[perf_df['timestamp'] <= cutoff]

        if recent.empty or historical.empty:
            return None, {"status": "insufficient_window", "message": "One of the analysis windows (recent/historical) is empty"}

        return recent, historical

    def _calculate_drift_metrics(self, recent_data: pd.DataFrame, historical_data: pd.DataFrame) -> bool:
        """Calculate drift detection metrics."""
        drift = False
        for metric in ['win_rate', 'sharpe_ratio']:
            z_score = abs(recent_data[metric].mean() - historical_data[metric].mean()) / (historical_data[metric].std() + 1e-6)
            if z_score > 2.0:
                drift = True
                break
        return drift
