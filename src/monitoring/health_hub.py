
"""
Моніторинг стану системи за допомогою машинного навчання для прогнозування системних збоїв та фінансового дрейфу.
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.analytics.data_managers.model_results_manager import ModelResultsManager
from src.data.management.data_manager import DataManager
from src.core.logging.notifier import UniversalNotifier as Notifier
from src.core.cache.cache_manager import CacheManager
from src.monitoring.infrastructure.resource_monitor import ResourceMonitor

class HealthHub:
    """Моніторить стан системи, прогнозує проблеми, детектує фінансовий дрейф та генерує рекомендації."""
    
    def __init__(self, config_manager: UnifiedConfigManager, data_manager: DataManager, results_manager: ModelResultsManager, notifier: Optional[Notifier] = None):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger("HealthHub")
        self.data_manager = data_manager
        self.results_manager = results_manager
        self.notifier = notifier
        self.cache_manager = CacheManager()
        self.resource_monitor = ResourceMonitor()
        self.models = {}
        self.scalers = {}

        # --- Diagnostic Change ---
        self.logger.info("Attempting to retrieve 'paths' configuration for HealthHub...")
        paths_config = self.config_manager.get_config('paths')
        self.logger.info(f"Retrieved 'paths' config for HealthHub: {paths_config}")

        models_path = paths_config.get('models') if paths_config else None
        if not models_path:
            self.logger.error("Failed to resolve models_path, it is None. Defaulting to 'trained_models'.")
            models_path = 'trained_models'
        
        self.model_dir = Path(models_path) / "system_health_monitor"
        # --- End Diagnostic Change ---

        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_ml_models()
        self.logger.info("HealthHub ініціалізовано з підтримкою DuckDB")
    
    def load_ml_models(self):
        """Завантаження ML моделей для моніторингу."""
        try:
            model_files = {
                "performance_predictor": "performance_predictor.pkl",
                "memory_predictor": "memory_predictor.pkl", 
                "disk_predictor": "disk_predictor.pkl",
                "network_predictor": "network_predictor.pkl",
                "anomaly_detector": "anomaly_detector.pkl"
            }
            
            for model_name, filename in model_files.items():
                model_path = self.model_dir / filename
                if model_path.exists():
                    self.models[model_name] = joblib.load(model_path)
                    self.logger.info(f"Завантажено модель: {model_name}")
            
            scaler_files = {
                "resource_scaler": "resource_scaler.pkl"
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_path = self.model_dir / filename
                if scaler_path.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    self.logger.info(f"Завантажено скалер: {scaler_name}")
            
        except Exception as e:
            self.logger.error(f"Не вдалося завантажити ML моделі: {e}")

    def check_system_health(self) -> Dict[str, Any]:
        """Отримує метрики з ResourceMonitor та запускає ML прогнозування."""
        try:
            # Get the new, detailed metrics dictionary
            current_metrics = self.resource_monitor.get_health_status()
            
            if not current_metrics or current_metrics.get('overall_status') == 'error':
                self.logger.warning("Не вдалося отримати поточні метрики від ResourceMonitor")
                return {"status": "failed", "error": "Unable to get current metrics"}
            
            # Automatic cache clearing based on detailed memory metrics
            mem_usage = current_metrics.get('system', {}).get('memory', {}).get('percent', 0)
            if mem_usage > 90.0:
                self.logger.warning("Використання пам\'яті > 90%. Запуск очищення кешу.")
                self.cache_manager.clear()
                if self.notifier:
                    self.notifier.send_info("Система: Високе навантаження на пам\'ять. Кеш очищено автоматично.")

            # Extract features for ML predictions using the detailed metrics
            features = self.extract_features_from_metrics(current_metrics)
            predictions = {}
            
            problem_types = ["performance", "memory", "disk", "network"]
            for problem_type in problem_types:
                model_name = f"{problem_type}_predictor"
                if model_name in self.models:
                    model = self.models[model_name]
                    scaler = self.scalers.get("resource_scaler")
                    features_scaled = scaler.transform([features]) if scaler else [features]
                    
                    prob = model.predict_proba(features_scaled)[0][1]
                    risk_level = self.calculate_risk_level(prob)
                    
                    predictions[problem_type] = {
                        "probability": float(prob),
                        "risk_level": risk_level
                    }

                    if risk_level in ["high", "critical"] and self.notifier:
                        self.notifier.send_alert(f"ALERT: Ризик {problem_type.upper()} становить {risk_level.upper()} ({prob:.1%})")

            anomaly_result = self.detect_anomalies(features)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "metrics": current_metrics, # Store the full detailed metrics
                "predictions": predictions,
                "anomalies": anomaly_result,
                "overall_risk": self.calculate_overall_risk(predictions),
                "recommendations": self.generate_ml_recommendations(predictions, anomaly_result)
            }
            
            return result
        except Exception as e:
            self.logger.error(f"Перевірка здоров\'я системи провалилася: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def check_model_drift(self, model_name: str, window_days: int = 7) -> Dict:
        """Порівнює поточну продуктивність моделі з історичною для детекції дрейфу."""
        try:
            query = f"""
                SELECT win_rate, sharpe_ratio, timestamp 
                FROM model_performance 
                WHERE model_name = '{model_name}' 
                ORDER BY timestamp DESC
            """
            perf_df = self.data_manager.load_data(query)
            
            if len(perf_df) < 20:
                return {"status": "insufficient_data"}

            perf_df['timestamp'] = pd.to_datetime(perf_df['timestamp'])
            recent = perf_df[perf_df['timestamp'] > (datetime.now() - timedelta(days=window_days))]
            historical = perf_df[perf_df['timestamp'] <= (datetime.now() - timedelta(days=window_days))]
            
            if recent.empty or historical.empty:
                return {"status": "insufficient_window_data"}

            metrics = ['win_rate', 'sharpe_ratio']
            drift_detected = False
            alerts = []

            for metric in metrics:
                hist_mean = historical[metric].mean()
                hist_std = historical[metric].std()
                curr_mean = recent[metric].mean()
                
                z_score = abs(curr_mean - hist_mean) / (hist_std if hist_std > 0 else 0.001)
                
                if z_score > 2.0:
                    drift_detected = True
                    msg = f"Виявлено дрейф моделі для {model_name} [{metric}]: Z-Score={z_score:.2f}"
                    alerts.append(msg)
                    self.logger.warning(f"[FINANCIAL] {msg}")

            if drift_detected and self.notifier:
                self.notifier.send_alert(f"CRITICAL: Дрейф моделі {model_name}\n" + "\n".join(alerts))

            return {
                "model_name": model_name,
                "drift_detected": drift_detected,
                "alerts": alerts,
                "recent_metrics": recent[metrics].mean().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Помилка детекції дрейфу для {model_name}: {e}")
            return {"status": "error", "message": str(e)}

    def extract_features_from_metrics(self, metrics: Dict) -> List[float]:
        """Extracts features for ML models from the detailed metrics dictionary."""
        self.logger.critical("Feature extraction for HealthHub is not fully implemented and relies on placeholder data.")
        raise NotImplementedError("The 'extract_features_from_metrics' method is not implemented. Critical metrics for pipeline performance, model accuracy, and market context are missing. Using this for predictions would be misleading.")

    def calculate_risk_level(self, prob: float) -> str:
        if prob >= 0.8: return "critical"
        if prob >= 0.6: return "high"
        if prob >= 0.4: return "medium"
        return "low"

    def calculate_overall_risk(self, predictions: Dict) -> str:
        probs = [p["probability"] for p in predictions.values()]
        max_p = max(probs) if probs else 0
        return self.calculate_risk_level(max_p)

    def detect_anomalies(self, features: List[float]) -> Dict:
        if "anomaly_detector" not in self.models: return {"is_anomaly": False}
        score = self.models["anomaly_detector"].decision_function([features])[0]
        is_anomaly = self.models["anomaly_detector"].predict([features])[0] == -1
        return {"is_anomaly": bool(is_anomaly), "score": float(score)}

    def generate_ml_recommendations(self, predictions: Dict, anomaly: Dict) -> List[str]:
        recs = []
        for k, v in predictions.items():
            if v["probability"] > 0.6: recs.append(f"Дія: Терміново зменшити ризик {k}.")
        if anomaly["is_anomaly"]: recs.append("Дія: Перевірте нетипову поведінку системи, виявлену IsolationForest.")
        return recs
