# src/monitoring/ml_analytics.py

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from src.core.logging.logger import ProjectLogger
from src.analytics.data_managers.model_results_manager import ModelResultsManager as ResultsManager
from src.monitoring.infrastructure.resource_monitor import get_resource_monitor
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger("MLAnalytics")

class MLAnalytics:
    """Machine Learning for analyzing system performance and predicting infrastructure issues."""
    
    def __init__(self, results_manager: ResultsManager, data_manager: Optional[DataManager] = None):
        self.results_manager = results_manager
        self.data_manager = data_manager
        self.resource_monitor = get_resource_monitor()
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("models/ml_analytics")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_ml_models()
        logger.info("MLAnalytics initialized with real ResourceMonitor integration")
    
    def load_ml_models(self):
        """Loads specialized monitoring models from disk."""
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
                    logger.debug(f"Loaded ML monitoring model: {model_name}")
            
            scaler_path = self.model_dir / "resource_scaler.pkl"
            if scaler_path.exists():
                self.scalers["resource_scaler"] = joblib.load(scaler_path)
            
        except Exception as e:
            logger.error(f"Failed to load monitoring ML models: {e}")
    
    def train_models(self, force_retrain: bool = False) -> Dict[str, Any]:
        """Trains monitoring models on historical system execution logs."""
        try:
            historical_data = self.load_historical_data(days=90)
            
            if len(historical_data) < 30:
                logger.warning("Insufficient historical logs for training MLAnalytics models.")
                return {"status": "failed", "error": "Insufficient data"}
            
            features_df, targets_df = self.prepare_training_data(historical_data)
            
            problem_models = ["performance", "memory", "disk", "network"]
            results = {"timestamp": datetime.now().isoformat(), "models_trained": {}}
            
            for problem_type in problem_models:
                results["models_trained"][problem_type] = self.train_problem_predictor(
                    features_df, targets_df, problem_type, force_retrain
                )
            
            results["models_trained"]["anomaly_detector"] = self.train_anomaly_detector(features_df, force_retrain)
            
            logger.info("Monitoring ML training session completed.")
            return results
            
        except Exception as e:
            logger.error(f"Training of monitoring models failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
    
    def predict_system_issues(self) -> Dict[str, Any]:
        """Predicts potential system bottlenecks using real-time resource metrics."""
        try:
            health_status = self.resource_monitor.get_health_status()
            if health_status['status'] == 'unknown':
                return {"status": "failed", "error": "Real-time metrics unavailable"}
            
            # Using raw data from ResourceMonitor instead of mock data
            current_metrics = self.resource_monitor.collect_all_metrics()
            features = self.extract_features_from_metrics(current_metrics)
            
            predictions = {}
            problem_types = ["performance", "memory", "disk", "network"]
            
            for p_type in problem_types:
                model_name = f"{p_type}_predictor"
                if model_name in self.models:
                    model = self.models[model_name]
                    scaler = self.scalers.get("resource_scaler")
                    feat_scaled = scaler.transform([features]) if scaler else [features]
                    
                    prob = model.predict_proba(feat_scaled)[0]
                    # Probability of class 1 (Problem exists)
                    p_val = float(prob[1]) if len(prob) > 1 else float(prob[0])
                    
                    predictions[p_type] = {
                        "probability": p_val,
                        "risk_level": self.calculate_risk_level(p_val)
                    }
            
            anomaly = self.detect_anomalies(features)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "metrics": health_status,
                "predictions": predictions,
                "anomalies": anomaly,
                "overall_risk": self.calculate_overall_risk(predictions),
                "recommendations": self.generate_ml_recommendations(predictions, anomaly)
            }
            
        except Exception as e:
            logger.error(f"Real-time issue prediction failed: {e}")
            return {"status": "failed", "error": str(e)}

    def check_model_drift(self, model_name: str, window_days: int = 7) -> Dict[str, Any]:
        """Detects performance degradation by comparing recent vs historical metrics."""
        if not self.data_manager:
            return {"status": "error", "message": "DataManager not provided"}

        try:
            # Query real performance records from DuckDB
            query = f"SELECT accuracy, timestamp FROM model_performance_logs WHERE model_id = '{model_name}'"
            df = self.data_manager.load_data(query)
            
            if len(df) < 10:
                return {"status": "insufficient_data"}

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff = datetime.now() - timedelta(days=window_days)
            
            recent = df[df['timestamp'] >= cutoff]['accuracy']
            historical = df[df['timestamp'] < cutoff]['accuracy']
            
            if recent.empty or historical.empty:
                return {"status": "missing_window_data"}

            z_score = abs(recent.mean() - historical.mean()) / (historical.std() + 1e-6)
            
            return {
                "model": model_name,
                "drift_detected": bool(z_score > 2.0),
                "z_score": float(z_score),
                "recent_avg": float(recent.mean()),
                "baseline_avg": float(historical.mean())
            }
        except Exception as e:
            logger.error(f"Drift detection failed for {model_name}: {e}")
            return {"status": "error", "error": str(e)}

    def extract_features_from_metrics(self, metrics: Dict[str, Any]) -> List[float]:
        """Converts raw ResourceMonitor dictionaries into ML feature vectors."""
        try:
            sys = metrics.get('system', {})
            disk = metrics.get('disk', {}).get('usage', {})
            proc = metrics.get('processes', {})
            
            features = [
                float(sys.get('memory', {}).get('percent', 0)),
                float(sys.get('cpu', {}).get('percent', 0)),
                float(disk.get('percent', 0)),
                float(proc.get('total', 0)),
                float(datetime.now().hour),
                float(datetime.now().dayofweek)
            ]
            # Pad to match trained model input dimension if necessary (e.g., 17)
            while len(features) < 17:
                features.append(0.0)
            return features
        except Exception as e:
            logger.error(f"Metric feature extraction failed: {e}")
            return [0.0] * 17

    def calculate_risk_level(self, prob: float) -> str:
        if prob >= 0.8: return "critical"
        if prob >= 0.6: return "high"
        if prob >= 0.4: return "medium"
        return "low"

    def calculate_overall_risk(self, predictions: Dict) -> str:
        probs = [p["probability"] for p in predictions.values()]
        return self.calculate_risk_level(max(probs) if probs else 0.0)

    def detect_anomalies(self, features: List[float]) -> Dict[str, Any]:
        if "anomaly_detector" not in self.models:
            return {"is_anomaly": False, "score": 0.0}
        
        score = float(self.models["anomaly_detector"].decision_function([features])[0])
        is_anomaly = self.models["anomaly_detector"].predict([features])[0] == -1
        return {"is_anomaly": bool(is_anomaly), "score": score}

    def generate_ml_recommendations(self, predictions: Dict, anomaly: Dict) -> List[str]:
        recs = []
        for issue, pred in predictions.items():
            if pred['probability'] > 0.6:
                recs.append(f"PREVENTIVE: High risk of {issue} issues. Check resource limits.")
        if anomaly['is_anomaly']:
            recs.append("CRITICAL: Unidentified system anomaly detected. Possible hardware or data corruption.")
        return recs

    def load_historical_data(self, days: int = 90) -> List[Dict[str, Any]]:
        """Loads past execution reports for training data generation."""
        if not self.results_manager:
            return []

        try:
            if hasattr(self.results_manager, "load_recent_results"):
                return self.results_manager.load_recent_results(days=days)
            if hasattr(self.results_manager, "load_all_results"):
                return self.results_manager.load_all_results()
        except Exception as e:
            logger.warning(f"Could not load historical monitoring data: {e}")

        return []

    def prepare_training_data(self, historical_data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Builds a conservative tabular dataset from historical monitoring records."""
        rows = []
        targets = []

        for record in historical_data:
            metrics = record.get("metrics", record)
            rows.append(self.extract_features_from_metrics(metrics))
            targets.append({
                "performance": int(record.get("performance_issue", False)),
                "memory": int(record.get("memory_issue", False)),
                "disk": int(record.get("disk_issue", False)),
                "network": int(record.get("network_issue", False)),
            })

        return pd.DataFrame(rows), pd.DataFrame(targets)

    def train_problem_predictor(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        problem_type: str,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        """Trains a simple monitoring classifier when enough labeled data exists."""
        if problem_type not in targets_df.columns:
            return {"status": "skipped", "reason": f"Missing target '{problem_type}'"}
        if features_df.empty or targets_df[problem_type].nunique() < 2:
            return {"status": "skipped", "reason": "Insufficient labeled classes"}

        try:
            from sklearn.ensemble import RandomForestClassifier

            X_train, X_test, y_train, y_test = train_test_split(
                features_df,
                targets_df[problem_type],
                test_size=0.2,
                shuffle=False,
            )
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            accuracy = accuracy_score(y_test, model.predict(X_test_scaled)) if len(y_test) else 0.0

            model_path = self.model_dir / f"{problem_type}_predictor.pkl"
            scaler_path = self.model_dir / "resource_scaler.pkl"
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            self.models[f"{problem_type}_predictor"] = model
            self.scalers["resource_scaler"] = scaler

            return {"status": "trained", "accuracy": float(accuracy), "model_path": str(model_path)}
        except Exception as e:
            logger.error(f"Training failed for {problem_type} predictor: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}

    def train_anomaly_detector(self, features_df: pd.DataFrame, force_retrain: bool = False) -> Dict[str, Any]:
        """Trains an isolation-forest anomaly detector for infrastructure metrics."""
        if features_df.empty:
            return {"status": "skipped", "reason": "No feature data"}

        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(features_df)
            model_path = self.model_dir / "anomaly_detector.pkl"
            joblib.dump(model, model_path)
            self.models["anomaly_detector"] = model
            return {"status": "trained", "model_path": str(model_path)}
        except Exception as e:
            logger.error(f"Anomaly detector training failed: {e}", exc_info=True)
            return {"status": "failed", "error": str(e)}
