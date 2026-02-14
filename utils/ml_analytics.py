"""
Machine Learning for аналandwithу withвandтandв and прогноwithування системних problems
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from .results_manager import ResultsManager

logger = logging.getLogger(__name__)

class MLAnalytics:
    """Machine Learning for аналandwithу withвandтandв"""
    
    def __init__(self, results_manager: ResultsManager):
        self.results_manager = results_manager
        self.models = {}
        self.scalers = {}
        self.model_dir = Path("models/ml_analytics")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.load_ml_models()
        logger.info("[MLAnalytics] Initialized")
    
    def load_ml_models(self):
        """Заванandження ML моwhereлей"""
        try:
            # Спроба forванandжити andснуючand моwhereлand
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
                    logger.info(f"[MLAnalytics] Loaded model: {model_name}")
                else:
                    logger.warning(f"[MLAnalytics] Model not found: {model_name}")
            
            # Заванandжити скейлери
            scaler_files = {
                "performance_scaler": "performance_scaler.pkl",
                "resource_scaler": "resource_scaler.pkl"
            }
            
            for scaler_name, filename in scaler_files.items():
                scaler_path = self.model_dir / filename
                if scaler_path.exists():
                    self.scalers[scaler_name] = joblib.load(scaler_path)
                    logger.info(f"[MLAnalytics] Loaded scaler: {scaler_name}")
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to load ML models: {e}")
    
    def train_models(self, force_retrain: bool = False) -> Dict:
        """
        Тренування ML моwhereлей
        
        Args:
            force_retrain: Примусове перетренування
            
        Returns:
            Словник with реwithульandandми тренування
        """
        try:
            training_results = {
                "timestamp": datetime.now().isoformat(),
                "training_session_id": f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "models_trained": {},
                "training_data_stats": {},
                "model_performance": {}
            }
            
            # Заванandжити andсторичнand данand
            historical_data = self.load_historical_data(days=90)
            
            if len(historical_data) < 30:
                logger.warning("[MLAnalytics] Insufficient historical data for training")
                return {
                    "status": "failed",
                    "error": "Insufficient historical data",
                    "data_points": len(historical_data)
                }
            
            # Пandдготовка data
            features_df, targets_df = self.prepare_training_data(historical_data)
            
            training_results["training_data_stats"] = {
                "total_samples": len(features_df),
                "feature_columns": list(features_df.columns),
                "target_columns": list(targets_df.columns),
                "date_range": {
                    "start": historical_data[0].get("timestamp") if historical_data else None,
                    "end": historical_data[-1].get("timestamp") if historical_data else None
                }
            }
            
            # Тренування моwhereлей for прогноwithування problems
            problem_models = ["performance", "memory", "disk", "network"]
            
            for problem_type in problem_models:
                try:
                    model_result = self.train_problem_predictor(
                        features_df, targets_df, problem_type, force_retrain
                    )
                    training_results["models_trained"][problem_type] = model_result
                    
                    if model_result.get("status") == "success":
                        training_results["model_performance"][problem_type] = model_result.get("performance", {})
                    
                except Exception as e:
                    logger.error(f"[MLAnalytics] Failed to train {problem_type} model: {e}")
                    training_results["models_trained"][problem_type] = {
                        "status": "failed",
                        "error": str(e)
                    }
            
            # Тренування whereтектора аномалandй
            try:
                anomaly_result = self.train_anomaly_detector(features_df, force_retrain)
                training_results["models_trained"]["anomaly_detector"] = anomaly_result
                
                if anomaly_result.get("status") == "success":
                    training_results["model_performance"]["anomaly_detector"] = anomaly_result.get("performance", {})
                
            except Exception as e:
                logger.error(f"[MLAnalytics] Failed to train anomaly detector: {e}")
                training_results["models_trained"]["anomaly_detector"] = {
                    "status": "failed",
                    "error": str(e)
                }
            
            # Зберегти реwithульandти тренування
            filename = f"ml_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.results_manager.save_results_to_output(training_results, filename)
            
            logger.info(f"[MLAnalytics] Training completed. Models trained: {len([m for m in training_results['models_trained'].values() if m.get('status') == 'success'])}")
            return training_results
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Training failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def predict_system_issues(self) -> Dict:
        """
        Прогноwithування системних problems
        
        Returns:
            Словник with прогноforми problems
        """
        try:
            # Отримати поточнand метрики
            current_metrics = self.get_current_system_metrics()
            
            if not current_metrics:
                return {
                    "status": "failed",
                    "error": "Unable to get current metrics"
                }
            
            # Пandдготовка оwithнак
            features = self.extract_features_from_metrics(current_metrics)
            
            predictions = {}
            prediction_details = {}
            
            # Прогноwithування for кожного типу problems
            problem_types = ["performance", "memory", "disk", "network"]
            
            for problem_type in problem_types:
                model_name = f"{problem_type}_predictor"
                
                if model_name in self.models:
                    try:
                        model = self.models[model_name]
                        scaler = self.scalers.get("resource_scaler")
                        
                        # Масшandбування оwithнак
                        if scaler:
                            features_scaled = scaler.transform([features])
                        else:
                            features_scaled = [features]
                        
                        # Прогноwithування
                        probability = model.predict_proba(features_scaled)[0]
                        prediction = model.predict(features_scaled)[0]
                        
                        predictions[problem_type] = {
                            "probability": float(probability[1]) if len(probability) > 1 else float(probability[0]),
                            "prediction": int(prediction),
                            "risk_level": self.calculate_risk_level(probability[1] if len(probability) > 1 else probability[0]),
                            "confidence": self.calculate_confidence(probability)
                        }
                        
                        prediction_details[problem_type] = {
                            "model_used": model_name,
                            "features_used": len(features),
                            "scaling_applied": scaler is not None
                        }
                        
                    except Exception as e:
                        logger.error(f"[MLAnalytics] Prediction failed for {problem_type}: {e}")
                        predictions[problem_type] = {
                            "probability": 0.0,
                            "prediction": 0,
                            "risk_level": "unknown",
                            "error": str(e)
                        }
                else:
                    predictions[problem_type] = {
                        "probability": 0.0,
                        "prediction": 0,
                        "risk_level": "model_not_available",
                        "error": "Model not trained"
                    }
            
            # Детекцandя аномалandй
            anomaly_result = self.detect_anomalies(features)
            
            result = {
                "timestamp": datetime.now().isoformat(),
                "prediction_type": "system_issues",
                "current_metrics": current_metrics,
                "predictions": predictions,
                "prediction_details": prediction_details,
                "anomaly_detection": anomaly_result,
                "overall_risk": self.calculate_overall_risk(predictions),
                "recommendations": self.generate_ml_recommendations(predictions, anomaly_result)
            }
            
            # Зберегти прогноwithи
            filename = f"ml_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.results_manager.save_results_to_output(result, filename)
            
            logger.info(f"[MLAnalytics] Generated predictions for {len(predictions)} problem types")
            return result
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Prediction failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def generate_ml_recommendations(self, predictions: Dict, anomaly_result: Dict) -> List[Dict]:
        """
        Геnotрацandя ML рекомендацandй
        
        Args:
            predictions: Прогноwithи problems
            anomaly_result: Реwithульandти whereтекцandї аномалandй
            
        Returns:
            Список рекомендацandй
        """
        recommendations = []
        
        # Рекомендацandї на основand прогноwithandв
        for problem_type, prediction in predictions.items():
            probability = prediction.get("probability", 0)
            risk_level = prediction.get("risk_level", "")
            
            if probability > 0.7:
                recommendations.append({
                    "type": "PREVENTIVE",
                    "issue": problem_type,
                    "probability": probability,
                    "risk_level": risk_level,
                    "action": self.get_preventive_action(problem_type),
                    "urgency": "high" if probability > 0.8 else "medium",
                    "timestamp": datetime.now().isoformat()
                })
            elif probability > 0.5:
                recommendations.append({
                    "type": "MONITORING",
                    "issue": problem_type,
                    "probability": probability,
                    "risk_level": risk_level,
                    "action": f"Increase monitoring for {problem_type}",
                    "urgency": "low",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Рекомендацandї на основand аномалandй
        if anomaly_result.get("is_anomaly", False):
            recommendations.append({
                "type": "ANOMALY",
                "issue": "system_behavior",
                "anomaly_score": anomaly_result.get("anomaly_score", 0),
                "action": "Investigate unusual system behavior",
                "urgency": "medium",
                "timestamp": datetime.now().isoformat()
            })
        
        # Загальнand рекомендацandї
        high_risk_count = sum(1 for p in predictions.values() if p.get("risk_level") in ["high", "critical"])
        if high_risk_count > 2:
            recommendations.append({
                "type": "SYSTEM_WIDE",
                "issue": "multiple_risks",
                "action": "Consider system-wide optimization",
                "urgency": "high",
                "timestamp": datetime.now().isoformat()
            })
        
        return recommendations
    
    def detect_anomalies(self, features: List[float]) -> Dict:
        """
        Детекцandя аномалandй
        
        Args:
            features: Оwithнаки for аналandwithу
            
        Returns:
            Словник with реwithульandandми whereтекцandї аномалandй
        """
        try:
            if "anomaly_detector" not in self.models:
                return {
                    "is_anomaly": False,
                    "anomaly_score": 0.0,
                    "status": "model_not_available"
                }
            
            model = self.models["anomaly_detector"]
            
            # Детекцandя аномалandй
            anomaly_score = model.decision_function([features])[0]
            is_anomaly = model.predict([features])[0] == -1
            
            return {
                "is_anomaly": bool(is_anomaly),
                "anomaly_score": float(anomaly_score),
                "threshold": model.contamination,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Anomaly detection failed: {e}")
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "status": "failed",
                "error": str(e)
            }
    
    # Helper methods
    def load_historical_data(self, days: int = 90) -> List[Dict]:
        """Заванandжити andсторичнand данand"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            historical_data = []
            
            # Заванandжити with output директорandї
            output_files = list(self.results_manager.output_dir.glob("*.json"))
            
            for file_path in output_files:
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time >= cutoff_date:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # Додати часову мandтку fileу
                        if "timestamp" not in data:
                            data["timestamp"] = file_time.isoformat()
                        
                        historical_data.append(data)
                
                except Exception as e:
                    logger.warning(f"[MLAnalytics] Failed to load {file_path}: {e}")
            
            # Сортування for часом
            historical_data.sort(key=lambda x: x.get("timestamp", ""))
            
            logger.info(f"[MLAnalytics] Loaded {len(historical_data)} historical records")
            return historical_data
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to load historical data: {e}")
            return []
    
    def prepare_training_data(self, historical_data: List[Dict]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Пandдготовка тренувальних data
        
        Args:
            historical_data: Історичнand данand
            
        Returns:
            Кортеж (features_df, targets_df)
        """
        features_list = []
        targets_list = []
        
        for record in historical_data:
            # Витягnotння оwithнак
            features = self.extract_features(record)
            if features:
                features_list.append(features)
                
                # Витягnotння цandльових withмandнних
                targets = self.extract_targets(record)
                if targets:
                    targets_list.append(targets)
        
        features_df = pd.DataFrame(features_list)
        targets_df = pd.DataFrame(targets_list)
        
        # Обробка пропущених withначень
        features_df = features_df.fillna(features_df.mean())
        targets_df = targets_df.fillna(0)
        
        logger.info(f"[MLAnalytics] Prepared training data: {len(features_df)} samples, {len(features_df.columns)} features")
        return features_df, targets_df
    
    def extract_features(self, record: Dict) -> Optional[List[float]]:
        """Витягnotння оwithнак with forпису"""
        try:
            features = []
            
            # Системнand метрики
            system_status = record.get("system_status", {})
            features.extend([
                system_status.get("memory_usage", 0),
                system_status.get("cpu_usage", 0),
                system_status.get("disk_usage", 0),
                system_status.get("active_processes", 0)
            ])
            
            # Метрики продуктивностand
            performance_metrics = record.get("performance_metrics", {})
            if "pipeline_execution_time" in performance_metrics:
                pipeline_times = performance_metrics["pipeline_execution_time"]
                if isinstance(pipeline_times, dict):
                    features.extend([
                        pipeline_times.get("stage_1", 0),
                        pipeline_times.get("stage_2", 0),
                        pipeline_times.get("stage_3", 0),
                        pipeline_times.get("stage_4", 0),
                        pipeline_times.get("stage_5", 0)
                    ])
                else:
                    features.extend([pipeline_times] + [0] * 4)
            else:
                features.extend([0] * 5)
            
            # Метрики моwhereлей
            model_performance = record.get("model_performance", {})
            features.extend([
                model_performance.get("avg_accuracy", 0),
                model_performance.get("training_efficiency", 0),
                model_performance.get("inference_speed", 0)
            ])
            
            # Часовand оwithнаки
            timestamp = record.get("timestamp")
            if timestamp:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                features.extend([
                    dt.hour,
                    dt.dayofweek,
                    dt.day
                ])
            else:
                features.extend([0, 0, 0])
            
            return features if len(features) == 17 else None  # 17 оwithнак
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Feature extraction failed: {e}")
            return None
    
    def extract_targets(self, record: Dict) -> Optional[Dict]:
        """Витягnotння цandльових withмandнних"""
        try:
            targets = {}
            
            # Проблеми продуктивностand
            performance_issues = record.get("issues", [])
            targets["performance_problem"] = 1 if any("PERFORMANCE" in issue.get("type", "") for issue in performance_issues) else 0
            
            # Проблеми пам'ятand
            targets["memory_problem"] = 1 if any("MEMORY" in issue.get("type", "") for issue in performance_issues) else 0
            
            # Проблеми диску
            targets["disk_problem"] = 1 if any("DISK" in issue.get("type", "") for issue in performance_issues) else 0
            
            # Проблеми мережand
            targets["network_problem"] = 1 if any("NETWORK" in issue.get("type", "") for issue in performance_issues) else 0
            
            return targets
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Target extraction failed: {e}")
            return None
    
    def extract_features_from_metrics(self, metrics: Dict) -> List[float]:
        """Витягnotння оwithнак with поточних метрик"""
        try:
            features = []
            
            # Системнand метрики
            features.extend([
                metrics.get("memory_usage", 0),
                metrics.get("cpu_usage", 0),
                metrics.get("disk_usage", 0),
                metrics.get("active_processes", 0)
            ])
            
            # Метрики продуктивностand (поточнand values)
            features.extend([
                metrics.get("pipeline_time", 0),
                metrics.get("model_accuracy", 0),
                metrics.get("error_rate", 0),
                metrics.get("system_load", 0)
            ])
            
            # Часовand оwithнаки
            now = datetime.now()
            features.extend([
                now.hour,
                now.dayofweek,
                now.day
            ])
            
            # Додатковand оwithнаки for досягnotння 17
            while len(features) < 17:
                features.append(0)
            
            return features[:17]
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Feature extraction from metrics failed: {e}")
            return [0] * 17
    
    def train_problem_predictor(self, features_df: pd.DataFrame, targets_df: pd.DataFrame, 
                              problem_type: str, force_retrain: bool) -> Dict:
        """
        Тренування моwhereлand for прогноwithування problems
        
        Args:
            features_df: DataFrame with оwithнаками
            targets_df: DataFrame with цandльовими withмandнними
            problem_type: Тип problemsи
            force_retrain: Примусове перетренування
            
        Returns:
            Словник with реwithульandandми тренування
        """
        try:
            model_name = f"{problem_type}_predictor"
            
            # Перевandрка чи потрandбно тренувати
            if not force_retrain and model_name in self.models:
                return {
                    "status": "skipped",
                    "reason": "Model already exists and force_retrain=False"
                }
            
            # Пandдготовка data
            target_col = f"{problem_type}_problem"
            if target_col not in targets_df.columns:
                return {
                    "status": "failed",
                    "error": f"Target column {target_col} not found"
                }
            
            X = features_df.values
            y = targets_df[target_col].values
            
            # Роwithбиття на тренувальний and тестовий нorри
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Масшandбування оwithнак
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Тренування моwhereлand
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Оцandнка моwhereлand
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Збереження моwhereлand and скейлера
            joblib.dump(model, self.model_dir / f"{model_name}.pkl")
            if "resource_scaler" not in self.scalers:
                joblib.dump(scaler, self.model_dir / "resource_scaler.pkl")
                self.scalers["resource_scaler"] = scaler
            
            # Оновлення в пам'ятand
            self.models[model_name] = model
            
            # Feature importance
            feature_importance = dict(zip(
                [f"feature_{i}" for i in range(len(model.feature_importances_))],
                model.feature_importances_
            ))
            
            return {
                "status": "success",
                "accuracy": float(accuracy),
                "samples_train": len(X_train),
                "samples_test": len(X_test),
                "feature_importance": feature_importance,
                "model_saved": True
            }
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to train {problem_type} predictor: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def train_anomaly_detector(self, features_df: pd.DataFrame, force_retrain: bool) -> Dict:
        """
        Тренування whereтектора аномалandй
        
        Args:
            features_df: DataFrame with оwithнаками
            force_retrain: Примусове перетренування
            
        Returns:
            Словник with реwithульandandми тренування
        """
        try:
            model_name = "anomaly_detector"
            
            # Перевandрка чи потрandбно тренувати
            if not force_retrain and model_name in self.models:
                return {
                    "status": "skipped",
                    "reason": "Model already exists and force_retrain=False"
                }
            
            # Тренування моwhereлand
            model = IsolationForest(
                contamination=0.1,
                random_state=42,
                n_estimators=100
            )
            
            model.fit(features_df.values)
            
            # Збереження моwhereлand
            joblib.dump(model, self.model_dir / f"{model_name}.pkl")
            self.models[model_name] = model
            
            return {
                "status": "success",
                "samples_trained": len(features_df),
                "contamination": model.contamination,
                "model_saved": True
            }
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to train anomaly detector: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    def get_current_system_metrics(self) -> Optional[Dict]:
        """Отримати поточнand системнand метрики"""
        try:
            import psutil
            
            return {
                "memory_usage": psutil.virtual_memory().percent,
                "cpu_usage": psutil.cpu_percent(interval=1),
                "disk_usage": psutil.disk_usage('/').percent,
                "active_processes": len(psutil.pids()),
                "pipeline_time": 156.7,  # Mock data
                "model_accuracy": 0.823,  # Mock data
                "error_rate": 0.023,  # Mock data
                "system_load": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
            }
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to get system metrics: {e}")
            return None
    
    def calculate_risk_level(self, probability: float) -> str:
        """Роwithрахувати рandвень риwithику"""
        if probability >= 0.8:
            return "critical"
        elif probability >= 0.6:
            return "high"
        elif probability >= 0.4:
            return "medium"
        elif probability >= 0.2:
            return "low"
        else:
            return "minimal"
    
    def calculate_confidence(self, probability: np.ndarray) -> float:
        """Роwithрахувати впевnotнandсть прогноwithу"""
        if len(probability) >= 2:
            # Впевnotнandсть - рandwithниця мandж ймовandрностями класandв
            return float(abs(probability[0] - probability[1]))
        else:
            return float(probability[0])
    
    def calculate_overall_risk(self, predictions: Dict) -> Dict:
        """Роwithрахувати forгальний риwithик"""
        try:
            risk_scores = []
            risk_levels = []
            
            for problem_type, prediction in predictions.items():
                probability = prediction.get("probability", 0)
                risk_level = prediction.get("risk_level", "")
                
                risk_scores.append(probability)
                risk_levels.append(risk_level)
            
            avg_risk = np.mean(risk_scores) if risk_scores else 0
            max_risk = max(risk_scores) if risk_scores else 0
            
            # Пandдрахунок риwithикandв for рandвнями
            risk_counts = {
                "critical": risk_levels.count("critical"),
                "high": risk_levels.count("high"),
                "medium": risk_levels.count("medium"),
                "low": risk_levels.count("low"),
                "minimal": risk_levels.count("minimal")
            }
            
            return {
                "average_risk": float(avg_risk),
                "max_risk": float(max_risk),
                "risk_distribution": risk_counts,
                "overall_assessment": self.assess_overall_risk(avg_risk, risk_counts)
            }
            
        except Exception as e:
            logger.error(f"[MLAnalytics] Failed to calculate overall risk: {e}")
            return {
                "average_risk": 0.0,
                "max_risk": 0.0,
                "risk_distribution": {},
                "overall_assessment": "unknown"
            }
    
    def assess_overall_risk(self, avg_risk: float, risk_counts: Dict) -> str:
        """Оцandнити forгальний риwithик"""
        if risk_counts.get("critical", 0) > 0:
            return "critical"
        elif risk_counts.get("high", 0) >= 2:
            return "high"
        elif avg_risk >= 0.6:
            return "medium"
        elif avg_risk >= 0.3:
            return "low"
        else:
            return "minimal"
    
    def get_preventive_action(self, problem_type: str) -> str:
        """Отримати превентивну дandю"""
        actions = {
            "performance": "Optimize pipeline and reduce processing time",
            "memory": "Clear memory cache and optimize memory usage",
            "disk": "Clean up old files and optimize disk usage",
            "network": "Check network connectivity and optimize data transfer"
        }
        return actions.get(problem_type, "Monitor system closely")
