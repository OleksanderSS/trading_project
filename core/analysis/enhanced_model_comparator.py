# core/analysis/enhanced_model_comparator.py - Покращена система порandвняння моwhereлей

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
import json
import time
from datetime import datetime
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, roc_auc_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from core.models.model_interface import ModelFactory, BaseModel
from core.analysis.model_comparison_engine import ModelComparisonEngine

logger = logging.getLogger(__name__)

class EnhancedModelComparator:
    """Покращена система порandвняння моwhereлей with whereandльною аналandтикою"""
    
    def __init__(self, results_dir: str = "results/model_comparison"):
        self.results_dir = results_dir
        self.comparison_results = {}
        self.model_results = {}
        self.statistical_tests = {}
        
        import os
        os.makedirs(results_dir, exist_ok=True)
    
    def compare_models(self, models_config: Dict[str, Dict], 
                      X_train: np.ndarray, X_test: np.ndarray, 
                      y_train: np.ndarray, y_test: np.ndarray,
                      ticker: str = "DEFAULT", timeframe: str = "1d") -> Dict[str, Any]:
        """
        Порandвняння моwhereлей with whereandльною аналandтикою
        
        Args:
            models_config: Конфandгурацandя моwhereлей {model_name: {type, category, task_type}}
            X_train, X_test, y_train, y_test: Данand for тренування and тестування
            ticker, timeframe: Параметри for andwhereнтифandкацandї
            
        Returns:
            Dict with реwithульandandми порandвняння
        """
        logger.info(f"[EnhancedModelComparator] Starting comparison for {len(models_config)} models")
        
        start_time = time.time()
        comparison_id = f"{ticker}_{timeframe}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Тренування and оцandнка allх моwhereлей
        model_results = {}
        
        for model_name, config in models_config.items():
            try:
                logger.info(f"[EnhancedModelComparator] Training {model_name}")
                
                # Створення моwhereлand
                model = ModelFactory.create_model(
                    model_type=config["type"],
                    model_category=config["category"],
                    task_type=config.get("task_type", "regression")
                )
                
                # Тренування
                train_result = model.train(X_train, y_train, ticker, timeframe)
                
                if train_result.get("status") != "success":
                    logger.warning(f"[EnhancedModelComparator] {model_name} training failed")
                    continue
                
                # Прогноwithування
                y_pred = model.predict(X_test)
                
                # Метрики якостand
                metrics = model.evaluate(X_test, y_test)
                
                # Додатковand метрики
                if config.get("task_type", "regression") == "regression":
                    metrics.update(self._calculate_regression_metrics(y_test, y_pred))
                else:
                    metrics.update(self._calculate_classification_metrics(y_test, y_pred))
                
                model_results[model_name] = {
                    "model": model,
                    "config": config,
                    "metrics": metrics,
                    "predictions": y_pred,
                    "actual": y_test,
                    "training_time": train_result.get("training_time", 0),
                    "status": "success"
                }
                
                logger.info(f"[EnhancedModelComparator] {model_name} completed - R2: {metrics.get('r2', 'N/A'):.4f}")
                
            except Exception as e:
                logger.error(f"[EnhancedModelComparator] Error with {model_name}: {e}")
                model_results[model_name] = {
                    "model": None,
                    "config": config,
                    "metrics": {},
                    "predictions": np.array([]),
                    "actual": y_test,
                    "training_time": 0,
                    "status": "error",
                    "error": str(e)
                }
        
        # Сandтистичнand тести
        statistical_results = self._perform_statistical_tests(model_results)
        
        # Рейтинг моwhereлей
        model_ranking = self._rank_models(model_results)
        
        # Вandwithуалandforцandя
        plots = self._create_visualizations(model_results, comparison_id)
        
        # Збереження реwithульandтandв
        comparison_results = {
            "comparison_id": comparison_id,
            "ticker": ticker,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "total_time": time.time() - start_time,
            "model_results": model_results,
            "statistical_tests": statistical_results,
            "model_ranking": model_ranking,
            "plots": plots,
            "summary": self._generate_summary(model_results, statistical_results, model_ranking)
        }
        
        # Збереження
        self._save_comparison_results(comparison_results, comparison_id)
        
        logger.info(f"[EnhancedModelComparator] Comparison completed in {time.time() - start_time:.2f}s")
        
        return comparison_results
    
    def _calculate_regression_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Роwithрахунок додаткових регресandйних метрик"""
        try:
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))) * 100
            
            # Сигна про помилку
            residuals = y_true - y_pred
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Нормалandwithованand метрики
            norm_rmse = rmse / (np.max(y_true) - np.min(y_true))
            
            # Кореляцandя
            correlation = np.corrcoef(y_true, y_pred)[0, 1]
            
            return {
                "mape": mape,
                "rmse": rmse,
                "norm_rmse": norm_rmse,
                "correlation": correlation,
                "residuals_mean": np.mean(residuals),
                "residuals_std": np.std(residuals)
            }
            
        except Exception as e:
            logger.error(f"Error calculating regression metrics: {e}")
            return {}
    
    def _calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Роwithрахунок додаткових класифandкацandйних метрик"""
        try:
            from sklearn.metrics import precision_score, recall_score, confusion_matrix
            
            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            return {
                "precision": precision,
                "recall": recall,
                "confusion_matrix": cm.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating classification metrics: {e}")
            return {}
    
    def _perform_statistical_tests(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Виконання сandтистичних тестandв мandж моwhereлями"""
        try:
            successful_models = {k: v for k, v in model_results.items() if v["status"] == "success"}
            
            if len(successful_models) < 2:
                return {"error": "Need at least 2 successful models for statistical tests"}
            
            statistical_results = {}
            
            # Порandвняння попарно
            model_names = list(successful_models.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], i+1):
                    pair_key = f"{model1}_vs_{model2}"
                    
                    # Тест на рandwithницю середнandх
                    actual1 = successful_models[model1]["actual"]
                    actual2 = successful_models[model2]["actual"]
                    pred1 = successful_models[model1]["predictions"]
                    pred2 = successful_models[model2]["predictions"]
                    
                    # Errors
                    errors1 = np.abs(actual1 - pred1)
                    errors2 = np.abs(actual2 - pred2)
                    
                    # T-test
                    t_stat, p_value = stats.ttest_rel(errors1, errors2)
                    
                    # Wilcoxon test
                    w_stat, w_p_value = stats.wilcoxon(errors1, errors2)
                    
                    statistical_results[pair_key] = {
                        "t_test": {"statistic": t_stat, "p_value": p_value},
                        "wilcoxon": {"statistic": w_stat, "p_value": w_p_value},
                        "mean_error_diff": np.mean(errors1) - np.mean(errors2),
                        "significant_difference": p_value < 0.05
                    }
            
            return statistical_results
            
        except Exception as e:
            logger.error(f"Error performing statistical tests: {e}")
            return {"error": str(e)}
    
    def _rank_models(self, model_results: Dict[str, Dict]) -> Dict[str, List[Tuple[str, float]]]:
        """Ранжування моwhereлей for рandwithними метриками"""
        try:
            successful_models = {k: v for k, v in model_results.items() if v["status"] == "success"}
            
            if not successful_models:
                return {}
            
            rankings = {}
            
            # Ранжування for R2 (for регресandї)
            r2_ranking = sorted(
                [(name, result["metrics"].get("r2", 0)) for name, result in successful_models.items()],
                key=lambda x: x[1], reverse=True
            )
            rankings["r2"] = r2_ranking
            
            # Ранжування for RMSE (менше краще)
            rmse_ranking = sorted(
                [(name, result["metrics"].get("rmse", float('inf'))) for name, result in successful_models.items()],
                key=lambda x: x[1]
            )
            rankings["rmse"] = rmse_ranking
            
            # Ранжування for часом тренування (менше краще)
            time_ranking = sorted(
                [(name, result["training_time"]) for name, result in successful_models.items()],
                key=lambda x: x[1]
            )
            rankings["training_time"] = time_ranking
            
            # Комбandнований рейтинг
            combined_scores = {}
            for name, result in successful_models.items():
                # Нормалandwithованand метрики (0-1, where 1 - найкращий)
                r2_score = result["metrics"].get("r2", 0)
                rmse_score = 1 / (1 + result["metrics"].get("rmse", float('inf')))
                time_score = 1 / (1 + result["training_time"])
                
                combined_scores[name] = (r2_score * 0.5 + rmse_score * 0.3 + time_score * 0.2)
            
            combined_ranking = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
            rankings["combined"] = combined_ranking
            
            return rankings
            
        except Exception as e:
            logger.error(f"Error ranking models: {e}")
            return {}
    
    def _create_visualizations(self, model_results: Dict[str, Dict], comparison_id: str) -> Dict[str, str]:
        """Створення вandwithуалandforцandй"""
        try:
            plots = {}
            successful_models = {k: v for k, v in model_results.items() if v["status"] == "success"}
            
            if not successful_models:
                return plots
            
            # Графandк порandвняння прогноwithandв
            plt.figure(figsize=(12, 8))
            
            for name, result in successful_models.items():
                plt.scatter(result["actual"], result["predictions"], alpha=0.6, label=name, s=20)
            
            # Іwhereальна лandнandя
            min_val = min([min(result["actual"].min(), result["predictions"].min()) for result in successful_models.values()])
            max_val = max([max(result["actual"].max(), result["predictions"].max()) for result in successful_models.values()])
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Model Predictions vs Actual Values')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plot_path = f"{self.results_dir}/predictions_comparison_{comparison_id}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots["predictions_comparison"] = plot_path
            
            # Графandк метрик
            metrics_data = []
            for name, result in successful_models.items():
                metrics_data.append({
                    "model": name,
                    "r2": result["metrics"].get("r2", 0),
                    "rmse": result["metrics"].get("rmse", 0),
                    "training_time": result["training_time"]
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # R2
            axes[0].bar(metrics_df["model"], metrics_df["r2"])
            axes[0].set_title('R Score')
            axes[0].set_ylabel('R')
            axes[0].tick_params(axis='x', rotation=45)
            
            # RMSE
            axes[1].bar(metrics_df["model"], metrics_df["rmse"])
            axes[1].set_title('RMSE')
            axes[1].set_ylabel('RMSE')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Training Time
            axes[2].bar(metrics_df["model"], metrics_df["training_time"])
            axes[2].set_title('Training Time (s)')
            axes[2].set_ylabel('Time (s)')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            metrics_plot_path = f"{self.results_dir}/metrics_comparison_{comparison_id}.png"
            plt.savefig(metrics_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            plots["metrics_comparison"] = metrics_plot_path
            
            return plots
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return {}
    
    def _generate_summary(self, model_results: Dict[str, Dict], 
                         statistical_tests: Dict[str, Any], 
                         model_ranking: Dict[str, List[Tuple[str, float]]]) -> Dict[str, Any]:
        """Геnotрацandя пandдсумку порandвняння"""
        try:
            successful_models = {k: v for k, v in model_results.items() if v["status"] == "success"}
            failed_models = {k: v for k, v in model_results.items() if v["status"] != "success"}
            
            summary = {
                "total_models": len(model_results),
                "successful_models": len(successful_models),
                "failed_models": len(failed_models),
                "best_models": {},
                "significant_differences": [],
                "recommendations": []
            }
            
            # Найкращand моwhereлand for рandwithними метриками
            if model_ranking:
                for metric, ranking in model_ranking.items():
                    if ranking:
                        summary["best_models"][metric] = ranking[0][0]
            
            # Значущand рandwithницand
            if statistical_tests and not statistical_tests.get("error"):
                for pair, tests in statistical_tests.items():
                    if tests.get("significant_difference"):
                        summary["significant_differences"].append({
                            "pair": pair,
                            "p_value": tests["t_test"]["p_value"],
                            "mean_error_diff": tests["mean_error_diff"]
                        })
            
            # Рекомендацandї
            if summary["successful_models"] > 0:
                best_overall = summary["best_models"].get("combined")
                if best_overall:
                    summary["recommendations"].append(f"Best overall model: {best_overall}")
                
                best_r2 = summary["best_models"].get("r2")
                if best_r2:
                    summary["recommendations"].append(f"Best R model: {best_r2}")
                
                fastest = summary["best_models"].get("training_time")
                if fastest:
                    summary["recommendations"].append(f"Fastest model: {fastest}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def _save_comparison_results(self, results: Dict[str, Any], comparison_id: str):
        """Збереження реwithульandтandв порandвняння"""
        try:
            # Збереження JSON
            json_path = f"{self.results_dir}/comparison_{comparison_id}.json"
            
            # Конверandцandя numpy arrays for JSON
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(v) for v in obj]
                return obj
            
            json_results = convert_numpy(results)
            
            with open(json_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            # Збереження CSV with метриками
            if results.get("model_results"):
                metrics_data = []
                for name, result in results["model_results"].items():
                    if result["status"] == "success":
                        metrics_row = {"model": name}
                        metrics_row.update(result["metrics"])
                        metrics_row["training_time"] = result["training_time"]
                        metrics_data.append(metrics_row)
                
                if metrics_data:
                    metrics_df = pd.DataFrame(metrics_data)
                    csv_path = f"{self.results_dir}/metrics_{comparison_id}.csv"
                    metrics_df.to_csv(csv_path, index=False)
            
            logger.info(f"[EnhancedModelComparator] Results saved for {comparison_id}")
            
        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")
    
    def load_comparison_results(self, comparison_id: str) -> Optional[Dict[str, Any]]:
        """Заванandження реwithульandтandв порandвняння"""
        try:
            json_path = f"{self.results_dir}/comparison_{comparison_id}.json"
            
            with open(json_path, 'r') as f:
                results = json.load(f)
            
            return results
            
        except Exception as e:
            logger.error(f"Error loading comparison results: {e}")
            return None
    
    def get_comparison_history(self) -> List[Dict[str, Any]]:
        """Отримати andсторandю порandвнянь"""
        try:
            import os
            json_files = [f for f in os.listdir(self.results_dir) if f.startswith("comparison_") and f.endswith(".json")]
            
            history = []
            for file in sorted(json_files, reverse=True):
                comparison_id = file.replace("comparison_", "").replace(".json", "")
                results = self.load_comparison_results(comparison_id)
                if results:
                    history.append({
                        "comparison_id": comparison_id,
                        "timestamp": results.get("timestamp"),
                        "ticker": results.get("ticker"),
                        "timeframe": results.get("timeframe"),
                        "total_models": results.get("summary", {}).get("total_models", 0),
                        "successful_models": results.get("summary", {}).get("successful_models", 0),
                        "best_model": results.get("summary", {}).get("best_models", {}).get("combined")
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting comparison history: {e}")
            return []
