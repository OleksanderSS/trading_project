"""
MODEL COMPARISON ENGINE
Двигун порandвняння моwhereлей for аналandwithу продуктивностand
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ComparisonMetric(Enum):
    """Метрики порandвняння"""
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1_SCORE = "f1_score"
    MAE = "mae"
    MSE = "mse"
    RMSE = "rmse"
    R2 = "r2"
    SHARPE_RATIO = "sharpe_ratio"
    MAX_DRAWDOWN = "max_drawdown"

@dataclass
class ModelResult:
    """Реwithульandт моwhereлand"""
    model_name: str
    model_type: str
    predictions: List[float]
    actual_values: List[float]
    metrics: Dict[str, float]
    training_time: float
    inference_time: float
    timestamp: datetime

class ModelComparisonEngine:
    """
    Двигун порandвняння моwhereлей
    """
    
    def __init__(self):
        """Інandцandалandforцandя"""
        self.logger = logging.getLogger(__name__)
        self.comparison_results = []
        self.best_models = {}
        
    def add_model_result(self, model_result: ModelResult) -> None:
        """Додати реwithульandт моwhereлand"""
        self.comparison_results.append(model_result)
        self.logger.info(f"Added result for model: {model_result.model_name}")
    
    def calculate_metrics(self, predictions: List[float], 
                         actual_values: List[float]) -> Dict[str, float]:
        """Роwithрахувати метрики"""
        try:
            predictions = np.array(predictions)
            actual_values = np.array(actual_values)
            
            metrics = {}
            
            # Basic regression metrics
            mse = np.mean((predictions - actual_values) ** 2)
            mae = np.mean(np.abs(predictions - actual_values))
            rmse = np.sqrt(mse)
            
            # R2 score
            ss_res = np.sum((actual_values - predictions) ** 2)
            ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Classification metrics (assuming binary classification threshold at 0)
            binary_preds = (predictions > 0).astype(int)
            binary_actual = (actual_values > 0).astype(int)
            
            tp = np.sum((binary_preds == 1) & (binary_actual == 1))
            tn = np.sum((binary_preds == 0) & (binary_actual == 0))
            fp = np.sum((binary_preds == 1) & (binary_actual == 0))
            fn = np.sum((binary_preds == 0) & (binary_actual == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
            # Financial metrics
            returns = predictions
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Max drawdown
            cumulative = np.cumprod(1 + returns / 100)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = np.min(drawdown)
            
            metrics.update({
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'r2': float(r2),
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1_score),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown)
            })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def compare_models(self, metric: ComparisonMetric = ComparisonMetric.F1_SCORE) -> Dict[str, Any]:
        """Порandвняти моwhereлand for метрикою"""
        if not self.comparison_results:
            return {'error': 'No model results to compare'}
        
        model_scores = {}
        for result in self.comparison_results:
            score = result.metrics.get(metric.value, 0)
            model_scores[result.model_name] = {
                'score': score,
                'model_type': result.model_type,
                'training_time': result.training_time,
                'inference_time': result.inference_time
            }
        
        # Sort by score
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        return {
            'metric': metric.value,
            'rankings': sorted_models,
            'best_model': sorted_models[0] if sorted_models else None,
            'total_models': len(sorted_models)
        }
    
    def get_best_model_by_metric(self, metric: ComparisonMetric) -> Optional[ModelResult]:
        """Отримати найкращу model for метрикою"""
        if not self.comparison_results:
            return None
        
        best_result = None
        best_score = float('-inf')
        
        for result in self.comparison_results:
            score = result.metrics.get(metric.value, 0)
            if score > best_score:
                best_score = score
                best_result = result
        
        return best_result
    
    def generate_comparison_report(self) -> Dict[str, Any]:
        """Згеnotрувати withвandт порandвняння"""
        if not self.comparison_results:
            return {'error': 'No model results to compare'}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(self.comparison_results),
            'model_types': list(set(r.model_type for r in self.comparison_results)),
            'metrics_comparison': {},
            'best_models': {},
            'detailed_results': []
        }
        
        # Compare by different metrics
        for metric in ComparisonMetric:
            comparison = self.compare_models(metric)
            report['metrics_comparison'][metric.value] = comparison
            
            if comparison.get('best_model'):
                best_model_name = comparison['best_model'][0]
                report['best_models'][metric.value] = best_model_name
        
        # Add detailed results
        for result in self.comparison_results:
            report['detailed_results'].append({
                'model_name': result.model_name,
                'model_type': result.model_type,
                'metrics': result.metrics,
                'training_time': result.training_time,
                'inference_time': result.inference_time,
                'timestamp': result.timestamp.isoformat()
            })
        
        return report
    
    def save_comparison_report(self, filepath: str) -> bool:
        """Зберегти withвandт порandвняння"""
        try:
            report = self.generate_comparison_report()
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Comparison report saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving comparison report: {e}")
            return False
    
    def load_comparison_report(self, filepath: str) -> bool:
        """Заванandжити withвandт порandвняння"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            # Reconstruct ModelResult objects
            for result_data in report.get('detailed_results', []):
                model_result = ModelResult(
                    model_name=result_data['model_name'],
                    model_type=result_data['model_type'],
                    predictions=[],  # Not stored in report
                    actual_values=[],  # Not stored in report
                    metrics=result_data['metrics'],
                    training_time=result_data['training_time'],
                    inference_time=result_data['inference_time'],
                    timestamp=datetime.fromisoformat(result_data['timestamp'])
                )
                self.comparison_results.append(model_result)
            
            self.logger.info(f"Comparison report loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading comparison report: {e}")
            return False
    
    def clear_results(self) -> None:
        """Очистити реwithульandти"""
        self.comparison_results.clear()
        self.best_models.clear()
        self.logger.info("Comparison results cleared")

def get_model_comparison_engine() -> ModelComparisonEngine:
    """Отримати екwithемпляр двигуна порandвняння моwhereлей"""
    return ModelComparisonEngine()

def main():
    """Тестова функцandя"""
    engine = ModelComparisonEngine()
    
    # Create test results
    test_results = [
        ModelResult(
            model_name="RandomForest_v1",
            model_type="RandomForest",
            predictions=np.random.randn(100).tolist(),
            actual_values=np.random.randn(100).tolist(),
            metrics={'accuracy': 0.85, 'mse': 0.12},
            training_time=120.5,
            inference_time=0.01,
            timestamp=datetime.now()
        ),
        ModelResult(
            model_name="XGBoost_v1",
            model_type="XGBoost",
            predictions=np.random.randn(100).tolist(),
            actual_values=np.random.randn(100).tolist(),
            metrics={'accuracy': 0.87, 'mse': 0.10},
            training_time=180.2,
            inference_time=0.005,
            timestamp=datetime.now()
        )
    ]
    
    for result in test_results:
        # Calculate actual metrics
        result.metrics = engine.calculate_metrics(result.predictions, result.actual_values)
        engine.add_model_result(result)
    
    # Generate report
    report = engine.generate_comparison_report()
    print("Model Comparison Report:")
    print(f"Total models: {report['total_models']}")
    print(f"Best by accuracy: {report['best_models'].get('accuracy', 'N/A')}")
    print(f"Best by MSE: {report['best_models'].get('mse', 'N/A')}")

if __name__ == "__main__":
    main()
