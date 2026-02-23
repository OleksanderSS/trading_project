# core/analysis/unified_model_comparison.py - Об'єднана система порandвняння моwhereлей

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import json

logger = logging.getLogger(__name__)

class UnifiedModelComparison:
    """
    Об'єднана система порandвняння моwhereлей
    """
    
    def __init__(self, results_dir: str = "results/comparison"):
        self.model_results = {}
        self.context_vectors = {}
        self.results_dir = results_dir
        self.comparison_history = []
        
        logger.info(f"[UnifiedModelComparison] Initialized with results_dir: {results_dir}")
    
    def add_model_result(self, model_name: str, model_type: str, 
                        ticker: str, interval: str, predictions: np.ndarray,
                        actual: np.ndarray, context_features: Dict[str, float]):
        """Додати реwithульandти моwhereлand"""
        key = f"{model_name}_{model_type}_{ticker}_{interval}"
        
        self.model_results[key] = {
            'predictions': predictions,
            'actual': actual,
            'model_type': model_type,
            'ticker': ticker,
            'interval': interval,
            'context_features': context_features,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.debug(f"[UnifiedModelComparison] Added result for {key}")
    
    def compare_models(self, model_keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Порandвняти моwhereлand"""
        try:
            if model_keys is None:
                model_keys = list(self.model_results.keys())
            
            comparison_results = {
                'timestamp': datetime.now().isoformat(),
                'models_compared': model_keys,
                'metrics': {},
                'statistical_tests': {},
                'rankings': {},
                'recommendations': {}
            }
            
            # Calculating метрики for кожної моwhereлand
            for key in model_keys:
                if key not in self.model_results:
                    continue
                
                result = self.model_results[key]
                predictions = result['predictions']
                actual = result['actual']
                
                # Виwithначаємо тип forдачand
                task_type = self._determine_task_type(predictions, actual)
                
                # Calculating метрики
                metrics = self._calculate_metrics(predictions, actual, task_type)
                comparison_results['metrics'][key] = metrics
            
            # Сandтистичнand тести
            comparison_results['statistical_tests'] = self._perform_statistical_tests(model_keys)
            
            # Ранжування моwhereлей
            comparison_results['rankings'] = self._rank_models(comparison_results['metrics'])
            
            # Рекомендацandї
            comparison_results['recommendations'] = self._generate_recommendations(comparison_results)
            
            # Зберandгаємо в andсторandю
            self.comparison_history.append(comparison_results)
            
            logger.info(f"[UnifiedModelComparison] Compared {len(model_keys)} models")
            return comparison_results
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error comparing models: {e}")
            return {'error': str(e)}
    
    def _determine_task_type(self, predictions: np.ndarray, actual: np.ndarray) -> str:
        """Виwithначити тип forдачand"""
        try:
            # Перевandряємо, чи це класифandкацandя
            unique_preds = len(np.unique(predictions))
            unique_actual = len(np.unique(actual))
            
            if unique_preds <= 5 and unique_actual <= 5:
                return 'classification'
            else:
                return 'regression'
        except:
            return 'regression'
    
    def _calculate_metrics(self, predictions: np.ndarray, actual: np.ndarray, task_type: str) -> Dict[str, float]:
        """Роwithрахувати метрики"""
        try:
            metrics = {}
            
            if task_type == 'classification':
                # Метрики класифandкацandї
                metrics['accuracy'] = accuracy_score(actual, predictions)
                metrics['precision'] = precision_score(actual, predictions, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(actual, predictions, average='weighted', zero_division=0)
                metrics['f1'] = f1_score(actual, predictions, average='weighted', zero_division=0)
            else:
                # Метрики регресandї
                metrics['mae'] = mean_absolute_error(actual, predictions)
                metrics['mse'] = mean_squared_error(actual, predictions)
                metrics['rmse'] = np.sqrt(metrics['mse'])
                metrics['r2'] = r2_score(actual, predictions)
                
                # Додатковand метрики
                metrics['mape'] = np.mean(np.abs((actual - predictions) / (actual + 1e-6))) * 100
                metrics['direction_accuracy'] = self._calculate_direction_accuracy(predictions, actual)
            
            return metrics
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error calculating metrics: {e}")
            return {}
    
    def _calculate_direction_accuracy(self, predictions: np.ndarray, actual: np.ndarray) -> float:
        """Роwithрахувати точнandсть напрямку"""
        try:
            pred_directions = np.sign(predictions)
            actual_directions = np.sign(actual)
            
            return np.mean(pred_directions == actual_directions)
        except:
            return 0.0
    
    def _perform_statistical_tests(self, model_keys: List[str]) -> Dict[str, Any]:
        """Виконати сandтистичнand тести"""
        try:
            tests = {}
            
            # Порandвняння попарно
            for i, key1 in enumerate(model_keys):
                for key2 in model_keys[i+1:]:
                    if key1 not in self.model_results or key2 not in self.model_results:
                        continue
                    
                    result1 = self.model_results[key1]
                    result2 = self.model_results[key2]
                    
                    pair_key = f"{key1}_vs_{key2}"
                    
                    # t-test for середнandх withначень
                    try:
                        t_stat, p_value = stats.ttest_rel(result1['predictions'], result2['actual'])
                        tests[pair_key] = {
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
                    except:
                        tests[pair_key] = {'error': 'Statistical test failed'}
            
            return tests
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error performing statistical tests: {e}")
            return {}
    
    def _rank_models(self, metrics: Dict[str, Dict[str, float]]) -> Dict[str, List[Tuple[str, float]]]:
        """Ранжувати моwhereлand"""
        try:
            rankings = {}
            
            # Ранжування по кожнandй метрицand
            all_metrics = set()
            for model_metrics in metrics.values():
                all_metrics.update(model_metrics.keys())
            
            for metric in all_metrics:
                if metric == 'mse' or metric == 'rmse' or metric == 'mae' or metric == 'mape':
                    # Чим менше, тим краще
                    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get(metric, float('inf')))
                else:
                    # Чим бandльше, тим краще
                    sorted_models = sorted(metrics.items(), key=lambda x: x[1].get(metric, -float('inf')), reverse=True)
                
                rankings[metric] = [(model, model_metrics.get(metric, 0)) for model, model_metrics in sorted_models]
            
            return rankings
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error ranking models: {e}")
            return {}
    
    def _generate_recommendations(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї"""
        try:
            recommendations = {
                'best_overall': None,
                'best_by_metric': {},
                'context_specific': {},
                'ensemble_candidates': []
            }
            
            rankings = comparison_results.get('rankings', {})
            
            # Найкраща model forгалом (for R2 or accuracy)
            if 'r2' in rankings:
                recommendations['best_overall'] = rankings['r2'][0]
            elif 'accuracy' in rankings:
                recommendations['best_overall'] = rankings['accuracy'][0]
            
            # Найкращand по кожнandй метрицand
            for metric, ranking in rankings.items():
                if ranking:
                    recommendations['best_by_metric'][metric] = ranking[0]
            
            # Кандидати for ансамблю (топ-3 моwhereлand)
            if 'r2' in rankings:
                top_models = rankings['r2'][:3]
                recommendations['ensemble_candidates'] = [model for model, _ in top_models]
            elif 'accuracy' in rankings:
                top_models = rankings['accuracy'][:3]
                recommendations['ensemble_candidates'] = [model for model, _ in top_models]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error generating recommendations: {e}")
            return {}
    
    def compare_with_context(self, context_features: Dict[str, float]) -> Dict[str, Any]:
        """Порandвняти моwhereлand with урахуванням контексту"""
        try:
            # Знаходимо моwhereлand withand схожим контекстом
            similar_models = self._find_similar_context_models(context_features)
            
            # Порandвнюємо моwhereлand
            comparison_results = self.compare_models()
            
            # Додаємо контекстну andнформацandю
            comparison_results['context_analysis'] = {
                'current_context': context_features,
                'similar_models': similar_models,
                'context_adjusted_rankings': self._adjust_rankings_by_context(comparison_results, context_features)
            }
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error in context comparison: {e}")
            return {'error': str(e)}
    
    def _find_similar_context_models(self, current_context: Dict[str, float]) -> List[Dict]:
        """Find моwhereлand withand схожим контекстом"""
        try:
            similar_models = []
            
            for key, result in self.model_results.items():
                model_context = result.get('context_features', {})
                
                similarity = self._calculate_context_similarity(current_context, model_context)
                
                if similarity > 0.7:  # Порandг схожостand
                    similar_models.append({
                        'model_key': key,
                        'similarity': similarity,
                        'context': model_context
                    })
            
            # Сортуємо for схожandстю
            similar_models.sort(key=lambda x: x['similarity'], reverse=True)
            return similar_models[:5]  # Поверandємо топ-5
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error finding similar models: {e}")
            return []
    
    def _calculate_context_similarity(self, ctx1: Dict[str, float], ctx2: Dict[str, float]) -> float:
        """Роwithрахувати схожandсть контекстandв"""
        try:
            common_features = set(ctx1.keys()) & set(ctx2.keys())
            
            if not common_features:
                return 0.0
            
            similarity = 0.0
            for feature in common_features:
                val1 = ctx1.get(feature, 0)
                val2 = ctx2.get(feature, 0)
                
                if abs(val1) + abs(val2) > 0:
                    diff = abs(val1 - val2) / (abs(val1) + abs(val2))
                    similarity += 1.0 - diff
            
            return similarity / len(common_features)
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error calculating context similarity: {e}")
            return 0.0
    
    def _adjust_rankings_by_context(self, comparison_results: Dict[str, Any], context: Dict[str, float]) -> Dict[str, List[Tuple[str, float]]]:
        """Коригувати ранжування with урахуванням контексту"""
        try:
            rankings = comparison_results.get('rankings', {})
            adjusted_rankings = {}
            
            for metric, ranking in rankings.items():
                adjusted_ranking = []
                
                for model_key, score in ranking:
                    # Знаходимо контекст моwhereлand
                    model_context = self.model_results.get(model_key, {}).get('context_features', {})
                    
                    # Calculating схожandсть контекстandв
                    similarity = self._calculate_context_similarity(context, model_context)
                    
                    # Коригуємо скор with урахуванням схожостand
                    adjusted_score = score * (0.7 + 0.3 * similarity)
                    adjusted_ranking.append((model_key, adjusted_score))
                
                # Пересортуємо
                adjusted_ranking.sort(key=lambda x: x[1], reverse=True)
                adjusted_rankings[metric] = adjusted_ranking
            
            return adjusted_rankings
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error adjusting rankings: {e}")
            return comparison_results.get('rankings', {})
    
    def save_comparison_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Зберегти реwithульandти порandвняння"""
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"model_comparison_{timestamp}.json"
            
            filepath = f"{self.results_dir}/{filename}"
            
            # Створюємо папку якщо not andснує
            import os
            os.makedirs(self.results_dir, exist_ok=True)
            
            # Конвертуємо for JSON
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                return obj
            
            json_results = convert_for_json(results)
            
            with open(filepath, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"[UnifiedModelComparison] Results saved to {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error saving results: {e}")
            return ""
    
    def get_comparison_summary(self) -> Dict[str, Any]:
        """Отримати пandдсумок порandвнянь"""
        try:
            summary = {
                'total_comparisons': len(self.comparison_history),
                'models_compared': set(),
                'avg_performance_by_model': {},
                'best_models_by_metric': {},
                'recent_comparison': self.comparison_history[-1] if self.comparison_history else None
            }
            
            # Збираємо all моwhereлand
            for comparison in self.comparison_history:
                models = comparison.get('models_compared', [])
                summary['models_compared'].update(models)
            
            summary['models_compared'] = list(summary['models_compared'])
            
            return summary
            
        except Exception as e:
            logger.error(f"[UnifiedModelComparison] Error getting summary: {e}")
            return {'error': str(e)}

# Функцandї for сумandсностand
def create_model_comparison_engine() -> UnifiedModelComparison:
    """Create движок порandвняння моwhereлей (сумandснandсть)"""
    return UnifiedModelComparison()

def create_module_comparison_analyzer() -> UnifiedModelComparison:
    """Create аналandforтор порandвняння модулandв (сумandснandсть)"""
    return UnifiedModelComparison()

if __name__ == "__main__":
    # Тестування
    comparison = UnifiedModelComparison()
    
    # Тестовand данand
    predictions1 = np.array([1, 2, 3, 4, 5])
    predictions2 = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
    actual = np.array([1, 2, 3, 4, 5])
    
    # Додаємо реwithульandти
    comparison.add_model_result("model1", "regression", "SPY", "1d", predictions1, actual, {})
    comparison.add_model_result("model2", "regression", "SPY", "1d", predictions2, actual, {})
    
    # Порandвнюємо
    results = comparison.compare_models()
    print(f"Comparison completed: {len(results.get('metrics', {}))} models compared")
