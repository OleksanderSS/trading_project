#!/usr/bin/env python3
"""
Correlation Engine - Model correlation and diversity calculations
Handles mathematical calculations for model correlation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mutual_info_score

from src.core.logging.logger import ProjectLogger
from src.core.error_handling.error_handler import IErrorHandler, ErrorHandler

logger = ProjectLogger.get_logger("CorrelationEngine")


class CorrelationEngine:
    """
    Correlation engine for model correlation analysis.
    
    Handles:
    - Prediction correlation matrix calculation
    - Diversity metrics calculation (entropy, disagreement, variance)
    - Redundancy detection
    - Optimal subset selection
    - Weight adjustment based on correlation
    - Analysis summary and trends
    """
    
    def __init__(self, correlation_method: str = "pearson", diversity_threshold: float = 0.7, 
                 error_handler: Optional[IErrorHandler] = None):
        """
        Initialize Correlation Engine.
        
        Args:
            correlation_method: Method for correlation calculation
            diversity_threshold: Threshold for diversity filtering
            error_handler: Error handler instance
        """
        self.logger = logger
        self.correlation_method = correlation_method
        self.diversity_threshold = diversity_threshold
        self.error_handler = error_handler or ErrorHandler()
        
        # Analysis cache
        self.correlation_cache = {}
        self.diversity_cache = {}
        self.analysis_history = []
        
        # Correlation methods configuration
        self.CORRELATION_METHODS = {
            'pearson': {'description': 'Pearson correlation coefficient', 'range': (-1, 1)},
            'spearman': {'description': 'Spearman rank correlation', 'range': (-1, 1)},
            'mutual_info': {'description': 'Mutual information score', 'range': (0, float('inf'))},
            'disagreement': {'description': 'Prediction disagreement measure', 'range': (0, 1)}
        }
        
        self.logger.info(f"✅ CorrelationEngine initialized with method: {correlation_method}")
    
    def analyze_correlation(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                          sample_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze prediction correlation between models.
        
        Args:
            models: Dictionary of model_name -> model_object
            X: Test features
            y: True targets
            sample_size: Number of samples to use (None for all)
            
        Returns:
            Dict with correlation analysis results
        """
        self.logger.info(f"🔍 Analyzing correlation between {len(models)} models")
        
        results = {
            'timestamp': datetime.now(),
            'correlation_method': self.correlation_method,
            'n_models': len(models),
            'sample_size': sample_size or len(X),
            'correlation_matrix': {},
            'diversity_metrics': {},
            'redundant_pairs': [],
            'optimal_subsets': {}
        }
        
        try:
            # Sample data if requested
            if sample_size and sample_size < len(X):
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X.iloc[indices]
                y_sample = y.iloc[indices]
            else:
                X_sample = X
                y_sample = y
            
            # Get predictions from all models
            predictions = self.get_model_predictions(models, X_sample)
            
            if len(predictions) < 2:
                raise ValueError('Need at least 2 models for correlation analysis')
            
            # Calculate correlation matrix
            correlation_matrix = self.calculate_correlation_matrix(predictions)
            results['correlation_matrix'] = correlation_matrix
            
            # Calculate diversity metrics
            diversity_metrics = self.calculate_diversity_metrics(predictions, correlation_matrix)
            results['diversity_metrics'] = diversity_metrics
            
            # Find redundant pairs
            redundant_pairs = self.find_redundant_pairs(correlation_matrix, diversity_metrics)
            results['redundant_pairs'] = redundant_pairs
            
            # Select optimal subsets
            optimal_subsets = self.select_optimal_subsets(predictions, correlation_matrix, diversity_metrics)
            results['optimal_subsets'] = optimal_subsets
            
            # Store in cache and history
            cache_key = f"{len(models)}_{sample_size or len(X)}"
            self.correlation_cache[cache_key] = results
            self.analysis_history.append(results)
            
            self.logger.info(f"✅ Correlation analysis complete. Redundant pairs: {len(redundant_pairs)}")
            return results
            
        except (ValueError, TypeError, KeyError, Exception) as e:
            self.logger.error(f"Error in correlation analysis: {e}", exc_info=True)
            self.error_handler.handle_error(e, context={'models_count': len(models)})
            raise RuntimeError(f"Correlation analysis failed: {e}") from e
    
    def get_model_predictions(self, models: Dict[str, Any], X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        predictions = {}
        for model_name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                # Keep behavior (skip model), but preserve stack trace for debugging.
                self.logger.error(
                    f"Error getting predictions from {model_name}: {e}",
                    exc_info=True,
                )
                continue
        return predictions
    
    def calculate_correlation_matrix(self, predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Calculate correlation matrix between model predictions."""
        try:
            model_names = list(predictions.keys())
            correlation_matrix = {}
            
            for i, model1 in enumerate(model_names):
                correlation_matrix[model1] = {}
                pred1 = predictions[model1]
                
                for j, model2 in enumerate(model_names):
                    pred2 = predictions[model2]
                    
                    if i == j:
                        correlation_matrix[model1][model2] = 1.0
                    else:
                        correlation = self.calculate_correlation(pred1, pred2)
                        correlation_matrix[model1][model2] = correlation
            
            return correlation_matrix
        except Exception as e:
            # Returning {} is a valid fallback for callers, but don't hide why.
            self.logger.error(
                f"Error calculating correlation matrix: {e}", exc_info=True
            )
            raise RuntimeError("Failed to calculate correlation matrix") from e
    
    def calculate_correlation(self, pred1: np.ndarray, pred2: np.ndarray) -> float:
        """Calculate correlation between two prediction arrays."""
        try:
            if self.correlation_method == 'pearson':
                correlation, _ = pearsonr(pred1, pred2)
            elif self.correlation_method == 'spearman':
                correlation, _ = spearmanr(pred1, pred2)
            elif self.correlation_method == 'mutual_info':
                pred1_discrete = pd.qcut(pred1, q=10, labels=False, duplicates='drop')
                pred2_discrete = pd.qcut(pred2, q=10, labels=False, duplicates='drop')
                correlation = mutual_info_score(pred1_discrete, pred2_discrete)
            elif self.correlation_method == 'disagreement':
                agreement = np.mean(pred1 == pred2)
                correlation = 1 - agreement
            else:
                correlation, _ = pearsonr(pred1, pred2)
            
            if np.isnan(correlation):
                self.logger.warning("Correlation result is NaN, returning 0.0")
                return 0.0
            return float(correlation)
        except (ValueError, TypeError, Exception) as e:
            self.logger.error(f"Error calculating correlation: {e}",
                exc_info=True)
            self.error_handler.handle_error(e, context={'method': self.correlation_method})
            raise RuntimeError(f"Correlation calculation failed: {e}") from e
    
    def calculate_diversity_metrics(self, predictions: Dict[str, np.ndarray],
                                    correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate diversity metrics for the ensemble."""
        try:
            metrics = {}
            pred_df = pd.DataFrame(predictions)
            
            metrics['entropy'] = self.calculate_prediction_entropy(pred_df)
            metrics['disagreement'] = self.calculate_average_disagreement(correlation_matrix)
            metrics['variance'] = self.calculate_prediction_variance(pred_df)
            metrics['correlation_penalty'] = self.calculate_correlation_penalty(correlation_matrix)
            metrics['overall_diversity'] = self.calculate_overall_diversity_score(metrics)
            
            return metrics
        except Exception as e:
            self.logger.error(
                f"Error calculating diversity metrics: {e}", exc_info=True
            )
            raise RuntimeError("Failed to calculate diversity metrics") from e
    
    def calculate_prediction_entropy(self, pred_df: pd.DataFrame) -> float:
        """Calculate entropy of prediction distribution."""
        try:
            entropies = []
            for col in pred_df.columns:
                discrete_preds = pd.qcut(pred_df[col], q=10, labels=False, duplicates='drop')
                value_counts = pd.Series(discrete_preds).value_counts(normalize=True)
                entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                entropies.append(entropy)
            return np.mean(entropies) if entropies else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating prediction entropy: {e}",
                exc_info=True)
            return 0.0
    
    def calculate_average_disagreement(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average pairwise disagreement."""
        try:
            model_names = list(correlation_matrix.keys())
            disagreements = []
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        corr = correlation_matrix[model1][model2]
                        disagreement = 1 - abs(corr)
                        disagreements.append(disagreement)
            
            return np.mean(disagreements) if disagreements else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating average disagreement: {e}",
                exc_info=True)
            return 0.0
    
    def calculate_prediction_variance(self, pred_df: pd.DataFrame) -> float:
        """Calculate variance of predictions across models."""
        try:
            sample_variances = pred_df.var(axis=1)
            return np.mean(sample_variances)
        except Exception as e:
            self.logger.error(f"Error calculating prediction variance: {e}",
                exc_info=True)
            return 0.0
    
    def calculate_correlation_penalty(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate penalty for high correlation."""
        try:
            model_names = list(correlation_matrix.keys())
            correlations = []
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        corr = abs(correlation_matrix[model1][model2])
                        correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0.0
            return avg_correlation
        except Exception as e:
            self.logger.error(f"Error calculating correlation penalty: {e}",
                exc_info=True)
            return 0.0
    
    def calculate_overall_diversity_score(self, diversity_metrics: Dict[str, float]) -> float:
        """Calculate overall diversity score from individual metrics."""
        try:
            normalized_metrics = {}
            
            for metric in ['entropy', 'disagreement', 'variance']:
                if metric in diversity_metrics:
                    normalized_metrics[metric] = min(diversity_metrics[metric] / 2.0, 1.0)
            
            if 'correlation_penalty' in diversity_metrics:
                normalized_metrics['correlation_penalty'] = 1.0 - min(diversity_metrics['correlation_penalty'], 1.0)
            
            if normalized_metrics:
                overall_score = np.mean(list(normalized_metrics.values()))
            else:
                overall_score = 0.0
            
            return overall_score
        except Exception as e:
            self.logger.error(f"Error calculating overall diversity score: {e}"
                , exc_info=True)
            return 0.0
    
    def find_redundant_pairs(self, correlation_matrix: Dict[str, Dict[str, float]],
                            diversity_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find redundant model pairs based on correlation."""
        try:
            redundant_pairs = []
            model_names = list(correlation_matrix.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        correlation = abs(correlation_matrix[model1][model2])
                        
                        if correlation > self.diversity_threshold:
                            redundant_pairs.append({
                                'model1': model1,
                                'model2': model2,
                                'correlation': correlation,
                                'redundancy_level': 'high' if correlation > 0.9 else 'medium'
                            })
            
            return redundant_pairs
        except Exception as e:
            self.logger.error(f"Error finding redundant pairs: {e}")
            raise RuntimeError("Failed to find redundant model pairs") from e
    
    def select_optimal_subsets(self, predictions: Dict[str, np.ndarray],
                              correlation_matrix: Dict[str, Dict[str, float]],
                              diversity_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Select optimal subsets of models for different ensemble sizes."""
        try:
            optimal_subsets = {}
            model_names = list(predictions.keys())
            
            for subset_size in range(2, min(len(model_names), 8) + 1):
                best_subset = self.find_best_subset(
                    model_names, predictions, correlation_matrix, subset_size
                )
                if best_subset:
                    optimal_subsets[f'size_{subset_size}'] = best_subset
            
            return optimal_subsets
        except Exception as e:
            self.logger.error(f"Error selecting optimal subsets: {e}")
            raise RuntimeError("Failed to select optimal model subsets") from e
    
    def find_best_subset(self, model_names: List[str], predictions: Dict[str, np.ndarray],
                        correlation_matrix: Dict[str, Dict[str, float]], subset_size: int) -> Optional[Dict[str, Any]]:
        """Find best subset of given size using greedy algorithm."""
        try:
            best_subset = None
            best_score = -float('inf')
            
            remaining_models = model_names.copy()
            selected_models = []
            
            while len(selected_models) < subset_size and remaining_models:
                best_candidate = None
                best_candidate_score = -float('inf')
                
                for candidate in remaining_models:
                    test_subset = selected_models + [candidate]
                    subset_predictions = {name: predictions[name] for name in test_subset}
                    subset_correlation = {
                        name1: {name2: correlation_matrix[name1][name2] for name2 in test_subset}
                        for name1 in test_subset
                    }
                    
                    diversity_score = self.calculate_subset_diversity_score(
                        subset_predictions, subset_correlation
                    )
                    
                    if np.isfinite(diversity_score) and diversity_score > best_candidate_score:
                        best_candidate_score = diversity_score
                        best_candidate = candidate
                
                if best_candidate is None:
                    self.logger.warning(
                        "No finite diversity candidate found; stopping subset selection early."
                    )
                    break
                selected_models.append(best_candidate)
                remaining_models.remove(best_candidate)
            
            if selected_models:
                subset_predictions = {name: predictions[name] for name in selected_models}
                subset_correlation = {
                    name1: {name2: correlation_matrix[name1][name2] for name2 in selected_models}
                    for name1 in selected_models
                }
                
                final_diversity = self.calculate_subset_diversity_score(
                    subset_predictions, subset_correlation
                )
                
                best_subset = {
                    'models': selected_models,
                    'diversity_score': final_diversity,
                    'size': len(selected_models),
                    'method': 'greedy_selection'
                }
            
            return best_subset
        except Exception as e:
            self.logger.error(f"Error finding best subset: {e}")
            raise RuntimeError("Failed to find best model subset") from e
    
    def calculate_subset_diversity_score(self, predictions: Dict[str, np.ndarray],
                                        correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate diversity score for a subset of models."""
        try:
            subset_metrics = self.calculate_diversity_metrics(predictions, correlation_matrix)
            return subset_metrics.get('overall_diversity', 0.0)
        except Exception as e:
            self.logger.error(f"Error calculating subset diversity score: {e}",
                exc_info=True)
            return 0.0
    
    def select_diverse_subset(self, models: Dict[str, Any], X: pd.DataFrame, y: pd.Series,
                             max_models: int = 5, diversity_threshold: Optional[float] = None) -> List[str]:
        """Select diverse subset of models for ensemble."""
        try:
            threshold = diversity_threshold or self.diversity_threshold
            analysis = self.analyze_correlation(models, X, y)
            
            if 'error' in analysis:
                return list(models.keys())[:max_models]
            
            subset_key = f'size_{max_models}'
            if subset_key in analysis.get('optimal_subsets', {}):
                optimal_subset = analysis['optimal_subsets'][subset_key]
                return optimal_subset['models']
            
            return list(models.keys())[:max_models]
        except Exception as e:
            self.logger.error(f"Error selecting diverse subset: {e}")
            return list(models.keys())[:max_models]
    
    def adjust_weights_by_correlation(self, base_weights: Dict[str, float],
                                     correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Adjust ensemble weights based on model correlation."""
        try:
            adjusted_weights = {}
            
            for model_name in base_weights:
                base_weight = base_weights[model_name]
                
                if model_name in correlation_matrix:
                    correlations = list(correlation_matrix[model_name].values())
                    avg_correlation = np.mean([abs(c) for c in correlations if c != 1.0])
                    correlation_penalty = avg_correlation ** 2
                    adjusted_weight = base_weight * (1 - correlation_penalty)
                else:
                    adjusted_weight = base_weight
                
                adjusted_weights[model_name] = max(adjusted_weight, 0.01)
            
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {name: weight / total_weight for name, weight in adjusted_weights.items()}
            
            return adjusted_weights
        except Exception as e:
            self.logger.error(f"Error adjusting weights by correlation: {e}")
            return base_weights
    
    def get_analysis_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of correlation analysis over time period."""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis['timestamp'] >= cutoff_time
            ]
            
            if not recent_analyses:
                return {'error': 'No recent correlation analysis data available'}
            
            summary = {
                'period_days': days,
                'total_analyses': len(recent_analyses),
                'average_models': np.mean([a['n_models'] for a in recent_analyses]),
                'average_diversity': np.mean([
                    a.get('diversity_metrics', {}).get('overall_diversity', 0)
                    for a in recent_analyses
                ]),
                'most_common_redundant_pairs': self.get_most_common_redundant_pairs(recent_analyses),
                'correlation_trends': self.analyze_correlation_trends(recent_analyses)
            }
            
            return summary
        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {e}")
            return {'error': str(e)}
    
    def get_most_common_redundant_pairs(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Get most commonly redundant model pairs."""
        try:
            pair_counts = {}
            
            for analysis in analyses:
                redundant_pairs = analysis.get('redundant_pairs', [])
                for pair in redundant_pairs:
                    pair_key = f"{pair['model1']}-{pair['model2']}"
                    pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            return [pair[0] for pair in sorted_pairs[:5]]
        except Exception as e:
            self.logger.error(f"Error getting most common redundant pairs: {e}")
            raise RuntimeError("Failed to get most common redundant pairs") from e
    
    def analyze_correlation_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trends in correlation patterns."""
        try:
            if len(analyses) < 5:
                return {'status': 'insufficient_data'}
            
            diversity_scores = [
                analysis.get('diversity_metrics', {}).get('overall_diversity', 0)
                for analysis in analyses
            ]
            
            if len(diversity_scores) >= 3:
                x = np.arange(len(diversity_scores))
                slope = np.polyfit(x, diversity_scores, 1)[0]
                
                if slope > 0.01:
                    trend = 'improving'
                elif slope < -0.01:
                    trend = 'decreasing'
                else:
                    trend = 'stable'
                
                return {'status': 'analyzed', 'trend': trend, 'slope': float(slope)}
            
            return {'status': 'insufficient_data'}
        except Exception as e:
            self.logger.error(f"Error analyzing correlation trends: {e}")
            return {'status': 'error', 'error': str(e)}


# Factory function
def get_correlation_engine(correlation_method: str = "pearson", 
                           diversity_threshold: float = 0.7, 
                           error_handler: Optional[IErrorHandler] = None) -> CorrelationEngine:
    """Factory function to get CorrelationEngine instance."""
    return CorrelationEngine(correlation_method, diversity_threshold, error_handler)
