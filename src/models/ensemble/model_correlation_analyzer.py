#!/usr/bin/env python3
"""
Model Correlation Analyzer - Analyzes model correlation and diversity for ensemble optimization
Identifies redundant models and selects optimal diverse subsets for ensemble performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelCorrelationAnalyzer")

class ModelCorrelationAnalyzer:
    """
    Model correlation analyzer for ensemble optimization.
    
    This analyzer provides:
    - Prediction correlation matrix analysis
    - Diversity metrics calculation (entropy, disagreement)
    - Redundancy detection and elimination
    - Optimal subset selection algorithms
    - Correlation-based weight adjustment
    - Ensemble diversity optimization
    
    Critical for building effective and diverse ensembles.
    """
    
    # Correlation methods
    CORRELATION_METHODS = {
        'pearson': {
            'description': 'Pearson correlation coefficient',
            'suitable_for': ['linear_relationships'],
            'range': (-1, 1)
        },
        'spearman': {
            'description': 'Spearman rank correlation',
            'suitable_for': ['monotonic_relationships'],
            'range': (-1, 1)
        },
        'mutual_info': {
            'description': 'Mutual information score',
            'suitable_for': ['nonlinear_relationships'],
            'range': (0, float('inf'))
        },
        'disagreement': {
            'description': 'Prediction disagreement measure',
            'suitable_for': ['diversity_analysis'],
            'range': (0, 1)
        }
    }
    
    # Diversity metrics
    DIVERSITY_METRICS = {
        'entropy': {
            'description': 'Entropy of prediction distribution',
            'higher_better': True
        },
        'disagreement': {
            'description': 'Average pairwise disagreement',
            'higher_better': True
        },
        'variance': {
            'description': 'Variance of predictions',
            'higher_better': True
        },
        'correlation_penalty': {
            'description': 'Penalty for high correlation',
            'higher_better': False
        }
    }
    
    def __init__(self, 
                 correlation_method: str = "pearson",
                 diversity_threshold: float = 0.7):
        """
        Initialize Model Correlation Analyzer.
        
        Args:
            correlation_method: Method for correlation calculation
            diversity_threshold: Threshold for diversity filtering
        """
        self.logger = logger
        self.correlation_method = correlation_method
        self.diversity_threshold = diversity_threshold
        
        # Analysis cache
        self.correlation_cache = {}
        self.diversity_cache = {}
        
        # Analysis history
        self.analysis_history = []
        
        self.logger.info(f"✅ ModelCorrelationAnalyzer initialized with method: {correlation_method}")
    
    def analyze_correlation(self, 
                           models: Dict[str, Any],
                           X: pd.DataFrame,
                           y: pd.Series,
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
            predictions = self._get_model_predictions(models, X_sample)
            
            if len(predictions) < 2:
                return {'error': 'Need at least 2 models for correlation analysis'}
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(predictions)
            results['correlation_matrix'] = correlation_matrix
            
            # Calculate diversity metrics
            diversity_metrics = self._calculate_diversity_metrics(predictions, correlation_matrix)
            results['diversity_metrics'] = diversity_metrics
            
            # Find redundant pairs
            redundant_pairs = self._find_redundant_pairs(correlation_matrix, diversity_metrics)
            results['redundant_pairs'] = redundant_pairs
            
            # Select optimal subsets
            optimal_subsets = self._select_optimal_subsets(predictions, correlation_matrix, diversity_metrics)
            results['optimal_subsets'] = optimal_subsets
            
            # Store in cache
            cache_key = f"{len(models)}_{sample_size or len(X)}"
            self.correlation_cache[cache_key] = results
            
            # Store in history
            self.analysis_history.append(results)
            
            self.logger.info(f"✅ Correlation analysis complete. Redundant pairs: {len(redundant_pairs)}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _get_model_predictions(self, 
                             models: Dict[str, Any],
                             X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from all models."""
        
        predictions = {}
        
        for model_name, model in models.items():
            try:
                pred = model.predict(X)
                predictions[model_name] = pred
            except Exception as e:
                self.logger.error(f"Error getting predictions from {model_name}: {e}")
                continue
        
        return predictions
    
    def _calculate_correlation_matrix(self, 
                                   predictions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
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
                        correlation = self._calculate_correlation(pred1, pred2)
                        correlation_matrix[model1][model2] = correlation
            
            return correlation_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return {}
    
    def _calculate_correlation(self, 
                             pred1: np.ndarray,
                             pred2: np.ndarray) -> float:
        """Calculate correlation between two prediction arrays."""
        
        try:
            if self.correlation_method == 'pearson':
                correlation, _ = pearsonr(pred1, pred2)
            elif self.correlation_method == 'spearman':
                correlation, _ = spearmanr(pred1, pred2)
            elif self.correlation_method == 'mutual_info':
                # For mutual information, need discrete values
                pred1_discrete = pd.qcut(pred1, q=10, labels=False, duplicates='drop')
                pred2_discrete = pd.qcut(pred2, q=10, labels=False, duplicates='drop')
                correlation = mutual_info_score(pred1_discrete, pred2_discrete)
            elif self.correlation_method == 'disagreement':
                # Disagreement measure: 1 - agreement rate
                agreement = np.mean(pred1 == pred2)
                correlation = 1 - agreement
            else:
                # Default to Pearson
                correlation, _ = pearsonr(pred1, pred2)
            
            # Handle NaN values
            if np.isnan(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _calculate_diversity_metrics(self, 
                                    predictions: Dict[str, np.ndarray],
                                    correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate diversity metrics for the ensemble."""
        
        try:
            metrics = {}
            
            # Convert to DataFrame for easier calculation
            pred_df = pd.DataFrame(predictions)
            
            # 1. Entropy of predictions
            entropy = self._calculate_prediction_entropy(pred_df)
            metrics['entropy'] = entropy
            
            # 2. Average pairwise disagreement
            disagreement = self._calculate_average_disagreement(correlation_matrix)
            metrics['disagreement'] = disagreement
            
            # 3. Variance of predictions
            variance = self._calculate_prediction_variance(pred_df)
            metrics['variance'] = variance
            
            # 4. Correlation penalty (lower is better)
            correlation_penalty = self._calculate_correlation_penalty(correlation_matrix)
            metrics['correlation_penalty'] = correlation_penalty
            
            # 5. Overall diversity score (normalized)
            diversity_score = self._calculate_overall_diversity_score(metrics)
            metrics['overall_diversity'] = diversity_score
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating diversity metrics: {e}")
            return {}
    
    def _calculate_prediction_entropy(self, pred_df: pd.DataFrame) -> float:
        """Calculate entropy of prediction distribution."""
        
        try:
            # For regression, use discretized predictions
            if pred_df.shape[1] > 1:
                # Multi-class case
                entropies = []
                for col in pred_df.columns:
                    # Discretize predictions
                    discrete_preds = pd.qcut(pred_df[col], q=10, labels=False, duplicates='drop')
                    value_counts = pd.Series(discrete_preds).value_counts(normalize=True)
                    entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                    entropies.append(entropy)
                return np.mean(entropies)
            else:
                # Binary/regression case
                entropies = []
                for col in pred_df.columns:
                    discrete_preds = pd.qcut(pred_df[col], q=10, labels=False, duplicates='drop')
                    value_counts = pd.Series(discrete_preds).value_counts(normalize=True)
                    entropy = -np.sum(value_counts * np.log2(value_counts + 1e-10))
                    entropies.append(entropy)
                return np.mean(entropies)
                
        except Exception as e:
            self.logger.error(f"Error calculating prediction entropy: {e}")
            return 0.0
    
    def _calculate_average_disagreement(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate average pairwise disagreement."""
        
        try:
            model_names = list(correlation_matrix.keys())
            disagreements = []
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Avoid double counting
                        corr = correlation_matrix[model1][model2]
                        # Disagreement = 1 - |correlation|
                        disagreement = 1 - abs(corr)
                        disagreements.append(disagreement)
            
            return np.mean(disagreements) if disagreements else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating average disagreement: {e}")
            return 0.0
    
    def _calculate_prediction_variance(self, pred_df: pd.DataFrame) -> float:
        """Calculate variance of predictions across models."""
        
        try:
            # Calculate variance for each sample across models
            sample_variances = pred_df.var(axis=1)
            return np.mean(sample_variances)
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction variance: {e}")
            return 0.0
    
    def _calculate_correlation_penalty(self, correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate penalty for high correlation."""
        
        try:
            model_names = list(correlation_matrix.keys())
            correlations = []
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Avoid double counting
                        corr = abs(correlation_matrix[model1][model2])
                        correlations.append(corr)
            
            # Penalty is high correlation (close to 1)
            avg_correlation = np.mean(correlations) if correlations else 0.0
            penalty = avg_correlation  # Higher correlation = higher penalty
            
            return penalty
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation penalty: {e}")
            return 0.0
    
    def _calculate_overall_diversity_score(self, diversity_metrics: Dict[str, float]) -> float:
        """Calculate overall diversity score from individual metrics."""
        
        try:
            # Normalize individual metrics to 0-1 scale
            normalized_metrics = {}
            
            # Higher is better metrics
            for metric in ['entropy', 'disagreement', 'variance']:
                if metric in diversity_metrics:
                    # Simple normalization (could be improved with min/max scaling)
                    normalized_metrics[metric] = min(diversity_metrics[metric] / 2.0, 1.0)
            
            # Lower is better metrics
            if 'correlation_penalty' in diversity_metrics:
                # Invert correlation penalty (lower correlation = higher diversity)
                normalized_metrics['correlation_penalty'] = 1.0 - min(diversity_metrics['correlation_penalty'], 1.0)
            
            # Calculate overall score (average of normalized metrics)
            if normalized_metrics:
                overall_score = np.mean(list(normalized_metrics.values()))
            else:
                overall_score = 0.0
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"Error calculating overall diversity score: {e}")
            return 0.0
    
    def _find_redundant_pairs(self, 
                               correlation_matrix: Dict[str, Dict[str, float]],
                               diversity_metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """Find redundant model pairs based on correlation."""
        
        try:
            redundant_pairs = []
            model_names = list(correlation_matrix.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:  # Avoid double counting
                        correlation = abs(correlation_matrix[model1][model2])
                        
                        # Check if correlation is above threshold
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
            return []
    
    def _select_optimal_subsets(self, 
                               predictions: Dict[str, np.ndarray],
                               correlation_matrix: Dict[str, Dict[str, float]],
                               diversity_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Select optimal subsets of models for different ensemble sizes."""
        
        try:
            optimal_subsets = {}
            model_names = list(predictions.keys())
            
            # Test different subset sizes
            for subset_size in range(2, min(len(model_names), 8) + 1):
                best_subset = self._find_best_subset(
                    model_names, predictions, correlation_matrix, subset_size
                )
                
                if best_subset:
                    optimal_subsets[f'size_{subset_size}'] = best_subset
            
            return optimal_subsets
            
        except Exception as e:
            self.logger.error(f"Error selecting optimal subsets: {e}")
            return {}
    
    def _find_best_subset(self, 
                         model_names: List[str],
                         predictions: Dict[str, np.ndarray],
                         correlation_matrix: Dict[str, Dict[str, float]],
                         subset_size: int) -> Optional[Dict[str, Any]]:
        """Find best subset of given size using greedy algorithm."""
        
        try:
            best_subset = None
            best_score = -float('inf')
            
            # Use greedy algorithm for efficiency
            remaining_models = model_names.copy()
            selected_models = []
            
            while len(selected_models) < subset_size and remaining_models:
                best_candidate = None
                best_candidate_score = -float('inf')
                
                # Try each remaining model
                for candidate in remaining_models:
                    test_subset = selected_models + [candidate]
                    
                    # Calculate diversity score for this subset
                    subset_predictions = {name: predictions[name] for name in test_subset}
                    subset_correlation = {
                        name1: {name2: correlation_matrix[name1][name2] 
                                  for name2 in test_subset}
                        for name1 in test_subset
                    }
                    
                    diversity_score = self._calculate_subset_diversity_score(
                        subset_predictions, subset_correlation
                    )
                    
                    if diversity_score > best_candidate_score:
                        best_candidate_score = diversity_score
                        best_candidate = candidate
                
                if best_candidate:
                    selected_models.append(best_candidate)
                    remaining_models.remove(best_candidate)
            
            if selected_models:
                # Calculate final metrics for best subset
                subset_predictions = {name: predictions[name] for name in selected_models}
                subset_correlation = {
                    name1: {name2: correlation_matrix[name1][name2] 
                              for name2 in selected_models}
                    for name1 in selected_models
                }
                
                final_diversity = self._calculate_subset_diversity_score(
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
            return None
    
    def _calculate_subset_diversity_score(self, 
                                         predictions: Dict[str, np.ndarray],
                                         correlation_matrix: Dict[str, Dict[str, float]]) -> float:
        """Calculate diversity score for a subset of models."""
        
        try:
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            
            # Calculate diversity metrics for subset
            subset_metrics = self._calculate_diversity_metrics(predictions, correlation_matrix)
            
            # Return overall diversity score
            return subset_metrics.get('overall_diversity', 0.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating subset diversity score: {e}")
            return 0.0
    
    def select_diverse_subset(self, 
                             models: Dict[str, Any],
                             X: pd.DataFrame,
                             y: pd.Series,
                             max_models: int = 5,
                             diversity_threshold: Optional[float] = None) -> List[str]:
        """
        Select diverse subset of models for ensemble.
        
        Args:
            models: Dictionary of models
            X: Test features
            y: True targets
            max_models: Maximum number of models to select
            diversity_threshold: Override default diversity threshold
            
        Returns:
            List of selected model names
        """
        try:
            # Use provided threshold or default
            threshold = diversity_threshold or self.diversity_threshold
            
            # Analyze correlation
            analysis = self.analyze_correlation(models, X, y)
            
            if 'error' in analysis:
                self.logger.warning(f"Correlation analysis failed: {analysis['error']}")
                return list(models.keys())[:max_models]  # Fallback to first N models
            
            # Get optimal subset for requested size
            subset_key = f'size_{max_models}'
            if subset_key in analysis.get('optimal_subsets', {}):
                optimal_subset = analysis['optimal_subsets'][subset_key]
                return optimal_subset['models']
            
            # Fallback: use greedy selection
            model_names = list(models.keys())
            selected = []
            remaining = model_names.copy()
            
            while len(selected) < max_models and remaining:
                best_model = remaining[0]  # Default to first
                best_diversity = -1
                
                for candidate in remaining:
                    test_subset = selected + [candidate]
                    
                    # Calculate diversity for test subset
                    test_predictions = {name: models[name].predict(X) for name in test_subset}
                    test_correlation = self._calculate_correlation_matrix(test_predictions)
                    test_diversity = self._calculate_diversity_metrics(test_predictions, test_correlation)
                    
                    diversity_score = test_diversity.get('overall_diversity', 0)
                    
                    if diversity_score > best_diversity:
                        best_diversity = diversity_score
                        best_model = candidate
                
                selected.append(best_model)
                remaining.remove(best_model)
            
            return selected
            
        except Exception as e:
            self.logger.error(f"Error selecting diverse subset: {e}")
            return list(models.keys())[:max_models]
    
    def adjust_weights_by_correlation(self, 
                                    base_weights: Dict[str, float],
                                    correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """
        Adjust ensemble weights based on model correlation.
        
        Args:
            base_weights: Base weights for models
            correlation_matrix: Correlation matrix between models
            
        Returns:
            Adjusted weights (sum to 1.0)
        """
        try:
            adjusted_weights = {}
            
            for model_name in base_weights:
                base_weight = base_weights[model_name]
                
                # Calculate correlation penalty for this model
                if model_name in correlation_matrix:
                    correlations = list(correlation_matrix[model_name].values())
                    avg_correlation = np.mean([abs(c) for c in correlations if c != 1.0])
                    
                    # Reduce weight for highly correlated models
                    correlation_penalty = avg_correlation ** 2  # Square for stronger penalty
                    adjusted_weight = base_weight * (1 - correlation_penalty)
                else:
                    adjusted_weight = base_weight
                
                adjusted_weights[model_name] = max(adjusted_weight, 0.01)  # Minimum weight
            
            # Normalize to sum to 1
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                adjusted_weights = {name: weight / total_weight 
                                  for name, weight in adjusted_weights.items()}
            
            return adjusted_weights
            
        except Exception as e:
            self.logger.error(f"Error adjusting weights by correlation: {e}")
            return base_weights
    
    def plot_correlation_matrix(self, 
                               correlation_matrix: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap."""
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Convert to DataFrame
            model_names = list(correlation_matrix.keys())
            corr_df = pd.DataFrame(correlation_matrix, index=model_names, columns=model_names)
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f', cbar_kws={'label': 'Correlation'})
            
            plt.title(f'Model Correlation Matrix ({self.correlation_method.capitalize()})')
            plt.xlabel('Models')
            plt.ylabel('Models')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Correlation matrix plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {e}")
    
    def get_analysis_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of correlation analysis over time period."""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent analyses
            recent_analyses = [
                analysis for analysis in self.analysis_history
                if analysis['timestamp'] >= cutoff_time
            ]
            
            if not recent_analyses:
                return {'error': 'No recent correlation analysis data available'}
            
            # Calculate summary statistics
            summary = {
                'period_days': days,
                'total_analyses': len(recent_analyses),
                'average_models': np.mean([a['n_models'] for a in recent_analyses]),
                'average_diversity': np.mean([
                    a.get('diversity_metrics', {}).get('overall_diversity', 0) 
                    for a in recent_analyses
                ]),
                'most_common_redundant_pairs': self._get_most_common_redundant_pairs(recent_analyses),
                'correlation_trends': self._analyze_correlation_trends(recent_analyses)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting analysis summary: {e}")
            return {'error': str(e)}
    
    def _get_most_common_redundant_pairs(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Get most commonly redundant model pairs."""
        
        try:
            pair_counts = {}
            
            for analysis in analyses:
                redundant_pairs = analysis.get('redundant_pairs', [])
                for pair in redundant_pairs:
                    pair_key = f"{pair['model1']}-{pair['model2']}"
                    pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1
            
            # Return top 5 most common pairs
            sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)
            return [pair[0] for pair in sorted_pairs[:5]]
            
        except Exception as e:
            self.logger.error(f"Error getting most common redundant pairs: {e}")
            return []
    
    def _analyze_correlation_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, str]:
        """Analyze trends in correlation patterns."""
        
        try:
            if len(analyses) < 5:
                return {'status': 'insufficient_data'}
            
            # Extract diversity scores over time
            diversity_scores = [
                analysis.get('diversity_metrics', {}).get('overall_diversity', 0)
                for analysis in analyses
            ]
            
            # Calculate trend
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


# Factory function for easy instantiation
def get_model_correlation_analyzer(correlation_method: str = "pearson",
                                 diversity_threshold: float = 0.7) -> ModelCorrelationAnalyzer:
    """Factory function to get ModelCorrelationAnalyzer instance."""
    return ModelCorrelationAnalyzer(correlation_method, diversity_threshold)


# Convenience function for quick analysis
def analyze_model_correlation_quick(models: Dict[str, Any],
                                 X: pd.DataFrame,
                                 y: pd.Series,
                                 correlation_method: str = "pearson") -> Dict[str, Any]:
    """
    Quick model correlation analysis.
    
    Args:
        models: Dictionary of models to analyze
        X: Test features
        y: True targets
        correlation_method: Correlation calculation method
        
    Returns:
        Correlation analysis result dictionary
    """
    analyzer = get_model_correlation_analyzer(correlation_method)
    return analyzer.analyze_correlation(models, X, y)
