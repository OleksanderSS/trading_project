# core/pipeline/adaptive_ensemble.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from utils.logger import ProjectLogger
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from models.ensemble_model import EnsembleModel

logger = ProjectLogger.get_logger(__name__)

class AdaptiveEnsemble:
    """
    Adaptive ensemble that:
    1. Combines model predictions using various methods
    2. Learns from past performance to weight models
    3. Adapts weights based on context and recent performance
    """
    
    def __init__(self):
        self.logger = logger
        self.model_weights = {}  # Dynamic weights for each model
        self.performance_history = {}  # Track performance over time
        self.context_weights = {}  # Weights for different contexts
        self.ensemble_methods = ['weighted_average', 'adaptive_weighted', 'context_aware', 'meta_learner']
        
    def create_ensemble_predictions(self,
                                  model_predictions: Dict[str, np.ndarray],
                                  y_true: np.ndarray,
                                  context_features: Optional[pd.DataFrame] = None,
                                  method: str = 'adaptive_weighted') -> Dict[str, np.ndarray]:
        """
        Create ensemble predictions using different methods.
        
        Args:
            model_predictions: Dict of model_name -> predictions
            y_true: True values for learning
            context_features: Context features for context-aware weighting
            method: Ensemble method to use
            
        Returns:
            Dict of ensemble_method -> ensemble_predictions
        """
        results = {}
        
        # Simple average (baseline)
        results['simple_average'] = self._simple_average(model_predictions)
        
        # Weighted average (based on historical performance)
        results['weighted_average'] = self._weighted_average(model_predictions, y_true)
        
        # Adaptive weighted (recent performance weighted more)
        results['adaptive_weighted'] = self._adaptive_weighted_average(model_predictions, y_true)
        
        # Context-aware (weights change based on context)
        if context_features is not None:
            results['context_aware'] = self._context_aware_weighting(model_predictions, y_true, context_features)
        # Використовуй gap-based ваги якщо є гепи
        if context_features is not None:
            gap_cols = [col for col in context_features.columns if "gap_percent" in col]
            if gap_cols:
                method = 'gap_based'
        
        # Gap-based model selection (new method)
        if context_features is not None:
            results['gap_based'] = self._gap_based_weighting(model_predictions, context_features)
        
        # Meta-learner (train a model to predict best weights)
        results['meta_learner'] = self._meta_learner_ensemble(model_predictions, y_true)
        
        # Update performance tracking
        self._update_performance_history(model_predictions, y_true, results)
        
        return results
    
    def _simple_average(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Simple average of all predictions"""
        if not predictions:
            return np.array([])
            
        # Stack predictions and calculate mean
        stacked = np.column_stack(list(predictions.values()))
        return np.mean(stacked, axis=1)
    
    def _weighted_average(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """Weighted average based on model performance"""
        if not predictions:
            return np.array([])
            
        # Calculate weights based on inverse error
        weights = {}
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                error = mean_absolute_error(y_true, pred)
                weights[name] = 1.0 / (error + 1e-8)  # Add small epsilon to avoid division by zero
            else:
                weights[name] = 1.0
                
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Apply weights
        weighted_pred = np.zeros_like(y_true)
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                weighted_pred += pred * weights[name]
                
        return weighted_pred
    
    def _adaptive_weighted_average(self, predictions: Dict[str, np.ndarray], y_true: np.ndarray) -> np.ndarray:
        """Adaptive weighting that emphasizes recent performance"""
        if not predictions:
            return np.array([])
            
        # Calculate weights with exponential decay for recent performance
        weights = {}
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                # Calculate errors with more weight on recent predictions
                errors = np.abs(y_true - pred)
                
                # Apply exponential decay (recent errors weighted more)
                decay_factor = 0.95
                decayed_errors = errors * (decay_factor ** np.arange(len(errors))[::-1])
                
                # Weight is inverse of average decayed error
                avg_error = np.mean(decayed_errors)
                weights[name] = 1.0 / (avg_error + 1e-8)
            else:
                weights[name] = 1.0
                
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Apply weights
        weighted_pred = np.zeros_like(y_true)
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                weighted_pred += pred * weights[name]
                
        return weighted_pred
    
    def _context_aware_weighting(self,
                                 predictions: Dict[str, np.ndarray],
                                 y_true: np.ndarray,
                                 context_features: pd.DataFrame) -> np.ndarray:
        """Context-aware weighting that adapts based on market conditions"""
        if not predictions:
            return np.array([])
            
        # Initialize weights
        weights = {name: 1.0 for name in predictions.keys()}
        
        # Adjust weights based on context
        if 'market_phase' in context_features.columns:
            # Different models perform better in different market phases
            phase_weights = self._get_market_phase_weights()
            current_phase = context_features['market_phase'].iloc[-1]
            
            for name in predictions.keys():
                if current_phase in phase_weights and name in phase_weights[current_phase]:
                    weights[name] *= phase_weights[current_phase][name]
        
        if 'macro_VIX' in context_features.columns:
            # Adjust weights based on volatility
            vix_level = context_features['macro_VIX'].iloc[-1]
            volatility_weights = self._get_volatility_weights(vix_level)
            
            for name in predictions.keys():
                if name in volatility_weights:
                    weights[name] *= volatility_weights[name]
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Apply weights
        weighted_pred = np.zeros_like(y_true)
        for name, pred in predictions.items():
            if len(pred) == len(y_true):
                weighted_pred += pred * weights[name]
                
        return weighted_pred
    
    def _meta_learner_ensemble(self,
                              predictions: Dict[str, np.ndarray],
                              y_true: np.ndarray) -> np.ndarray:
        """Use a meta-learner to learn optimal combination of models"""
        if not predictions:
            return np.array([])
            
        # Prepare training data for meta-learner
        X_meta = np.column_stack(list(predictions.values()))
        y_meta = y_true
        
        # Split for meta-training
        split_point = len(y_meta) // 2
        X_train_meta = X_meta[:split_point]
        y_train_meta = y_meta[:split_point]
        X_test_meta = X_meta[split_point:]
        y_test_meta = y_meta[split_point:]
        
        # Train simple meta-learner (linear regression)
        try:
            from sklearn.linear_model import LinearRegression
            meta_learner = LinearRegression()
            meta_learner.fit(X_train_meta, y_train_meta)
            
            # Get meta-predictions
            meta_pred = meta_learner.predict(X_test_meta)
            
            # For the full prediction, we need to handle the split
            # For simplicity, return the meta-predictions for the test portion
            full_pred = np.zeros_like(y_meta)
            full_pred[:split_point] = y_train_meta  # Use actual values for training portion
            full_pred[split_point:] = meta_pred
            
            return full_pred
            
        except Exception as e:
            self.logger.warning(f"Meta-learner failed: {e}, falling back to weighted average")
            return self._weighted_average(predictions, y_true)
    
    def _get_market_phase_weights(self) -> Dict[str, Dict[str, float]]:
        """Get model weights for different market phases"""
        # These would be learned from historical data
        # For now, use example weights
        return {
            'bull': {
                'trend_following_model': 1.2,
                'momentum_model': 1.1,
                'mean_reversion_model': 0.8
            },
            'bear': {
                'trend_following_model': 1.1,
                'momentum_model': 0.9,
                'mean_reversion_model': 1.2
            },
            'neutral': {
                'trend_following_model': 1.0,
                'momentum_model': 1.0,
                'mean_reversion_model': 1.0
            }
        }
    
    def _get_volatility_weights(self, vix_level: float) -> Dict[str, float]:
        """Get model weights based on volatility level"""
        if vix_level > 30:  # High volatility
            return {
                'volatility_model': 1.3,
                'trend_following_model': 0.8,
                'momentum_model': 0.9
            }
        elif vix_level < 15:  # Low volatility
            return {
                'volatility_model': 0.7,
                'trend_following_model': 1.1,
                'momentum_model': 1.2
            }
        else:  # Normal volatility
            return {
                'volatility_model': 1.0,
                'trend_following_model': 1.0,
                'momentum_model': 1.0
            }
    
    def _update_performance_history(self,
                                   model_predictions: Dict[str, np.ndarray],
                                   y_true: np.ndarray,
                                   ensemble_results: Dict[str, np.ndarray]):
        """Update performance history for learning"""
        timestamp = datetime.now()
        
        # Update individual model performance
        for name, pred in model_predictions.items():
            if len(pred) == len(y_true):
                if name not in self.performance_history:
                    self.performance_history[name] = []
                    
                performance = {
                    'timestamp': timestamp,
                    'mae': mean_absolute_error(y_true, pred),
                    'r2': r2_score(y_true, pred)
                }
                self.performance_history[name].append(performance)
                
                # Keep only last 100 records
                if len(self.performance_history[name]) > 100:
                    self.performance_history[name] = self.performance_history[name][-100:]
        
        # Update ensemble performance
        for method, pred in ensemble_results.items():
            if len(pred) == len(y_true):
                ensemble_key = f'ensemble_{method}'
                if ensemble_key not in self.performance_history:
                    self.performance_history[ensemble_key] = []
                    
                performance = {
                    'timestamp': timestamp,
                    'mae': mean_absolute_error(y_true, pred),
                    'r2': r2_score(y_true, pred)
                }
                self.performance_history[ensemble_key].append(performance)
                
                # Keep only last 100 records
                if len(self.performance_history[ensemble_key]) > 100:
                    self.performance_history[ensemble_key] = self.performance_history[ensemble_key][-100:]
    
    def get_best_ensemble_method(self) -> str:
        """Get the best performing ensemble method based on recent history"""
        method_performance = {}
        
        for key, history in self.performance_history.items():
            if key.startswith('ensemble_') and history:
                method = key.replace('ensemble_', '')
                recent_performance = history[-10:]  # Last 10 records
                avg_mae = np.mean([h['mae'] for h in recent_performance])
                method_performance[method] = avg_mae
        
        if method_performance:
            best_method = min(method_performance, key=method_performance.get)
            self.logger.info(f"Best ensemble method: {best_method} (MAE: {method_performance[best_method]:.4f})")
            return best_method
        
        return 'adaptive_weighted'  # Default
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get summary of model and ensemble performance"""
        summary = {}
        
        for key, history in self.performance_history.items():
            if history:
                recent_performance = history[-20:]  # Last 20 records
                
                summary[key] = {
                    'recent_mae': np.mean([h['mae'] for h in recent_performance]),
                    'recent_r2': np.mean([h['r2'] for h in recent_performance]),
                    'total_predictions': len(history),
                    'last_updated': history[-1]['timestamp']
                }
        
        return summary

    def _gap_based_weighting(self, predictions: Dict[str, np.ndarray],
                             context_features: pd.DataFrame) -> np.ndarray:
        """????????? ???? ?? ?????? ?????"""
        if context_features is None or context_features.empty:
            return self._simple_average(predictions)

        gap_cols = [col for col in context_features.columns if 'gap_percent' in col]
        if not gap_cols:
            return self._simple_average(predictions)

        max_gap = context_features[gap_cols].abs().max().iloc[0]

        if max_gap > 0.02:
            gap_weights = {
                'lgbm': 1.2, 'xgb': 1.2, 'catboost': 1.1,
                'rf': 0.8, 'linear': 0.7, 'mlp': 0.9,
                'gru': 1.3, 'transformer': 1.4, 'tabnet': 1.3
            }
        else:
            gap_weights = {
                'lgbm': 0.9, 'xgb': 0.9, 'catboost': 0.8,
                'rf': 1.1, 'linear': 1.2, 'mlp': 1.1,
                'gru': 0.8, 'transformer': 0.9, 'tabnet': 0.8
            }

        final_weights = {}
        for model in predictions.keys():
            weight = gap_weights.get(model.lower(), 1.0)
            final_weights[model] = weight

        total_weight = sum(final_weights.values())
        normalized_weights = {model: weight/total_weight for model, weight in final_weights.items()}

        stacked = np.column_stack(list(predictions.values()))
        weights_array = np.array(list(normalized_weights.values()))
        return np.average(stacked, axis=1, weights=weights_array)
