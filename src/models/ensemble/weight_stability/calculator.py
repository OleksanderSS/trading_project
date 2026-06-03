import numpy as np
from typing import Dict, List, Any, Optional
import logging
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilityCalculator")

class WeightStabilityCalculator:
    """Calculates stability metrics for ensemble weights."""
    
    def __init__(self, config: Any):
        self.logger = logger
        self.config = config

    def calculate_weight_changes(self, 
                               new_weights: Dict[str, float],
                               old_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate weight changes between updates."""
        changes = {}
        for model_name in new_weights:
            new_weight = new_weights[model_name]
            old_weight = old_weights.get(model_name, new_weight)
            changes[model_name] = new_weight - old_weight
        return changes

    def check_excessive_changes(self, 
                               weight_changes: Dict[str, float]) -> Dict[str, Any]:
        """Check for excessive weight changes."""
        excessive_changes = {
            'has_excessive': False,
            'excessive_models': [],
            'max_change': 0.0,
            'average_change': 0.0
        }
        
        if not weight_changes:
            return excessive_changes
        
        changes_list = [abs(c) for c in weight_changes.values()]
        excessive_changes['max_change'] = max(changes_list) if changes_list else 0.0
        excessive_changes['average_change'] = np.mean(changes_list) if changes_list else 0.0
        
        for model_name, change in weight_changes.items():
            if abs(change) > self.config.max_change_per_update:
                excessive_changes['has_excessive'] = True
                excessive_changes['excessive_models'].append({
                    'model': model_name,
                    'change': change,
                    'threshold': self.config.max_change_per_update
                })
        
        return excessive_changes

    def calculate_weight_volatility(self, 
                                  weight_changes_history: List[Dict[str, float]],
                                  current_models: List[str]) -> Dict[str, Any]:
        """Calculate weight volatility over recent window."""
        try:
            if len(weight_changes_history) < self.config.window_size:
                return {'average_volatility': 0.0, 'max_volatility': 0.0, 'model_volatilities': {}}
            
            recent_changes = weight_changes_history[-self.config.window_size:]
            model_volatilities = {}
            
            for model_name in current_models:
                model_changes = [
                    change.get(model_name, 0.0) 
                    for change in recent_changes
                    if model_name in change
                ]
                
                if len(model_changes) >= 2:
                    model_volatilities[model_name] = np.std(model_changes)
            
            avg_volatility = np.mean(list(model_volatilities.values())) if model_volatilities else 0.0
            max_volatility = max(model_volatilities.values()) if model_volatilities else 0.0
            
            return {
                'average_volatility': avg_volatility,
                'max_volatility': max_volatility,
                'model_volatilities': model_volatilities
            }
        except Exception as e:
            self.logger.error(f"Error calculating weight volatility: {e}")
            return {'average_volatility': 0.0, 'max_volatility': 0.0}

    def calculate_weight_drift(self, 
                             weight_history: List[Dict[str, Any]],
                             current_weights: Dict[str, float]) -> Dict[str, Any]:
        """Calculate cumulative weight drift over time."""
        try:
            if len(weight_history) < 2:
                return {'total_drift': 0.0, 'drift_rate': 0.0, 'model_drifts': {}}
            
            first_weights = weight_history[0]['weights']
            model_drifts = {}
            
            for model_name in current_weights:
                if model_name in first_weights:
                    model_drifts[model_name] = abs(current_weights[model_name] - first_weights[model_name])
            
            total_drift = np.mean(list(model_drifts.values())) if model_drifts else 0.0
            time_span = (weight_history[-1]['timestamp'] - weight_history[0]['timestamp'])
            days = time_span.total_seconds() / (24 * 3600)
            drift_rate = total_drift / days if days > 0 else 0.0
            
            return {
                'total_drift': total_drift,
                'drift_rate': drift_rate,
                'model_drifts': model_drifts
            }
        except Exception as e:
            self.logger.error(f"Error calculating weight drift: {e}")
            return {'total_drift': 0.0, 'drift_rate': 0.0}

    def calculate_weight_consistency(self, 
                                   weight_history: List[Dict[str, Any]],
                                   current_models: List[str]) -> float:
        """Calculate weight consistency score."""
        try:
            if len(weight_history) < self.config.window_size:
                return 1.0
            
            recent_weights = [record['weights'] for record in weight_history[-self.config.window_size:]]
            model_consistencies = []
            
            for model_name in current_models:
                model_vals = [w.get(model_name, 0.0) for w in recent_weights if model_name in w]
                if len(model_vals) >= 2:
                    mean_val = np.mean(model_vals)
                    std_val = np.std(model_vals)
                    consistency = max(0.0, 1.0 - (std_val / mean_val)) if mean_val > 0 else 1.0
                    model_consistencies.append(consistency)
            
            return np.mean(model_consistencies) if model_consistencies else 1.0
        except Exception as e:
            self.logger.error(f"Error calculating weight consistency: {e}")
            return 1.0

    def calculate_reversal_frequency(self, 
                                   weight_changes_history: List[Dict[str, float]],
                                   current_models: List[str]) -> float:
        """Calculate frequency of weight direction reversals."""
        try:
            if len(weight_changes_history) < 3:
                return 0.0
            
            recent_changes = weight_changes_history[-20:]
            model_reversals = []
            
            for model_name in current_models:
                model_changes = [c.get(model_name, 0.0) for c in recent_changes if model_name in c]
                if len(model_changes) >= 3:
                    reversals = 0
                    for i in range(1, len(model_changes)):
                        if (model_changes[i-1] > 0 and model_changes[i] < 0) or \
                           (model_changes[i-1] < 0 and model_changes[i] > 0):
                            reversals += 1
                    model_reversals.append(reversals / (len(model_changes) - 1))
            
            return np.mean(model_reversals) if model_reversals else 0.0
        except Exception as e:
            self.logger.error(f"Error calculating reversal frequency: {e}")
            return 0.0

    def calculate_overall_stability_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall stability score from individual metrics."""
        try:
            normalized_scores = {}
            
            if 'volatility' in metrics:
                vol = metrics['volatility'].get('average_volatility', 0.0)
                normalized_scores['volatility'] = max(0.0, 1.0 - (vol / self.config.stability_threshold))
            
            if 'drift' in metrics:
                drift = metrics['drift'].get('total_drift', 0.0)
                threshold = self.config.STABILITY_METRICS['drift']['threshold']
                normalized_scores['drift'] = max(0.0, 1.0 - (drift / threshold))
            
            if 'consistency' in metrics:
                normalized_scores['consistency'] = metrics['consistency']
            
            if 'reversal_frequency' in metrics:
                rev = metrics['reversal_frequency']
                threshold = self.config.STABILITY_METRICS['reversal_frequency']['threshold']
                normalized_scores['reversal_frequency'] = max(0.0, 1.0 - (rev / threshold))
            
            return np.mean(list(normalized_scores.values())) if normalized_scores else 1.0
        except Exception as e:
            self.logger.error(f"Error calculating overall stability score: {e}")
            return 1.0
