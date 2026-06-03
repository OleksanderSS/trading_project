from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from src.core.logging.logger import ProjectLogger

from .config import WeightStabilityConfig
from .calculator import WeightStabilityCalculator
from .analyzer import WeightStabilityAnalyzer
from .stabilizer import WeightStabilizer
from .visualizer import WeightStabilityVisualizer

logger = ProjectLogger.get_logger("WeightStabilityMonitor")

class WeightStabilityMonitor:
    """Orchestrator for the Weight Stability Monitor system."""
    
    def __init__(self, 
                 stability_threshold: float = 0.1,
                 window_size: int = 10,
                 max_change_per_update: float = 0.15):
        self.logger = logger
        self.config = WeightStabilityConfig(stability_threshold, window_size, max_change_per_update)
        self.calculator = WeightStabilityCalculator(self.config)
        self.analyzer = WeightStabilityAnalyzer(self.config)
        self.stabilizer = WeightStabilizer(self.config)
        self.visualizer = WeightStabilityVisualizer(self.config)
        
        # State
        self.weight_history = []
        self.weight_changes = []
        self.stability_events = []
        self.current_weights = {}
        self.last_weights = {}
        self.stability_status = 'stable'
        
        self.logger.info(f"✅ WeightStabilityMonitor (Modular) initialized")

    def update_weights(self, 
                     new_weights: Dict[str, float],
                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Update weights and monitor stability."""
        if timestamp is None: timestamp = datetime.now()
        
        self.last_weights = self.current_weights.copy()
        weight_changes = self.calculator.calculate_weight_changes(new_weights, self.last_weights)
        excessive_changes = self.calculator.check_excessive_changes(weight_changes)
        
        self.weight_history.append({'timestamp': timestamp, 'weights': new_weights.copy(), 'changes': weight_changes})
        self.weight_changes.append(weight_changes)
        
        if len(self.weight_history) > 100:
            self.weight_history = self.weight_history[-100:]
            self.weight_changes = self.weight_changes[-100:]
        
        self.current_weights = new_weights.copy()
        
        # Analyze
        stability_metrics = self._run_analysis()
        recommendations = self.analyzer.generate_stability_recommendations(stability_metrics, excessive_changes)
        self.stability_status = self.analyzer.determine_stability_status(stability_metrics.get('overall_stability', 1.0))
        
        results = {
            'timestamp': timestamp,
            'new_weights': new_weights,
            'stability_analysis': {'status': 'completed', 'metrics': stability_metrics},
            'recommendations': recommendations,
            'stability_status': self.stability_status,
            'action_required': self.analyzer.is_action_required(recommendations)
        }
        
        if results['action_required']:
            self._store_stability_event(results)
            
        return results

    def _run_analysis(self) -> Dict[str, Any]:
        if len(self.weight_history) < 2: return {}
        
        models = list(self.current_weights.keys())
        metrics = {
            'volatility': self.calculator.calculate_weight_volatility(self.weight_changes, models),
            'drift': self.calculator.calculate_weight_drift(self.weight_history, self.current_weights),
            'consistency': self.calculator.calculate_weight_consistency(self.weight_history, models),
            'reversal_frequency': self.calculator.calculate_reversal_frequency(self.weight_changes, models)
        }
        metrics['overall_stability'] = self.calculator.calculate_overall_stability_score(metrics)
        return metrics

    def _store_stability_event(self, results: Dict[str, Any]):
        event = {
            'timestamp': results['timestamp'],
            'stability_status': results['stability_status'],
            'stability_score': results['stability_analysis']['metrics'].get('overall_stability', 1.0),
            'recommendations': results['recommendations'],
            'weights': results['new_weights'].copy()
        }
        self.stability_events.append(event)
        if len(self.stability_events) > 100: self.stability_events = self.stability_events[-100:]

    def stabilize_weights(self, proposed_weights: Dict[str, float], method: str = "constrained") -> Dict[str, float]:
        if not self.last_weights: return proposed_weights
        
        if method == "constrained":
            return self.stabilizer.apply_constrained_stabilization(proposed_weights, self.last_weights)
        elif method == "exponential_smoothing":
            return self.stabilizer.apply_exponential_smoothing(proposed_weights, self.last_weights)
        elif method == "volatility_based":
            vols = self.calculator.calculate_weight_volatility(self.weight_changes, list(proposed_weights.keys()))
            return self.stabilizer.apply_volatility_based_stabilization(proposed_weights, self.last_weights, vols.get('model_volatilities', {}))
        return proposed_weights

    def get_stability_summary(self, days: int = 30) -> Dict[str, Any]:
        cutoff = datetime.now() - timedelta(days=days)
        recent = [e for e in self.stability_events if e['timestamp'] >= cutoff]
        if not recent: return {'error': 'No data'}
        
        scores = [e['stability_score'] for e in recent]
        return {
            'period_days': days,
            'total_events': len(recent),
            'average_stability_score': sum(scores)/len(scores),
            'stability_trend': self.analyzer.analyze_stability_trend(scores)
        }

    def plot_stability_metrics(self, save_path: Optional[str] = None) -> None:
        self.visualizer.plot_stability_metrics(self.weight_history, self.weight_changes, list(self.current_weights.keys()), save_path)
