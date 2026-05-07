#!/usr/bin/env python3
"""
Weight Stability Monitor - Monitors and ensures ensemble weight stability
Detects excessive weight fluctuations and provides stabilization mechanisms.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilityMonitor")

class WeightStabilityMonitor:
    """
    Weight stability monitor for ensemble models.
    
    This monitor provides:
    - Weight volatility tracking and analysis
    - Stability metrics calculation
    - Weight drift detection
    - Automatic weight rebalancing
    - Historical stability analysis
    - Stability constraint enforcement
    
    Critical for maintaining consistent ensemble performance.
    """
    
    # Stability metrics
    STABILITY_METRICS = {
        'volatility': {
            'description': 'Standard deviation of weight changes',
            'lower_better': True,
            'threshold': 0.1
        },
        'drift': {
            'description': 'Cumulative weight drift over time',
            'lower_better': True,
            'threshold': 0.2
        },
        'consistency': {
            'description': 'Weight consistency score',
            'higher_better': True,
            'threshold': 0.8
        },
        'reversal_frequency': {
            'description': 'Frequency of weight direction reversals',
            'lower_better': True,
            'threshold': 0.3
        }
    }
    
    def __init__(self, 
                 stability_threshold: float = 0.1,
                 window_size: int = 10,
                 max_change_per_update: float = 0.15):
        """
        Initialize Weight Stability Monitor.
        
        Args:
            stability_threshold: Threshold for stability alerts
            window_size: Window size for stability calculations
            max_change_per_update: Maximum allowed weight change per update
        """
        self.logger = logger
        self.stability_threshold = stability_threshold
        self.window_size = window_size
        self.max_change_per_update = max_change_per_update
        
        # Weight history storage
        self.weight_history = []
        self.weight_changes = []
        self.stability_events = []
        
        # Current state
        self.current_weights = {}
        self.last_weights = {}
        self.stability_status = 'stable'
        
        # Analysis cache
        self.stability_cache = {}
        
        self.logger.info(f"✅ WeightStabilityMonitor initialized")
    
    def update_weights(self, 
                     new_weights: Dict[str, float],
                     timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Update weights and monitor stability.
        
        Args:
            new_weights: New weight dictionary
            timestamp: Update timestamp (uses now if None)
            
        Returns:
            Dict with stability analysis and recommendations
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        self.logger.info(f"📊 Updating weight stability at {timestamp}")
        
        results = {
            'timestamp': timestamp,
            'new_weights': new_weights,
            'stability_analysis': {},
            'recommendations': [],
            'action_required': False
        }
        
        try:
            # Store last weights
            self.last_weights = self.current_weights.copy()
            
            # Calculate weight changes
            weight_changes = self._calculate_weight_changes(new_weights, self.last_weights)
            
            # Check for excessive changes
            excessive_changes = self._check_excessive_changes(weight_changes)
            
            # Update history
            self.weight_history.append({
                'timestamp': timestamp,
                'weights': new_weights.copy(),
                'changes': weight_changes
            })
            
            self.weight_changes.append(weight_changes)
            
            # Keep only recent history
            if len(self.weight_history) > 100:
                self.weight_history = self.weight_history[-100:]
                self.weight_changes = self.weight_changes[-100:]
            
            # Update current weights
            self.current_weights = new_weights.copy()
            
            # Perform stability analysis
            stability_analysis = self._analyze_stability()
            results['stability_analysis'] = stability_analysis
            
            # Generate recommendations
            recommendations = self._generate_stability_recommendations(
                stability_analysis, excessive_changes
            )
            results['recommendations'] = recommendations
            
            # Update stability status
            self._update_stability_status(stability_analysis)
            results['stability_status'] = self.stability_status
            
            # Check if action is required
            results['action_required'] = self._is_action_required(recommendations)
            
            # Store stability event if needed
            if results['action_required']:
                self._store_stability_event(results)
            
            self.logger.info(f"✅ Weight stability update complete. Status: {self.stability_status}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error updating weight stability: {e}", exc_info=True)
            results['error'] = str(e)
            return results
    
    def _calculate_weight_changes(self, 
                                new_weights: Dict[str, float],
                                old_weights: Dict[str, float]) -> Dict[str, float]:
        """Calculate weight changes between updates."""
        
        changes = {}
        
        for model_name in new_weights:
            new_weight = new_weights[model_name]
            old_weight = old_weights.get(model_name, new_weight)
            
            change = new_weight - old_weight
            changes[model_name] = change
        
        return changes
    
    def _check_excessive_changes(self, 
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
        
        # Calculate statistics
        changes_list = list(weight_changes.values())
        excessive_changes['max_change'] = max(abs(c) for c in changes_list)
        excessive_changes['average_change'] = np.mean([abs(c) for c in changes_list])
        
        # Check each model
        for model_name, change in weight_changes.items():
            if abs(change) > self.max_change_per_update:
                excessive_changes['has_excessive'] = True
                excessive_changes['excessive_models'].append({
                    'model': model_name,
                    'change': change,
                    'threshold': self.max_change_per_update
                })
        
        return excessive_changes
    
    def _analyze_stability(self) -> Dict[str, Any]:
        """Analyze weight stability metrics."""
        
        try:
            if len(self.weight_history) < 2:
                return {'status': 'insufficient_data'}
            
            stability_metrics = {}
            
            # 1. Weight volatility
            volatility = self._calculate_weight_volatility()
            stability_metrics['volatility'] = volatility
            
            # 2. Weight drift
            drift = self._calculate_weight_drift()
            stability_metrics['drift'] = drift
            
            # 3. Weight consistency
            consistency = self._calculate_weight_consistency()
            stability_metrics['consistency'] = consistency
            
            # 4. Reversal frequency
            reversal_freq = self._calculate_reversal_frequency()
            stability_metrics['reversal_frequency'] = reversal_freq
            
            # 5. Overall stability score
            overall_stability = self._calculate_overall_stability_score(stability_metrics)
            stability_metrics['overall_stability'] = overall_stability
            
            return {
                'status': 'completed',
                'metrics': stability_metrics,
                'analysis_timestamp': datetime.now()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing stability: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _calculate_weight_volatility(self) -> Dict[str, float]:
        """Calculate weight volatility over recent window."""
        
        try:
            if len(self.weight_changes) < self.window_size:
                return {'average_volatility': 0.0, 'max_volatility': 0.0}
            
            # Get recent changes
            recent_changes = self.weight_changes[-self.window_size:]
            
            # Calculate volatility for each model
            model_volatilities = {}
            
            for model_name in self.current_weights:
                model_changes = [
                    change.get(model_name, 0.0) 
                    for change in recent_changes
                    if model_name in change
                ]
                
                if len(model_changes) >= 2:
                    volatility = np.std(model_changes)
                    model_volatilities[model_name] = volatility
            
            # Calculate overall volatility metrics
            if model_volatilities:
                avg_volatility = np.mean(list(model_volatilities.values()))
                max_volatility = max(model_volatilities.values())
            else:
                avg_volatility = 0.0
                max_volatility = 0.0
            
            return {
                'average_volatility': avg_volatility,
                'max_volatility': max_volatility,
                'model_volatilities': model_volatilities
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating weight volatility: {e}")
            return {'average_volatility': 0.0, 'max_volatility': 0.0}
    
    def _calculate_weight_drift(self) -> Dict[str, float]:
        """Calculate cumulative weight drift over time."""
        
        try:
            if len(self.weight_history) < 2:
                return {'total_drift': 0.0, 'drift_rate': 0.0}
            
            # Get first and last weights
            first_weights = self.weight_history[0]['weights']
            current_weights = self.current_weights
            
            # Calculate drift for each model
            model_drifts = {}
            
            for model_name in current_weights:
                if model_name in first_weights:
                    initial_weight = first_weights[model_name]
                    current_weight = current_weights[model_name]
                    drift = abs(current_weight - initial_weight)
                    model_drifts[model_name] = drift
            
            # Calculate overall drift metrics
            if model_drifts:
                total_drift = np.mean(list(model_drifts.values()))
                
                # Calculate drift rate (drift per day)
                time_span = (self.weight_history[-1]['timestamp'] - self.weight_history[0]['timestamp'])
                days = time_span.total_seconds() / (24 * 3600)
                drift_rate = total_drift / days if days > 0 else 0.0
            else:
                total_drift = 0.0
                drift_rate = 0.0
            
            return {
                'total_drift': total_drift,
                'drift_rate': drift_rate,
                'model_drifts': model_drifts
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating weight drift: {e}")
            return {'total_drift': 0.0, 'drift_rate': 0.0}
    
    def _calculate_weight_consistency(self) -> float:
        """Calculate weight consistency score."""
        
        try:
            if len(self.weight_history) < self.window_size:
                return 1.0  # Perfect consistency by default
            
            # Get recent weights
            recent_weights = [
                record['weights'] for record in self.weight_history[-self.window_size:]
            ]
            
            # Calculate consistency for each model
            model_consistencies = []
            
            for model_name in self.current_weights:
                model_weights = [
                    weights.get(model_name, 0.0) 
                    for weights in recent_weights
                    if model_name in weights
                ]
                
                if len(model_weights) >= 2:
                    # Consistency = 1 - coefficient of variation
                    mean_weight = np.mean(model_weights)
                    std_weight = np.std(model_weights)
                    
                    if mean_weight > 0:
                        cv = std_weight / mean_weight
                        consistency = max(0.0, 1.0 - cv)
                    else:
                        consistency = 1.0
                    
                    model_consistencies.append(consistency)
            
            # Overall consistency
            if model_consistencies:
                return np.mean(model_consistencies)
            else:
                return 1.0
                
        except Exception as e:
            self.logger.error(f"Error calculating weight consistency: {e}")
            return 1.0
    
    def _calculate_reversal_frequency(self) -> float:
        """Calculate frequency of weight direction reversals."""
        
        try:
            if len(self.weight_changes) < 3:
                return 0.0
            
            # Get recent changes
            recent_changes = self.weight_changes[-20:]  # Last 20 changes
            
            # Calculate reversals for each model
            model_reversals = []
            
            for model_name in self.current_weights:
                model_changes = [
                    change.get(model_name, 0.0) 
                    for change in recent_changes
                    if model_name in change
                ]
                
                if len(model_changes) >= 3:
                    reversals = 0
                    for i in range(1, len(model_changes)):
                        # Check for sign change
                        if (model_changes[i-1] > 0 and model_changes[i] < 0) or \
                           (model_changes[i-1] < 0 and model_changes[i] > 0):
                            reversals += 1
                    
                    reversal_freq = reversals / (len(model_changes) - 1)
                    model_reversals.append(reversal_freq)
            
            # Overall reversal frequency
            if model_reversals:
                return np.mean(model_reversals)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating reversal frequency: {e}")
            return 0.0
    
    def _calculate_overall_stability_score(self, stability_metrics: Dict[str, Any]) -> float:
        """Calculate overall stability score from individual metrics."""
        
        try:
            if 'metrics' not in stability_metrics:
                return 1.0
            
            metrics = stability_metrics['metrics']
            
            # Normalize individual metrics
            normalized_scores = {}
            
            # Volatility (lower is better)
            if 'volatility' in metrics:
                volatility = metrics['volatility'].get('average_volatility', 0.0)
                normalized_scores['volatility'] = max(0.0, 1.0 - (volatility / self.stability_threshold))
            
            # Drift (lower is better)
            if 'drift' in metrics:
                drift = metrics['drift'].get('total_drift', 0.0)
                drift_threshold = self.STABILITY_METRICS['drift']['threshold']
                normalized_scores['drift'] = max(0.0, 1.0 - (drift / drift_threshold))
            
            # Consistency (higher is better)
            if 'consistency' in metrics:
                consistency = metrics['consistency']
                normalized_scores['consistency'] = consistency
            
            # Reversal frequency (lower is better)
            if 'reversal_frequency' in metrics:
                reversal_freq = metrics['reversal_frequency']
                rev_threshold = self.STABILITY_METRICS['reversal_frequency']['threshold']
                normalized_scores['reversal_frequency'] = max(0.0, 1.0 - (reversal_freq / rev_threshold))
            
            # Calculate overall score
            if normalized_scores:
                overall_score = np.mean(list(normalized_scores.values()))
            else:
                overall_score = 1.0
            
            return overall_score
            
        except Exception as e:
            self.logger.error(f"Error calculating overall stability score: {e}")
            return 1.0
    
    def _generate_stability_recommendations(self, 
                                         stability_analysis: Dict[str, Any],
                                         excessive_changes: Dict[str, Any]) -> List[str]:
        """Generate stability recommendations."""
        
        recommendations = []
        
        try:
            if stability_analysis.get('status') != 'completed':
                return recommendations
            
            metrics = stability_analysis.get('metrics', {})
            
            # Check overall stability
            overall_stability = metrics.get('overall_stability', 1.0)
            
            if overall_stability < 0.5:
                recommendations.append(
                    f"🚨 CRITICAL: Very low stability score ({overall_stability:.3f}). "
                    "Immediate action required."
                )
            elif overall_stability < 0.7:
                recommendations.append(
                    f"⚠️ WARNING: Low stability score ({overall_stability:.3f}). "
                    "Consider stabilization measures."
                )
            elif overall_stability >= 0.8:
                recommendations.append(
                    f"✅ GOOD: High stability score ({overall_stability:.3f}). "
                    "Weights are stable."
                )
            
            # Check specific metrics
            if 'volatility' in metrics:
                volatility = metrics['volatility'].get('average_volatility', 0.0)
                if volatility > self.stability_threshold:
                    recommendations.append(
                        f"📊 HIGH VOLATILITY: Weight volatility is {volatility:.4f}. "
                        "Consider reducing update frequency or increasing smoothing."
                    )
            
            if 'drift' in metrics:
                drift = metrics['drift'].get('total_drift', 0.0)
                if drift > self.STABILITY_METRICS['drift']['threshold']:
                    recommendations.append(
                        f"📈 HIGH DRIFT: Weight drift is {drift:.4f}. "
                        "Consider weight rebalancing or reset."
                    )
            
            if 'consistency' in metrics:
                consistency = metrics['consistency']
                if consistency < self.STABILITY_METRICS['consistency']['threshold']:
                    recommendations.append(
                        f"🔄 LOW CONSISTENCY: Weight consistency is {consistency:.3f}. "
                        "Consider increasing smoothing factor."
                    )
            
            if 'reversal_frequency' in metrics:
                reversal_freq = metrics['reversal_frequency']
                if reversal_freq > self.STABILITY_METRICS['reversal_frequency']['threshold']:
                    recommendations.append(
                        f"🔄 HIGH REVERSALS: Reversal frequency is {reversal_freq:.3f}. "
                        "Consider reducing update sensitivity."
                    )
            
            # Check excessive changes
            if excessive_changes.get('has_excessive', False):
                recommendations.append(
                    f"⚠️ EXCESSIVE CHANGES: {len(excessive_changes['excessive_models'])} models "
                    f"exceeded change threshold of {self.max_change_per_update}."
                )
                
                for model_info in excessive_changes['excessive_models']:
                    recommendations.append(
                        f"   • {model_info['model']}: {model_info['change']:.4f} "
                        f"(threshold: {model_info['threshold']:.4f})"
                    )
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating stability recommendations: {e}")
            return recommendations
    
    def _update_stability_status(self, stability_analysis: Dict[str, Any]) -> None:
        """Update current stability status."""
        
        try:
            if stability_analysis.get('status') != 'completed':
                return
            
            metrics = stability_analysis.get('metrics', {})
            overall_stability = metrics.get('overall_stability', 1.0)
            
            if overall_stability >= 0.8:
                self.stability_status = 'stable'
            elif overall_stability >= 0.6:
                self.stability_status = 'moderately_stable'
            elif overall_stability >= 0.4:
                self.stability_status = 'unstable'
            else:
                self.stability_status = 'highly_unstable'
                
        except Exception as e:
            self.logger.error(f"Error updating stability status: {e}")
    
    def _is_action_required(self, recommendations: List[str]) -> bool:
        """Determine if action is required based on recommendations."""
        
        critical_keywords = [
            'CRITICAL', 'IMMEDIATE', 'HIGH VOLATILITY', 
            'HIGH DRIFT', 'EXCESSIVE CHANGES'
        ]
        
        return any(
            keyword in recommendation.upper()
            for recommendation in recommendations
            for keyword in critical_keywords
        )
    
    def _store_stability_event(self, analysis_results: Dict[str, Any]) -> None:
        """Store stability event for historical tracking."""
        
        try:
            event = {
                'timestamp': analysis_results['timestamp'],
                'stability_status': analysis_results['stability_status'],
                'stability_score': analysis_results['stability_analysis'].get('metrics', {}).get('overall_stability', 1.0),
                'recommendations': analysis_results['recommendations'],
                'weights': analysis_results['new_weights'].copy()
            }
            
            self.stability_events.append(event)
            
            # Keep only last 100 events
            if len(self.stability_events) > 100:
                self.stability_events = self.stability_events[-100:]
                
        except Exception as e:
            self.logger.error(f"Error storing stability event: {e}")
    
    def stabilize_weights(self, 
                        proposed_weights: Dict[str, float],
                        stabilization_method: str = "constrained") -> Dict[str, float]:
        """
        Apply stability constraints to weight changes.
        
        Args:
            proposed_weights: Proposed new weights
            stabilization_method: Method for stabilization
            
        Returns:
            Stabilized weights (sum to 1.0)
        """
        try:
            if not self.last_weights:
                return proposed_weights
            
            if stabilization_method == "constrained":
                return self._apply_constrained_stabilization(proposed_weights)
            elif stabilization_method == "exponential_smoothing":
                return self._apply_exponential_smoothing(proposed_weights)
            elif stabilization_method == "volatility_based":
                return self._apply_volatility_based_stabilization(proposed_weights)
            else:
                return proposed_weights
                
        except Exception as e:
            self.logger.error(f"Error stabilizing weights: {e}")
            return proposed_weights
    
    def _apply_constrained_stabilization(self, proposed_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply constrained stabilization to weight changes."""
        
        stabilized = {}
        
        for model_name, weight in proposed_weights.items():
            last_weight = self.last_weights.get(model_name, weight)
            
            # Limit weight changes
            change = weight - last_weight
            if abs(change) > self.max_change_per_update:
                if change > 0:
                    stabilized[model_name] = last_weight + self.max_change_per_update
                else:
                    stabilized[model_name] = last_weight - self.max_change_per_update
            else:
                stabilized[model_name] = weight
        
        # Normalize to sum to 1
        total = sum(stabilized.values())
        if total > 0:
            stabilized = {m: w/total for m, w in stabilized.items()}
        
        return stabilized
    
    def _apply_exponential_smoothing(self, proposed_weights: Dict[str, float], alpha: float = 0.7) -> Dict[str, float]:
        """Apply exponential smoothing to weight changes."""
        
        smoothed = {}
        
        for model_name, weight in proposed_weights.items():
            last_weight = self.last_weights.get(model_name, weight)
            
            # EMA: new = alpha * current + (1-alpha) * old
            smoothed_weight = alpha * weight + (1 - alpha) * last_weight
            smoothed[model_name] = smoothed_weight
        
        # Normalize to sum to 1
        total = sum(smoothed.values())
        if total > 0:
            smoothed = {m: w/total for m, w in smoothed.items()}
        
        return smoothed
    
    def _apply_volatility_based_stabilization(self, proposed_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply volatility-based stabilization."""
        
        # Calculate recent volatility for each model
        volatility_metrics = self._calculate_weight_volatility()
        model_volatilities = volatility_metrics.get('model_volatilities', {})
        
        stabilized = {}
        
        for model_name, weight in proposed_weights.items():
            last_weight = self.last_weights.get(model_name, weight)
            model_volatility = model_volatilities.get(model_name, 0.0)
            
            # Adjust maximum change based on volatility
            volatility_adjusted_threshold = self.max_change_per_update * (1 + model_volatility)
            
            change = weight - last_weight
            if abs(change) > volatility_adjusted_threshold:
                if change > 0:
                    stabilized[model_name] = last_weight + volatility_adjusted_threshold
                else:
                    stabilized[model_name] = last_weight - volatility_adjusted_threshold
            else:
                stabilized[model_name] = weight
        
        # Normalize to sum to 1
        total = sum(stabilized.values())
        if total > 0:
            stabilized = {m: w/total for m, w in stabilized.items()}
        
        return stabilized
    
    def get_stability_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of weight stability over time period."""
        
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            # Filter recent events
            recent_events = [
                event for event in self.stability_events
                if event['timestamp'] >= cutoff_time
            ]
            
            if not recent_events:
                return {'error': 'No recent stability data available'}
            
            # Calculate summary statistics
            summary = {
                'period_days': days,
                'total_events': len(recent_events),
                'average_stability_score': np.mean([
                    event['stability_score'] for event in recent_events
                ]),
                'stability_distribution': self._calculate_stability_distribution(recent_events),
                'most_common_status': self._get_most_common_stability_status(recent_events),
                'stability_trend': self._analyze_stability_trend(recent_events)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting stability summary: {e}")
            return {'error': str(e)}
    
    def _calculate_stability_distribution(self, events: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of stability statuses."""
        
        distribution = {}
        
        for event in events:
            status = event.get('stability_status', 'unknown')
            distribution[status] = distribution.get(status, 0) + 1
        
        return distribution
    
    def _get_most_common_stability_status(self, events: List[Dict[str, Any]]) -> str:
        """Get most common stability status."""
        
        status_counts = {}
        
        for event in events:
            status = event.get('stability_status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            return max(status_counts.items(), key=lambda x: x[1])[0]
        else:
            return 'unknown'
    
    def _analyze_stability_trend(self, events: List[Dict[str, Any]]) -> str:
        """Analyze trend in stability scores."""
        
        if len(events) < 5:
            return 'insufficient_data'
        
        # Extract stability scores over time
        stability_scores = [event['stability_score'] for event in events]
        
        # Calculate trend
        x = np.arange(len(stability_scores))
        slope = np.polyfit(x, stability_scores, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'degrading'
        else:
            return 'stable'
    
    def plot_stability_metrics(self, save_path: Optional[str] = None) -> None:
        """Plot stability metrics over time."""
        
        try:
            if len(self.weight_history) < 2:
                return
            
            import matplotlib.pyplot as plt
            
            # Extract data for plotting
            timestamps = [record['timestamp'] for record in self.weight_history]
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Weight Stability Metrics Over Time')
            
            # Plot 1: Weight volatility
            volatilities = []
            for i in range(len(self.weight_history)):
                if i >= self.window_size:
                    window_changes = [
                        change.get(list(self.current_weights.keys())[0], 0.0)
                        for change in self.weight_changes[max(0, i-self.window_size):i]
                    ]
                    vol = np.std(window_changes) if len(window_changes) > 1 else 0.0
                    volatilities.append(vol)
                else:
                    volatilities.append(0.0)
            
            axes[0, 0].plot(timestamps, volatilities)
            axes[0, 0].set_title('Weight Volatility')
            axes[0, 0].set_ylabel('Volatility')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Weight drift
            drifts = []
            initial_weights = self.weight_history[0]['weights']
            for record in self.weight_history:
                current_weights = record['weights']
                total_drift = np.mean([
                    abs(current_weights[model] - initial_weights.get(model, 0.0))
                    for model in current_weights
                ])
                drifts.append(total_drift)
            
            axes[0, 1].plot(timestamps, drifts)
            axes[0, 1].set_title('Cumulative Weight Drift')
            axes[0, 1].set_ylabel('Drift')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Weight consistency
            consistencies = []
            for i in range(len(self.weight_history)):
                if i >= self.window_size:
                    window_weights = [
                        record['weights'] for record in self.weight_history[max(0, i-self.window_size):i+1]
                    ]
                    
                    model_consistencies = []
                    for model_name in self.current_weights:
                        model_weights = [
                            weights.get(model_name, 0.0) for weights in window_weights
                            if model_name in weights
                        ]
                        
                        if len(model_weights) >= 2:
                            mean_weight = np.mean(model_weights)
                            std_weight = np.std(model_weights)
                            if mean_weight > 0:
                                cv = std_weight / mean_weight
                                consistency = max(0.0, 1.0 - cv)
                            else:
                                consistency = 1.0
                            model_consistencies.append(consistency)
                    
                    if model_consistencies:
                        consistencies.append(np.mean(model_consistencies))
                    else:
                        consistencies.append(1.0)
                else:
                    consistencies.append(1.0)
            
            axes[1, 0].plot(timestamps, consistencies)
            axes[1, 0].set_title('Weight Consistency')
            axes[1, 0].set_ylabel('Consistency')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Overall stability score
            stability_scores = []
            for i in range(len(self.weight_history)):
                if i >= self.window_size:
                    # Calculate stability score for this point
                    window_data = self.weight_history[max(0, i-self.window_size):i+1]
                    
                    # Simplified stability calculation
                    window_changes = [
                        change for record in window_data 
                        for change in record['changes'].values()
                    ]
                    
                    if window_changes:
                        avg_volatility = np.mean([abs(c) for c in window_changes])
                        stability = max(0.0, 1.0 - (avg_volatility / self.stability_threshold))
                    else:
                        stability = 1.0
                    
                    stability_scores.append(stability)
                else:
                    stability_scores.append(1.0)
            
            axes[1, 1].plot(timestamps, stability_scores)
            axes[1, 1].set_title('Overall Stability Score')
            axes[1, 1].set_ylabel('Stability Score')
            axes[1, 1].grid(True, alpha=0.3)
            
            # Format x-axis
            for ax in axes.flat:
                ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Stability metrics plot saved to {save_path}")
            else:
                plt.show()
            
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error plotting stability metrics: {e}")


# Factory function for easy instantiation
def get_weight_stability_monitor(stability_threshold: float = 0.1,
                               window_size: int = 10,
                               max_change_per_update: float = 0.15) -> WeightStabilityMonitor:
    """Factory function to get WeightStabilityMonitor instance."""
    return WeightStabilityMonitor(stability_threshold, window_size, max_change_per_update)


# Convenience function for quick monitoring
def monitor_weight_stability_quick(new_weights: Dict[str, float],
                                 last_weights: Dict[str, float],
                                 stability_threshold: float = 0.1) -> Dict[str, Any]:
    """
    Quick weight stability monitoring.
    
    Args:
        new_weights: New weight dictionary
        last_weights: Previous weight dictionary
        stability_threshold: Threshold for stability alerts
        
    Returns:
        Stability monitoring result dictionary
    """
    monitor = get_weight_stability_monitor(stability_threshold)
    return monitor.update_weights(new_weights, datetime.now())
