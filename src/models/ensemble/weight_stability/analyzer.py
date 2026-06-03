import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WeightStabilityAnalyzer")

class WeightStabilityAnalyzer:
    """Analyzes weight stability and provides recommendations."""
    
    def __init__(self, config: Any):
        self.logger = logger
        self.config = config

    def generate_stability_recommendations(self, 
                                         metrics: Dict[str, Any],
                                         excessive_changes: Dict[str, Any]) -> List[str]:
        """Generate stability recommendations."""
        recommendations = []
        try:
            overall_stability = metrics.get('overall_stability', 1.0)
            
            if overall_stability < 0.5:
                recommendations.append(f"🚨 CRITICAL: Very low stability score ({overall_stability:.3f}). Immediate action required.")
            elif overall_stability < 0.7:
                recommendations.append(f"⚠️ WARNING: Low stability score ({overall_stability:.3f}). Consider stabilization measures.")
            elif overall_stability >= 0.8:
                recommendations.append(f"✅ GOOD: High stability score ({overall_stability:.3f}). Weights are stable.")
            
            # Check specific metrics
            if 'volatility' in metrics:
                vol = metrics['volatility'].get('average_volatility', 0.0)
                if vol > self.config.stability_threshold:
                    recommendations.append(f"📊 HIGH VOLATILITY: Weight volatility is {vol:.4f}. Consider reducing update frequency or increasing smoothing.")
            
            if 'drift' in metrics:
                drift = metrics['drift'].get('total_drift', 0.0)
                if drift > self.config.STABILITY_METRICS['drift']['threshold']:
                    recommendations.append(f"📈 HIGH DRIFT: Weight drift is {drift:.4f}. Consider weight rebalancing or reset.")
            
            if 'consistency' in metrics:
                cons = metrics['consistency']
                if cons < self.config.STABILITY_METRICS['consistency']['threshold']:
                    recommendations.append(f"🔄 LOW CONSISTENCY: Weight consistency is {cons:.3f}. Consider increasing smoothing factor.")
            
            if excessive_changes.get('has_excessive', False):
                recommendations.append(f"⚠️ EXCESSIVE CHANGES: {len(excessive_changes['excessive_models'])} models exceeded threshold.")
            
            return recommendations
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return recommendations

    def determine_stability_status(self, overall_stability: float) -> str:
        """Update current stability status."""
        if overall_stability >= 0.8: return 'stable'
        elif overall_stability >= 0.6: return 'moderately_stable'
        elif overall_stability >= 0.4: return 'unstable'
        else: return 'highly_unstable'

    def is_action_required(self, recommendations: List[str]) -> bool:
        """Determine if action is required based on recommendations."""
        critical_keywords = ['CRITICAL', 'IMMEDIATE', 'HIGH VOLATILITY', 'HIGH DRIFT', 'EXCESSIVE CHANGES']
        return any(keyword in rec.upper() for rec in recommendations for keyword in critical_keywords)

    def analyze_stability_trend(self, scores: List[float]) -> str:
        """Analyze trend in stability scores."""
        if len(scores) < 5:
            return 'insufficient_data'
        x = np.arange(len(scores))
        slope = np.polyfit(x, scores, 1)[0]
        if slope > 0.01: return 'improving'
        elif slope < -0.01: return 'degrading'
        else: return 'stable'
