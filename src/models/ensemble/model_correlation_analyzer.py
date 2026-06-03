#!/usr/bin/env python3
"""
Model Correlation Analyzer - Facade for Modular Model Correlation Analysis.
Maintains backward compatibility with the original ModelCorrelationAnalyzer.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

from .correlation.correlation_engine import CorrelationEngine, get_correlation_engine
from .correlation.correlation_visualizer import CorrelationVisualizer, get_correlation_visualizer

class ModelCorrelationAnalyzer:
    """
    Facade for ModelCorrelationAnalyzer.
    Delegates to modular components in the 'correlation' subdirectory.
    """
    
    def __init__(self, 
                 correlation_method: str = "pearson",
                 diversity_threshold: float = 0.7):
        """
        Initialize Model Correlation Analyzer.
        
        Args:
            correlation_method: Method for correlation calculation
            diversity_threshold: Threshold for diversity filtering
        """
        self.correlation_method = correlation_method
        self.diversity_threshold = diversity_threshold
        
        # Initialize components
        self.engine = get_correlation_engine(correlation_method, diversity_threshold)
        self.visualizer = get_correlation_visualizer()
        
        # Analysis history for backward compatibility
        self.analysis_history = []

    def analyze_correlation(self, 
                           models: Dict[str, Any],
                           X: pd.DataFrame,
                           y: pd.Series,
                           sample_size: Optional[int] = None) -> Dict[str, Any]:
        """Analyze prediction correlation between models."""
        results = self.engine.analyze_correlation(models, X, y, sample_size)
        if 'error' not in results:
            self.analysis_history.append(results)
        return results

    def select_diverse_subset(self, 
                             models: Dict[str, Any],
                             X: pd.DataFrame,
                             y: pd.Series,
                             max_models: int = 5,
                             diversity_threshold: Optional[float] = None) -> List[str]:
        """Select diverse subset of models for ensemble."""
        return self.engine.select_diverse_subset(models, X, y, max_models, diversity_threshold)

    def adjust_weights_by_correlation(self, 
                                    base_weights: Dict[str, float],
                                    correlation_matrix: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Adjust ensemble weights based on model correlation."""
        return self.engine.adjust_weights_by_correlation(base_weights, correlation_matrix)

    def plot_correlation_matrix(self, 
                               correlation_matrix: Dict[str, Dict[str, float]],
                               save_path: Optional[str] = None) -> None:
        """Plot correlation matrix heatmap."""
        self.visualizer.plot_correlation_matrix(correlation_matrix, self.correlation_method, save_path)

    def get_analysis_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get summary of correlation analysis over time period."""
        return self.engine.get_analysis_summary(self.analysis_history, days)

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
    """Quick model correlation analysis."""
    analyzer = get_model_correlation_analyzer(correlation_method)
    return analyzer.analyze_correlation(models, X, y)
