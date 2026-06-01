#!/usr/bin/env python3
"""
Drift Visualizer - Visualization of drift monitoring results
Handles matplotlib/seaborn visualizations for drift analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DriftVisualizer")


class DriftVisualizer:
    """
    Visualization for drift monitoring results.
    
    Provides visualization methods for:
    - Distribution drift plots
    - Performance trend charts
    - Confidence drift visualizations
    - Drift summary dashboards
    """
    
    def __init__(self):
        """Initialize Drift Visualizer."""
        self.logger = logger
        self.logger.info("✅ DriftVisualizer initialized")
    
    def plot_distribution_drift(self, 
                               current_predictions: np.ndarray,
                               reference_predictions: np.ndarray,
                               method: str = 'ks_test',
                               save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot distribution drift between current and reference predictions.
        
        Args:
            current_predictions: Current prediction distribution
            reference_predictions: Reference prediction distribution
            method: Drift detection method used
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with plot information
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Histogram comparison
            axes[0].hist(reference_predictions, bins=50, alpha=0.5, label='Reference', color='blue')
            axes[0].hist(current_predictions, bins=50, alpha=0.5, label='Current', color='red')
            axes[0].set_xlabel('Prediction Value')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title(f'Distribution Comparison ({method})')
            axes[0].legend()
            
            # KDE plot
            sns.kdeplot(reference_predictions, ax=axes[1], label='Reference', color='blue')
            sns.kdeplot(current_predictions, ax=axes[1], label='Current', color='red')
            axes[1].set_xlabel('Prediction Value')
            axes[1].set_ylabel('Density')
            axes[1].set_title('Kernel Density Estimation')
            axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved distribution drift plot to {save_path}")
            
            result = {
                'status': 'success',
                'plot_type': 'distribution_drift',
                'method': method,
                'save_path': save_path
            }
            
            plt.close()
            return result
            
        except ImportError:
            self.logger.warning("matplotlib/seaborn not available, skipping visualization")
            return {'status': 'skipped', 'reason': 'visualization_libraries_not_available'}
        except Exception as e:
            self.logger.error(f"Error plotting distribution drift: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def plot_performance_trend(self, 
                             performance_history: List[Dict[str, Any]],
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot performance trend over time.
        
        Args:
            performance_history: List of performance records
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with plot information
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(performance_history) < 2:
                return {'status': 'skipped', 'reason': 'insufficient_data'}
            
            # Extract data
            timestamps = [record['timestamp'] for record in performance_history]
            mse_values = [record['metrics'].get('mse', 0) for record in performance_history]
            mae_values = [record['metrics'].get('mae', 0) for record in performance_history]
            r2_values = [record['metrics'].get('r2', 0) for record in performance_history]
            
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            
            # MSE trend
            axes[0].plot(timestamps, mse_values, marker='o', color='red')
            axes[0].set_ylabel('MSE')
            axes[0].set_title('Mean Squared Error Trend')
            axes[0].grid(True, alpha=0.3)
            
            # MAE trend
            axes[1].plot(timestamps, mae_values, marker='o', color='blue')
            axes[1].set_ylabel('MAE')
            axes[1].set_title('Mean Absolute Error Trend')
            axes[1].grid(True, alpha=0.3)
            
            # R2 trend
            axes[2].plot(timestamps, r2_values, marker='o', color='green')
            axes[2].set_ylabel('R²')
            axes[2].set_title('R² Score Trend')
            axes[2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved performance trend plot to {save_path}")
            
            result = {
                'status': 'success',
                'plot_type': 'performance_trend',
                'data_points': len(performance_history),
                'save_path': save_path
            }
            
            plt.close()
            return result
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualization")
            return {'status': 'skipped', 'reason': 'visualization_libraries_not_available'}
        except Exception as e:
            self.logger.error(f"Error plotting performance trend: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def plot_confidence_drift(self, 
                             current_confidences: np.ndarray,
                             reference_confidences: np.ndarray,
                             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot confidence drift between current and reference distributions.
        
        Args:
            current_confidences: Current confidence distribution
            reference_confidences: Reference confidence distribution
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with plot information
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Box plot comparison
            data_to_plot = [reference_confidences, current_confidences]
            bp = axes[0].boxplot(data_to_plot, labels=['Reference', 'Current'], patch_artist=True)
            bp['boxes'][0].set_facecolor('blue')
            bp['boxes'][1].set_facecolor('red')
            axes[0].set_ylabel('Confidence')
            axes[0].set_title('Confidence Distribution Comparison')
            axes[0].grid(True, alpha=0.3)
            
            # Violin plot
            violin_parts = axes[1].violinplot(data_to_plot, positions=[1, 2], showmeans=True)
            for pc in violin_parts['bodies']:
                pc.set_facecolor('lightblue')
            axes[1].set_xticks([1, 2])
            axes[1].set_xticklabels(['Reference', 'Current'])
            axes[1].set_ylabel('Confidence')
            axes[1].set_title('Confidence Distribution (Violin Plot)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved confidence drift plot to {save_path}")
            
            result = {
                'status': 'success',
                'plot_type': 'confidence_drift',
                'save_path': save_path
            }
            
            plt.close()
            return result
            
        except ImportError:
            self.logger.warning("matplotlib/seaborn not available, skipping visualization")
            return {'status': 'skipped', 'reason': 'visualization_libraries_not_available'}
        except Exception as e:
            self.logger.error(f"Error plotting confidence drift: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _plot_drift_score_gauge(self, ax, drift_analysis: Dict[str, Any]) -> None:
        """Plot drift score gauge."""
        drift_score = drift_analysis.get('overall_drift_score', 0)
        severity = drift_analysis.get('drift_severity', 'none')
        colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'lightgreen', 'none': 'green'}
        ax.bar(['Drift Score'], [drift_score], color=colors.get(severity, 'gray'))
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title(f'Overall Drift Score ({severity.upper()})')

    def _plot_performance_degradation(self, ax, performance_analysis: Dict[str, Any]) -> None:
        """Plot performance degradation."""
        degrad_score = performance_analysis.get('degradation_score', 0)
        degrad_detected = performance_analysis.get('degradation_detected', False)
        ax.bar(['Degradation'], [degrad_score], color='red' if degrad_detected else 'green')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Performance Degradation')

    def _plot_confidence_drift(self, ax, confidence_analysis: Dict[str, Any]) -> None:
        """Plot confidence drift."""
        conf_drift = confidence_analysis.get('drift_score', 0)
        conf_detected = confidence_analysis.get('confidence_drift_detected', False)
        ax.bar(['Confidence Drift'], [conf_drift], color='red' if conf_detected else 'green')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.set_title('Confidence Drift')

    def _plot_drift_methods_comparison(self, ax, drift_analysis: Dict[str, Any]) -> None:
        """Plot drift methods comparison."""
        drift_methods = drift_analysis.get('drift_methods', {})
        method_names = list(drift_methods.keys())
        method_scores = [drift_methods[m].get('drift_score', 0) for m in method_names]
        method_detected = [drift_methods[m].get('drift_detected', False) for m in method_names]
        colors_list = ['red' if d else 'green' for d in method_detected]
        ax.bar(method_names, method_scores, color=colors_list)
        ax.set_ylabel('Drift Score')
        ax.set_title('Drift Detection Methods Comparison')
        ax.set_ylim(0, 1)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    def _save_plot(self, fig, save_path: Optional[str]) -> None:
        """Save plot if path provided."""
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Saved drift summary dashboard to {save_path}")

    def plot_drift_summary_dashboard(self, 
                                    drift_analysis: Dict[str, Any],
                                    performance_analysis: Dict[str, Any],
                                    confidence_analysis: Dict[str, Any],
                                    save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot comprehensive drift summary dashboard.
        
        Args:
            drift_analysis: Drift analysis results
            performance_analysis: Performance analysis results
            confidence_analysis: Confidence analysis results
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with plot information
        """
        try:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=(16, 10))
            
            # Create grid layout
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            # Drift score gauge
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_drift_score_gauge(ax1, drift_analysis)
            
            # Performance degradation
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_performance_degradation(ax2, performance_analysis)
            
            # Confidence drift
            ax3 = fig.add_subplot(gs[0, 2])
            self._plot_confidence_drift(ax3, confidence_analysis)
            
            # Drift methods comparison
            ax4 = fig.add_subplot(gs[1, :])
            self._plot_drift_methods_comparison(ax4, drift_analysis)
            
            self._save_plot(fig, save_path)
            
            result = {
                'status': 'success',
                'plot_type': 'drift_summary_dashboard',
                'save_path': save_path
            }
            
            plt.close()
            return result
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualization")
            return {'status': 'skipped', 'reason': 'visualization_libraries_not_available'}
        except Exception as e:
            self.logger.error(f"Error plotting drift summary dashboard: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def plot_drift_timeline(self, 
                           drift_history: List[Dict[str, Any]],
                           save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Plot drift detection timeline.
        
        Args:
            drift_history: List of drift detection records
            save_path: Optional path to save the plot
            
        Returns:
            Dictionary with plot information
        """
        try:
            import matplotlib.pyplot as plt
            
            if len(drift_history) < 2:
                return {'status': 'skipped', 'reason': 'insufficient_data'}
            
            # Extract data
            timestamps = [record['timestamp'] for record in drift_history]
            drift_scores = [record.get('drift_score', 0) for record in drift_history]
            drift_detected = [record.get('drift_detected', False) for record in drift_history]
            
            fig, ax = plt.subplots(figsize=(14, 6))
            
            # Plot drift scores
            ax.plot(timestamps, drift_scores, marker='o', color='blue', label='Drift Score')
            
            # Highlight detected drifts
            for i, (ts, score, detected) in enumerate(zip(timestamps, drift_scores, drift_detected)):
                if detected:
                    ax.scatter(ts, score, color='red', s=100, zorder=5)
            
            ax.set_xlabel('Time')
            ax.set_ylabel('Drift Score')
            ax.set_title('Drift Detection Timeline')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                self.logger.info(f"Saved drift timeline plot to {save_path}")
            
            result = {
                'status': 'success',
                'plot_type': 'drift_timeline',
                'data_points': len(drift_history),
                'save_path': save_path
            }
            
            plt.close()
            return result
            
        except ImportError:
            self.logger.warning("matplotlib not available, skipping visualization")
            return {'status': 'skipped', 'reason': 'visualization_libraries_not_available'}
        except Exception as e:
            self.logger.error(f"Error plotting drift timeline: {e}")
            return {'status': 'error', 'error': str(e)}


# Factory function
def get_drift_visualizer() -> DriftVisualizer:
    """Factory function to get DriftVisualizer instance."""
    return DriftVisualizer()
