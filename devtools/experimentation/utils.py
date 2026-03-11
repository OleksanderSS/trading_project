# experiments/experiment_utils.py

"""
Utility functions for experiments
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)

class PerformanceTracker:
    """Track performance metrics during experiments"""
    
    def __init__(self):
        self.start_time = None
        self.checkpoints = []
        self.memory_usage = []
        self.cpu_usage = []
    
    def start(self):
        """Start tracking"""
        self.start_time = time.time()
        self.checkpoints.append(("start", self.start_time, self._get_system_stats()))
    
    def checkpoint(self, name: str):
        """Add a checkpoint"""
        current_time = time.time()
        elapsed = current_time - self.start_time if self.start_time else 0
        stats = self._get_system_stats()
        
        self.checkpoints.append((name, current_time, stats))
        
        logger.info(f" Checkpoint '{name}': {elapsed:.2f}s, Memory: {stats['memory_mb']:.1f}MB")
    
    def _get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            "memory_mb": psutil.virtual_memory().used / 1024 / 1024,
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent()
        }
    
    def get_summary(self) -> Dict:
        """Get performance summary"""
        if not self.checkpoints:
            return {}
        
        total_time = self.checkpoints[-1][1] - self.checkpoints[0][1]
        
        return {
            "total_time_seconds": total_time,
            "total_time_formatted": f"{total_time:.2f}s",
            "checkpoints": len(self.checkpoints),
            "peak_memory_mb": max([cp[2]["memory_mb"] for cp in self.checkpoints]),
            "avg_cpu_percent": np.mean([cp[2]["cpu_percent"] for cp in self.checkpoints])
        }

class ExperimentVisualizer:
    """Create visualizations for experiment results"""
    
    def __init__(self, style: str = "seaborn"):
        self.style = style
        self.setup_style()
    
    def setup_style(self):
        """Setup plotting style"""
        if self.style == "seaborn":
            sns.set_style("whitegrid")
            sns.set_palette("viridis")
        plt.rcParams.update({
            "font.size": 12,
            "figure.titlesize": 14,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 10
        })
    
    def plot_metric_comparison(self, df: pd.DataFrame, metrics: List[str], 
                          group_col: str = "layer_count", 
                          save_path: Optional[Path] = None) -> None:
        """Plot metrics comparison by group"""
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            # Group by specified column
            grouped = df.groupby(group_col)[metric].agg(['mean', 'std']).reset_index()
            
            # Plot
            axes[i].errorbar(grouped[group_col], grouped['mean'], 
                            yerr=grouped['std'], marker='o', capsize=5)
            axes[i].set_title(f'{metric} vs {group_col}')
            axes[i].set_xlabel(group_col.replace('_', ' ').title())
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(metrics), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metric comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame, metrics: List[str],
                               save_path: Optional[Path] = None) -> None:
        """Plot correlation heatmap of metrics"""
        
        # Filter numeric columns
        numeric_df = df[metrics].select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            logger.error("No numeric columns for correlation heatmap")
            return
        
        # Calculate correlation
        corr_matrix = numeric_df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0,
                   square=True, fmt='.3f')
        plt.title('Metrics Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def plot_best_results(self, df: pd.DataFrame, metric: str, 
                        top_n: int = 10, save_path: Optional[Path] = None) -> None:
        """Plot top N results for a metric"""
        
        # Sort by metric
        if metric in df.columns:
            sorted_df = df.sort_values(metric, ascending=False).head(top_n)
        else:
            logger.error(f"Metric '{metric}' not found in dataframe")
            return
        
        # Create horizontal bar plot
        plt.figure(figsize=(12, 8))
        
        # Create labels
        labels = [f"{row['ticker']} {row['time_frame']}\n{row['layers']}" 
                 for _, row in sorted_df.iterrows()]
        
        plt.barh(range(len(sorted_df)), sorted_df[metric])
        plt.yticks(range(len(sorted_df)), labels)
        plt.xlabel(metric)
        plt.title(f'Top {top_n} Results by {metric}')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Best results plot saved to {save_path}")
        
        plt.show()

def analyze_experiment_results(results: List[Dict], metrics: List[str]) -> Dict:
    """Analyze experiment results and return insights"""
    
    if not results:
        return {"error": "No results to analyze"}
    
    df = pd.DataFrame(results)
    analysis = {}
    
    # Success rate
    success_count = len(df[df['success'] == True])
    total_count = len(df)
    analysis['success_rate'] = success_count / total_count if total_count > 0 else 0
    
    # Metric analysis (only successful results)
    successful_df = df[df['success'] == True]
    
    if len(successful_df) > 0:
        for metric in metrics:
            if metric in successful_df.columns:
                metric_data = successful_df[metric].dropna()
                if len(metric_data) > 0:
                    analysis[metric] = {
                        'mean': metric_data.mean(),
                        'std': metric_data.std(),
                        'min': metric_data.min(),
                        'max': metric_data.max(),
                        'median': metric_data.median(),
                        'count': len(metric_data)
                    }
    
    # Best results
    for metric in metrics:
        if metric in successful_df.columns:
            best_idx = successful_df[metric].idxmax() if metric in ['R2', 'Sharpe'] else successful_df[metric].idxmin()
            if best_idx is not None and best_idx in successful_df.index:
                best_result = successful_df.loc[best_idx]
                analysis[f'best_{metric.lower()}'] = {
                    'value': best_result[metric],
                    'ticker': best_result['ticker'],
                    'time_frame': best_result['time_frame'],
                    'layers': best_result['layers']
                }
    
    return analysis

def create_experiment_summary(results: List[Dict], experiment_name: str, 
                         output_dir: Path) -> Dict:
    """Create comprehensive experiment summary"""
    
    from datetime import datetime
    
    summary = {
        'experiment_name': experiment_name,
        'timestamp': datetime.now().isoformat(),
        'total_tests': len(results),
        'successful_tests': len([r for r in results if r.get('success', False)]),
        'output_directory': str(output_dir),
        'results_file': str(output_dir / f"{experiment_name}_results.json")
    }
    
    # Add performance analysis
    if results:
        analysis = analyze_experiment_results(results, ['MAE', 'RMSE', 'R2', 'Sharpe'])
        summary['analysis'] = analysis
    
    return summary

def save_experiment_summary(summary: Dict, output_dir: Path):
    """Save experiment summary to file"""
    
    summary_path = output_dir / "experiment_summary.json"
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"Experiment summary saved to {summary_path}")
    return summary_path
