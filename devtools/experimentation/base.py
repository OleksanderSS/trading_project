# experiments/experiment_base.py

"""
Base class for experiments with common functionality
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import logging
import json
import os
from pathlib import Path

from config.unified_config_manager import UnifiedConfigManager

logger = logging.getLogger(__name__)

class BaseExperiment(ABC):
    """Base class for all experiments"""
    
    def __init__(self, name: str):
        self.name = name
        self.config_manager = UnifiedConfigManager()
        self.config = self.config_manager.get_config('experiments')
        self.output_dir = Path(self.config['base']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        self.start_time = None
        
    @abstractmethod
    def run_experiment(self, **kwargs) -> List[Dict]:
        """Run the experiment and return results"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> List[str]:
        """Get list of metrics this experiment produces"""
        pass
    
    def setup(self, **kwargs):
        """Setup experiment before running"""
        self.start_time = datetime.now()
        logger.info(f"Starting experiment: {self.name}")
        
    def teardown(self):
        """Cleanup after experiment"""
        duration = datetime.now() - self.start_time if self.start_time else timedelta(0)
        logger.info(f"Experiment {self.name} completed in {duration}")
    
    def save_results(self, format: str = "both"):
        """Save results to file"""
        if not self.results:
            logger.warning("No results to save")
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{self.name}_{timestamp}"
        
        if format in ["csv", "both"]:
            csv_path = self.output_dir / f"{base_filename}.csv"
            df = pd.DataFrame(self.results)
            df.to_csv(csv_path, index=False)
            logger.info(f"Results saved to {csv_path}")
        
        if format in ["json", "both"]:
            json_path = self.output_dir / f"{base_filename}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to {json_path}")
    
    def generate_report(self) -> str:
        """Generate markdown report"""
        if not self.results:
            return "No results to report"
            
        df = pd.DataFrame(self.results)
        
        report = f"# {self.name} Experiment Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"**Total Results:** {len(self.results)}\n\n"
        
        # Summary statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report += "## Summary Statistics\n\n"
            for col in numeric_cols:
                report += f"- **{col}:** Mean={df[col].mean():.4f}, Std={df[col].std():.4f}\n"
            report += "\n"
        
        # Detailed results table
        report += "## Detailed Results\n\n"
        report += df.to_markdown(index=False)
        
        return report
    
    def save_report(self):
        """Save markdown report"""
        report = self.generate_report()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"{self.name}_{timestamp}_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def run(self, save_results: bool = True, save_report: bool = True, **kwargs):
        """Complete experiment workflow"""
        try:
            self.setup(**kwargs)
            
            # Run experiment
            self.results = self.run_experiment(**kwargs)
            
            # Save results
            if save_results:
                self.save_results()
            
            # Generate and save report
            if save_report:
                self.save_report()
            
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment {self.name} failed: {e}")
            raise
        finally:
            self.teardown()
    
    def get_best_result(self, metric: str, higher_is_better: bool = True) -> Optional[Dict]:
        """Get best result based on metric"""
        if not self.results:
            return None
            
        df = pd.DataFrame(self.results)
        if metric not in df.columns:
            return None
            
        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
            
        return self.results[best_idx]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics of results"""
        if not self.results:
            return {}
            
        df = pd.DataFrame(self.results)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        stats = {}
        for col in numeric_cols:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'count': df[col].count()
            }
        
        return stats
