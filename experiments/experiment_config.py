# experiments/experiment_config.py

"""
Configuration for experiments
"""

from pathlib import Path
from typing import Dict, List, Optional
import os

class ExperimentConfig:
    """Centralized configuration for experiments"""
    
    # Output configuration
    OUTPUT = {
        "base_dir": "experiments/results",
        "create_timestamp_dirs": True,
        "save_formats": ["csv", "json"],  # csv, json, parquet
        "generate_plots": True,
        "plot_format": "png",
        "plot_dpi": 300
    }
    
    # Parallel processing
    PARALLEL = {
        "enabled": True,
        "max_workers": None,  # None = auto-detect CPU count
        "chunk_size": 10,  # Number of test cases per worker
        "timeout_seconds": 300  # 5 minutes per test case
    }
    
    # Experiment parameters
    DEFAULT_PARAMS = {
        "days": 365,
        "tickers": None,  # None = use config default
        "timeframes": None,  # None = use config default
        "metrics": ["MAE", "RMSE", "R2", "Sharpe"],
        "higher_is_better": {
            "MAE": False,
            "RMSE": False,
            "R2": True,
            "Sharpe": True
        }
    }
    
    # Visualization
    VISUALIZATION = {
        "figure_size": (12, 8),
        "style": "seaborn",
        "color_palette": "viridis",
        "font_size": 12,
        "save_plots": True,
        "plot_formats": ["png", "pdf"]
    }
    
    # Logging
    LOGGING = {
        "level": "INFO",
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "save_to_file": True,
        "log_dir": "logs/experiments"
    }
    
    # Performance tracking
    TRACKING = {
        "track_execution_time": True,
        "track_memory_usage": True,
        "track_system_resources": True,
        "save_performance_stats": True
    }
    
    @classmethod
    def get_output_dir(cls, experiment_name: str) -> Path:
        """Get output directory for experiment"""
        base_dir = Path(cls.OUTPUT["base_dir"])
        
        if cls.OUTPUT["create_timestamp_dirs"]:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = base_dir / experiment_name / timestamp
        else:
            output_dir = base_dir / experiment_name
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    
    @classmethod
    def get_max_workers(cls) -> int:
        """Get maximum number of workers"""
        if not cls.PARALLEL["enabled"]:
            return 1
        
        max_workers = cls.PARALLEL["max_workers"]
        if max_workers is None:
            import multiprocessing as mp
            max_workers = mp.cpu_count()
        
        return max(1, max_workers)
    
    @classmethod
    def get_default_params(cls) -> Dict:
        """Get default experiment parameters"""
        return cls.DEFAULT_PARAMS.copy()
    
    @classmethod
    def get_metrics_info(cls) -> Dict:
        """Get information about metrics"""
        return {
            "metrics": cls.DEFAULT_PARAMS["metrics"],
            "higher_is_better": cls.DEFAULT_PARAMS["higher_is_better"]
        }
    
    @classmethod
    def setup_logging(cls, experiment_name: str):
        """Setup logging for experiment"""
        import logging
        
        log_dir = Path(cls.LOGGING["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"{experiment_name}.log"
        
        logging.basicConfig(
            level=getattr(logging, cls.LOGGING["level"]),
            format=cls.LOGGING["format"],
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return issues"""
        issues = []
        
        # Check output directory
        try:
            base_dir = Path(cls.OUTPUT["base_dir"])
            base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            issues.append(f"Output directory issue: {e}")
        
        # Check parallel processing
        if cls.PARALLEL["enabled"] and cls.PARALLEL["max_workers"] is not None:
            if cls.PARALLEL["max_workers"] < 1:
                issues.append("max_workers must be >= 1")
        
        # Check metrics
        required_metrics = ["MAE", "RMSE", "R2", "Sharpe"]
        for metric in required_metrics:
            if metric not in cls.DEFAULT_PARAMS["metrics"]:
                issues.append(f"Missing metric: {metric}")
        
        return issues
