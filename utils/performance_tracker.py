# utils/performance_tracker.py

import json
import time
from datetime import datetime
from .logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger("PerformanceTracker")

class PerformanceTracker:
    """Вandдстежує продуктивнandсть pipeline and моwhereлей"""
    
    def __init__(self):
        self.start_time = None
        self.stage_times = {}
        self.model_performance = {}
        
    def start_pipeline(self):
        """Початок вandдстеження pipeline"""
        self.start_time = time.time()
        logger.info("[START] Pipeline tracking started")
        
    def start_stage(self, stage_name):
        """Початок вandдстеження stage"""
        self.stage_times[stage_name] = {"start": time.time()}
        
    def end_stage(self, stage_name):
        """Кandnotць вandдстеження stage"""
        if stage_name in self.stage_times:
            duration = time.time() - self.stage_times[stage_name]["start"]
            self.stage_times[stage_name]["duration"] = duration
            logger.info(f" {stage_name} completed in {duration:.2f}s")
            
    def track_model(self, model_name, ticker, interval, metrics):
        """Вandдстеження продуктивностand моwhereлand"""
        key = f"{model_name}_{ticker}_{interval}"
        self.model_performance[key] = {
            "timestamp": datetime.now().isoformat(),
            "model": model_name,
            "ticker": ticker,
            "interval": interval,
            "metrics": metrics
        }
        
    def get_summary(self):
        """Отримати пandдсумок продуктивностand"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        summary = {
            "total_pipeline_time": total_time,
            "stage_times": self.stage_times,
            "model_count": len(self.model_performance),
            "best_models": self._get_best_models(),
            "timestamp": datetime.now().isoformat()
        }
        
        return summary
        
    def _get_best_models(self):
        """Find найкращand моwhereлand"""
        best = {}
        for key, data in self.model_performance.items():
            ticker = data["ticker"]
            metrics = data.get("metrics", {})
            
            # Для класифandкацandї використовуємо F1
            score = metrics.get("F1", metrics.get("accuracy", 0))
            
            if ticker not in best or score > best[ticker]["score"]:
                best[ticker] = {
                    "model": data["model"],
                    "score": score,
                    "interval": data["interval"]
                }
                
        return best
        
    def save_report(self, filepath="output/performance_report.json"):
        """Зберегти withвandт продуктивностand"""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        summary = self.get_summary()
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"[DATA] Performance report saved: {filepath}")
        return summary

# Global tracker
performance_tracker = PerformanceTracker()