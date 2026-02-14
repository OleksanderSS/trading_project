"""
Colab-Optimized Pipeline for Large Ticker Sets
Оптимandwithований pipeline for Colab with великими нorрами тandкерandв
"""

import os
import json
import logging
import time
import gc
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Додаємо шлях до проекту
current_dir = Path(__file__).parent
import sys
sys.path.append(str(current_dir.parent))

@dataclass
class ColabConfig:
    """Конфandгурацandя for Colab оптимandforцandї"""
    max_tickers_per_run: int = 20  # Максимальна кandлькandсть тandкерandв for forпуск
    max_memory_usage_gb: float = 10.0  # Максимальна пам'ять
    auto_save_interval: int = 5  # Інтервал автоwithбереження (хвилин)
    cleanup_memory: bool = True  # Очищати пам'ять мandж батчами
    use_lightweight_models: bool = True  # Використовувати легкand моwhereлand
    checkpoint_frequency: int = 3  # Частоand withбереження чекпоandнтandв
    
class ColabOptimizer:
    """Оптимandforтор for Colab середовища"""
    
    def __init__(self, config: ColabConfig = None):
        self.config = config or ColabConfig()
        self.logger = logging.getLogger("ColabOptimizer")
        
        # Створюємо директорandї
        self.colab_dir = Path("data/colab")
        self.models_dir = Path("models/colab")
        self.checkpoints_dir = Path("models/colab/checkpoints")
        
        for dir_path in [self.colab_dir, self.models_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Вandдстеження ресурсandв
        self.start_time = time.time()
        self.processed_tickers = []
        self.current_batch = []
        
    def check_colab_environment(self) -> Dict[str, Any]:
        """Check environment Colab"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            
            return {
                "is_colab": "google.colab" in sys.modules,
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "used_memory_gb": memory.used / (1024**3),
                "memory_percent": memory.percent
            }
        except ImportError:
            return {"is_colab": False, "error": "psutil not available"}
    
    def optimize_for_colab(self) -> Dict[str, Any]:
        """Оптимandwithувати settings for Colab"""
        env_info = self.check_colab_environment()
        
        if env_info.get("is_colab", False):
            self.logger.info("Colab environment detected, applying optimizations...")
            
            # Налаштування for Colab
            optimizations = {
                "reduce_batch_size": True,
                "enable_memory_cleanup": True,
                "use_lightweight_models": True,
                "disable_verbose_logging": True,
                "limit_data_loading": True
            }
            
            # Очищуємо пам'ять
            if self.config.cleanup_memory:
                self.cleanup_memory()
            
            return optimizations
        else:
            self.logger.info("Local environment detected")
            return {}
    
    def cleanup_memory(self):
        """Очищення пам'ятand"""
        self.logger.info("Cleaning up memory...")
        
        # Збираємо смandття
        gc.collect()
        
        # Очищуємо кеш pandas
        if hasattr(pd, 'core'):
            pd.core.common._ALLOWED_OPS = set()
        
        # Очищуємо NumPy кеш
        if hasattr(np, 'clear_cache'):
            np.clear_cache()
        
        self.logger.info("Memory cleanup completed")
    
    def create_ticker_batches_for_colab(self, tickers: List[str]) -> List[List[str]]:
        """Create батчand оптимandwithованand for Colab"""
        batches = []
        
        # Роwithбиваємо на notвеликand батчand
        batch_size = min(self.config.max_tickers_per_run, len(tickers))
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            
            # Перевandряємо роwithмandр батчу
            if len(batch) > self.config.max_tickers_per_run:
                # Роwithбиваємо ще дрandбнandше
                for j in range(0, len(batch), self.config.max_tickers_per_run):
                    batches.append(batch[j:j + self.config.max_tickers_per_run])
            else:
                batches.append(batch)
        
        return batches
    
    def estimate_colab_runtime(self, tickers: List[str]) -> Dict[str, float]:
        """Оцandнити час виконання в Colab"""
        # Приблиwithнand роwithрахунки for Colab
        time_per_ticker_minutes = 15  # 15 хвилин на тandкер в Colab
        setup_time_minutes = 10  # Час на settings
        
        total_time = len(tickers) * time_per_ticker_minutes + setup_time_minutes
        
        return {
            "estimated_minutes": total_time,
            "estimated_hours": total_time / 60,
            "safe_session_hours": 11.5,  # Colab сесandя ~12 годин
            "recommended_batches": max(1, int(total_time / 60 / 10))  # Батчand по 10 годин
        }
    
    def create_colab_training_plan(self, tickers: List[str]) -> Dict[str, Any]:
        """Create план тренування for Colab"""
        batches = self.create_ticker_batches_for_colab(tickers)
        runtime_estimate = self.estimate_colab_runtime(tickers)
        
        plan = {
            "total_tickers": len(tickers),
            "total_batches": len(batches),
            "batch_size": self.config.max_tickers_per_run,
            "runtime_estimate": runtime_estimate,
            "batches": [],
            "optimizations": self.optimize_for_colab()
        }
        
        for i, batch in enumerate(batches):
            batch_info = {
                "batch_id": i + 1,
                "tickers": batch,
                "ticker_count": len(batch),
                "estimated_time_minutes": len(batch) * 15,
                "memory_estimate_gb": len(batch) * 0.5
            }
            plan["batches"].append(batch_info)
        
        return plan
    
    def save_colab_plan(self, plan: Dict[str, Any], filename: str = None):
        """Зберегти план for Colab"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"colab_plan_{timestamp}.json"
        
        filepath = self.colab_dir / filename
        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=2)
        
        self.logger.info(f"Colab plan saved to {filepath}")
        return filepath
    
    def execute_colab_batch(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Виконати батч в Colab"""
        batch_id = batch_info["batch_id"]
        tickers = batch_info["tickers"]
        
        self.logger.info(f"Executing Colab batch {batch_id}: {tickers}")
        
        start_time = time.time()
        
        try:
            # Очищуємо пам'ять перед батчем
            if self.config.cleanup_memory:
                self.cleanup_memory()
            
            # Симуляцandя тренування (replacementти на реальний code)
            result = self._simulate_colab_training(batch_info)
            
            # Зберandгаємо промandжнand реwithульandти
            self._save_batch_checkpoint(batch_id, result)
            
            # Очищуємо пам'ять пandсля батчу
            if self.config.cleanup_memory:
                self.cleanup_memory()
            
            end_time = time.time()
            
            return {
                "batch_id": batch_id,
                "status": "completed",
                "tickers": tickers,
                "execution_time": end_time - start_time,
                "memory_peak": self._get_memory_usage(),
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Colab batch {batch_id} failed: {e}")
            return {
                "batch_id": batch_id,
                "status": "failed",
                "tickers": tickers,
                "error": str(e)
            }
    
    def _simulate_colab_training(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Симуляцandя тренування в Colab"""
        # Симуляцandя часу тренування
        time.sleep(2)  # Симуляцandя
        
        return {
            "models_trained": len(batch_info["tickers"]) * 3,
            "accuracy": 0.82 + np.random.random() * 0.1,
            "loss": 0.3 + np.random.random() * 0.1,
            "data_processed": len(batch_info["tickers"]) * 1000
        }
    
    def _save_batch_checkpoint(self, batch_id: int, result: Dict[str, Any]):
        """Зберегти чекпоandнт батчу"""
        checkpoint = {
            "batch_id": batch_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }
        
        filepath = self.checkpoints_dir / f"batch_{batch_id}_checkpoint.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _get_memory_usage(self) -> float:
        """Отримати поточnot викорисandння пам'ятand"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except ImportError:
            return 0.0
    
    def create_colab_notebook(self, plan: Dict[str, Any]) -> str:
        """Create Colab notebook for тренування"""
        notebook_content = f"""
# Colab Training Notebook
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# Install dependencies
!pip install -q pandas numpy scikit-learn tensorflow torch

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Setup paths
import sys
import os
sys.path.append('/content/drive/MyDrive/trading_project')

# Import modules
from colab.colab_optimized_pipeline import ColabOptimizer, ColabConfig

# Configuration
config = ColabConfig(
    max_tickers_per_run={self.config.max_tickers_per_run},
    max_memory_usage_gb={self.config.max_memory_usage_gb},
    auto_save_interval={self.config.auto_save_interval},
    cleanup_memory={self.config.cleanup_memory},
    use_lightweight_models={self.config.use_lightweight_models}
)

# Initialize optimizer
optimizer = ColabOptimizer(config)

# Load training plan
plan = {json.dumps(plan, indent=2)}

# Execute training
results = []
for batch_info in plan["batches"]:
    result = optimizer.execute_colab_batch(batch_info)
    results.append(result)
    
    # Check if we're approaching time limit
    import time
    if time.time() - optimizer.start_time > 10 * 3600:  # 10 hours
        print("Approaching time limit, saving progress...")
        break

# Save results
import json
with open('/content/drive/MyDrive/trading_project/results/colab_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Training completed!")
"""
        
        # Зберandгаємо notebook
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        notebook_path = self.colab_dir / f"colab_training_{timestamp}.ipynb"
        
        with open(notebook_path, 'w') as f:
            f.write(notebook_content)
        
        self.logger.info(f"Colab notebook created: {notebook_path}")
        return str(notebook_path)

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Colab Optimized Pipeline')
    parser.add_argument('--tickers', default='all', help='Ticker category or list')
    parser.add_argument('--max-tickers', type=int, default=20, help='Max tickers per run')
    parser.add_argument('--create-notebook', action='store_true', help='Create Colab notebook')
    parser.add_argument('--check-env', action='store_true', help='Check Colab environment')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Імпортуємо тandкери
    try:
        from config.tickers import get_tickers
        if args.tickers == 'all':
            tickers = get_tickers('all')
        else:
            tickers = get_tickers(args.tickers)
    except ImportError:
        # Якщо not mayмо andмпортувати, використовуємо тестовand данand
        tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Створюємо конфandгурацandю
    config = ColabConfig(max_tickers_per_run=args.max_tickers)
    optimizer = ColabOptimizer(config)
    
    # Перевandряємо environment
    if args.check_env:
        env_info = optimizer.check_colab_environment()
        print(f"Environment info: {env_info}")
    
    # Створюємо план
    plan = optimizer.create_colab_training_plan(tickers)
    
    # Зберandгаємо план
    plan_file = optimizer.save_colab_plan(plan)
    print(f"Colab plan saved to: {plan_file}")
    
    # Виводимо сandтистику
    print(f"\n=== Colab Training Plan ===")
    print(f"Total tickers: {plan['total_tickers']}")
    print(f"Total batches: {plan['total_batches']}")
    print(f"Max tickers per batch: {plan['batch_size']}")
    print(f"Estimated runtime: {plan['runtime_estimate']['estimated_hours']:.1f} hours")
    print(f"Recommended batches: {plan['runtime_estimate']['recommended_batches']}")
    
    # Створюємо notebook
    if args.create_notebook:
        notebook_path = optimizer.create_colab_notebook(plan)
        print(f"Colab notebook created: {notebook_path}")

if __name__ == "__main__":
    main()
