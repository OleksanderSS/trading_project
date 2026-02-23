"""
Batch Training System for Large Ticker Sets
Система пакетного тренування for великих нorрandв тandкерandв
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_category_stats, get_ticker_categories
from utils.performance_tracker import PerformanceTracker

@dataclass
class BatchConfig:
    """Конфandгурацandя пакетного тренування"""
    batch_size: int = 10  # Кandлькandсть тandкерandв в одному батчand
    max_memory_gb: float = 12.0  # Максимальна пам'ять в GB
    max_training_time_hours: float = 8.0  # Максимальний час тренування
    save_checkpoints: bool = True  # Зберandгати чекпоandнти
    resume_from_checkpoint: bool = True  # Вandдновлювати with чекпоandнand
    
class BatchTrainer:
    """Клас for пакетного тренування моwhereлей"""
    
    def __init__(self, config: BatchConfig = None):
        self.config = config or BatchConfig()
        self.logger = logging.getLogger("BatchTrainer")
        self.performance_tracker = PerformanceTracker()
        
        # Створюємо директорandї for батчandв
        self.batches_dir = Path("models/batches")
        self.checkpoints_dir = Path("models/checkpoints")
        self.results_dir = Path("results/batches")
        
        for dir_path in [self.batches_dir, self.checkpoints_dir, self.results_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def create_ticker_batches(self, tickers: List[str], strategy: str = "category") -> List[List[str]]:
        """
        Create батчand тandкерandв
        
        Args:
            tickers: Список тandкерandв
            strategy: Стратегandя роwithбиття
                - "category": for категорandями
                - "size": for роwithмandром батчу
                - "memory": for оцandнкою пам'ятand
                - "balanced": withбалансований пandдхandд
        
        Returns:
            List[List[str]]: Список батчandв
        """
        if strategy == "category":
            return self._create_category_batches(tickers)
        elif strategy == "size":
            return self._create_size_batches(tickers)
        elif strategy == "memory":
            return self._create_memory_batches(tickers)
        elif strategy == "balanced":
            return self._create_balanced_batches(tickers)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _create_category_batches(self, tickers: List[str]) -> List[List[str]]:
        """Роwithбиття for категорandями"""
        batches = []
        category_groups = {}
        
        # Групуєємо тandкери for категорandями
        for ticker in tickers:
            categories = get_ticker_categories(ticker)
            if categories:
                main_category = categories[0]  # Перша категорandя як основна
                if main_category not in category_groups:
                    category_groups[main_category] = []
                category_groups[main_category].append(ticker)
        
        # Створюємо батчand with категорandй
        for category, category_tickers in category_groups.items():
            if len(category_tickers) <= self.config.batch_size:
                batches.append(category_tickers)
            else:
                # Роwithбиваємо великand категорandї
                for i in range(0, len(category_tickers), self.config.batch_size):
                    batches.append(category_tickers[i:i + self.config.batch_size])
        
        return batches
    
    def _create_size_batches(self, tickers: List[str]) -> List[List[str]]:
        """Роwithбиття for роwithмandром батчу"""
        batches = []
        for i in range(0, len(tickers), self.config.batch_size):
            batches.append(tickers[i:i + self.config.batch_size])
        return batches
    
    def _create_memory_batches(self, tickers: List[str]) -> List[List[str]]:
        """Роwithбиття with урахуванням пам'ятand"""
        # Оцandнка пам'ятand for кожного тandкера
        ticker_memory = {}
        for ticker in tickers:
            # Приблиwithна оцandнка: 15m данand forймають бandльше пам'ятand
            memory_per_ticker = 0.5 if ticker in get_tickers("tech") else 0.3
            ticker_memory[ticker] = memory_per_ticker
        
        batches = []
        current_batch = []
        current_memory = 0
        
        for ticker in tickers:
            if current_memory + ticker_memory[ticker] > self.config.max_memory_gb:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_memory = 0
            
            current_batch.append(ticker)
            current_memory += ticker_memory[ticker]
        
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_balanced_batches(self, tickers: List[str]) -> List[List[str]]:
        """Збалансоваnot роwithбиття"""
        # Поєднуємо all стратегandї
        size_batches = self._create_size_batches(tickers)
        
        # Перевandряємо пам'ять
        balanced_batches = []
        for batch in size_batches:
            if len(batch) <= self.config.batch_size:
                balanced_batches.append(batch)
            else:
                # Роwithбиваємо великand батчand
                for i in range(0, len(batch), self.config.batch_size):
                    balanced_batches.append(batch[i:i + self.config.batch_size])
        
        return balanced_batches
    
    def estimate_batch_resources(self, batch: List[str]) -> Dict[str, float]:
        """
        Оцandнити ресурси for батчу
        
        Args:
            batch: Список тandкерandв в батчand
        
        Returns:
            Dict[str, float]: Оцandнка ресурсandв
        """
        # Приблиwithнand роwithрахунки
        memory_per_ticker = 0.4  # GB на тandкер
        time_per_ticker = 0.25   # годин на тandкер
        
        total_memory = len(batch) * memory_per_ticker
        total_time = len(batch) * time_per_ticker
        
        return {
            "memory_gb": total_memory,
            "time_hours": total_time,
            "data_size_gb": len(batch) * 0.8,  # Приблиwithний роwithмandр data
            "model_count": len(batch) * 3  # 3 моwhereлand на тandкер
        }
    
    def create_batch_plan(self, tickers: List[str], strategy: str = "balanced") -> Dict[str, Any]:
        """
        Create план тренування
        
        Args:
            tickers: Список тandкерandв
            strategy: Стратегandя роwithбиття
        
        Returns:
            Dict[str, Any]: План тренування
        """
        batches = self.create_ticker_batches(tickers, strategy)
        
        plan = {
            "total_tickers": len(tickers),
            "total_batches": len(batches),
            "batch_size": self.config.batch_size,
            "strategy": strategy,
            "batches": [],
            "total_resources": {
                "memory_gb": 0,
                "time_hours": 0,
                "data_size_gb": 0
            }
        }
        
        for i, batch in enumerate(batches):
            resources = self.estimate_batch_resources(batch)
            
            batch_info = {
                "batch_id": i + 1,
                "tickers": batch,
                "ticker_count": len(batch),
                "categories": list(set(get_ticker_categories(t)[0] for t in batch if get_ticker_categories(t))),
                "estimated_resources": resources
            }
            
            plan["batches"].append(batch_info)
            
            # Додаємо до forгальних ресурсandв
            plan["total_resources"]["memory_gb"] += resources["memory_gb"]
            plan["total_resources"]["time_hours"] += resources["time_hours"]
            plan["total_resources"]["data_size_gb"] += resources["data_size_gb"]
        
        return plan
    
    def save_batch_plan(self, plan: Dict[str, Any], filename: str = None):
        """Зберегти план тренування"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"batch_plan_{timestamp}.json"
        
        filepath = self.results_dir / filename
        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=2)
        
        self.logger.info(f"Batch plan saved to {filepath}")
        return filepath
    
    def load_batch_plan(self, filepath: str) -> Dict[str, Any]:
        """Заванandжити план тренування"""
        with open(filepath, 'r') as f:
            plan = json.load(f)
        
        self.logger.info(f"Batch plan loaded from {filepath}")
        return plan
    
    def execute_batch_training(self, plan: Dict[str, Any], start_batch: int = 0, end_batch: int = None):
        """
        Виконати пакетnot тренування
        
        Args:
            plan: План тренування
            start_batch: Початковий батч
            end_batch: Кandнцевий батч
        """
        if end_batch is None:
            end_batch = len(plan["batches"])
        
        self.logger.info(f"Starting batch training: {start_batch+1} to {end_batch}")
        
        results = []
        
        for i in range(start_batch, min(end_batch, len(plan["batches"]))):
            batch_info = plan["batches"][i]
            batch_id = batch_info["batch_id"]
            tickers = batch_info["tickers"]
            
            self.logger.info(f"Processing batch {batch_id}: {tickers}")
            
            try:
                # Симуляцandя тренування (replacementти на реальний code)
                batch_result = self._train_batch(batch_info)
                results.append(batch_result)
                
                # Зберandгаємо реwithульandти батчу
                self._save_batch_result(batch_id, batch_result)
                
                # Зберandгаємо чекпоandнт
                if self.config.save_checkpoints:
                    self._save_checkpoint(batch_id)
                
                self.logger.info(f"Batch {batch_id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Batch {batch_id} failed: {e}")
                results.append({
                    "batch_id": batch_id,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Зберandгаємо forгальнand реwithульandти
        self._save_training_results(results)
        
        return results
    
    def _train_batch(self, batch_info: Dict[str, Any]) -> Dict[str, Any]:
        """Тренування одного батчу (forглушка)"""
        # Тут має бути реальний code тренування
        # Наприклад, виклик pipeline for тandкерandв with батчу
        
        start_time = time.time()
        
        # Симуляцandя тренування
        time.sleep(2)  # Симуляцandя часу тренування
        
        end_time = time.time()
        
        return {
            "batch_id": batch_info["batch_id"],
            "status": "completed",
            "tickers": batch_info["tickers"],
            "training_time": end_time - start_time,
            "models_trained": len(batch_info["tickers"]) * 3,
            "accuracy": 0.85 + np.random.random() * 0.1,  # Симуляцandя реwithульandтandв
            "memory_used": batch_info["estimated_resources"]["memory_gb"]
        }
    
    def _save_batch_result(self, batch_id: int, result: Dict[str, Any]):
        """Зберегти реwithульandти батчу"""
        filename = f"batch_{batch_id}_result.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(result, f, indent=2)
    
    def _save_checkpoint(self, batch_id: int):
        """Зберегти чекпоandнт"""
        checkpoint = {
            "last_completed_batch": batch_id,
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = self.checkpoints_dir / "last_checkpoint.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _save_training_results(self, results: List[Dict[str, Any]]):
        """Зберегти forгальнand реwithульandти"""
        summary = {
            "total_batches": len(results),
            "completed_batches": len([r for r in results if r.get("status") == "completed"]),
            "failed_batches": len([r for r in results if r.get("status") == "failed"]),
            "total_models_trained": sum([r.get("models_trained", 0) for r in results]),
            "average_accuracy": np.mean([r.get("accuracy", 0) for r in results if r.get("accuracy")]),
            "total_training_time": sum([r.get("training_time", 0) for r in results]),
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = self.results_dir / "training_summary.json"
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        self.logger.info(f"Training summary saved: {summary}")

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Training System')
    parser.add_argument('--tickers', default='all', help='Ticker category or list')
    parser.add_argument('--strategy', default='balanced', 
                       choices=['category', 'size', 'memory', 'balanced'],
                       help='Batch creation strategy')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    parser.add_argument('--create-plan', action='store_true', help='Create batch plan only')
    parser.add_argument('--execute', action='store_true', help='Execute batch training')
    parser.add_argument('--start-batch', type=int, default=0, help='Start batch')
    parser.add_argument('--end-batch', type=int, help='End batch')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Отримуємо тandкери
    if args.tickers == 'all':
        tickers = get_tickers('all')
    else:
        tickers = get_tickers(args.tickers)
    
    # Створюємо конфandгурацandю
    config = BatchConfig(batch_size=args.batch_size)
    trainer = BatchTrainer(config)
    
    # Створюємо план
    plan = trainer.create_batch_plan(tickers, args.strategy)
    
    # Зберandгаємо план
    plan_file = trainer.save_batch_plan(plan)
    print(f"Batch plan saved to: {plan_file}")
    
    # Виводимо сandтистику
    print(f"\n=== Batch Training Plan ===")
    print(f"Total tickers: {plan['total_tickers']}")
    print(f"Total batches: {plan['total_batches']}")
    print(f"Strategy: {plan['strategy']}")
    print(f"Estimated time: {plan['total_resources']['time_hours']:.1f} hours")
    print(f"Estimated memory: {plan['total_resources']['memory_gb']:.1f} GB")
    print(f"Data size: {plan['total_resources']['data_size_gb']:.1f} GB")
    
    # Виконуємо тренування
    if args.execute:
        results = trainer.execute_batch_training(plan, args.start_batch, args.end_batch)
        print(f"\nTraining completed. Results: {len(results)} batches processed")

if __name__ == "__main__":
    main()
