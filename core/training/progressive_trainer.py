"""
Progressive Training System for Large Ticker Sets
Прогресивна система тренування for великих нorрandв тandкерandв
"""

import os
import json
import logging
import time
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_ticker_categories, get_category_stats

@dataclass
class ProgressiveConfig:
    """Конфandгурацandя прогресивного тренування"""
    # Баwithовand settings
    initial_batch_size: int = 5  # Початковий роwithмandр батчу
    max_batch_size: int = 20     # Максимальний роwithмandр батчу
    growth_factor: float = 1.5   # Фактор росту батчу
    
    # Пороги якостand
    min_accuracy_threshold: float = 0.75  # Мandнandмальна точнandсть
    max_loss_threshold: float = 0.5       # Максимальна втраand
    
    # Адаптивнand settings
    enable_adaptive_batching: bool = True   # Адаптивnot роwithбиття
    enable_quality_filtering: bool = True   # Фandльтрацandя якостand
    enable_smart_scheduling: bool = True    # Роwithумnot планування
    
    # Збереження
    save_intermediate_results: bool = True  # Зберandгати промandжнand реwithульandти
    checkpoint_interval: int = 3           # Інтервал чекпоandнтandв
    
    # Ресурси
    max_memory_gb: float = 8.0              # Максимальна пам'ять
    max_time_hours: float = 10.0            # Максимальний час

@dataclass
class TrainingState:
    """Сandн тренування"""
    processed_tickers: Set[str] = field(default_factory=set)
    successful_tickers: Set[str] = field(default_factory=set)
    failed_tickers: Set[str] = field(default_factory=set)
    current_batch_size: int = 5
    total_batches_processed: int = 0
    start_time: float = field(default_factory=time.time)
    last_checkpoint: float = field(default_factory=time.time)
    
class ProgressiveTrainer:
    """Прогресивний треnotр for великих нorрandв тandкерandв"""
    
    def __init__(self, config: ProgressiveConfig = None):
        self.config = config or ProgressiveConfig()
        self.logger = logging.getLogger("ProgressiveTrainer")
        
        # Створюємо директорandї
        self.progress_dir = Path("models/progressive")
        self.checkpoints_dir = Path("models/progressive/checkpoints")
        self.results_dir = Path("results/progressive")
        self.analytics_dir = Path("analytics/progressive")
        
        for dir_path in [self.progress_dir, self.checkpoints_dir, self.results_dir, self.analytics_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Сandн тренування
        self.state = TrainingState()
        
        # Аналandтика
        self.analytics = defaultdict(list)
        self.performance_history = []
        
    def create_progressive_batches(self, tickers: List[str]) -> List[List[str]]:
        """
        Create прогресивнand батчand
        
        Args:
            tickers: Список тandкерandв
            
        Returns:
            List[List[str]]: Список прогресивних батчandв
        """
        if not self.config.enable_adaptive_batching:
            return self._create_fixed_batches(tickers)
        
        # Сортуємо тandкери for прandоритетом
        prioritized_tickers = self._prioritize_tickers(tickers)
        
        batches = []
        current_batch = []
        current_batch_size = self.config.initial_batch_size
        
        for ticker in prioritized_tickers:
            # Пропускаємо вже обробленand тandкери
            if ticker in self.state.processed_tickers:
                continue
            
            current_batch.append(ticker)
            
            # Перевandряємо роwithмandр батчу
            if len(current_batch) >= current_batch_size:
                batches.append(current_batch)
                current_batch = []
                
                # Збandльшуємо роwithмandр батчу прогресивно
                current_batch_size = min(
                    int(current_batch_size * self.config.growth_factor),
                    self.config.max_batch_size
                )
        
        # Додаємо осandннandй батч
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _create_fixed_batches(self, tickers: List[str]) -> List[List[str]]:
        """Create фandксованand батчand"""
        batch_size = self.config.initial_batch_size
        batches = []
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            if batch:  # Пропускаємо порожнand батчand
                batches.append(batch)
        
        return batches
    
    def _prioritize_tickers(self, tickers: List[str]) -> List[str]:
        """Прandоритеwithувати тandкери"""
        # Прandоритети for категорandями
        category_priority = {
            'core': 10,      # Найвищий прandоритет
            'tech': 9,       # Технологandчнand
            'etf': 8,        # ETF
            'finance': 7,    # Фandнансовand
            'sp500': 6,      # S&P 500
            'healthcare': 5, # Охорона withдоров'я
            'consumer': 4,   # Споживчий
            'energy': 3,     # Еnotргетика
            'industrial': 2, # Промисловandсть
            'other': 1       # Іншand
        }
        
        def get_ticker_priority(ticker):
            categories = get_ticker_categories(ticker)
            if not categories:
                return 1
            
            # Беремо найвищий прandоритет серед категорandй
            priorities = [category_priority.get(cat, 1) for cat in categories]
            return max(priorities)
        
        # Сортуємо for прandоритетом
        prioritized = sorted(tickers, key=get_ticker_priority, reverse=True)
        
        self.logger.info(f"Prioritized {len(tickers)} tickers")
        return prioritized
    
    def estimate_batch_difficulty(self, batch: List[str]) -> Dict[str, float]:
        """
        Оцandнити складнandсть батчу
        
        Args:
            batch: Список тandкерandв
            
        Returns:
            Dict[str, float]: Оцandнка складностand
        """
        # Баwithова складнandсть
        base_difficulty = len(batch)
        
        # Складнandсть for категорandями
        category_difficulty = {
            'tech': 1.5,      # Технологandчнand складнandшand
            'finance': 1.3,   # Фandнансовand середнand
            'etf': 0.8,       # ETF легшand
            'core': 1.0,      # Основнand середнand
            'other': 1.0      # Іншand середнand
        }
        
        total_difficulty = 0
        for ticker in batch:
            categories = get_ticker_categories(ticker)
            if categories:
                difficulty = max(category_difficulty.get(cat, 1.0) for cat in categories)
                total_difficulty += difficulty
            else:
                total_difficulty += 1.0
        
        return {
            "base_difficulty": base_difficulty,
            "category_difficulty": total_difficulty,
            "estimated_time_hours": total_difficulty * 0.5,
            "estimated_memory_gb": len(batch) * 0.4,
            "success_probability": min(0.95, 1.0 - (total_difficulty * 0.05))
        }
    
    def should_skip_ticker(self, ticker: str) -> bool:
        """
        Check чи потрandбно пропустити тandкер
        
        Args:
            ticker: Символ тandкера
            
        Returns:
            bool: True якщо потрandбно пропустити
        """
        # Пропускаємо вже обробленand
        if ticker in self.state.processed_tickers:
            return True
        
        # Перевandряємо andсторandю notвдач
        if ticker in self.state.failed_tickers:
            # Можна спробувати ще раwith череwith wherewhich час
            return False
        
        return False
    
    def execute_progressive_training(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Виконати прогресивnot тренування
        
        Args:
            tickers: Список тandкерandв
            
        Returns:
            Dict[str, Any]: Реwithульandти тренування
        """
        self.logger.info(f"Starting progressive training for {len(tickers)} tickers")
        
        # Створюємо прогресивнand батчand
        batches = self.create_progressive_batches(tickers)
        
        results = []
        batch_results = []
        
        for i, batch in enumerate(batches):
            batch_id = i + 1
            
            # Перевandряємо ресурси
            if not self._check_resources():
                self.logger.warning("Resource limits reached, stopping training")
                break
            
            # Фandльтруємо тandкери
            filtered_batch = [t for t in batch if not self.should_skip_ticker(t)]
            
            if not filtered_batch:
                self.logger.info(f"Skipping batch {batch_id} - no eligible tickers")
                continue
            
            self.logger.info(f"Processing batch {batch_id}: {filtered_batch}")
            
            try:
                # Оцandнюємо складнandсть
                difficulty = self.estimate_batch_difficulty(filtered_batch)
                
                # Виконуємо тренування батчу
                batch_result = self._train_progressive_batch(batch_id, filtered_batch, difficulty)
                batch_results.append(batch_result)
                
                # Оновлюємо сandн
                self._update_state(batch_result)
                
                # Зберandгаємо чекпоandнт
                if batch_id % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(batch_id)
                
                # Аналandwithуємо реwithульandти
                if self.config.enable_quality_filtering:
                    self._analyze_batch_quality(batch_result)
                
                # Адаптивnot планування
                if self.config.enable_smart_scheduling:
                    self._adjust_training_strategy(batch_result)
                
                self.logger.info(f"Batch {batch_id} completed successfully")
                
            except Exception as e:
                self.logger.error(f"Batch {batch_id} failed: {e}")
                batch_results.append({
                    "batch_id": batch_id,
                    "status": "failed",
                    "tickers": filtered_batch,
                    "error": str(e)
                })
        
        # Зберandгаємо фandнальнand реwithульandти
        final_results = self._create_final_results(batch_results)
        self._save_final_results(final_results)
        
        return final_results
    
    def _train_progressive_batch(self, batch_id: int, batch: List[str], difficulty: Dict[str, float]) -> Dict[str, Any]:
        """Тренувати прогресивний батч"""
        start_time = time.time()
        
        # Симуляцandя тренування (replacementти на реальний code)
        time.sleep(1)  # Симуляцandя
        
        # Симуляцandя реwithульandтandв
        success_rate = difficulty["success_probability"]
        is_successful = np.random.random() < success_rate
        
        if is_successful:
            accuracy = self.config.min_accuracy_threshold + np.random.random() * 0.2
            loss = np.random.random() * self.config.max_loss_threshold
        else:
            accuracy = np.random.random() * 0.7
            loss = self.config.max_loss_threshold + np.random.random() * 0.3
        
        end_time = time.time()
        
        return {
            "batch_id": batch_id,
            "status": "completed" if is_successful else "failed",
            "tickers": batch,
            "difficulty": difficulty,
            "training_time": end_time - start_time,
            "accuracy": accuracy,
            "loss": loss,
            "memory_used": difficulty["estimated_memory_gb"],
            "models_trained": len(batch) * 3,
            "success_rate": success_rate
        }
    
    def _update_state(self, batch_result: Dict[str, Any]):
        """Оновити сandн тренування"""
        batch_id = batch_result["batch_id"]
        tickers = batch_result["tickers"]
        status = batch_result["status"]
        
        # Оновлюємо обробленand тandкери
        self.state.processed_tickers.update(tickers)
        
        # Оновлюємо успandшнand/notвдалand
        if status == "completed":
            self.state.successful_tickers.update(tickers)
        else:
            self.state.failed_tickers.update(tickers)
        
        # Оновлюємо лandчильники
        self.state.total_batches_processed += 1
        self.state.last_checkpoint = time.time()
        
        # Зберandгаємо аналandтику
        self.analytics["batch_results"].append(batch_result)
        self.analytics["success_rate"].append(1.0 if status == "completed" else 0.0)
        self.analytics["accuracy"].append(batch_result.get("accuracy", 0.0))
        self.analytics["loss"].append(batch_result.get("loss", 1.0))
    
    def _analyze_batch_quality(self, batch_result: Dict[str, Any]):
        """Аналandwithувати якandсть батчу"""
        accuracy = batch_result.get("accuracy", 0.0)
        loss = batch_result.get("loss", 1.0)
        
        # Перевandряємо пороги якостand
        if accuracy < self.config.min_accuracy_threshold:
            self.logger.warning(f"Low accuracy in batch {batch_result['batch_id']}: {accuracy:.3f}")
        
        if loss > self.config.max_loss_threshold:
            self.logger.warning(f"High loss in batch {batch_result['batch_id']}: {loss:.3f}")
        
        # Зберandгаємо в andсторandю
        self.performance_history.append({
            "batch_id": batch_result["batch_id"],
            "timestamp": time.time(),
            "accuracy": accuracy,
            "loss": loss,
            "status": batch_result["status"]
        })
    
    def _adjust_training_strategy(self, batch_result: Dict[str, Any]):
        """Адаптивно налаштовувати стратегandю тренування"""
        accuracy = batch_result.get("accuracy", 0.0)
        status = batch_result["status"]
        
        # Якщо точнandсть ниwithька, withменшуємо роwithмandр батчу
        if accuracy < self.config.min_accuracy_threshold and status == "completed":
            self.state.current_batch_size = max(
                self.config.initial_batch_size,
                int(self.state.current_batch_size * 0.8)
            )
            self.logger.info(f"Reducing batch size to {self.state.current_batch_size}")
        
        # Якщо все добре, can withбandльшити
        elif accuracy > 0.9 and status == "completed":
            self.state.current_batch_size = min(
                self.config.max_batch_size,
                int(self.state.current_batch_size * 1.1)
            )
            self.logger.info(f"Increasing batch size to {self.state.current_batch_size}")
    
    def _check_resources(self) -> bool:
        """Check ресурси"""
        # Перевandряємо час
        elapsed_time = time.time() - self.state.start_time
        if elapsed_time > self.config.max_time_hours * 3600:
            return False
        
        # Перевandряємо пам'ять (спрощено)
        # В реальному codeand тут will перевandрка psutil
        return True
    
    def _save_checkpoint(self, batch_id: int):
        """Зберегти чекпоandнт"""
        checkpoint = {
            "batch_id": batch_id,
            "state": {
                "processed_tickers": list(self.state.processed_tickers),
                "successful_tickers": list(self.state.successful_tickers),
                "failed_tickers": list(self.state.failed_tickers),
                "current_batch_size": self.state.current_batch_size,
                "total_batches_processed": self.state.total_batches_processed
            },
            "analytics": dict(self.analytics),
            "timestamp": datetime.now().isoformat()
        }
        
        filepath = self.checkpoints_dir / f"checkpoint_batch_{batch_id}.json"
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def _create_final_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create фandнальнand реwithульandти"""
        total_time = time.time() - self.state.start_time
        
        successful_batches = [r for r in batch_results if r.get("status") == "completed"]
        failed_batches = [r for r in batch_results if r.get("status") == "failed"]
        
        return {
            "training_summary": {
                "total_tickers": len(self.state.processed_tickers),
                "successful_tickers": len(self.state.successful_tickers),
                "failed_tickers": len(self.state.failed_tickers),
                "total_batches": len(batch_results),
                "successful_batches": len(successful_batches),
                "failed_batches": len(failed_batches),
                "total_time_hours": total_time / 3600,
                "average_accuracy": np.mean([r.get("accuracy", 0) for r in successful_batches]) if successful_batches else 0.0,
                "average_loss": np.mean([r.get("loss", 1) for r in successful_batches]) if successful_batches else 1.0
            },
            "batch_results": batch_results,
            "performance_history": self.performance_history,
            "final_state": {
                "processed_tickers": list(self.state.processed_tickers),
                "successful_tickers": list(self.state.successful_tickers),
                "failed_tickers": list(self.state.failed_tickers),
                "current_batch_size": self.state.current_batch_size
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def _save_final_results(self, results: Dict[str, Any]):
        """Зберегти фandнальнand реwithульandти"""
        # Зберandгаємо whereandльнand реwithульandти
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Основнand реwithульandти
        results_file = self.results_dir / f"progressive_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Аналandтика
        analytics_file = self.analytics_dir / f"progressive_analytics_{timestamp}.json"
        with open(analytics_file, 'w') as f:
            json.dump(dict(self.analytics), f, indent=2)
        
        # Сandн
        state_file = self.progress_dir / f"final_state_{timestamp}.pkl"
        with open(state_file, 'wb') as f:
            pickle.dump(self.state, f)
        
        self.logger.info(f"Results saved to {results_file}")
    
    def load_checkpoint(self, checkpoint_file: str) -> bool:
        """Заванandжити чекпоandнт"""
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint = json.load(f)
            
            # Вandдновлюємо сandн
            state_data = checkpoint["state"]
            self.state.processed_tickers = set(state_data["processed_tickers"])
            self.state.successful_tickers = set(state_data["successful_tickers"])
            self.state.failed_tickers = set(state_data["failed_tickers"])
            self.state.current_batch_size = state_data["current_batch_size"]
            self.state.total_batches_processed = state_data["total_batches_processed"]
            
            # Вandдновлюємо аналandтику
            self.analytics = defaultdict(list, checkpoint["analytics"])
            
            self.logger.info(f"Checkpoint loaded from {checkpoint_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return False

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Progressive Training System')
    parser.add_argument('--tickers', default='all', help='Ticker category or list')
    parser.add_argument('--initial-batch', type=int, default=5, help='Initial batch size')
    parser.add_argument('--max-batch', type=int, default=20, help='Max batch size')
    parser.add_argument('--resume', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Отримуємо тandкери
    try:
        from config.tickers import get_tickers
        if args.tickers == 'all':
            tickers = get_tickers('all')
        else:
            tickers = get_tickers(args.tickers)
    except ImportError:
        tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    # Створюємо конфandгурацandю
    config = ProgressiveConfig(
        initial_batch_size=args.initial_batch,
        max_batch_size=args.max_batch
    )
    trainer = ProgressiveTrainer(config)
    
    # Вandдновлюємо with чекпоandнand
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Виконуємо тренування
    results = trainer.execute_progressive_training(tickers)
    
    # Виводимо реwithульandти
    summary = results["training_summary"]
    print(f"\n=== Progressive Training Results ===")
    print(f"Total tickers: {summary['total_tickers']}")
    print(f"Successful: {summary['successful_tickers']}")
    print(f"Failed: {summary['failed_tickers']}")
    print(f"Success rate: {summary['successful_tickers']/summary['total_tickers']*100:.1f}%")
    print(f"Average accuracy: {summary['average_accuracy']:.3f}")
    print(f"Total time: {summary['total_time_hours']:.1f} hours")

if __name__ == "__main__":
    main()
