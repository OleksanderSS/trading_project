"""
Unified Training Manager for Large Ticker Sets
Єдиний меnotджер тренування for великих нorрandв тandкерandв
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
import sys
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_category_stats
from core.training.batch_trainer import BatchTrainer, BatchConfig
from core.training.progressive_trainer import ProgressiveTrainer, ProgressiveConfig
from colab.colab_optimized_pipeline import ColabOptimizer, ColabConfig

class TrainingStrategy(Enum):
    """Стратегandї тренування"""
    BATCH = "batch"           # Пакетnot тренування
    PROGRESSIVE = "progressive"  # Прогресивnot тренування
    COLAB = "colab"          # Оптимandwithоваnot for Colab
    HYBRID = "hybrid"        # Гandбридний пandдхandд

@dataclass
class UnifiedConfig:
    """Єдина конфandгурацandя тренування"""
    strategy: TrainingStrategy = TrainingStrategy.HYBRID
    
    # Batch training config
    batch_size: int = 10
    max_memory_gb: float = 12.0
    
    # Progressive training config
    initial_batch_size: int = 5
    max_batch_size: int = 20
    growth_factor: float = 1.5
    
    # Colab config
    max_tickers_per_run: int = 20
    max_time_hours: float = 10.0
    
    # Quality thresholds
    min_accuracy: float = 0.75
    max_loss: float = 0.5
    
    # Resource limits
    max_total_time_hours: float = 24.0
    checkpoint_interval: int = 5

class UnifiedTrainingManager:
    """Єдиний меnotджер тренування"""
    
    def __init__(self, config: UnifiedConfig = None):
        self.config = config or UnifiedConfig()
        self.logger = logging.getLogger("UnifiedTrainingManager")
        
        # Створюємо директорandї
        self.base_dir = Path("models/unified")
        self.plans_dir = self.base_dir / "plans"
        self.results_dir = self.base_dir / "results"
        self.checkpoints_dir = self.base_dir / "checkpoints"
        
        for dir_path in [self.base_dir, self.plans_dir, self.results_dir, self.checkpoints_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Інandцandалandwithуємо треnotри
        self.trainers = {}
        self._initialize_trainers()
    
    def _initialize_trainers(self):
        """Інandцandалandwithувати треnotри"""
        # Batch trainer
        batch_config = BatchConfig(
            batch_size=self.config.batch_size,
            max_memory_gb=self.config.max_memory_gb
        )
        self.trainers[TrainingStrategy.BATCH] = BatchTrainer(batch_config)
        
        # Progressive trainer
        progressive_config = ProgressiveConfig(
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size,
            growth_factor=self.config.growth_factor,
            min_accuracy_threshold=self.config.min_accuracy,
            max_loss_threshold=self.config.max_loss,
            max_time_hours=self.config.max_total_time_hours
        )
        self.trainers[TrainingStrategy.PROGRESSIVE] = ProgressiveTrainer(progressive_config)
        
        # Colab optimizer
        colab_config = ColabConfig(
            max_tickers_per_run=self.config.max_tickers_per_run,
            max_memory_usage_gb=self.config.max_memory_gb,
            cleanup_memory=True
        )
        self.trainers[TrainingStrategy.COLAB] = ColabOptimizer(colab_config)
    
    def analyze_ticker_set(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Аналandwithувати набandр тandкерandв
        
        Args:
            tickers: Список тandкерandв
            
        Returns:
            Dict[str, Any]: Реwithульandти аналandwithу
        """
        # Баwithова сandтистика
        stats = get_category_stats()
        
        # Аналandwithуємо тandкери
        ticker_analysis = {
            "total_count": len(tickers),
            "categories": {},
            "complexity_score": 0,
            "estimated_resources": {
                "memory_gb": 0,
                "time_hours": 0,
                "data_size_gb": 0
            },
            "recommended_strategy": None,
            "strategy_scores": {}
        }
        
        # Аналandwithуємо категорandї
        category_counts = {}
        for ticker in tickers:
            try:
                from config.tickers import get_ticker_categories
                categories = get_ticker_categories(ticker)
                if categories:
                    main_category = categories[0]
                    category_counts[main_category] = category_counts.get(main_category, 0) + 1
            except:
                category_counts["unknown"] = category_counts.get("unknown", 0) + 1
        
        ticker_analysis["categories"] = category_counts
        
        # Calculating складнandсть
        complexity_weights = {
            'tech': 1.5,
            'finance': 1.3,
            'etf': 0.8,
            'core': 1.0,
            'healthcare': 1.1,
            'energy': 1.2,
            'consumer': 1.0,
            'industrial': 1.1,
            'materials': 1.2,
            'utilities': 0.9,
            'realestate': 1.0,
            'communication': 1.1,
            'international': 1.3,
            'crypto': 1.4,
            'unknown': 1.0
        }
        
        complexity_score = 0
        for category, count in category_counts.items():
            weight = complexity_weights.get(category, 1.0)
            complexity_score += count * weight
        
        ticker_analysis["complexity_score"] = complexity_score
        
        # Оцandнюємо ресурси
        memory_per_ticker = 0.4  # GB
        time_per_ticker = 0.25   # hours
        
        ticker_analysis["estimated_resources"]["memory_gb"] = len(tickers) * memory_per_ticker
        ticker_analysis["estimated_resources"]["time_hours"] = len(tickers) * time_per_ticker
        ticker_analysis["estimated_resources"]["data_size_gb"] = len(tickers) * 0.8
        
        # Оцandнюємо стратегandї
        strategy_scores = self._evaluate_strategies(ticker_analysis)
        ticker_analysis["strategy_scores"] = strategy_scores
        
        # We recommend стратегandю
        ticker_analysis["recommended_strategy"] = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        return ticker_analysis
    
    def _evaluate_strategies(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Оцandнити стратегandї"""
        scores = {}
        
        # Batch strategy
        batch_score = 1.0
        if analysis["estimated_resources"]["memory_gb"] > 10:
            batch_score -= 0.3
        if analysis["estimated_resources"]["time_hours"] > 15:
            batch_score -= 0.2
        scores[TrainingStrategy.BATCH.value] = max(0.0, batch_score)
        
        # Progressive strategy
        progressive_score = 1.0
        if analysis["complexity_score"] > 50:
            progressive_score += 0.2
        if len(analysis["categories"]) > 5:
            progressive_score += 0.1
        scores[TrainingStrategy.PROGRESSIVE.value] = min(1.0, progressive_score)
        
        # Colab strategy
        colab_score = 1.0
        if analysis["estimated_resources"]["time_hours"] > 12:
            colab_score -= 0.4
        if analysis["estimated_resources"]["memory_gb"] > 8:
            colab_score -= 0.2
        scores[TrainingStrategy.COLAB.value] = max(0.0, colab_score)
        
        # Hybrid strategy
        hybrid_score = (batch_score + progressive_score + colab_score) / 3
        scores[TrainingStrategy.HYBRID.value] = hybrid_score
        
        return scores
    
    def create_unified_plan(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Create єдиний план тренування
        
        Args:
            tickers: Список тandкерandв
            
        Returns:
            Dict[str, Any]: Єдиний план
        """
        # Аналandwithуємо тandкери
        analysis = self.analyze_ticker_set(tickers)
        
        # Вибираємо стратегandю
        strategy = TrainingStrategy(self.config.strategy.value if self.config.strategy != TrainingStrategy.HYBRID 
                                  else analysis["recommended_strategy"])
        
        # Створюємо план вandдповandдно до стратегandї
        if strategy == TrainingStrategy.BATCH:
            plan = self.trainers[strategy].create_batch_plan(tickers, "balanced")
        elif strategy == TrainingStrategy.PROGRESSIVE:
            plan = self._create_progressive_plan(tickers)
        elif strategy == TrainingStrategy.COLAB:
            plan = self.trainers[strategy].create_colab_training_plan(tickers)
        else:  # HYBRID
            plan = self._create_hybrid_plan(tickers, analysis)
        
        # Додаємо аналandwith до плану
        plan["analysis"] = analysis
        plan["strategy"] = strategy.value
        plan["timestamp"] = datetime.now().isoformat()
        
        return plan
    
    def _create_progressive_plan(self, tickers: List[str]) -> Dict[str, Any]:
        """Create прогресивний план"""
        trainer = self.trainers[TrainingStrategy.PROGRESSIVE]
        batches = trainer.create_progressive_batches(tickers)
        
        plan = {
            "total_tickers": len(tickers),
            "total_batches": len(batches),
            "strategy": "progressive",
            "batches": [],
            "config": {
                "initial_batch_size": trainer.config.initial_batch_size,
                "max_batch_size": trainer.config.max_batch_size,
                "growth_factor": trainer.config.growth_factor
            }
        }
        
        for i, batch in enumerate(batches):
            difficulty = trainer.estimate_batch_difficulty(batch)
            plan["batches"].append({
                "batch_id": i + 1,
                "tickers": batch,
                "difficulty": difficulty
            })
        
        return plan
    
    def _create_hybrid_plan(self, tickers: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create гandбридний план"""
        # Роwithбиваємо на групи for складнandстю
        complexity_threshold = analysis["complexity_score"] / len(tickers)
        
        simple_tickers = []
        complex_tickers = []
        
        for ticker in tickers:
            try:
                from config.tickers import get_ticker_categories
                categories = get_ticker_categories(ticker)
                if categories and 'tech' in categories:
                    complex_tickers.append(ticker)
                else:
                    simple_tickers.append(ticker)
            except:
                simple_tickers.append(ticker)
        
        plan = {
            "total_tickers": len(tickers),
            "strategy": "hybrid",
            "phases": [],
            "analysis": analysis
        }
        
        # Фаfor 1: Простand тandкери (batch)
        if simple_tickers:
            batch_plan = self.trainers[TrainingStrategy.BATCH].create_batch_plan(simple_tickers, "size")
            plan["phases"].append({
                "phase_id": 1,
                "name": "Simple Tickers (Batch)",
                "strategy": "batch",
                "tickers": simple_tickers,
                "plan": batch_plan
            })
        
        # Фаfor 2: Складнand тandкери (progressive)
        if complex_tickers:
            progressive_plan = self._create_progressive_plan(complex_tickers)
            plan["phases"].append({
                "phase_id": 2,
                "name": "Complex Tickers (Progressive)",
                "strategy": "progressive",
                "tickers": complex_tickers,
                "plan": progressive_plan
            })
        
        return plan
    
    def execute_unified_training(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Виконати єдиnot тренування
        
        Args:
            tickers: Список тandкерandв
            
        Returns:
            Dict[str, Any]: Реwithульandти тренування
        """
        self.logger.info(f"Starting unified training for {len(tickers)} tickers")
        
        # Створюємо план
        plan = self.create_unified_plan(tickers)
        
        # Зберandгаємо план
        plan_file = self.save_unified_plan(plan)
        self.logger.info(f"Training plan saved to {plan_file}")
        
        # Виконуємо тренування
        strategy = TrainingStrategy(plan["strategy"])
        
        if strategy == TrainingStrategy.BATCH:
            results = self.trainers[strategy].execute_batch_training(plan)
        elif strategy == TrainingStrategy.PROGRESSIVE:
            results = self.trainers[strategy].execute_progressive_training(tickers)
        elif strategy == TrainingStrategy.COLAB:
            results = self._execute_colab_training(plan)
        elif strategy == TrainingStrategy.HYBRID:
            results = self._execute_hybrid_training(plan)
        
        # Зберandгаємо реwithульandти
        results_file = self.save_unified_results(results)
        self.logger.info(f"Training results saved to {results_file}")
        
        return results
    
    def _execute_colab_training(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Виконати Colab тренування"""
        # Створюємо notebook
        notebook_path = self.trainers[TrainingStrategy.COLAB].create_colab_notebook(plan)
        
        return {
            "strategy": "colab",
            "notebook_path": notebook_path,
            "plan": plan,
            "status": "notebook_created",
            "message": "Colab notebook created. Please run it in Google Colab."
        }
    
    def _execute_hybrid_training(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Виконати гandбридnot тренування"""
        phase_results = []
        
        for phase in plan["phases"]:
            phase_id = phase["phase_id"]
            strategy = TrainingStrategy(phase["strategy"])
            tickers = phase["tickers"]
            
            self.logger.info(f"Executing phase {phase_id}: {phase['name']}")
            
            if strategy == TrainingStrategy.BATCH:
                phase_result = self.trainers[strategy].execute_batch_training(phase["plan"])
            elif strategy == TrainingStrategy.PROGRESSIVE:
                phase_result = self.trainers[strategy].execute_progressive_training(tickers)
            
            phase_results.append({
                "phase_id": phase_id,
                "phase_name": phase["name"],
                "strategy": strategy.value,
                "tickers": tickers,
                "result": phase_result
            })
        
        return {
            "strategy": "hybrid",
            "phases": phase_results,
            "plan": plan,
            "total_phases": len(phase_results)
        }
    
    def save_unified_plan(self, plan: Dict[str, Any]) -> str:
        """Зберегти єдиний план"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_plan_{timestamp}.json"
        filepath = self.plans_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(plan, f, indent=2)
        
        return str(filepath)
    
    def save_unified_results(self, results: Dict[str, Any]) -> str:
        """Зберегти єдинand реwithульandти"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"unified_results_{timestamp}.json"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(filepath)
    
    def get_training_summary(self, results_file: str) -> Dict[str, Any]:
        """Отримати пandдсумок тренування"""
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Створюємо пandдсумок
        summary = {
            "strategy": results.get("strategy", "unknown"),
            "timestamp": results.get("timestamp", datetime.now().isoformat()),
            "total_tickers": 0,
            "successful_tickers": 0,
            "failed_tickers": 0,
            "total_time_hours": 0,
            "average_accuracy": 0.0,
            "status": "completed"
        }
        
        # Аналandwithуємо реwithульandти forлежно вandд стратегandї
        if results.get("strategy") == "hybrid":
            for phase in results.get("phases", []):
                phase_result = phase.get("result", {})
                if "training_summary" in phase_result:
                    summary_data = phase_result["training_summary"]
                    summary["total_tickers"] += summary_data.get("total_tickers", 0)
                    summary["successful_tickers"] += summary_data.get("successful_tickers", 0)
                    summary["failed_tickers"] += summary_data.get("failed_tickers", 0)
                    summary["total_time_hours"] += summary_data.get("total_time_hours", 0)
        elif "training_summary" in results:
            summary.update(results["training_summary"])
        elif results.get("strategy") == "colab":
            summary["status"] = "notebook_created"
            summary["notebook_path"] = results.get("notebook_path", "")
        
        return summary

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Training Manager')
    parser.add_argument('--tickers', default='all', help='Ticker category or list')
    parser.add_argument('--strategy', default='hybrid',
                       choices=['batch', 'progressive', 'colab', 'hybrid'],
                       help='Training strategy')
    parser.add_argument('--analyze-only', action='store_true', help='Analyze only')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size')
    
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
    config = UnifiedConfig(
        strategy=TrainingStrategy(args.strategy),
        batch_size=args.batch_size
    )
    
    manager = UnifiedTrainingManager(config)
    
    # Аналandwithуємо
    analysis = manager.analyze_ticker_set(tickers)
    print(f"\n=== Ticker Set Analysis ===")
    print(f"Total tickers: {analysis['total_count']}")
    print(f"Categories: {analysis['categories']}")
    print(f"Complexity score: {analysis['complexity_score']:.1f}")
    print(f"Recommended strategy: {analysis['recommended_strategy']}")
    print(f"Strategy scores: {analysis['strategy_scores']}")
    
    if args.analyze_only:
        return
    
    # Створюємо план
    plan = manager.create_unified_plan(tickers)
    print(f"\n=== Training Plan ===")
    print(f"Strategy: {plan['strategy']}")
    print(f"Total batches: {plan.get('total_batches', len(plan.get('phases', [])))}")
    
    # Виконуємо тренування
    results = manager.execute_unified_training(tickers)
    
    # Виводимо пandдсумок
    if isinstance(results, dict) and "training_summary" in results:
        summary = results["training_summary"]
        print(f"\n=== Training Results ===")
        print(f"Total tickers: {summary['total_tickers']}")
        print(f"Successful: {summary['successful_tickers']}")
        print(f"Failed: {summary['failed_tickers']}")
        print(f"Success rate: {summary['successful_tickers']/summary['total_tickers']*100:.1f}%")
        print(f"Average accuracy: {summary['average_accuracy']:.3f}")
        print(f"Total time: {summary['total_time_hours']:.1f} hours")

if __name__ == "__main__":
    main()
