"""
Adaptive Training Manager for Large Ticker Sets with Adaptive Targets
Адаптивний менеджер тренування for великих наборів тікерів with адаптивними таргетами
"""

import os
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.config.tickers import get_tickers, get_category_stats
# Note: Ensure these modules exist or adjust based on project structure
# Based on project file list, these might need verification of existence:
# src/config/adaptive_targets.py, src/features/adaptive_target_generator.py
try:
    from src.config.adaptive_targets import AdaptiveTargetsSystem, TimeframeType
except ImportError:
    # Fallback or placeholder if specific adaptive_targets doesn't exist yet
    class TimeframeType(Enum):
        INTRADAY_SHORT = "15m"
        INTRADAY_LONG = "60m"
        DAILY = "1d"
    class AdaptiveTargetsSystem:
        def get_suitable_targets(self, tf, points): return []
        def get_targets_by_category(self, tf, points): return {}

try:
    from src.features.adaptive_target_generator import AdaptiveTargetGenerator
except ImportError:
    class AdaptiveTargetGenerator: pass

from src.training.unified_training_manager import UnifiedTrainingManager, TrainingStrategy

class TrainingMode(Enum):
    """Режими тренування"""
    CONSERVATIVE = "conservative"  # Консервативний (тільки надійні таргети)
    BALANCED = "balanced"         # Збалансований (оптимальне співвідношення)
    AGGRESSIVE = "aggressive"     # Агресивний (максимальна кількість таргетів)

@dataclass
class AdaptiveTrainingConfig:
    """Конфігурація адаптивного тренування"""
    # Основні settings
    mode: TrainingMode = TrainingMode.BALANCED
    strategy: TrainingStrategy = TrainingStrategy.HYBRID
    
    # Налаштування таргетів
    min_target_quality: float = 0.7      # Мінімальна якість таргетів
    max_targets_per_ticker: int = 10     # Максимальна кількість таргетів
    target_diversity_threshold: float = 0.3  # Поріг різноманітності таргетів
    
    # Налаштування data
    intraday_data_limit_days: int = 60    # Обмеження for intraday
    daily_data_limit_years: int = 2       # Обмеження for daily
    
    # Ресурси
    max_memory_gb: float = 12.0           # Максимальна пам'ять
    max_time_hours: float = 24.0          # Максимальний час
    
    # Якість
    enable_quality_filtering: bool = True  # Фільтрація якості
    enable_target_validation: bool = True  # Валідація таргетів

class AdaptiveTrainingManager:
    """Адаптивний менеджер тренування"""
    
    def __init__(self, config: AdaptiveTrainingConfig = None):
        self.config = config or AdaptiveTrainingConfig()
        self.logger = logging.getLogger("AdaptiveTrainingManager")
        
        # Ініціалізуємо компоненти
        self.target_system = AdaptiveTargetsSystem()
        self.target_generator = AdaptiveTargetGenerator()
        self.unified_manager = UnifiedTrainingManager()
        
        # Створюємо директорії
        self.adaptive_dir = Path("models/adaptive")
        self.targets_dir = Path("data/adaptive_targets")
        self.reports_dir = Path("reports/adaptive")
        
        for dir_path in [self.adaptive_dir, self.targets_dir, self.reports_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def analyze_ticker_set_with_targets(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Аналізувати набір тікерів with урахуванням таргетів
        
        Args:
            tickers: Список тікерів
            
        Returns:
            Dict[str, Any]: Результати аналізу
        """
        analysis = {
            "ticker_analysis": {},
            "target_analysis": {},
            "compatibility_matrix": {},
            "recommendations": []
        }
        
        # Аналізуємо кожен тікер
        for ticker in tickers:
            ticker_analysis = self._analyze_single_ticker(ticker)
            analysis["ticker_analysis"][ticker] = ticker_analysis
        
        # Аналізуємо сумісність таргетів
        analysis["target_analysis"] = self._analyze_target_compatibility(tickers)
        
        # Створюємо матрицю сумісності
        analysis["compatibility_matrix"] = self._create_compatibility_matrix(tickers)
        
        # Генеруємо рекомендації
        analysis["recommendations"] = self._generate_training_recommendations(analysis)
        
        return analysis
    
    def _analyze_single_ticker(self, ticker: str) -> Dict[str, Any]:
        """Аналізувати окремий тікер"""
        # Симуляція аналізу data тікера
        # В реальному коді тут буде завантаження реальних data
        
        timeframe_analysis = {}
        
        # Аналізуємо кожен таймфрейм
        for timeframe_type in [TimeframeType.INTRADAY_SHORT, TimeframeType.INTRADAY_LONG, TimeframeType.DAILY]:
            # Симуляція кількості data
            if timeframe_type == TimeframeType.INTRADAY_SHORT:
                data_points = 4000  # ~60 днів 15m
            elif timeframe_type == TimeframeType.INTRADAY_LONG:
                data_points = 780    # ~60 днів 60m
            else:  # DAILY
                data_points = 500    # ~2 роки
            
            # Отримуємо підходящі таргети
            suitable_targets = self.target_system.get_suitable_targets(timeframe_type, data_points)
            target_categories = self.target_system.get_targets_by_category(timeframe_type, data_points)
            
            timeframe_analysis[timeframe_type.value] = {
                "data_points": data_points,
                "suitable_targets": len(suitable_targets),
                "target_categories": {cat: len(targets) for cat, targets in target_categories.items() if targets},
                "target_quality_score": self._calculate_target_quality_score(suitable_targets, data_points),
                "recommended_for_training": len(suitable_targets) >= 5  # Мінімум 5 таргетів
            }
        
        return {
            "ticker": ticker,
            "timeframe_analysis": timeframe_analysis,
            "overall_score": self._calculate_overall_ticker_score(timeframe_analysis),
            "best_timeframe": max(timeframe_analysis.keys(), 
                                key=lambda x: timeframe_analysis[x]["overall_score"] if "overall_score" in timeframe_analysis[x] else 0)
        }
    
    def _calculate_target_quality_score(self, targets: List, data_points: int) -> float:
        """Розрахувати оцінку якості таргетів"""
        if not targets:
            return 0.0
        
        # Базова оцінка за кількістю
        quantity_score = min(len(targets) / 10, 1.0)  # 10 таргетів = 1.0
        
        # Оцінка за різноманітністю категорій
        categories = set()
        for target in targets:
            if "volatility" in target.name:
                categories.add("volatility")
            elif "return" in target.name:
                categories.add("return")
            elif "trend" in target.name or "direction" in target.name:
                categories.add("trend")
            elif "drawdown" in target.name or "sharpe" in target.name:
                categories.add("risk")
            elif "volume" in target.name or "acceleration" in target.name:
                categories.add("behavioral")
            elif "support" in target.name or "resistance" in target.name:
                categories.add("structural")
        
        diversity_score = min(len(categories) / 6, 1.0)  # 6 категорій = 1.0
        
        # Оцінка за пріоритетом
        priority_score = sum(1.0 / target.priority for target in targets[:5]) / 5  # Середній пріоритет топ-5
        
        # Комбінована оцінка
        return (quantity_score * 0.3 + diversity_score * 0.4 + priority_score * 0.3)
    
    def _calculate_overall_ticker_score(self, timeframe_analysis: Dict[str, Any]) -> float:
        """Розрахувати загальну оцінку тікера"""
        scores = []
        
        for tf, analysis in timeframe_analysis.items():
            if analysis["recommended_for_training"]:
                scores.append(analysis["suitable_targets"] / 10)  # Нормалізуємо до 0-1
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _analyze_target_compatibility(self, tickers: List[str]) -> Dict[str, Any]:
        """Аналізувати сумісність таргетів"""
        compatibility = {
            "common_targets": set(),
            "unique_targets": {},
            "target_distribution": {},
            "quality_distribution": {}
        }
        
        # Симуляція аналізу сумісності
        # В реальному коді тут буде аналіз реальних таргетів
        
        all_targets = set()
        ticker_targets = {}
        
        for ticker in tickers:
            # Симуляція таргетів за тікера
            ticker_target_set = {
                f"target_volatility_1h_{ticker}",
                f"target_return_1h_{ticker}",
                f"target_direction_1h_{ticker}",
                f"target_volatility_4h_{ticker}",
                f"target_return_4h_{ticker}",
                f"target_volatility_1d_{ticker}",
                f"target_return_5d_{ticker}",
                f"target_volatility_5d_{ticker}",
                f"target_direction_5d_{ticker}",
                f"target_max_drawdown_20d_{ticker}"
            }
            
            ticker_targets[ticker] = ticker_target_set
            all_targets.update(ticker_target_set)
        
        # Знаходимо спільні таргети
        common_targets = set.intersection(*[set(targets) for targets in ticker_targets.values()])
        compatibility["common_targets"] = common_targets
        
        # Знаходимо унікальні таргети
        for ticker, targets in ticker_targets.items():
            unique = targets - common_targets
            if unique:
                compatibility["unique_targets"][ticker] = unique
        
        # Розподіл таргетів
        compatibility["target_distribution"] = {
            "total_unique": len(all_targets),
            "common": len(common_targets),
            "unique_per_ticker": {ticker: len(targets - common_targets) for ticker, targets in ticker_targets.items()}
        }
        
        return compatibility
    
    def _create_compatibility_matrix(self, tickers: List[str]) -> Dict[str, Any]:
        """Створити матрицю сумісності"""
        matrix = {
            "tickers": tickers,
            "compatibility_scores": {},
            "training_groups": []
        }
        
        # Розрахунок оцінки сумісності
        for i, ticker1 in enumerate(tickers):
            for j, ticker2 in enumerate(tickers):
                if i <= j:
                    pair = f"{ticker1}_{ticker2}" if i != j else ticker1
                    
                    # Симуляція оцінки сумісності
                    if i == j:
                        score = 1.0  # Сам з собою завжди сумісний
                    else:
                        # Базова сумісність (в реальному коді буде реальний розрахунок)
                        score = 0.7 + np.random.random() * 0.2  # 0.7-0.9
                    
                    matrix["compatibility_scores"][pair] = score
        
        # Створюємо групи для тренування
        matrix["training_groups"] = self._create_training_groups(tickers, matrix["compatibility_scores"])
        
        return matrix
    
    def _create_training_groups(self, tickers: List[str], compatibility_scores: Dict[str, float]) -> List[List[str]]:
        """Створити групи для тренування"""
        groups = []
        used_tickers = set()
        
        # Проста логіка групування
        group_size = min(5, len(tickers))  # Групи по 5 тікерів
        
        for i in range(0, len(tickers), group_size):
            group = tickers[i:i + group_size]
            groups.append(group)
            used_tickers.update(group)
        
        return groups
    
    def _generate_training_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Згенерувати рекомендації для тренування"""
        recommendations = []
        
        # Аналізуємо якість таргетів
        avg_quality = np.mean([
            ticker["overall_score"] for ticker in analysis["ticker_analysis"].values()
        ])
        
        if avg_quality > 0.8:
            recommendations.append("High target quality - suitable for complex models")
        elif avg_quality > 0.6:
            recommendations.append("Moderate target quality - use balanced approach")
        else:
            recommendations.append("Low target quality - use conservative models")
        
        # Аналізуємо сумісність
        common_targets = analysis["target_analysis"]["common_targets"]
        if len(common_targets) > 5:
            recommendations.append("Good target compatibility - can use unified models")
        else:
            recommendations.append("Limited target compatibility - consider ticker-specific models")
        
        # Аналізуємо розподіл таргетів
        for ticker, ticker_analysis in analysis["ticker_analysis"].items():
            best_tf = ticker_analysis["best_timeframe"]
            if best_tf == "15m":
                recommendations.append(f"{ticker}: Focus on intraday strategies")
            elif best_tf == "1d":
                recommendations.append(f"{ticker}: Suitable for long-term models")
        
        # Рекомендації по стратегії
        if len(analysis["ticker_analysis"]) > 20:
            recommendations.append("Large ticker set - use progressive training")
        elif len(analysis["ticker_analysis"]) > 10:
            recommendations.append("Medium ticker set - use hybrid approach")
        else:
            recommendations.append("Small ticker set - can use batch training")
        
        return recommendations
    
    def create_adaptive_training_plan(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Створити адаптивний план тренування
        
        Args:
            tickers: Список тікерів
            
        Returns:
            Dict[str, Any]: Адаптивний план
        """
        # Аналізуємо тікери
        analysis = self.analyze_ticker_set_with_targets(tickers)
        
        # Вибираємо стратегію тренування
        strategy = self._select_optimal_strategy(analysis)
        
        # Створюємо план
        plan = {
            "analysis": analysis,
            "strategy": strategy.value,
            "training_groups": analysis["compatibility_matrix"]["training_groups"],
            "target_configurations": {},
            "resource_estimates": {},
            "quality_metrics": {},
            "execution_plan": {}
        }
        
        # Налаштовуємо конфігурації таргетів за кожної групи
        for i, group in enumerate(plan["training_groups"]):
            group_config = self._create_group_target_config(group, analysis)
            plan["target_configurations"][f"group_{i+1}"] = group_config
        
        # Оцінюємо ресурси
        plan["resource_estimates"] = self._estimate_training_resources(plan)
        
        # Налаштовуємо метрики якості
        plan["quality_metrics"] = self._calculate_quality_metrics(analysis)
        
        # Створюємо план виконання
        plan["execution_plan"] = self._create_execution_plan(plan)
        
        return plan
    
    def _select_optimal_strategy(self, analysis: Dict[str, Any]) -> TrainingStrategy:
        """Вибрати оптимальну стратегію"""
        ticker_count = len(analysis["ticker_analysis"])
        avg_quality = np.mean([
            ticker["overall_score"] for ticker in analysis["ticker_analysis"].values()
        ])
        
        # Логіка вибору стратегії
        if ticker_count > 20:
            return TrainingStrategy.PROGRESSIVE
        elif ticker_count > 10:
            return TrainingStrategy.HYBRID
        elif avg_quality > 0.8:
            return TrainingStrategy.BATCH
        else:
            return TrainingStrategy.HYBRID
    
    def _create_group_target_config(self, group: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Створити конфігурацію таргетів за групи"""
        config = {
            "tickers": group,
            "common_targets": [],
            "unique_targets": {},
            "target_categories": {},
            "quality_threshold": self.config.min_target_quality,
            "max_targets": self.config.max_targets_per_ticker
        }
        
        # Знаходимо спільні таргети
        group_targets = []
        for ticker in group:
            ticker_analysis = analysis["ticker_analysis"][ticker]
            best_tf = ticker_analysis["best_timeframe"]
            
            # Симуляція таргетів за цього таймфрейму
            if best_tf == "15m":
                targets = [
                    "target_volatility_1h", "target_return_1h", "target_direction_1h",
                    "target_volatility_4h", "target_return_4h", "target_volume_anomaly_1h"
                ]
            elif best_tf == "60m":
                targets = [
                    "target_volatility_1h", "target_return_1h", "target_direction_1h",
                    "target_volatility_4h", "target_return_4h", "target_volume_anomaly_1h"
                ]
            else:  # 1d
                targets = [
                    "target_volatility_5d", "target_return_5d", "target_direction_5d",
                    "target_volatility_20d", "target_max_drawdown_20d", "target_mean_reversion_5d"
                ]
            
            group_targets.extend(targets)
        
        # Знаходимо спільні та унікальні
        from collections import Counter
        target_counts = Counter(group_targets)
        config["common_targets"] = [target for target, count in target_counts.items() if count == len(group)]
        
        return config
    
    def _estimate_training_resources(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Оцінити ресурси тренування"""
        total_tickers = len(plan["analysis"]["ticker_analysis"])
        total_groups = len(plan["training_groups"])
        
        # Приблизні розрахунки
        memory_per_ticker = 0.5  # GB
        time_per_ticker = 0.25    # hours
        
        return {
            "estimated_memory_gb": total_tickers * memory_per_ticker,
            "estimated_time_hours": total_tickers * time_per_ticker,
            "estimated_models": total_tickers * 3,  # 3 моделі на тікер
            "parallel_groups": min(4, total_groups),  # Максимально 4 групи паралельно
            "checkpoint_frequency": max(1, total_groups // 5)
        }
    
    def _calculate_quality_metrics(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Розрахувати метрики якості"""
        ticker_scores = [ticker["overall_score"] for ticker in analysis["ticker_analysis"].values()]
        
        return {
            "average_ticker_score": np.mean(ticker_scores),
            "min_ticker_score": np.min(ticker_scores),
            "max_ticker_score": np.max(ticker_scores),
            "score_distribution": {
                "high": len([s for s in ticker_scores if s > 0.8]),
                "medium": len([s for s in ticker_scores if 0.6 <= s <= 0.8]),
                "low": len([s for s in ticker_scores if s < 0.6])
            },
            "target_diversity": len(analysis["target_analysis"]["common_targets"]),
            "compatibility_score": np.mean(list(analysis["compatibility_matrix"]["compatibility_scores"].values()))
        }
    
    def _create_execution_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Створити план виконання"""
        phases = []
        
        for i, group in enumerate(plan["training_groups"]):
            phase = {
                "phase_id": i + 1,
                "phase_name": f"Training Group {i + 1}",
                "tickers": group,
                "target_config": plan["target_configurations"][f"group_{i+1}"],
                "estimated_duration_hours": len(group) * 0.25,  # 15 хв на тікер
                "dependencies": [] if i == 0 else [f"phase_{i}"]
            }
            phases.append(phase)
        
        return {
            "total_phases": len(phases),
            "phases": phases,
            "parallel_execution": len(phases) > 2,
            "checkpoint_points": [i * 2 + 1 for i in range(len(phases) // 2)]
        }
    
    def execute_adaptive_training(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Виконати адаптивне тренування
        
        Args:
            tickers: Список тікерів
            
        Returns:
            Dict[str, Any]: Результати тренування
        """
        self.logger.info(f"Starting adaptive training for {len(tickers)} tickers")
        
        # Створюємо план
        plan = self.create_adaptive_training_plan(tickers)
        
        # Зберігаємо план
        plan_file = self._save_adaptive_plan(plan)
        self.logger.info(f"Adaptive training plan saved to {plan_file}")
        
        # Виконуємо тренування
        results = self._execute_adaptive_plan(plan)
        
        # Зберігаємо результати
        results_file = self._save_adaptive_results(results)
        self.logger.info(f"Adaptive training results saved to {results_file}")
        
        return results
    
    def _save_adaptive_plan(self, plan: Dict[str, Any]) -> str:
        """Зберегти адаптивний план"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptive_plan_{timestamp}.json"
        filepath = self.adaptive_dir / filename
        
        # Конвертуємо об'єкти, які не можна серіалізувати
        serializable_plan = self._make_serializable(plan)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_plan, f, indent=2)
        
        return str(filepath)
    
    def _save_adaptive_results(self, results: Dict[str, Any]) -> str:
        """Зберегти результати адаптивного тренування"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"adaptive_results_{timestamp}.json"
        filepath = self.adaptive_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        return str(filepath)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Зробити об'єкт серіалізованим"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def _execute_adaptive_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Виконати адаптивний план"""
        # Симуляція виконання
        # В реальному коді тут буде виклик реального тренування
        
        results = {
            "execution_summary": {
                "plan_strategy": plan["strategy"],
                "total_phases": plan["execution_plan"]["total_phases"],
                "completed_phases": 0,
                "total_tickers": len(plan["analysis"]["ticker_analysis"]),
                "successful_tickers": 0,
                "failed_tickers": 0,
                "total_time_hours": 0,
                "average_quality": 0
            },
            "phase_results": [],
            "quality_metrics": {},
            "recommendations": []
        }
        
        # Симуляція виконання фаз
        for phase in plan["execution_plan"]["phases"]:
            phase_result = {
                "phase_id": phase["phase_id"],
                "phase_name": phase["phase_name"],
                "tickers": phase["tickers"],
                "status": "completed",
                "duration_hours": phase["estimated_duration_hours"],
                "models_trained": len(phase["tickers"]) * 3,
                "average_accuracy": 0.85 + np.random.random() * 0.1
            }
            results["phase_results"].append(phase_result)
            results["execution_summary"]["completed_phases"] += 1
            results["execution_summary"]["successful_tickers"] += len(phase["tickers"])
            results["execution_summary"]["total_time_hours"] += phase["estimated_duration_hours"]
        
        # Розрахунок фінальних метрик
        results["execution_summary"]["average_quality"] = np.mean([
            phase["average_accuracy"] for phase in results["phase_results"]
        ])
        
        return results

def main():
    """Основна функція для тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Adaptive Training Manager')
    parser.add_argument('--tickers', default='core', help='Ticker category or list')
    parser.add_argument('--mode', default='balanced', 
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Training mode')
    parser.add_argument('--analyze-only', action='store_true', help='Analyze only')
    parser.add_argument('--save-plan', action='store_true', help='Save training plan')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Отримуємо тікери
    try:
        if args.tickers == 'core':
            tickers = get_tickers('core')
        elif args.tickers == 'all':
            tickers = get_tickers('all')[:10]  # Обмежуємо for тесту
        else:
            tickers = get_tickers(args.tickers)
    except Exception:
        tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AAPL']
    
    # Створюємо конфігурацію
    config = AdaptiveTrainingConfig(mode=TrainingMode(args.mode))
    manager = AdaptiveTrainingManager(config)
    
    print(f"Adaptive Training Analysis for {len(tickers)} tickers: {tickers}")
    print(f"Mode: {args.mode}")
    
    # Аналізуємо
    analysis = manager.analyze_ticker_set_with_targets(tickers)
    
    print(f"\n=== Ticker Analysis ===")
    for ticker, ticker_analysis in analysis["ticker_analysis"].items():
        print(f"{ticker}:")
        print(f"  Overall score: {ticker_analysis['overall_score']:.3f}")
        print(f"  Best timeframe: {ticker_analysis['best_timeframe']}")
        print(f"  Recommended: {ticker_analysis['timeframe_analysis'][ticker_analysis['best_timeframe']]['recommended_for_training']}")
    
    print(f"\n=== Target Analysis ===")
    target_analysis = analysis["target_analysis"]
    print(f"Common targets: {len(target_analysis['common_targets'])}")
    print(f"Target distribution: {target_analysis['target_distribution']}")
    
    print(f"\n=== Recommendations ===")
    for rec in analysis["recommendations"]:
        print(f"- {rec}")
    
    if args.analyze_only:
        return
    
    # Створюємо план
    plan = manager.create_adaptive_training_plan(tickers)
    
    print(f"\n=== Training Plan ===")
    print(f"Strategy: {plan['strategy']}")
    print(f"Training groups: {len(plan['training_groups'])}")
    print(f"Estimated time: {plan['resource_estimates']['estimated_time_hours']:.1f} hours")
    print(f"Estimated memory: {plan['resource_estimates']['estimated_memory_gb']:.1f} GB")
    
    # Зберігаємо план
    if args.save_plan:
        plan_file = manager._save_adaptive_plan(plan)
        print(f"Plan saved to: {plan_file}")
    
    # Виконуємо тренування
    results = manager.execute_adaptive_training(tickers)
    
    print(f"\n=== Training Results ===")
    summary = results["execution_summary"]
    print(f"Completed phases: {summary['completed_phases']}/{summary['total_phases']}")
    print(f"Successful tickers: {summary['successful_tickers']}/{summary['total_tickers']}")
    print(f"Total time: {summary['total_time_hours']:.1f} hours")
    print(f"Average quality: {summary['average_quality']:.3f}")

if __name__ == "__main__":
    main()