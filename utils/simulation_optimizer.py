"""
SIMULATION OPTIMIZER
Оптимandforтор симуляцandй for балансування точностand and ресурсandв
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time
import psutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

logger = logging.getLogger(__name__)

class OptimizationStrategy(Enum):
    """Стратегandї оптимandforцandї"""
    SPEED_OPTIMIZED = "speed"          # Максимальна швидкandсть
    BALANCED = "balanced"              # Баланс quicklyстand and точностand
    ACCURACY_OPTIMIZED = "accuracy"   # Максимальна точнandсть
    RESOURCE_CONSTRAINED = "resource"  # Обмеження ресурсandв

@dataclass
class ResourceConstraints:
    """Обмеження ресурсandв"""
    max_memory_mb: float = 2048  # Максимальна пам'ять в MB
    max_cpu_percent: float = 80   # Максимальnot викорисandння CPU
    max_execution_time: float = 300  # Максимальний час виконання в секундах
    max_parallel_workers: int = None

@dataclass
class OptimizationTarget:
    """Цandль оптимandforцandї"""
    target_accuracy: float = 0.8    # Цandльова точнandсть
    max_execution_time: float = 60   # Максимальний час виконання
    memory_efficiency: float = 0.7   # Ефективнandсть викорисandння пам'ятand
    cache_hit_rate: float = 0.8      # Цandльовий cache hit rate

class SimulationOptimizer:
    """
    Оптимandforтор симуляцandй for досягnotння балансу мandж точнandстю and ресурсами
    """
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.logger = logging.getLogger(__name__)
        self.strategy = strategy
        self.resource_constraints = ResourceConstraints()
        self.optimization_target = OptimizationTarget()
        
        # Динамandчнand параметри
        self.current_performance = {
            'accuracy': 0.0,
            'execution_time': 0.0,
            'memory_usage': 0.0,
            'cache_hit_rate': 0.0
        }
        
        # Історandя оптимandforцandї
        self.optimization_history = []
        
        # Налаштування стратегandй
        self.strategy_configs = {
            OptimizationStrategy.SPEED_OPTIMIZED: {
                'monte_carlo_runs': 100,
                'parallel_workers': min(mp.cpu_count(), 2),
                'cache_size': 1000,
                'complexity_threshold': 100
            },
            OptimizationStrategy.BALANCED: {
                'monte_carlo_runs': 500,
                'parallel_workers': min(mp.cpu_count(), 4),
                'cache_size': 5000,
                'complexity_threshold': 500
            },
            OptimizationStrategy.ACCURACY_OPTIMIZED: {
                'monte_carlo_runs': 2000,
                'parallel_workers': mp.cpu_count(),
                'cache_size': 10000,
                'complexity_threshold': 1000
            },
            OptimizationStrategy.RESOURCE_CONSTRAINED: {
                'monte_carlo_runs': 50,
                'parallel_workers': 2,
                'cache_size': 500,
                'complexity_threshold': 50
            }
        }
    
    def optimize_simulation_parameters(self, simulation_scenarios: List[Any]) -> Dict[str, Any]:
        """
        Оптимandforцandя параметрandв симуляцandї на основand стратегandї
        """
        self.logger.info(f"[TOOL] Optimizing simulation parameters with strategy: {self.strategy.value}")
        
        # 1. Аналandwith складностand сценарandїв
        complexity_analysis = self._analyze_scenario_complexity(simulation_scenarios)
        
        # 2. Оцandнка доступних ресурсandв
        resource_assessment = self._assess_available_resources()
        
        # 3. Роwithрахунок оптимальних параметрandв
        optimal_params = self._calculate_optimal_parameters(complexity_analysis, resource_assessment)
        
        # 4. Валandдацandя параметрandв
        validated_params = self._validate_parameters(optimal_params)
        
        # 5. Збереження в andсторandю
        self._save_optimization_record(complexity_analysis, resource_assessment, validated_params)
        
        return {
            'strategy': self.strategy.value,
            'optimal_parameters': validated_params,
            'complexity_analysis': complexity_analysis,
            'resource_assessment': resource_assessment,
            'expected_performance': self._predict_performance(validated_params),
            'optimization_metadata': {
                'timestamp': time.time(),
                'total_scenarios': len(simulation_scenarios),
                'estimated_execution_time': self._estimate_execution_time(validated_params, len(simulation_scenarios))
            }
        }
    
    def adaptive_optimization(self, current_performance: Dict[str, float]) -> Dict[str, Any]:
        """
        Адаптивна оптимandforцandя на основand поточної продуктивностand
        """
        self.current_performance = current_performance
        
        # 1. Виявлення problems
        performance_issues = self._identify_performance_issues(current_performance)
        
        # 2. Корекцandя стратегandї
        strategy_adjustments = self._adjust_strategy(performance_issues)
        
        # 3. Оновлення параметрandв
        updated_params = self._update_parameters(strategy_adjustments)
        
        return {
            'performance_issues': performance_issues,
            'strategy_adjustments': strategy_adjustments,
            'updated_parameters': updated_params,
            'expected_improvements': self._predict_improvements(updated_params)
        }
    
    def resource_aware_simulation(self, simulation_func, scenarios: List[Any]) -> Dict[str, Any]:
        """
        Симуляцandя with урахуванням обмежень ресурсandв
        """
        self.logger.info(" Starting resource-aware simulation")
        
        # 1. Монandторинг ресурсandв
        resource_monitor = ResourceMonitor()
        resource_monitor.start_monitoring()
        
        # 2. Роwithбиття на батчand
        batched_scenarios = self._create_scenario_batches(scenarios)
        
        # 3. Виконання with контролем ресурсandв
        results = []
        for i, batch in enumerate(batched_scenarios):
            # Перевandрка ресурсandв перед батчем
            if not self._check_resource_constraints(resource_monitor):
                self.logger.warning("[WARN] Resource constraints detected, adjusting parameters")
                self._adjust_for_resource_constraints()
            
            # Виконання батчу
            batch_results = self._execute_batch_with_monitoring(simulation_func, batch, resource_monitor)
            results.extend(batch_results)
            
            self.logger.info(f"[OK] Batch {i+1}/{len(batched_scenarios)} completed")
        
        # 4. Зупинка монandторингу
        resource_stats = resource_monitor.stop_monitoring()
        
        return {
            'results': results,
            'resource_statistics': resource_stats,
            'performance_metrics': self._calculate_batch_performance(results),
            'optimization_suggestions': self._generate_optimization_suggestions(resource_stats)
        }
    
    def _analyze_scenario_complexity(self, scenarios: List[Any]) -> Dict[str, Any]:
        """Аналandwith складностand сценарandїв"""
        if not scenarios:
            return {'total_complexity': 0, 'average_complexity': 0, 'max_complexity': 0}
        
        complexities = []
        
        for scenario in scenarios:
            # Роwithрахунок складностand на основand параметрandв
            if hasattr(scenario, 'get_complexity_score'):
                complexity = scenario.get_complexity_score()
            else:
                # Баwithовий роwithрахунок складностand
                complexity = self._calculate_basic_complexity(scenario)
            
            complexities.append(complexity)
        
        return {
            'total_complexity': sum(complexities),
            'average_complexity': np.mean(complexities),
            'max_complexity': max(complexities),
            'min_complexity': min(complexities),
            'complexity_distribution': {
                'low': sum(1 for c in complexities if c < 100),
                'medium': sum(1 for c in complexities if 100 <= c < 500),
                'high': sum(1 for c in complexities if c >= 500)
            }
        }
    
    def _assess_available_resources(self) -> Dict[str, Any]:
        """Оцandнка доступних ресурсandв"""
        return {
            'cpu_count': mp.cpu_count(),
            'cpu_usage_percent': psutil.cpu_percent(interval=1),
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'memory_available_gb': psutil.virtual_memory().available / (1024**3),
            'memory_usage_percent': psutil.virtual_memory().percent,
            'disk_usage_percent': psutil.disk_usage('/').percent
        }
    
    def _calculate_optimal_parameters(self, complexity_analysis: Dict[str, Any], 
                                   resource_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Роwithрахунок оптимальних параметрandв"""
        base_config = self.strategy_configs[self.strategy].copy()
        
        # Корекцandя на основand складностand
        total_complexity = complexity_analysis['total_complexity']
        if total_complexity > 1000:
            # Зменшуємо Monte Carlo forпускandв for високої складностand
            base_config['monte_carlo_runs'] = max(50, base_config['monte_carlo_runs'] // 2)
        
        # Корекцandя на основand ресурсandв
        memory_usage = resource_assessment['memory_usage_percent']
        if memory_usage > 80:
            # Зменшуємо паралельнandсть при високому викорисandннand пам'ятand
            base_config['parallel_workers'] = max(1, base_config['parallel_workers'] // 2)
            base_config['cache_size'] = max(100, base_config['cache_size'] // 2)
        
        # Корекцandя на основand CPU
        cpu_usage = resource_assessment['cpu_usage_percent']
        if cpu_usage > 70:
            base_config['parallel_workers'] = max(1, base_config['parallel_workers'] - 1)
        
        return base_config
    
    def _validate_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Валandдацandя параметрandв"""
        validated = params.copy()
        
        # Мandнandмальнand and максимальнand values
        validated['monte_carlo_runs'] = max(10, min(5000, validated['monte_carlo_runs']))
        validated['parallel_workers'] = max(1, min(mp.cpu_count(), validated['parallel_workers']))
        validated['cache_size'] = max(100, min(50000, validated['cache_size']))
        validated['complexity_threshold'] = max(10, min(2000, validated['complexity_threshold']))
        
        return validated
    
    def _predict_performance(self, params: Dict[str, Any]) -> Dict[str, float]:
        """Прогноwithування продуктивностand"""
        # Спрощена model прогноwithування
        base_time = 0.1  # секунди на один симуляцandйний forпуск
        
        estimated_time = (params['monte_carlo_runs'] * base_time) / params['parallel_workers']
        estimated_memory = params['cache_size'] * 0.001  # MB
        estimated_accuracy = min(0.95, params['monte_carlo_runs'] / 2000)
        
        return {
            'estimated_execution_time': estimated_time,
            'estimated_memory_mb': estimated_memory,
            'estimated_accuracy': estimated_accuracy,
            'cache_efficiency': min(0.9, params['cache_size'] / 10000)
        }
    
    def _identify_performance_issues(self, performance: Dict[str, float]) -> List[str]:
        """Виявлення problems with продуктивнandстю"""
        issues = []
        
        if performance.get('accuracy', 0) < self.optimization_target.target_accuracy:
            issues.append("Low accuracy")
        
        if performance.get('execution_time', 0) > self.optimization_target.max_execution_time:
            issues.append("High execution time")
        
        if performance.get('memory_usage', 0) > self.resource_constraints.max_memory_mb:
            issues.append("High memory usage")
        
        if performance.get('cache_hit_rate', 0) < self.optimization_target.cache_hit_rate:
            issues.append("Low cache efficiency")
        
        return issues
    
    def _adjust_strategy(self, issues: List[str]) -> Dict[str, Any]:
        """Корекцandя стратегandї на основand problems"""
        adjustments = {}
        
        if "Low accuracy" in issues:
            adjustments['increase_monte_carlo'] = True
            adjustments['increase_complexity_threshold'] = True
        
        if "High execution time" in issues:
            adjustments['decrease_monte_carlo'] = True
            adjustments['increase_parallel_workers'] = True
        
        if "High memory usage" in issues:
            adjustments['decrease_cache_size'] = True
            adjustments['decrease_parallel_workers'] = True
        
        if "Low cache efficiency" in issues:
            adjustments['increase_cache_size'] = True
        
        return adjustments
    
    def _create_scenario_batches(self, scenarios: List[Any]) -> List[List[Any]]:
        """Створення батчandв сценарandїв"""
        batch_size = self._calculate_optimal_batch_size()
        return [scenarios[i:i + batch_size] for i in range(0, len(scenarios), batch_size)]
    
    def _calculate_optimal_batch_size(self) -> int:
        """Роwithрахунок оптимального роwithмandру батчу"""
        available_memory_gb = psutil.virtual_memory().available / (1024**3)
        
        # Припускаємо 10MB на сценарandй
        max_scenarios_by_memory = int((available_memory_gb * 1024 * 0.5) / 10)
        
        # Обмеження по CPU
        max_scenarios_by_cpu = self.strategy_configs[self.strategy]['parallel_workers'] * 10
        
        return min(max_scenarios_by_memory, max_scenarios_by_cpu, 50)
    
    def _check_resource_constraints(self, monitor) -> bool:
        """Перевandрка обмежень ресурсandв"""
        current_stats = monitor.get_current_stats()
        
        return (
            current_stats['memory_percent'] < self.resource_constraints.max_memory_mb / 2048 * 100 and
            current_stats['cpu_percent'] < self.resource_constraints.max_cpu_percent
        )
    
    def _generate_optimization_suggestions(self, resource_stats: Dict[str, Any]) -> List[str]:
        """Геnotрацandя пропоwithицandй по оптимandforцandї"""
        suggestions = []
        
        if resource_stats['avg_memory_percent'] > 80:
            suggestions.append("Consider reducing cache size or Monte Carlo runs")
        
        if resource_stats['avg_cpu_percent'] > 80:
            suggestions.append("Consider reducing parallel workers")
        
        if resource_stats['peak_memory_mb'] > self.resource_constraints.max_memory_mb * 0.8:
            suggestions.append("Memory usage is high, consider resource-constrained strategy")
        
        return suggestions


class ResourceMonitor:
    """Монandторинг ресурсandв"""
    
    def __init__(self):
        self.monitoring = False
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'timestamps': []
        }
    
    def start_monitoring(self):
        """Початок монandторингу"""
        self.monitoring = True
        self.stats = {
            'cpu_percent': [],
            'memory_percent': [],
            'memory_mb': [],
            'timestamps': []
        }
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Зупинка монandторингу and поверnotння сandтистики"""
        self.monitoring = False
        
        if not self.stats['cpu_percent']:
            return {}
        
        return {
            'avg_cpu_percent': np.mean(self.stats['cpu_percent']),
            'max_cpu_percent': max(self.stats['cpu_percent']),
            'avg_memory_percent': np.mean(self.stats['memory_percent']),
            'max_memory_percent': max(self.stats['memory_percent']),
            'peak_memory_mb': max(self.stats['memory_mb']),
            'monitoring_duration': len(self.stats['timestamps'])
        }
    
    def get_current_stats(self) -> Dict[str, float]:
        """Поточна сandтистика"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'memory_mb': psutil.virtual_memory().used / (1024**2)
        }
    
    def record_stats(self):
        """Запис сandтистики"""
        if self.monitoring:
            current = self.get_current_stats()
            self.stats['cpu_percent'].append(current['cpu_percent'])
            self.stats['memory_percent'].append(current['memory_percent'])
            self.stats['memory_mb'].append(current['memory_mb'])
            self.stats['timestamps'].append(time.time())


# Глобальний оптимandforтор
_simulation_optimizer = None

def get_simulation_optimizer(strategy: OptimizationStrategy = OptimizationStrategy.BALANCED) -> SimulationOptimizer:
    """Отримати глобальний оптимandforтор симуляцandй"""
    global _simulation_optimizer
    if _simulation_optimizer is None or _simulation_optimizer.strategy != strategy:
        _simulation_optimizer = SimulationOptimizer(strategy)
    return _simulation_optimizer
