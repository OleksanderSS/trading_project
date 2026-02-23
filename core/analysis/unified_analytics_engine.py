# core/analysis/unified_analytics_engine.py - Єдиний аналandтичний движок

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from abc import ABC, abstractmethod
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class IAnalyzer(ABC):
    """Інтерфейс for allх аналandтичних модулandв"""
    
    @abstractmethod
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """Основний метод аналandwithу"""
        pass
    
    @abstractmethod
    def get_analyzer_type(self) -> str:
        """Тип аналandforтора"""
        pass

class IComparator(ABC):
    """Інтерфейс for allх модулandв порandвняння"""
    
    @abstractmethod
    def compare(self, items: List[Any], **kwargs) -> Dict[str, Any]:
        """Основний метод порandвняння"""
        pass
    
    @abstractmethod
    def get_comparison_type(self) -> str:
        """Тип порandвняння"""
        pass

class IContextProcessor(ABC):
    """Інтерфейс for allх контекстних процесорandв"""
    
    @abstractmethod
    def process_context(self, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Обробка контексту"""
        pass
    
    @abstractmethod
    def get_processor_type(self) -> str:
        """Тип процесора"""
        pass

class UnifiedAnalyticsEngine:
    """
    Єдиний аналandтичний движок for allх модулandв
    """
    
    def __init__(self):
        self.analyzers = {}
        self.comparators = {}
        self.context_processors = {}
        self.cache = {}
        self.performance_metrics = {}
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("[UnifiedAnalyticsEngine] Initialized")
    
    def register_analyzer(self, analyzer: IAnalyzer, name: str = None) -> str:
        """Реєстрацandя аналandforтора"""
        if name is None:
            name = analyzer.get_analyzer_type()
        
        self.analyzers[name] = analyzer
        logger.info(f"[UnifiedAnalyticsEngine] Registered analyzer: {name}")
        return name
    
    def register_comparator(self, comparator: IComparator, name: str = None) -> str:
        """Реєстрацandя компаратора"""
        if name is None:
            name = comparator.get_comparison_type()
        
        self.comparators[name] = comparator
        logger.info(f"[UnifiedAnalyticsEngine] Registered comparator: {name}")
        return name
    
    def register_context_processor(self, processor: IContextProcessor, name: str = None) -> str:
        """Реєстрацandя контекстного процесора"""
        if name is None:
            name = processor.get_processor_type()
        
        self.context_processors[name] = processor
        logger.info(f"[UnifiedAnalyticsEngine] Registered context processor: {name}")
        return name
    
    def analyze(self, analyzer_name: str, data: Any, **kwargs) -> Dict[str, Any]:
        """Виконання аналandwithу"""
        if analyzer_name not in self.analyzers:
            raise ValueError(f"Analyzer {analyzer_name} not found")
        
        # Перевandряємо кеш
        cache_key = f"{analyzer_name}_{hash(str(data))}"
        if cache_key in self.cache:
            logger.debug(f"[UnifiedAnalyticsEngine] Using cached result for {analyzer_name}")
            return self.cache[cache_key]
        
        # Виконуємо аналandwith
        start_time = datetime.now()
        try:
            result = self.analyzers[analyzer_name].analyze(data, **kwargs)
            
            # Кешуємо реwithульandт
            self.cache[cache_key] = result
            
            # Записуємо метрики
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(analyzer_name, duration, True)
            
            logger.info(f"[UnifiedAnalyticsEngine] Analysis {analyzer_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(analyzer_name, duration, False)
            logger.error(f"[UnifiedAnalyticsEngine] Analysis {analyzer_name} failed: {e}")
            raise
    
    def compare(self, comparator_name: str, items: List[Any], **kwargs) -> Dict[str, Any]:
        """Виконання порandвняння"""
        if comparator_name not in self.comparators:
            raise ValueError(f"Comparator {comparator_name} not found")
        
        # Перевandряємо кеш
        cache_key = f"{comparator_name}_{hash(str(items))}"
        if cache_key in self.cache:
            logger.debug(f"[UnifiedAnalyticsEngine] Using cached comparison for {comparator_name}")
            return self.cache[cache_key]
        
        # Виконуємо порandвняння
        start_time = datetime.now()
        try:
            result = self.comparators[comparator_name].compare(items, **kwargs)
            
            # Кешуємо реwithульandт
            self.cache[cache_key] = result
            
            # Записуємо метрики
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(comparator_name, duration, True)
            
            logger.info(f"[UnifiedAnalyticsEngine] Comparison {comparator_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(comparator_name, duration, False)
            logger.error(f"[UnifiedAnalyticsEngine] Comparison {comparator_name} failed: {e}")
            raise
    
    def process_context(self, processor_name: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Обробка контексту"""
        if processor_name not in self.context_processors:
            raise ValueError(f"Context processor {processor_name} not found")
        
        # Перевandряємо кеш
        cache_key = f"{processor_name}_{hash(str(data))}_{hash(str(context))}"
        if cache_key in self.cache:
            logger.debug(f"[UnifiedAnalyticsEngine] Using cached context for {processor_name}")
            return self.cache[cache_key]
        
        # Виконуємо processing
        start_time = datetime.now()
        try:
            result = self.context_processors[processor_name].process_context(data, context)
            
            # Кешуємо реwithульandт
            self.cache[cache_key] = result
            
            # Записуємо метрики
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processor_name, duration, True)
            
            logger.info(f"[UnifiedAnalyticsEngine] Context processing {processor_name} completed in {duration:.2f}s")
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self._update_performance_metrics(processor_name, duration, False)
            logger.error(f"[UnifiedAnalyticsEngine] Context processing {processor_name} failed: {e}")
            raise
    
    def parallel_analyze(self, analyzer_configs: List[Tuple[str, Any, Dict]]) -> Dict[str, Any]:
        """Паралельний аналandwith"""
        results = {}
        
        futures = {}
        for analyzer_name, data, kwargs in analyzer_configs:
            if analyzer_name in self.analyzers:
                future = self.thread_pool.submit(self.analyze, analyzer_name, data, **kwargs)
                futures[analyzer_name] = future
        
        # Збираємо реwithульandти
        for analyzer_name, future in futures.items():
            try:
                results[analyzer_name] = future.result(timeout=300)  # 5 хв andймаут
            except Exception as e:
                logger.error(f"[UnifiedAnalyticsEngine] Parallel analysis {analyzer_name} failed: {e}")
                results[analyzer_name] = {"error": str(e)}
        
        return results
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Отримати метрики продуктивностand"""
        return self.performance_metrics
    
    def clear_cache(self):
        """Очистити кеш"""
        self.cache.clear()
        logger.info("[UnifiedAnalyticsEngine] Cache cleared")
    
    def _update_performance_metrics(self, component: str, duration: float, success: bool):
        """Оновлення метрик продуктивностand"""
        if component not in self.performance_metrics:
            self.performance_metrics[component] = {
                'total_calls': 0,
                'successful_calls': 0,
                'failed_calls': 0,
                'total_duration': 0.0,
                'avg_duration': 0.0,
                'max_duration': 0.0,
                'min_duration': float('inf'),
                'success_rate': 0.0
            }
        
        metrics = self.performance_metrics[component]
        metrics['total_calls'] += 1
        metrics['total_duration'] += duration
        
        if success:
            metrics['successful_calls'] += 1
        else:
            metrics['failed_calls'] += 1
        
        # Оновлюємо сandтистику
        metrics['avg_duration'] = metrics['total_duration'] / metrics['total_calls']
        metrics['max_duration'] = max(metrics['max_duration'], duration)
        metrics['min_duration'] = min(metrics['min_duration'], duration)
        metrics['success_rate'] = metrics['successful_calls'] / metrics['total_calls']
    
    def get_registered_components(self) -> Dict[str, List[str]]:
        """Отримати список forреєстрованих компоnotнтandв"""
        return {
            'analyzers': list(self.analyzers.keys()),
            'comparators': list(self.comparators.keys()),
            'context_processors': list(self.context_processors.keys())
        }
    
    def save_performance_report(self, filepath: str = None) -> str:
        """Зберегти withвandт про продуктивнandсть"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/unified_analytics_performance_{timestamp}.json"
        
        # Створюємо папку якщо not andснує
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Формуємо withвandт
        report = {
            'timestamp': datetime.now().isoformat(),
            'components': self.get_registered_components(),
            'performance_metrics': self.performance_metrics,
            'cache_size': len(self.cache),
            'summary': self._generate_performance_summary()
        }
        
        # Зберandгаємо
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"[UnifiedAnalyticsEngine] Performance report saved to {filepath}")
        return filepath
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """Геnotрувати пandдсумок продуктивностand"""
        if not self.performance_metrics:
            return {}
        
        total_calls = sum(m['total_calls'] for m in self.performance_metrics.values())
        total_successful = sum(m['successful_calls'] for m in self.performance_metrics.values())
        total_duration = sum(m['total_duration'] for m in self.performance_metrics.values())
        
        return {
            'total_components': len(self.performance_metrics),
            'total_calls': total_calls,
            'total_successful': total_successful,
            'overall_success_rate': total_successful / total_calls if total_calls > 0 else 0,
            'total_duration': total_duration,
            'avg_duration_per_call': total_duration / total_calls if total_calls > 0 else 0,
            'slowest_component': max(self.performance_metrics.items(), key=lambda x: x[1]['avg_duration'])[0] if self.performance_metrics else None,
            'fastest_component': min(self.performance_metrics.items(), key=lambda x: x[1]['avg_duration'])[0] if self.performance_metrics else None,
            'most_reliable': max(self.performance_metrics.items(), key=lambda x: x[1]['success_rate'])[0] if self.performance_metrics else None,
            'least_reliable': min(self.performance_metrics.items(), key=lambda x: x[1]['success_rate'])[0] if self.performance_metrics else None
        }

# Глобальний екwithемпляр
unified_engine = UnifiedAnalyticsEngine()

# Функцandї for withручностand
def register_analyzer(analyzer: IAnalyzer, name: str = None) -> str:
    """Зареєструвати аналandforтор"""
    return unified_engine.register_analyzer(analyzer, name)

def register_comparator(comparator: IComparator, name: str = None) -> str:
    """Зареєструвати компаратор"""
    return unified_engine.register_comparator(comparator, name)

def register_context_processor(processor: IContextProcessor, name: str = None) -> str:
    """Зареєструвати контекстний процесор"""
    return unified_engine.register_context_processor(processor, name)

def analyze_with_engine(analyzer_name: str, data: Any, **kwargs) -> Dict[str, Any]:
    """Виконати аналandwith череwith єдиний движок"""
    return unified_engine.analyze(analyzer_name, data, **kwargs)

def compare_with_engine(comparator_name: str, items: List[Any], **kwargs) -> Dict[str, Any]:
    """Виконати порandвняння череwith єдиний движок"""
    return unified_engine.compare(comparator_name, items, **kwargs)

def process_context_with_engine(processor_name: str, data: Any, context: Dict[str, Any]) -> Dict[str, Any]:
    """Обробити контекст череwith єдиний движок"""
    return unified_engine.process_context(processor_name, data, context)

if __name__ == "__main__":
    # Тестування
    print("Unified Analytics Engine - готовий до викорисandння")
    print(f"Зареєстрованand компоnotнти: {unified_engine.get_registered_components()}")
