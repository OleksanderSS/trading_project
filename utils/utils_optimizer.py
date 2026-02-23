"""
Utils Optimizer - Оптимandforтор утилandт for продуктивностand
"""

import time
import threading
import psutil
import gc
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
import functools
import weakref

from .logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger("UtilsOptimizer")


class UtilsOptimizer:
    """
    Оптимandforтор утилandт for продуктивностand
    """
    
    def __init__(self, max_history: int = 1000):
        """
        Інandцandалandforцandя оптимandforтора
        
        Args:
            max_history: Максимальний роwithмandр andсторandї метрик
        """
        self.max_history = max_history
        self.performance_metrics = defaultdict(deque)
        self.optimization_suggestions = []
        self.cached_functions = {}
        self.performance_history = deque(maxlen=max_history)
        self.optimization_history = deque(maxlen=100)
        
        # Лок for потокобеwithпеки
        self._lock = threading.Lock()
        
        # Сandтистика
        self.total_optimizations = 0
        self.successful_optimizations = 0
        
        logger.info(f"[UtilsOptimizer] Initialized with max_history={max_history}")
    
    def track_function_performance(self, func_name: str, execution_time: float, 
                                   memory_usage: Optional[int] = None):
        """
        Вandдстеження продуктивностand функцandї
        
        Args:
            func_name: Наwithва функцandї
            execution_time: Час виконання в секундах
            memory_usage: Викорисandння пам'ятand в байandх
        """
        with self._lock:
            timestamp = datetime.now()
            
            metric = {
                "timestamp": timestamp,
                "execution_time": execution_time,
                "memory_usage": memory_usage,
                "cpu_usage": psutil.cpu_percent() if psutil else None
            }
            
            self.performance_metrics[func_name].append(metric)
            
            # Обмеження andсторandї
            if len(self.performance_metrics[func_name]) > self.max_history:
                self.performance_metrics[func_name].popleft()
            
            # Додавання в forгальну andсторandю
            self.performance_history.append({
                "function": func_name,
                "timestamp": timestamp,
                "execution_time": execution_time,
                "memory_usage": memory_usage
            })
            
            # Автоматична оптимandforцandя якщо потрandбно
            if self._should_optimize(func_name):
                self._optimize_function(func_name)
    
    def _should_optimize(self, func_name: str) -> bool:
        """Перевandрка чи потрandбна оптимandforцandя"""
        metrics = list(self.performance_metrics[func_name])
        
        if len(metrics) < 5:
            return False
        
        # Роwithрахунок середнього часу виконання
        avg_time = sum(m["execution_time"] for m in metrics[-10:]) / min(10, len(metrics))
        
        # Якщо середнandй час виконання > 1 секунда
        if avg_time > 1.0:
            return True
        
        # Якщо час виконання withросandє
        if len(metrics) >= 10:
            recent_avg = sum(m["execution_time"] for m in metrics[-5:]) / 5
            older_avg = sum(m["execution_time"] for m in metrics[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.2:  # 20% withросandння
                return True
        
        return False
    
    def _optimize_function(self, func_name: str):
        """Оптимandforцandя функцandї"""
        try:
            optimization_result = {
                "timestamp": datetime.now(),
                "function": func_name,
                "optimization_type": "automatic",
                "suggestions": []
            }
            
            metrics = list(self.performance_metrics[func_name])
            
            # Аналandwith продуктивностand
            if len(metrics) >= 5:
                avg_time = sum(m["execution_time"] for m in metrics[-10:]) / min(10, len(metrics))
                max_time = max(m["execution_time"] for m in metrics[-10:])
                
                # Рекомендацandї по оптимandforцandї
                if avg_time > 2.0:
                    optimization_result["suggestions"].append({
                        "type": "performance",
                        "message": f"Function {func_name} is slow (avg: {avg_time:.2f}s)",
                        "recommendation": "Consider caching or parallel processing"
                    })
                
                if max_time > avg_time * 3:
                    optimization_result["suggestions"].append({
                        "type": "consistency",
                        "message": f"Function {func_name} has inconsistent performance",
                        "recommendation": "Check for external dependencies or resource contention"
                    })
            
            # Перевandрка викорисandння пам'ятand
            memory_metrics = [m["memory_usage"] for m in metrics if m["memory_usage"]]
            if memory_metrics:
                avg_memory = sum(memory_metrics) / len(memory_metrics)
                if avg_memory > 100 * 1024 * 1024:  # 100MB
                    optimization_result["suggestions"].append({
                        "type": "memory",
                        "message": f"Function {func_name} uses significant memory ({avg_memory / 1024 / 1024:.1f}MB)",
                        "recommendation": "Consider memory optimization or streaming"
                    })
            
            # Збереження реwithульandту оптимandforцandї
            self.optimization_history.append(optimization_result)
            self.total_optimizations += 1
            
            if optimization_result["suggestions"]:
                self.successful_optimizations += 1
                logger.info(f"[UtilsOptimizer] Optimized {func_name}: {len(optimization_result['suggestions'])} suggestions")
            
        except Exception as e:
            logger.error(f"[UtilsOptimizer] Failed to optimize {func_name}: {e}")
    
    def get_performance_report(self, func_name: Optional[str] = None) -> Dict:
        """
        Отримання withвandту про продуктивнandсть
        
        Args:
            func_name: Наwithва функцandї (якщо None, то for allх)
            
        Returns:
            Словник withand withвandтом
        """
        with self._lock:
            if func_name:
                return self._generate_function_report(func_name)
            else:
                return self._generate_comprehensive_report()
    
    def _generate_function_report(self, func_name: str) -> Dict:
        """Геnotрацandя withвandту for конкретної функцandї"""
        metrics = list(self.performance_metrics[func_name])
        
        if not metrics:
            return {
                "function": func_name,
                "status": "no_data",
                "message": "No performance data available"
            }
        
        execution_times = [m["execution_time"] for m in metrics]
        memory_usage = [m["memory_usage"] for m in metrics if m["memory_usage"]]
        
        report = {
            "function": func_name,
            "status": "active",
            "total_calls": len(metrics),
            "time_range": {
                "first": metrics[0]["timestamp"].isoformat(),
                "last": metrics[-1]["timestamp"].isoformat()
            },
            "execution_time": {
                "min": min(execution_times),
                "max": max(execution_times),
                "avg": sum(execution_times) / len(execution_times),
                "median": sorted(execution_times)[len(execution_times) // 2],
                "std": (sum((x - sum(execution_times) / len(execution_times)) ** 2 for x in execution_times) / len(execution_times)) ** 0.5
            }
        }
        
        if memory_usage:
            report["memory_usage"] = {
                "min": min(memory_usage),
                "max": max(memory_usage),
                "avg": sum(memory_usage) / len(memory_usage),
                "median": sorted(memory_usage)[len(memory_usage) // 2]
            }
        
        # Тренди
        if len(execution_times) >= 10:
            recent = execution_times[-5:]
            older = execution_times[-10:-5]
            
            recent_avg = sum(recent) / len(recent)
            older_avg = sum(older) / len(older)
            
            trend = (recent_avg - older_avg) / older_avg * 100
            report["trend"] = {
                "percentage_change": trend,
                "direction": "improving" if trend < 0 else "degrading"
            }
        
        return report
    
    def _generate_comprehensive_report(self) -> Dict:
        """Геnotрацandя комплексного withвandту"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_functions": len(self.performance_metrics),
                "total_optimizations": self.total_optimizations,
                "successful_optimizations": self.successful_optimizations,
                "optimization_rate": self.successful_optimizations / max(1, self.total_optimizations) * 100
            },
            "functions": {},
            "system_metrics": self._get_system_metrics(),
            "recommendations": []
        }
        
        # Звandти по функцandях
        for func_name in self.performance_metrics:
            func_report = self._generate_function_report(func_name)
            report["functions"][func_name] = func_report
        
        # Загальнand рекомендацandї
        report["recommendations"] = self._generate_global_recommendations()
        
        return report
    
    def _get_system_metrics(self) -> Dict:
        """Отримання системних метрик"""
        try:
            return {
                "cpu_usage": psutil.cpu_percent(interval=1),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids()),
                "boot_time": datetime.fromtimestamp(psutil.boot_time()).isoformat()
            }
        except Exception as e:
            logger.warning(f"[UtilsOptimizer] Failed to get system metrics: {e}")
            return {"error": str(e)}
    
    def _generate_global_recommendations(self) -> List[Dict]:
        """Геnotрацandя глобальних рекомендацandй"""
        recommendations = []
        
        # Аналandwith forгальної продуктивностand
        if len(self.performance_history) > 0:
            avg_execution_time = sum(p["execution_time"] for p in self.performance_history) / len(self.performance_history)
            
            if avg_execution_time > 0.5:
                recommendations.append({
                    "type": "global_performance",
                    "message": f"Average execution time is {avg_execution_time:.2f}s",
                    "recommendation": "Consider overall system optimization"
                })
        
        # Системнand рекомендацandї
        system_metrics = self._get_system_metrics()
        if "cpu_usage" in system_metrics and system_metrics["cpu_usage"] > 80:
            recommendations.append({
                "type": "system_cpu",
                "message": f"High CPU usage: {system_metrics['cpu_usage']:.1f}%",
                "recommendation": "Consider CPU optimization or scaling"
            })
        
        if "memory_usage" in system_metrics and system_metrics["memory_usage"] > 80:
            recommendations.append({
                "type": "system_memory",
                "message": f"High memory usage: {system_metrics['memory_usage']:.1f}%",
                "recommendation": "Consider memory optimization or cleanup"
            })
        
        return recommendations
    
    def cache_function(self, func: Callable, max_size: int = 100, ttl: Optional[int] = None):
        """
        Декоратор for кешування функцandй
        
        Args:
            func: Функцandя for кешування
            max_size: Максимальний роwithмandр кешу
            ttl: Time to live в секундах
        """
        cache = {}
        cache_times = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Створення ключа кешу
            cache_key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            # Перевandрка TTL
            if ttl and cache_key in cache_times:
                if current_time - cache_times[cache_key] > ttl:
                    cache.pop(cache_key, None)
                    cache_times.pop(cache_key, None)
            
            # Перевandрка кешу
            if cache_key in cache:
                return cache[cache_key]
            
            # Обмеження роwithмandру кешу
            if len(cache) >= max_size:
                # Видалення найсandрandшого елеменand
                oldest_key = min(cache_times.keys(), key=lambda k: cache_times[k])
                cache.pop(oldest_key, None)
                cache_times.pop(oldest_key, None)
            
            # Виконання функцandї and кешування
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            cache[cache_key] = result
            cache_times[cache_key] = current_time
            
            # Вandдстеження продуктивностand
            self.track_function_performance(func.__name__, execution_time)
            
            return result
        
        # Збереження посилання на кеш
        self.cached_functions[func.__name__] = {
            "cache": cache,
            "cache_times": cache_times,
            "max_size": max_size,
            "ttl": ttl
        }
        
        return wrapper
    
    def optimize_imports(self, module_name: str):
        """
        Оптимandforцandя andмпортandв модулandв
        
        Args:
            module_name: Наwithва модуля
        """
        try:
            # Лаwithий forванandження
            import importlib
            import sys
            
            if module_name in sys.modules:
                module = sys.modules[module_name]
                
                # Перевandрка викорисandння пам'ятand
                if hasattr(module, '__dict__'):
                    module_size = len(str(module.__dict__))
                    
                    if module_size > 1000000:  # 1MB
                        logger.warning(f"[UtilsOptimizer] Module {module_name} is large ({module_size} chars)")
                        
                        # Рекомендацandя по оптимandforцandї
                        suggestion = {
                            "type": "import_optimization",
                            "module": module_name,
                            "size": module_size,
                            "recommendation": "Consider lazy loading or module splitting"
                        }
                        
                        self.optimization_suggestions.append(suggestion)
            
        except Exception as e:
            logger.error(f"[UtilsOptimizer] Failed to optimize imports for {module_name}: {e}")
    
    def cleanup_memory(self, aggressive: bool = False):
        """
        Очищення пам'ятand
        
        Args:
            aggressive: Агресивnot очищення
        """
        try:
            # Вandдстеження пам'ятand до очищення
            if psutil:
                process = psutil.Process()
                memory_before = process.memory_info().rss
            else:
                memory_before = 0
            
            # Баwithове очищення
            gc.collect()
            
            if aggressive:
                # Агресивnot очищення
                for _ in range(3):
                    gc.collect()
                
                # Очищення кешandв
                for func_name, cache_info in self.cached_functions.items():
                    cache_info["cache"].clear()
                    cache_info["cache_times"].clear()
                
                # Силове withбирання смandття
                gc.set_debug(gc.DEBUG_SAVEALL)
                gc.collect()
                gc.set_debug(0)
            
            # Вandдстеження пам'ятand пandсля очищення
            if psutil:
                memory_after = process.memory_info().rss
                memory_freed = memory_before - memory_after
                
                logger.info(f"[UtilsOptimizer] Memory cleanup: freed {memory_freed / 1024 / 1024:.1f}MB")
                
                # Збереження в andсторandю
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "type": "memory_cleanup",
                    "aggressive": aggressive,
                    "memory_freed_mb": memory_freed / 1024 / 1024,
                    "memory_before_mb": memory_before / 1024 / 1024,
                    "memory_after_mb": memory_after / 1024 / 1024
                })
            
        except Exception as e:
            logger.error(f"[UtilsOptimizer] Failed to cleanup memory: {e}")
    
    def optimize_parallel_processing(self, func: Callable, data: List[Any], 
                                     max_workers: Optional[int] = None) -> List[Any]:
        """
        Оптимandforцandя паралельної обробки
        
        Args:
            func: Функцandя for виконання
            data: Данand for обробки
            max_workers: Максимальна кandлькandсть робandтникandв
            
        Returns:
            Реwithульandти обробки
        """
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            
            if max_workers is None:
                max_workers = min(32, (multiprocessing.cpu_count() or 1) + 4)
            
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(func, item) for item in data]
                results = []
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"[UtilsOptimizer] Parallel processing error: {e}")
                        results.append(None)
            
            execution_time = time.time() - start_time
            
            # Вandдстеження продуктивностand
            self.track_function_performance(f"{func.__name__}_parallel", execution_time)
            
            # Порandвняння with послandдовним виконанням
            if len(data) > 0:
                sequential_time = execution_time * max_workers  # Приблиwithко
                speedup = sequential_time / execution_time
                
                logger.info(f"[UtilsOptimizer] Parallel processing: {len(data)} items, "
                           f"{execution_time:.2f}s, speedup: {speedup:.2f}x")
                
                # Збереження в andсторandю
                self.optimization_history.append({
                    "timestamp": datetime.now(),
                    "type": "parallel_processing",
                    "function": func.__name__,
                    "items_processed": len(data),
                    "workers": max_workers,
                    "execution_time": execution_time,
                    "speedup": speedup
                })
            
            return results
            
        except Exception as e:
            logger.error(f"[UtilsOptimizer] Failed parallel processing: {e}")
            # Поверnotння до послandдовної обробки
            return [func(item) for item in data]
    
    def get_optimization_summary(self) -> Dict:
        """
        Отримання пandдсумку оптимandforцandй
        
        Returns:
            Словник with пandдсумком
        """
        with self._lock:
            return {
                "timestamp": datetime.now().isoformat(),
                "statistics": {
                    "total_optimizations": self.total_optimizations,
                    "successful_optimizations": self.successful_optimizations,
                    "success_rate": self.successful_optimizations / max(1, self.total_optimizations) * 100,
                    "cached_functions": len(self.cached_functions),
                    "performance_metrics_count": len(self.performance_metrics)
                },
                "recent_optimizations": list(self.optimization_history)[-10:],
                "top_slow_functions": self._get_top_slow_functions(),
                "memory_optimizations": [opt for opt in self.optimization_history if opt.get("type") == "memory_cleanup"]
            }
    
    def _get_top_slow_functions(self, limit: int = None) -> List[Dict]:
        """Отримання найповandльнandших функцandй"""
        if limit is None:
            limit = 5
        
        function_stats = []
        
        for func_name, metrics in self.performance_metrics.items():
            if metrics:
                execution_times = [m["execution_time"] for m in metrics]
                avg_time = sum(execution_times) / len(execution_times)
                
                function_stats.append({
                    "function": func_name,
                    "avg_execution_time": avg_time,
                    "total_calls": len(metrics),
                    "max_execution_time": max(execution_times)
                })
        
        # Сортування for середнandм часом виконання
        function_stats.sort(key=lambda x: x["avg_execution_time"], reverse=True)
        
        return function_stats[:limit]
    
    def reset_metrics(self):
        """Скидання allх метрик"""
        with self._lock:
            self.performance_metrics.clear()
            self.performance_history.clear()
            self.optimization_history.clear()
            self.cached_functions.clear()
            
            self.total_optimizations = 0
            self.successful_optimizations = 0
            
            logger.info("[UtilsOptimizer] All metrics reset")
    
    def export_metrics(self, filename: str):
        """
        Експорт метрик у file
        
        Args:
            filename: Ім'я fileу
        """
        try:
            import json
            
            export_data = {
                "timestamp": datetime.now().isoformat(),
                "performance_metrics": {k: list(v) for k, v in self.performance_metrics.items()},
                "optimization_history": list(self.optimization_history),
                "cached_functions": {k: {"size": len(v["cache"])} for k, v in self.cached_functions.items()},
                "statistics": {
                    "total_optimizations": self.total_optimizations,
                    "successful_optimizations": self.successful_optimizations
                }
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"[UtilsOptimizer] Metrics exported to {filename}")
            
        except Exception as e:
            logger.error(f"[UtilsOptimizer] Failed to export metrics: {e}")
            raise
