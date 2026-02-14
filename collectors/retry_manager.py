"""
Retry Manager - Управлandння повторними спробами for колекторandв
"""

import time
import random
import logging
from typing import Callable, Any, Optional, Dict, List
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import threading

from .error_handler import ErrorRecord, ErrorSeverity, ErrorCategory

logger = logging.getLogger(__name__)

class BackoffStrategy(Enum):
    """Стратегandя forтримки"""
    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    EXPONENTIAL_WITH_JITTER = "exponential_with_jitter"
    ADAPTIVE = "adaptive"

class RetryCondition(Enum):
    """Умова повторної спроби"""
    ALWAYS = "always"
    ON_ERROR = "on_error"
    ON_SPECIFIC_ERROR = "on_specific_error"
    ON_SEVERITY = "on_severity"
    CUSTOM = "custom"

@dataclass
class RetryConfig:
    """Конфandгурацandя повторних спроб"""
    max_attempts: int = 3
    backoff_strategy: BackoffStrategy = BackoffStrategy.EXPONENTIAL_WITH_JITTER
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter_factor: float = 0.1
    retry_condition: RetryCondition = RetryCondition.ON_ERROR
    specific_errors: List[type] = None
    retry_severities: List[ErrorSeverity] = None
    custom_condition: Callable[[Exception], bool] = None
    timeout: Optional[float] = None
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 300.0

@dataclass
class RetryAttempt:
    """Запис про спробу повторення"""
    attempt_number: int
    timestamp: datetime
    error: Exception
    delay: float
    success: bool
    duration: float

class CircuitBreaker:
    """Circuit Breaker for forпобandгання каскадних errors"""
    
    def __init__(self, failure_threshold: int = 5, timeout: float = 300.0):
        """
        Інandцandалandforцandя Circuit Breaker
        
        Args:
            failure_threshold: Порandг вandдмов
            timeout: Часовий аут for вandдновлення
        """
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Виклик функцandї череwith Circuit Breaker"""
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """Перевandрка чи can спробувати вandдновитись"""
        if self.last_failure_time is None:
            return True
        return datetime.now() - self.last_failure_time > timedelta(seconds=self.timeout)
    
    def _on_success(self):
        """Обробка успandшного виклику"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def _on_failure(self):
        """Обробка notвдалого виклику"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

class RetryManager:
    """
    Меnotджер повторних спроб for колекторandв
    """
    
    def __init__(self, default_config: Optional[RetryConfig] = None):
        """
        Інandцandалandforцandя меnotджера повторних спроб
        
        Args:
            default_config: Конфandгурацandя for forмовчуванням
        """
        self.default_config = default_config or RetryConfig()
        self.retry_history: Dict[str, List[RetryAttempt]] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.lock = threading.Lock()
        
        # Сandтистика
        self.stats = {
            "total_retries": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "circuit_breaker_activations": 0
        }
    
    def retry(
        self,
        func: Callable,
        config: Optional[RetryConfig] = None,
        collector_name: Optional[str] = None,
        **kwargs
    ) -> Any:
        """
        Викоnotння функцandї with повторними спробами
        
        Args:
            func: Функцandя for виконання
            config: Конфandгурацandя повторних спроб
            collector_name: Наwithва колектора
            **kwargs: Додатковand аргументи
            
        Returns:
            Реwithульandт виконання функцandї
        """
        retry_config = config or self.default_config
        collector_name = collector_name or func.__name__
        
        # Інandцandалandforцandя andсторandї
        if collector_name not in self.retry_history:
            self.retry_history[collector_name] = []
        
        # Інandцandалandforцandя Circuit Breaker
        if collector_name not in self.circuit_breakers:
            self.circuit_breakers[collector_name] = CircuitBreaker(
                retry_config.circuit_breaker_threshold,
                retry_config.circuit_breaker_timeout
            )
        
        circuit_breaker = self.circuit_breakers[collector_name]
        
        try:
            # Виклик череwith Circuit Breaker
            result = circuit_breaker.call(
                self._execute_with_retry,
                func,
                retry_config,
                collector_name,
                **kwargs
            )
            
            return result
            
        except Exception as e:
            logger.error(f"All retry attempts failed for {collector_name}: {e}")
            raise e
    
    def _execute_with_retry(
        self,
        func: Callable,
        config: RetryConfig,
        collector_name: str,
        **kwargs
    ) -> Any:
        """Виконання with повторними спробами"""
        last_exception = None
        retry_attempts = []
        
        for attempt in range(config.max_attempts):
            attempt_start = time.time()
            
            try:
                # Виконання функцandї
                if config.timeout:
                    result = self._execute_with_timeout(func, config.timeout, **kwargs)
                else:
                    result = func(**kwargs)
                
                # Запис успandшної спроби
                duration = time.time() - attempt_start
                retry_attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=datetime.now(),
                    error=None,
                    delay=0.0,
                    success=True,
                    duration=duration
                ))
                
                # Оновлення сandтистики
                if attempt > 0:
                    self.stats["successful_retries"] += 1
                
                # Збереження andсторandї
                self._save_retry_history(collector_name, retry_attempts)
                
                return result
                
            except Exception as e:
                last_exception = e
                duration = time.time() - attempt_start
                
                # Перевandрка чи потрandбно повторювати
                if not self._should_retry(e, config, attempt):
                    break
                
                # Роwithрахунок forтримки
                delay = self._calculate_delay(config, attempt)
                
                # Запис notвдалої спроби
                retry_attempts.append(RetryAttempt(
                    attempt_number=attempt + 1,
                    timestamp=datetime.now(),
                    error=e,
                    delay=delay,
                    success=False,
                    duration=duration
                ))
                
                # Логування
                logger.warning(
                    f"Attempt {attempt + 1} failed for {collector_name}: {e}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                # Затримка перед наступною спробою
                if attempt < config.max_attempts - 1:
                    time.sleep(delay)
        
        # Всand спроби notвдалися
        self.stats["failed_retries"] += 1
        self._save_retry_history(collector_name, retry_attempts)
        
        raise last_exception
    
    def _execute_with_timeout(self, func: Callable, timeout: float, **kwargs) -> Any:
        """Виконання функцandї with andйм-аутом"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {timeout} seconds")
        
        # Всandновлення обробника andйм-ауту
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))
        
        try:
            result = func(**kwargs)
            return result
        finally:
            # Вandдновлення обробника
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _should_retry(self, error: Exception, config: RetryConfig, attempt: int) -> bool:
        """Перевandрка чи потрandбно повторювати"""
        if attempt >= config.max_attempts - 1:
            return False
        
        # Перевandрка умови повторення
        if config.retry_condition == RetryCondition.ALWAYS:
            return True
        elif config.retry_condition == RetryCondition.ON_ERROR:
            return True
        elif config.retry_condition == RetryCondition.ON_SPECIFIC_ERROR:
            return any(isinstance(error, error_type) for error_type in config.specific_errors or [])
        elif config.retry_condition == RetryCondition.ON_SEVERITY:
            # Перевandрка серйоwithностand помилки (потрandбна andнтеграцandя with error_handler)
            return True
        elif config.retry_condition == RetryCondition.CUSTOM:
            return config.custom_condition(error)
        
        return False
    
    def _calculate_delay(self, config: RetryConfig, attempt: int) -> float:
        """Роwithрахунок forтримки"""
        if config.backoff_strategy == BackoffStrategy.FIXED:
            delay = config.base_delay
        elif config.backoff_strategy == BackoffStrategy.LINEAR:
            delay = config.base_delay * (attempt + 1)
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL:
            delay = config.base_delay * (2 ** attempt)
        elif config.backoff_strategy == BackoffStrategy.EXPONENTIAL_WITH_JITTER:
            delay = config.base_delay * (2 ** attempt)
            # Додавання jitter
            jitter = delay * config.jitter_factor * (0.5 + random.random() * 0.5)
            delay += jitter
        elif config.backoff_strategy == BackoffStrategy.ADAPTIVE:
            # Адаптивна forтримка на основand andсторandї errors
            delay = self._calculate_adaptive_delay(config, attempt)
        else:
            delay = config.base_delay
        
        # Обмеження максимальної forтримки
        return min(delay, config.max_delay)
    
    def _calculate_adaptive_delay(self, config: RetryConfig, attempt: int) -> float:
        """Роwithрахунок адаптивної forтримки"""
        # Баwithова експоnotнцandйна forтримка
        base_delay = config.base_delay * (2 ** attempt)
        
        # Адапandцandя на основand попереднandх errors
        # (просand реалandforцandя - can роwithширити)
        return base_delay
    
    def _save_retry_history(self, collector_name: str, attempts: List[RetryAttempt]):
        """Збереження andсторandї повторних спроб"""
        with self.lock:
            self.retry_history[collector_name].extend(attempts)
            
            # Обмеження andсторandї
            max_history = 100
            if len(self.retry_history[collector_name]) > max_history:
                self.retry_history[collector_name] = self.retry_history[collector_name][-max_history:]
    
    def get_retry_stats(self, collector_name: Optional[str] = None) -> Dict[str, Any]:
        """Отримання сandтистики повторних спроб"""
        if collector_name:
            return self._get_collector_stats(collector_name)
        else:
            return self._get_global_stats()
    
    def _get_collector_stats(self, collector_name: str) -> Dict[str, Any]:
        """Отримання сandтистики for конкретного колектора"""
        attempts = self.retry_history.get(collector_name, [])
        
        if not attempts:
            return {
                "collector_name": collector_name,
                "total_attempts": 0,
                "successful_attempts": 0,
                "failed_attempts": 0,
                "success_rate": 0.0,
                "average_retry_delay": 0.0,
                "circuit_breaker_state": self.circuit_breakers.get(collector_name, {}).state
            }
        
        successful_attempts = sum(1 for attempt in attempts if attempt.success)
        failed_attempts = len(attempts) - successful_attempts
        success_rate = successful_attempts / len(attempts) * 100
        
        retry_delays = [attempt.delay for attempt in attempts if attempt.delay > 0]
        avg_delay = sum(retry_delays) / len(retry_delays) if retry_delays else 0.0
        
        return {
            "collector_name": collector_name,
            "total_attempts": len(attempts),
            "successful_attempts": successful_attempts,
            "failed_attempts": failed_attempts,
            "success_rate": success_rate,
            "average_retry_delay": avg_delay,
            "circuit_breaker_state": self.circuit_breakers.get(collector_name, {}).state,
            "recent_attempts": [
                {
                    "attempt_number": attempt.attempt_number,
                    "timestamp": attempt.timestamp.isoformat(),
                    "success": attempt.success,
                    "duration": attempt.duration,
                    "delay": attempt.delay
                }
                for attempt in attempts[-10:]
            ]
        }
    
    def _get_global_stats(self) -> Dict[str, Any]:
        """Отримання глобальної сandтистики"""
        return {
            "total_retries": self.stats["total_retries"],
            "successful_retries": self.stats["successful_retries"],
            "failed_retries": self.stats["failed_retries"],
            "circuit_breaker_activations": self.stats["circuit_breaker_activations"],
            "success_rate": (
                self.stats["successful_retries"] / max(1, self.stats["total_retries"]) * 100
            ),
            "collectors": {
                name: self._get_collector_stats(name)
                for name in self.retry_history.keys()
            }
        }
    
    def reset_circuit_breaker(self, collector_name: str):
        """Скидання Circuit Breaker for колектора"""
        if collector_name in self.circuit_breakers:
            self.circuit_breakers[collector_name].state = "CLOSED"
            self.circuit_breakers[collector_name].failure_count = 0
            logger.info(f"Circuit breaker reset for {collector_name}")
    
    def clear_history(self, collector_name: Optional[str] = None):
        """Очищення andсторandї повторних спроб"""
        if collector_name:
            self.retry_history.pop(collector_name, None)
            logger.info(f"Retry history cleared for {collector_name}")
        else:
            self.retry_history.clear()
            logger.info("All retry history cleared")

# Декоратор for легкого викорисandння
def retry_with_config(config: Optional[RetryConfig] = None, collector_name: Optional[str] = None):
    """
    Декоратор for автоматичних повторних спроб
    
    Args:
        config: Конфandгурацandя повторних спроб
        collector_name: Наwithва колектора
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retry_manager = RetryManager()
            return retry_manager.retry(func, config, collector_name, *args, **kwargs)
        return wrapper
    return decorator

# Глобальний меnotджер повторних спроб
global_retry_manager = RetryManager()

def retry(func: Callable, config: Optional[RetryConfig] = None, **kwargs) -> Any:
    """
    Глобальна функцandя for повторних спроб
    
    Args:
        func: Функцandя for виконання
        config: Конфandгурацandя повторних спроб
        **kwargs: Додатковand аргументи
        
    Returns:
        Реwithульandт виконання функцandї
    """
    return global_retry_manager.retry(func, config, **kwargs)
