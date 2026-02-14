#!/usr/bin/env python3
"""
Покращена система обробки помилок для trading системи
Централізована обробка, логування та відновлення від помилок
"""

import logging
import traceback
import sys
from typing import Dict, Any, List, Optional, Callable, Type, Union
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
from functools import wraps
import threading
import queue
import time
from dataclasses import dataclass, field
from contextlib import contextmanager

from utils.common_utils import FileManager, CacheManager


class ErrorSeverity(str, Enum):
    """Рівні серйозності помилок"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ErrorCategory(str, Enum):
    """Категорії помилок"""
    DATA = "DATA"
    NETWORK = "NETWORK"
    VALIDATION = "VALIDATION"
    CALCULATION = "CALCULATION"
    SYSTEM = "SYSTEM"
    BUSINESS = "BUSINESS"
    UNKNOWN = "UNKNOWN"


@dataclass
class ErrorInfo:
    """Інформація про помилку"""
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    context: str
    timestamp: datetime
    traceback: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    additional_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертація в словник"""
        return {
            'error_type': self.error_type,
            'error_message': self.error_message,
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'additional_data': self.additional_data
        }


class ErrorClassifier:
    """Класифікатор помилок"""
    
    def __init__(self):
        self.classification_rules = {
            'ValueError': ErrorCategory.VALIDATION,
            'TypeError': ErrorCategory.VALIDATION,
            'KeyError': ErrorCategory.DATA,
            'IndexError': ErrorCategory.DATA,
            'ConnectionError': ErrorCategory.NETWORK,
            'TimeoutError': ErrorCategory.NETWORK,
            'MemoryError': ErrorCategory.SYSTEM,
            'FileNotFoundError': ErrorCategory.SYSTEM,
            'PermissionError': ErrorCategory.SYSTEM,
            'ZeroDivisionError': ErrorCategory.CALCULATION,
            'OverflowError': ErrorCategory.CALCULATION,
        }
        
        self.severity_rules = {
            ErrorCategory.SYSTEM: ErrorSeverity.HIGH,
            ErrorCategory.NETWORK: ErrorSeverity.MEDIUM,
            ErrorCategory.DATA: ErrorSeverity.MEDIUM,
            ErrorCategory.VALIDATION: ErrorSeverity.LOW,
            ErrorCategory.CALCULATION: ErrorSeverity.MEDIUM,
            ErrorCategory.BUSINESS: ErrorSeverity.HIGH,
        }
    
    def classify_error(self, error: Exception, context: str = "") -> tuple[ErrorCategory, ErrorSeverity]:
        """Класифікація помилки"""
        error_type = type(error).__name__
        
        # Класифікація за правилами
        category = self.classification_rules.get(error_type, ErrorCategory.UNKNOWN)
        
        # Спроба класифікації за контекстом
        if 'network' in context.lower() or 'api' in context.lower():
            category = ErrorCategory.NETWORK
        elif 'data' in context.lower() or 'database' in context.lower():
            category = ErrorCategory.DATA
        elif 'validation' in context.lower() or 'check' in context.lower():
            category = ErrorCategory.VALIDATION
        elif 'calculation' in context.lower() or 'compute' in context.lower():
            category = ErrorCategory.CALCULATION
        
        # Визначення серйозності
        severity = self.severity_rules.get(category, ErrorSeverity.MEDIUM)
        
        # Підвищення серйозності для критичних помилок
        if 'critical' in context.lower() or 'fatal' in context.lower():
            severity = ErrorSeverity.CRITICAL
        
        return category, severity


class ErrorRecoveryStrategy:
    """Стратегія відновлення від помилок"""
    
    def __init__(self):
        self.recovery_actions = {
            ErrorCategory.NETWORK: self._recover_network_error,
            ErrorCategory.DATA: self._recover_data_error,
            ErrorCategory.VALIDATION: self._recover_validation_error,
            ErrorCategory.CALCULATION: self._recover_calculation_error,
            ErrorCategory.SYSTEM: self._recover_system_error,
            ErrorCategory.BUSINESS: self._recover_business_error,
        }
    
    def recover(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Спроба відновлення від помилки"""
        recovery_action = self.recovery_actions.get(error_info.category)
        
        if recovery_action:
            try:
                return recovery_action(error_info)
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Recovery failed: {e}",
                    'action_taken': 'none'
                }
        
        return {
            'success': False,
            'message': "No recovery strategy available",
            'action_taken': 'none'
        }
    
    def _recover_network_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від мережевої помилки"""
        # Затримка перед повторною спробою
        time.sleep(1)
        
        return {
            'success': True,
            'message': "Network error recovery: retry with delay",
            'action_taken': 'retry_with_delay',
            'delay_seconds': 1
        }
    
    def _recover_data_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від помилки data"""
        return {
            'success': True,
            'message': "Data error recovery: use default values",
            'action_taken': 'use_defaults',
            'default_values': True
        }
    
    def _recover_validation_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від помилки валідації"""
        return {
            'success': True,
            'message': "Validation error recovery: sanitize input",
            'action_taken': 'sanitize_input'
        }
    
    def _recover_calculation_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від помилки обчислення"""
        return {
            'success': True,
            'message': "Calculation error recovery: use safe defaults",
            'action_taken': 'safe_defaults',
            'safe_computation': True
        }
    
    def _recover_system_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від системної помилки"""
        return {
            'success': False,
            'message': "System error requires manual intervention",
            'action_taken': 'manual_intervention_required'
        }
    
    def _recover_business_error(self, error_info: ErrorInfo) -> Dict[str, Any]:
        """Відновлення від бізнес-помилки"""
        return {
            'success': True,
            'message': "Business error recovery: skip operation",
            'action_taken': 'skip_operation'
        }


class EnhancedErrorHandler:
    """Покращений обробник помилок"""
    
    def __init__(self, logger: logging.Logger = None, config: Dict[str, Any] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.classifier = ErrorClassifier()
        self.recovery_strategy = ErrorRecoveryStrategy()
        self.file_manager = FileManager()
        self.cache_manager = CacheManager()
        
        # Конфігурація
        self.config = config or {
            'max_error_history': 1000,
            'error_log_file': 'errors.json',
            'enable_recovery': True,
            'enable_monitoring': True,
            'alert_threshold': 10  # Кількість помилок для алерту
        }
        
        # Статистика помилок
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[str, int] = {}
        self.error_counts_by_category: Dict[ErrorCategory, int] = {}
        self.error_counts_by_severity: Dict[ErrorSeverity, int] = {}
        
        # Моніторинг
        self.error_queue = queue.Queue()
        self.monitoring_thread = None
        self.alert_callbacks: List[Callable] = []
        
        # Блокування для потокоwithoutпекності
        self.lock = threading.Lock()
        
        # Запуск моніторингу
        if self.config['enable_monitoring']:
            self._start_monitoring()
    
    def handle_error(self, error: Exception, context: str = "", 
                    user_id: str = None, session_id: str = None,
                    additional_data: Dict[str, Any] = None) -> ErrorInfo:
        """Обробка помилки"""
        # Класифікація помилки
        category, severity = self.classifier.classify_error(error, context)
        
        # Створення інформації про помилку
        error_info = ErrorInfo(
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            category=category,
            context=context,
            timestamp=datetime.now(),
            traceback=traceback.format_exc(),
            user_id=user_id,
            session_id=session_id,
            additional_data=additional_data or {}
        )
        
        # Оновлення статистики
        self._update_statistics(error_info)
        
        # Логування
        self._log_error(error_info)
        
        # Спроба відновлення
        if self.config['enable_recovery']:
            recovery_result = self.recovery_strategy.recover(error_info)
            error_info.additional_data['recovery_result'] = recovery_result
        
        # Збереження в історію
        self._save_error_history(error_info)
        
        # Перевірка на алерти
        self._check_alerts(error_info)
        
        return error_info
    
    def _update_statistics(self, error_info: ErrorInfo):
        """Оновлення статистики помилок"""
        with self.lock:
            # Лічильники помилок
            error_key = f"{error_info.error_type}:{error_info.context}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
            
            # Лічильники по категоріях
            self.error_counts_by_category[error_info.category] = \
                self.error_counts_by_category.get(error_info.category, 0) + 1
            
            # Лічильники по серйозності
            self.error_counts_by_severity[error_info.severity] = \
                self.error_counts_by_severity.get(error_info.severity, 0) + 1
            
            # Додавання в історію
            self.error_history.append(error_info)
            
            # Обмеження історії
            if len(self.error_history) > self.config['max_error_history']:
                self.error_history.pop(0)
    
    def _log_error(self, error_info: ErrorInfo):
        """Логування помилки"""
        log_message = f"[{error_info.severity.value}] {error_info.category.value} in {error_info.context}: {error_info.error_message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
        
        # Логування traceback для серйозних помилок
        if error_info.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL] and error_info.traceback:
            self.logger.debug(f"Traceback for {error_info.error_type}: {error_info.traceback}")
    
    def _save_error_history(self, error_info: ErrorInfo):
        """Збереження історії помилок"""
        try:
            # Періодичне збереження в файл
            if len(self.error_history) % 10 == 0:  # Кожні 10 помилок
                error_log_file = Path(self.config['error_log_file'])
                errors_data = [error.to_dict() for error in self.error_history[-100:]]  # Останні 100 помилок
                self.file_manager.save_json(errors_data, error_log_file)
        except Exception as e:
            self.logger.error(f"Failed to save error history: {e}")
    
    def _check_alerts(self, error_info: ErrorInfo):
        """Перевірка на алерти"""
        # Алерт на велику кількість помилок
        total_errors = sum(self.error_counts.values())
        if total_errors >= self.config['alert_threshold']:
            self._trigger_alert('high_error_count', {
                'total_errors': total_errors,
                'threshold': self.config['alert_threshold']
            })
        
        # Алерт на критичні помилки
        if error_info.severity == ErrorSeverity.CRITICAL:
            self._trigger_alert('critical_error', error_info.to_dict())
        
        # Алерт на системні помилки
        if error_info.category == ErrorCategory.SYSTEM:
            self._trigger_alert('system_error', error_info.to_dict())
    
    def _trigger_alert(self, alert_type: str, alert_data: Dict[str, Any]):
        """Спрацювання алерту"""
        for callback in self.alert_callbacks:
            try:
                callback(alert_type, alert_data)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def _start_monitoring(self):
        """Запуск моніторингу помилок"""
        self.monitoring_thread = threading.Thread(target=self._monitor_errors, daemon=True)
        self.monitoring_thread.start()
    
    def _monitor_errors(self):
        """Моніторинг помилок в окремому потоці"""
        while True:
            try:
                # Отримання помилок з черги
                while not self.error_queue.empty():
                    error_info = self.error_queue.get_nowait()
                    self._process_error_monitoring(error_info)
                
                # Сон
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Error monitoring failed: {e}")
    
    def _process_error_monitoring(self, error_info: ErrorInfo):
        """Обробка помилки в моніторингу"""
        # Аналіз тенденцій
        self._analyze_error_trends(error_info)
        
        # Перевірка на аномалії
        self._check_error_anomalies(error_info)
    
    def _analyze_error_trends(self, error_info: ErrorInfo):
        """Аналіз тенденцій помилок"""
        # Аналіз останніх 100 помилок
        recent_errors = self.error_history[-100:]
        
        if len(recent_errors) >= 10:
            # Перевірка на сплеск помилок
            last_10_errors = recent_errors[-10:]
            error_types = [e.error_type for e in last_10_errors]
            
            # Якщо 7+ з 10 помилок одного типу
            if len(set(error_types)) <= 3:
                self._trigger_alert('error_spike', {
                    'error_types': error_types,
                    'count_by_type': {t: error_types.count(t) for t in set(error_types)}
                })
    
    def _check_error_anomalies(self, error_info: ErrorInfo):
        """Перевірка на аномалії помилок"""
        # Аномалія: новий тип помилки
        if error_info.error_type not in [e.error_type for e in self.error_history[:-1]]:
            self._trigger_alert('new_error_type', error_info.to_dict())
        
        # Аномалія: незвичайний контекст
        contexts = [e.context for e in self.error_history]
        if contexts.count(error_info.context) == 1 and len(contexts) > 50:
            self._trigger_alert('unusual_context', error_info.to_dict())
    
    def add_alert_callback(self, callback: Callable[[str, Dict[str, Any]], None]):
        """Додавання callback для алертів"""
        self.alert_callbacks.append(callback)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Отримання статистики помилок"""
        with self.lock:
            return {
                'total_errors': len(self.error_history),
                'error_counts': dict(self.error_counts),
                'errors_by_category': {k.value: v for k, v in self.error_counts_by_category.items()},
                'errors_by_severity': {k.value: v for k, v in self.error_counts_by_severity.items()},
                'recent_errors': [e.to_dict() for e in self.error_history[-10:]],
                'error_rate': self._calculate_error_rate()
            }
    
    def _calculate_error_rate(self) -> float:
        """Розрахунок частоти помилок"""
        if len(self.error_history) < 2:
            return 0.0
        
        first_error = self.error_history[0].timestamp
        last_error = self.error_history[-1].timestamp
        duration = (last_error - first_error).total_seconds()
        
        if duration > 0:
            return len(self.error_history) / duration * 3600  # Помилки на годину
        
        return 0.0
    
    def clear_error_history(self):
        """Очищення історії помилок"""
        with self.lock:
            self.error_history.clear()
            self.error_counts.clear()
            self.error_counts_by_category.clear()
            self.error_counts_by_severity.clear()
    
    def export_errors(self, file_path: str, format: str = 'json'):
        """Експорт помилок в файл"""
        errors_data = [error.to_dict() for error in self.error_history]
        
        if format == 'json':
            self.file_manager.save_json(errors_data, file_path)
        elif format == 'csv':
            df = pd.DataFrame(errors_data)
            self.file_manager.save_dataframe(df, file_path, 'csv')
        else:
            raise ValueError(f"Unsupported format: {format}")


# Декоратори для покращеної обробки помилок
def enhanced_error_handler(context: str = "", severity: ErrorSeverity = None, 
                          enable_recovery: bool = True, max_retries: int = 3):
    """Покращений декоратор обробки помилок"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = get_enhanced_error_handler()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_info = error_handler.handle_error(
                        e, context or f"{func.__module__}.{func.__name__}",
                        additional_data={'attempt': attempt + 1, 'max_retries': max_retries}
                    )
                    
                    # Перевірка на можливість відновлення
                    if enable_recovery and 'recovery_result' in error_info.additional_data:
                        recovery_result = error_info.additional_data['recovery_result']
                        if recovery_result.get('success', False):
                            # Спроба повторного виконання після відновлення
                            if recovery_result.get('action_taken') == 'retry_with_delay':
                                time.sleep(recovery_result.get('delay_seconds', 1))
                            continue
                    
                    # Остання спроба
                    if attempt == max_retries:
                        raise e
            
            return None
        return wrapper
    return decorator


def safe_execute(func: Callable, *args, default_return: Any = None, 
                context: str = "", **kwargs) -> Any:
    """Безпечне виконання функції"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler = get_enhanced_error_handler()
        error_handler.handle_error(e, context)
        return default_return


@contextmanager
def error_context(context: str, user_id: str = None, session_id: str = None):
    """Контекстний менеджер для обробки помилок"""
    error_handler = get_enhanced_error_handler()
    try:
        yield
    except Exception as e:
        error_handler.handle_error(e, context, user_id, session_id)
        raise


# Глобальний екземпляр
_enhanced_error_handler = None


def get_enhanced_error_handler(logger: logging.Logger = None, config: Dict[str, Any] = None) -> EnhancedErrorHandler:
    """Отримання глобального екземпляру покращеного обробника помилок"""
    global _enhanced_error_handler
    if _enhanced_error_handler is None:
        _enhanced_error_handler = EnhancedErrorHandler(logger, config)
    return _enhanced_error_handler


def setup_error_handling(config: Dict[str, Any] = None):
    """Налаштування обробки помилок"""
    global _enhanced_error_handler
    _enhanced_error_handler = EnhancedErrorHandler(config=config)


# Приклад callback для алертів
def default_alert_callback(alert_type: str, alert_data: Dict[str, Any]):
    """Стандартний callback для алертів"""
    logger = logging.getLogger(__name__)
    logger.warning(f"ALERT: {alert_type} - {alert_data}")


# Ініціалізація
if __name__ == "__main__":
    # Налаштування логування
    logging.basicConfig(level=logging.INFO)
    
    # Створення обробника помилок
    error_handler = EnhancedErrorHandler()
    
    # Додавання callback для алертів
    error_handler.add_alert_callback(default_alert_callback)
    
    # Тестування
    try:
        # Симуляція помилки
        raise ValueError("Test error for demonstration")
    except Exception as e:
        error_info = error_handler.handle_error(e, "test_context")
        print(f"Handled error: {error_info.error_type} - {error_info.severity.value}")
    
    # Отримання статистики
    stats = error_handler.get_error_statistics()
    print(f"Error statistics: {stats}")
