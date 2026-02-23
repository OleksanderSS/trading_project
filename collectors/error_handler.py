"""
Error Handler - Унandфandкована обробка errors for колекторandв
"""

import logging
import traceback
from typing import Dict, List, Optional, Any, Type
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import json
from pathlib import Path

from .collector_interface import (
    CollectorError, APIError, ConfigurationError, DataValidationError,
    ErrorSeverity, ErrorCategory
)

logger = logging.getLogger(__name__)

@dataclass
class ErrorRecord:
    """Запис помилки"""
    timestamp: datetime
    collector_name: str
    error_type: str
    error_message: str
    severity: ErrorSeverity
    category: ErrorCategory
    stack_trace: Optional[str]
    context: Dict[str, Any]
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    resolution_method: Optional[str] = None

class ErrorHandler:
    """
    Унandфandкований обробник errors for колекторandв
    """
    
    def __init__(self, log_file: Optional[str] = None, max_history: int = 1000):
        """
        Інandцandалandforцandя обробника errors
        
        Args:
            log_file: Шлях до fileу logs
            max_history: Максимальна кandлькandсть forписandв в andсторandї
        """
        self.log_file = log_file
        self.max_history = max_history
        self.error_history: List[ErrorRecord] = []
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        
        # Сandтистика errors
        self.error_stats = {
            "total_errors": 0,
            "resolved_errors": 0,
            "by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "by_category": {category.value: 0 for category in ErrorCategory},
            "by_collector": {}
        }
        
        # Заванandження попереднandх errors
        self._load_error_patterns()
    
    def handle_error(
        self,
        collector_name: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> ErrorRecord:
        """
        Обробка помилки
        
        Args:
            collector_name: Наwithва колектора
            error: Error
            context: Контекст помилки
            
        Returns:
            ErrorRecord: Запис помилки
        """
        # Виvalues типу and категорandї помилки
        error_type = type(error).__name__
        severity = self._determine_severity(error)
        category = self._determine_category(error)
        
        # Створення forпису помилки
        error_record = ErrorRecord(
            timestamp=datetime.now(),
            collector_name=collector_name,
            error_type=error_type,
            error_message=str(error),
            severity=severity,
            category=category,
            stack_trace=traceback.format_exc(),
            context=context or {}
        )
        
        # Додавання в andсторandю
        self._add_to_history(error_record)
        
        # Оновлення сandтистики
        self._update_stats(error_record)
        
        # Логування
        self._log_error(error_record)
        
        # Спроба automaticallyго вирandшення
        resolution = self._attempt_auto_resolution(error_record)
        if resolution:
            error_record.resolved = True
            error_record.resolution_time = datetime.now()
            error_record.resolution_method = resolution
        
        # Збереження в file
        if self.log_file:
            self._save_to_file(error_record)
        
        return error_record
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Виvalues рandвня серйоwithностand помилки"""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, APIError):
            return ErrorSeverity.HIGH
        elif isinstance(error, ConfigurationError):
            return ErrorSeverity.CRITICAL
        elif isinstance(error, DataValidationError):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, CollectorError):
            return ErrorSeverity.HIGH
        else:
            return ErrorSeverity.LOW
    
    def _determine_category(self, error: Exception) -> ErrorCategory:
        """Виvalues категорandї помилки"""
        if isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, APIError):
            return ErrorCategory.API
        elif isinstance(error, ConfigurationError):
            return ErrorCategory.CONFIGURATION
        elif isinstance(error, DataValidationError):
            return ErrorCategory.VALIDATION
        elif isinstance(error, CollectorError):
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN
    
    def _add_to_history(self, error_record: ErrorRecord):
        """Додавання forпису в andсторandю"""
        self.error_history.append(error_record)
        
        # Обмеження andсторandї
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
    
    def _update_stats(self, error_record: ErrorRecord):
        """Оновлення сandтистики"""
        self.error_stats["total_errors"] += 1
        
        if error_record.resolved:
            self.error_stats["resolved_errors"] += 1
        
        self.error_stats["by_severity"][error_record.severity.value] += 1
        self.error_stats["by_category"][error_record.category.value] += 1
        
        if error_record.collector_name not in self.error_stats["by_collector"]:
            self.error_stats["by_collector"][error_record.collector_name] = 0
        self.error_stats["by_collector"][error_record.collector_name] += 1
    
    def _log_error(self, error_record: ErrorRecord):
        """Логування помилки"""
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(error_record.severity, logging.ERROR)
        
        logger.log(
            log_level,
            f"[{error_record.severity.value.upper()}] "
            f"{error_record.collector_name}: {error_record.error_message}"
        )
        
        if error_record.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"Stack trace: {error_record.stack_trace}")
    
    def _attempt_auto_resolution(self, error_record: ErrorRecord) -> Optional[str]:
        """Спроба automaticallyго вирandшення помилки"""
        error_message = error_record.error_message.lower()
        
        # Мережевand помилки
        if error_record.category == ErrorCategory.NETWORK:
            if "timeout" in error_message:
                return "timeout_retry"
            elif "connection" in error_message:
                return "connection_retry"
        
        # API помилки
        elif error_record.category == ErrorCategory.API:
            if "rate limit" in error_message:
                return "rate_limit_backoff"
            elif "unauthorized" in error_message:
                return "auth_check"
            elif "not found" in error_message:
                return "endpoint_check"
        
        # Помилки конфandгурацandї
        elif error_record.category == ErrorCategory.CONFIGURATION:
            if "missing" in error_message:
                return "config_fix"
            elif "invalid" in error_message:
                return "config_validate"
        
        return None
    
    def _save_to_file(self, error_record: ErrorRecord):
        """Збереження forпису в file"""
        try:
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(log_path, 'a', encoding='utf-8') as f:
                error_data = {
                    "timestamp": error_record.timestamp.isoformat(),
                    "collector_name": error_record.collector_name,
                    "error_type": error_record.error_type,
                    "error_message": error_record.error_message,
                    "severity": error_record.severity.value,
                    "category": error_record.category.value,
                    "stack_trace": error_record.stack_trace,
                    "context": error_record.context,
                    "resolved": error_record.resolved,
                    "resolution_time": error_record.resolution_time.isoformat() if error_record.resolution_time else None,
                    "resolution_method": error_record.resolution_method
                }
                
                f.write(json.dumps(error_data, default=str) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to save error to file: {e}")
    
    def _load_error_patterns(self):
        """Заванandження патернandв errors"""
        self.error_patterns = {
            "network_timeout": {
                "keywords": ["timeout", "timed out"],
                "resolution": "retry_with_backoff",
                "max_retries": 3,
                "backoff_factor": 2.0
            },
            "rate_limit": {
                "keywords": ["rate limit", "too many requests"],
                "resolution": "exponential_backoff",
                "initial_delay": 60,
                "max_delay": 300
            },
            "auth_error": {
                "keywords": ["unauthorized", "authentication", "401"],
                "resolution": "refresh_credentials",
                "action": "check_api_keys"
            },
            "data_validation": {
                "keywords": ["validation", "invalid", "malformed"],
                "resolution": "data_cleaning",
                "action": "validate_input"
            }
        }
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Отримання пandдсумку errors"""
        return {
            "total_errors": self.error_stats["total_errors"],
            "resolved_errors": self.error_stats["resolved_errors"],
            "resolution_rate": (
                self.error_stats["resolved_errors"] / max(1, self.error_stats["total_errors"]) * 100
            ),
            "by_severity": self.error_stats["by_severity"],
            "by_category": self.error_stats["by_category"],
            "by_collector": dict(sorted(
                self.error_stats["by_collector"].items(),
                key=lambda x: x[1],
                reverse=True
            )),
            "recent_errors": [
                {
                    "timestamp": err.timestamp.isoformat(),
                    "collector": err.collector_name,
                    "message": err.error_message,
                    "severity": err.severity.value,
                    "resolved": err.resolved
                }
                for err in self.error_history[-10:]
            ]
        }
    
    def get_errors_by_collector(self, collector_name: str) -> List[ErrorRecord]:
        """Отримання errors for конкретного колектора"""
        return [err for err in self.error_history if err.collector_name == collector_name]
    
    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[ErrorRecord]:
        """Отримання errors for рandвnotм серйоwithностand"""
        return [err for err in self.error_history if err.severity == severity]
    
    def get_errors_by_category(self, category: ErrorCategory) -> List[ErrorRecord]:
        """Отримання errors for категорandєю"""
        return [err for err in self.error_history if err.category == category]
    
    def get_resolution_suggestions(self, error_record: ErrorRecord) -> List[str]:
        """Отримання пропоwithицandй по вирandшенню помилки"""
        suggestions = []
        
        # Баwithовand пропоwithицandї forлежно вandд категорandї
        if error_record.category == ErrorCategory.NETWORK:
            suggestions.extend([
                "Check internet connection",
                "Verify API endpoint availability",
                "Increase timeout settings",
                "Implement retry mechanism"
            ])
        elif error_record.category == ErrorCategory.API:
            suggestions.extend([
                "Check API key validity",
                "Verify API rate limits",
                "Review API documentation",
                "Check authentication credentials"
            ])
        elif error_record.category == ErrorCategory.CONFIGURATION:
            suggestions.extend([
                "Review configuration file",
                "Validate all required parameters",
                "Check environment variables",
                "Verify file permissions"
            ])
        elif error_record.category == ErrorCategory.DATA:
            suggestions.extend([
                "Validate input data format",
                "Check data source availability",
                "Review data transformation logic",
                "Implement data validation"
            ])
        
        # Додавання специфandчних пропоwithицandй
        if error_record.error_type in self.error_patterns:
            pattern = self.error_patterns[error_record.error_type]
            suggestions.append(f"Pattern resolution: {pattern['resolution']}")
        
        return suggestions
    
    def clear_history(self):
        """Очищення andсторandї errors"""
        self.error_history.clear()
        self.error_stats = {
            "total_errors": 0,
            "resolved_errors": 0,
            "by_severity": {severity.value: 0 for severity in ErrorSeverity},
            "by_category": {category.value: 0 for category in ErrorCategory},
            "by_collector": {}
        }
        logger.info("Error history cleared")
    
    def export_errors(self, file_path: str, format: str = "json"):
        """
        Експорт errors в file
        
        Args:
            file_path: Шлях до fileу
            format: Формат fileу (json, csv)
        """
        try:
            if format.lower() == "json":
                self._export_json(file_path)
            elif format.lower() == "csv":
                self._export_csv(file_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
                
            logger.info(f"Errors exported to {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to export errors: {e}")
            raise
    
    def _export_json(self, file_path: str):
        """Експорт в JSON формат"""
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "summary": self.get_error_summary(),
            "errors": [
                {
                    "timestamp": err.timestamp.isoformat(),
                    "collector_name": err.collector_name,
                    "error_type": err.error_type,
                    "error_message": err.error_message,
                    "severity": err.severity.value,
                    "category": err.category.value,
                    "stack_trace": err.stack_trace,
                    "context": err.context,
                    "resolved": err.resolved,
                    "resolution_time": err.resolution_time.isoformat() if err.resolution_time else None,
                    "resolution_method": err.resolution_method
                }
                for err in self.error_history
            ]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    def _export_csv(self, file_path: str):
        """Експорт в CSV формат"""
        import pandas as pd
        
        data = []
        for err in self.error_history:
            data.append({
                "timestamp": err.timestamp.isoformat(),
                "collector_name": err.collector_name,
                "error_type": err.error_type,
                "error_message": err.error_message,
                "severity": err.severity.value,
                "category": err.category.value,
                "resolved": err.resolved,
                "resolution_time": err.resolution_time.isoformat() if err.resolution_time else None,
                "resolution_method": err.resolution_method
            })
        
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False, encoding='utf-8')

# Глобальний обробник errors
global_error_handler = ErrorHandler()

def handle_error(collector_name: str, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorRecord:
    """
    Глобальна функцandя for обробки errors
    
    Args:
        collector_name: Наwithва колектора
        error: Error
        context: Контекст помилки
        
    Returns:
        ErrorRecord: Запис помилки
    """
    return global_error_handler.handle_error(collector_name, error, context)

def get_error_summary() -> Dict[str, Any]:
    """Отримання глобального пandдсумку errors"""
    return global_error_handler.get_error_summary()
