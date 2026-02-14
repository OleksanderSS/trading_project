"""
Config Validator - Валandдацandя конфandгурацandй system
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Рandвнand валandдацandї"""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationResult:
    """Реwithульandт валandдацandї"""
    level: ValidationLevel
    message: str
    config_path: Optional[str] = None
    field_path: Optional[str] = None
    suggestion: Optional[str] = None


class ConfigValidator:
    """
    Валandдатор конфandгурацandй system
    """
    
    def __init__(self):
        self.results: List[ValidationResult] = []
        self.config_dir = Path("config")
        
        # Виvalues схем валandдацandї
        self.schemas = self._define_schemas()
    
    def _define_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Виvalues схем валandдацandї"""
        return {
            "collectors_config": {
                "required_fields": ["global", "collectors"],
                "global_schema": {
                    "log_level": {"type": str, "choices": ["DEBUG", "INFO", "WARNING", "ERROR"]},
                    "max_concurrent_collectors": {"type": int, "min": 1, "max": 20},
                    "default_timeout": {"type": int, "min": 5, "max": 300},
                    "cache_dir": {"type": str, "required": True}
                },
                "collector_schema": {
                    "name": {"type": str, "required": True},
                    "type": {"type": str, "required": True, "choices": ["news", "economic", "financial", "events", "ai", "social", "sentiment", "data"]},
                    "enabled": {"type": bool},
                    "max_retries": {"type": int, "min": 0, "max": 10},
                    "timeout": {"type": int, "min": 5, "max": 600},
                    "rate_limit": {"type": int, "min": 1, "max": 10000},
                    "cache_ttl": {"type": int, "min": 0},
                    "batch_size": {"type": int, "min": 1, "max": 10000},
                    "validate_data": {"type": bool}
                }
            },
            "google_key": {
                "required_fields": ["type", "project_id", "private_key", "client_email"],
                "schema": {
                    "type": {"type": str, "required": True, "value": "service_account"},
                    "project_id": {"type": str, "required": True},
                    "private_key": {"type": str, "required": True, "pattern": "-----BEGIN PRIVATE KEY-----"},
                    "client_email": {"type": str, "required": True, "pattern": "@"}
                }
            },
            "thresholds": {
                "required_fields": [],
                "schema": {
                    "confidence": {"type": dict, "keys": ["high", "medium", "low"]},
                    "risk": {"type": dict, "keys": ["high", "medium", "low"]},
                    "gdelt": {"type": dict, "keys": ["min_mentions", "max_events_per_day"]}
                }
            },
            "news_sources": {
                "required_fields": ["sources"],
                "schema": {
                    "sources": {"type": dict, "min_items": 1}
                },
                "source_schema": {
                    "url": {"type": str, "required": True, "pattern": "^https?://"},
                    "priority": {"type": int, "min": 1, "max": 10},
                    "reliability": {"type": float, "min": 0.0, "max": 1.0}
                }
            }
        }
    
    def validate_all_configs(self) -> List[ValidationResult]:
        """
        Валandдацandя allх конфandгурацandйних fileandв
        
        Returns:
            List[ValidationResult]: Реwithульandти валandдацandї
        """
        self.results.clear()
        
        # Валandдацandя основних конфandгурацandй
        self._validate_collectors_config()
        self._validate_google_key()
        self._validate_thresholds()
        self._validate_news_sources()
        
        # Валandдацandя andснування fileandв
        self._validate_required_files()
        
        # Валandдацandя середовища
        self._validate_environment()
        
        # Валandдацandя логandчної консистентностand
        self._validate_logical_consistency()
        
        return self.results
    
    def _validate_collectors_config(self):
        """Валandдацandя конфandгурацandї колекторandв"""
        config_files = [
            "collectors/collectors_config.json",
            "unified_collectors_config.yaml"
        ]
        
        for config_file in config_files:
            file_path = self.config_dir / config_file
            if not file_path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Configuration file not found: {config_file}",
                    config_path=str(file_path),
                    suggestion="Create the configuration file or use default settings"
                ))
                continue
            
            try:
                # Заванandження конфandгурацandї
                with open(file_path, 'r', encoding='utf-8') as f:
                    if file_path.suffix.lower() == '.yaml':
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Валandдацandя структури
                self._validate_schema(config_data, self.schemas["collectors_config"], str(file_path))
                
                # Валandдацandя колекторandв
                if "collectors" in config_data:
                    self._validate_collectors(config_data["collectors"], str(file_path))
                
            except Exception as e:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Error loading configuration: {e}",
                    config_path=str(file_path)
                ))
    
    def _validate_collectors(self, collectors: Dict[str, Any], config_path: str):
        """Валandдацandя конфandгурацandй колекторandв"""
        schema = self.schemas["collectors_config"]["collector_schema"]
        
        for name, collector_config in collectors.items():
            # Валandдацandя обов'яwithкових полandв
            for field, rules in schema.items():
                if rules.get("required", False) and field not in collector_config:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Required field '{field}' missing in collector '{name}'",
                        config_path=config_path,
                        field_path=f"collectors.{name}.{field}"
                    ))
                
                # Валandдацandя типandв and withначень
                if field in collector_config:
                    self._validate_field(
                        collector_config[field], rules, 
                        f"collectors.{name}.{field}", config_path
                    )
            
            # Специфandчна валandдацandя for типandв колекторandв
            collector_type = collector_config.get("type")
            if collector_type:
                self._validate_collector_type_specific(name, collector_config, collector_type, config_path)
    
    def _validate_collector_type_specific(self, name: str, config: Dict[str, Any], collector_type: str, config_path: str):
        """Специфandчна валandдацandя for типandв колекторandв"""
        additional_params = config.get("additional_params", {})
        
        if collector_type == "gdelt":
            # Валandдацandя GDELT конфandгурацandї
            if "service_account_path" in additional_params:
                service_path = Path(additional_params["service_account_path"])
                if not service_path.exists():
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Service account file not found: {service_path}",
                        config_path=config_path,
                        field_path=f"collectors.{name}.additional_params.service_account_path",
                        suggestion="Ensure the Google service account file exists"
                    ))
            
            # Валandдацandя економandчних codeandв
            if "economic_event_codes" in additional_params:
                event_codes = additional_params["economic_event_codes"]
                if not isinstance(event_codes, list) or len(event_codes) == 0:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message=f"Invalid economic_event_codes in collector '{name}'",
                        config_path=config_path,
                        field_path=f"collectors.{name}.additional_params.economic_event_codes"
                    ))
        
        elif collector_type == "newsapi":
            # Валandдацandя NewsAPI конфandгурацandї
            api_key_env = additional_params.get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Environment variable not set: {api_key_env}",
                    config_path=config_path,
                    field_path=f"collectors.{name}.additional_params.api_key_env",
                    suggestion=f"Set the {api_key_env} environment variable"
                ))
        
        elif collector_type == "fred":
            # Валandдацandя FRED конфandгурацandї
            api_key_env = additional_params.get("api_key_env")
            if api_key_env and not os.getenv(api_key_env):
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Environment variable not set: {api_key_env}",
                    config_path=config_path,
                    field_path=f"collectors.{name}.additional_params.api_key_env",
                    suggestion=f"Set the {api_key_env} environment variable"
                ))
            
            # Валandдацandя andндикаторandв
            indicators = additional_params.get("default_indicators", [])
            if not isinstance(indicators, list) or len(indicators) == 0:
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"No default indicators configured for FRED collector",
                    config_path=config_path,
                    field_path=f"collectors.{name}.additional_params.default_indicators"
                ))
    
    def _validate_google_key(self):
        """Валandдацandя Google service account ключа"""
        key_file = self.config_dir / "google_key.json"
        
        if not key_file.exists():
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="Google service account key file not found",
                config_path=str(key_file),
                suggestion="Create google_key.json with service account credentials"
            ))
            return
        
        try:
            with open(key_file, 'r', encoding='utf-8') as f:
                key_data = json.load(f)
            
            # Валandдацandя структури
            self._validate_schema(key_data, self.schemas["google_key"], str(key_file))
            
            # Валandдацandя формату ключа
            private_key = key_data.get("private_key", "")
            if not private_key.startswith("-----BEGIN PRIVATE KEY-----"):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message="Invalid private key format",
                    config_path=str(key_file),
                    field_path="private_key"
                ))
            
            # Валandдацandя email
            client_email = key_data.get("client_email", "")
            if "@" not in client_email or ".iam.gserviceaccount.com" not in client_email:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message="Invalid client email format",
                    config_path=str(key_file),
                    field_path="client_email"
                ))
                
        except json.JSONDecodeError as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Invalid JSON format: {e}",
                config_path=str(key_file)
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Error reading Google key file: {e}",
                config_path=str(key_file)
            ))
    
    def _validate_thresholds(self):
        """Валandдацandя порогandв"""
        thresholds_file = self.config_dir / "thresholds.py"
        
        if not thresholds_file.exists():
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Thresholds file not found",
                config_path=str(thresholds_file),
                suggestion="Create thresholds.py with system thresholds"
            ))
            return
        
        try:
            # Спроба andмпортувати thresholds
            import sys
            sys.path.insert(0, str(self.config_dir))
            
            try:
                import thresholds
                # Валandдацandя наявностand основних порогandв
                if hasattr(thresholds, 'THRESHOLDS'):
                    self._validate_thresholds_data(thresholds.THRESHOLDS, str(thresholds_file))
                else:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message="THRESHOLDS constant not found in thresholds.py",
                        config_path=str(thresholds_file)
                    ))
            finally:
                sys.path.remove(str(self.config_dir))
                
        except ImportError as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Error importing thresholds: {e}",
                config_path=str(thresholds_file)
            ))
        except Exception as e:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Error validating thresholds: {e}",
                config_path=str(thresholds_file)
            ))
    
    def _validate_thresholds_data(self, thresholds_data: Dict[str, Any], config_path: str):
        """Валandдацandя data порогandв"""
        if not isinstance(thresholds_data, dict):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="THRESHOLDS must be a dictionary",
                config_path=config_path
            ))
            return
        
        # Валandдацandя порогandв впевnotностand
        if "confidence" in thresholds_data:
            confidence = thresholds_data["confidence"]
            if not isinstance(confidence, dict):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message="confidence thresholds must be a dictionary",
                    config_path=config_path,
                    field_path="confidence"
                ))
            else:
                for level, value in confidence.items():
                    if not isinstance(value, (int, float)) or not (0 <= value <= 1):
                        self.results.append(ValidationResult(
                            level=ValidationLevel.ERROR,
                            message=f"Invalid confidence threshold for {level}: {value}",
                            config_path=config_path,
                            field_path=f"confidence.{level}",
                            suggestion="Confidence thresholds must be between 0 and 1"
                        ))
    
    def _validate_news_sources(self):
        """Валandдацandя джерел новин"""
        news_files = [
            "news_sources.yaml",
            "source_quality.yaml"
        ]
        
        for news_file in news_files:
            file_path = self.config_dir / news_file
            
            if not file_path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"News sources file not found: {news_file}",
                    config_path=str(file_path)
                ))
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    sources_data = yaml.safe_load(f)
                
                # Валandдацandя структури
                if "sources" not in sources_data:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message="sources field not found",
                        config_path=str(file_path)
                    ))
                    continue
                
                # Валandдацandя джерел
                sources = sources_data["sources"]
                if not isinstance(sources, dict) or len(sources) == 0:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.WARNING,
                        message="No sources configured",
                        config_path=str(file_path),
                        field_path="sources"
                    ))
                    continue
                
                for source_name, source_config in sources.items():
                    self._validate_news_source(source_name, source_config, str(file_path))
                    
            except yaml.YAMLError as e:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Invalid YAML format: {e}",
                    config_path=str(file_path)
                ))
            except Exception as e:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Error reading news sources: {e}",
                    config_path=str(file_path)
                ))
    
    def _validate_news_source(self, source_name: str, source_config: Dict[str, Any], config_path: str):
        """Валandдацandя джерела новин"""
        # Валandдацandя URL
        url = source_config.get("url", "")
        if not url or not isinstance(url, str):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Invalid URL for source '{source_name}'",
                config_path=config_path,
                field_path=f"sources.{source_name}.url"
            ))
        elif not (url.startswith("http://") or url.startswith("https://")):
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message=f"URL should start with http:// or https:// for source '{source_name}'",
                config_path=config_path,
                field_path=f"sources.{source_name}.url"
            ))
        
        # Валandдацandя прandоритету
        priority = source_config.get("priority")
        if priority is not None:
            if not isinstance(priority, int) or not (1 <= priority <= 10):
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Invalid priority for source '{source_name}': {priority}",
                    config_path=config_path,
                    field_path=f"sources.{source_name}.priority",
                    suggestion="Priority should be an integer between 1 and 10"
                ))
        
        # Валandдацandя надandйностand
        reliability = source_config.get("reliability")
        if reliability is not None:
            if not isinstance(reliability, (int, float)) or not (0 <= reliability <= 1):
                self.results.append(ValidationResult(
                    level=ValidationLevel.WARNING,
                    message=f"Invalid reliability for source '{source_name}': {reliability}",
                    config_path=config_path,
                    field_path=f"sources.{source_name}.reliability",
                    suggestion="Reliability should be a float between 0 and 1"
                ))
    
    def _validate_required_files(self):
        """Валandдацandя наявностand обов'яwithкових fileandв"""
        required_files = [
            "__init__.py",
            "config.py",
            "secrets_manager.py",
            "logger_config.py"
        ]
        
        for file_name in required_files:
            file_path = self.config_dir / file_name
            if not file_path.exists():
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Required file not found: {file_name}",
                    config_path=str(file_path)
                ))
    
    def _validate_environment(self):
        """Валandдацandя середовища"""
        # Перевandрка Python шляхandв
        if str(self.config_dir) not in os.sys.path:
            self.results.append(ValidationResult(
                level=ValidationLevel.INFO,
                message="Config directory not in Python path",
                suggestion=f"Add '{self.config_dir}' to PYTHONPATH"
            ))
        
        # Перевandрка прав доступу
        if not os.access(self.config_dir, os.R_OK):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message="No read access to config directory",
                config_path=str(self.config_dir)
            ))
        
        # Перевandрка наявностand директорandї кешу
        cache_dir = Path("cache")
        if not cache_dir.exists():
            self.results.append(ValidationResult(
                level=ValidationLevel.WARNING,
                message="Cache directory not found",
                config_path=str(cache_dir),
                suggestion="Create cache directory for better performance"
            ))
    
    def _validate_logical_consistency(self):
        """Валandдацandя логandчної консистентностand"""
        # Перевandрка дублювання andмен колекторandв
        collector_names = set()
        for config_file in ["collectors/collectors_config.json", "unified_collectors_config.yaml"]:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.suffix.lower() == '.yaml':
                            config_data = yaml.safe_load(f)
                        else:
                            config_data = json.load(f)
                    
                    if "collectors" in config_data:
                        for name in config_data["collectors"].keys():
                            if name in collector_names:
                                self.results.append(ValidationResult(
                                    level=ValidationLevel.WARNING,
                                    message=f"Duplicate collector name: {name}",
                                    config_path=str(file_path),
                                    suggestion="Ensure collector names are unique across all config files"
                                ))
                            collector_names.add(name)
                            
                except Exception:
                    continue
        
        # Перевandрка уwithгодженостand типandв колекторandв
        valid_types = ["news", "economic", "financial", "events", "ai", "social", "sentiment", "data"]
        for config_file in ["collectors/collectors_config.json", "unified_collectors_config.yaml"]:
            file_path = self.config_dir / config_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        if file_path.suffix.lower() == '.yaml':
                            config_data = yaml.safe_load(f)
                        else:
                            config_data = json.load(f)
                    
                    if "collectors" in config_data:
                        for name, config in config_data["collectors"].items():
                            collector_type = config.get("type")
                            if collector_type and collector_type not in valid_types:
                                self.results.append(ValidationResult(
                                    level=ValidationLevel.WARNING,
                                    message=f"Unknown collector type: {collector_type}",
                                    config_path=str(file_path),
                                    field_path=f"collectors.{name}.type",
                                    suggestion=f"Use one of: {', '.join(valid_types)}"
                                ))
                                
                except Exception:
                    continue
    
    def _validate_schema(self, data: Dict[str, Any], schema: Dict[str, Any], config_path: str):
        """Валandдацandя data for схемою"""
        # Перевandрка обов'яwithкових полandв
        for field in schema.get("required_fields", []):
            if field not in data:
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Required field '{field}' missing",
                    config_path=config_path,
                    field_path=field
                ))
        
        # Валandдацandя полandв for схемою
        for field_name, field_schema in schema.items():
            if field_name in data and isinstance(field_schema, dict):
                self._validate_field(
                    data[field_name], field_schema, 
                    field_name, config_path
                )
    
    def _validate_field(self, value: Any, rules: Dict[str, Any], field_path: str, config_path: str):
        """Валandдацandя поля for правилами"""
        # Перевandрка типу
        expected_type = rules.get("type")
        if expected_type and not isinstance(value, expected_type):
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Invalid type for {field_path}: expected {expected_type.__name__}, got {type(value).__name__}",
                config_path=config_path,
                field_path=field_path
            ))
            return
        
        # Перевandрка вибору
        choices = rules.get("choices")
        if choices and value not in choices:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Invalid value for {field_path}: {value}. Must be one of: {choices}",
                config_path=config_path,
                field_path=field_path
            ))
        
        # Перевandрка мandнandмуму
        min_val = rules.get("min")
        if min_val is not None and value < min_val:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Value for {field_path} is below minimum: {value} < {min_val}",
                config_path=config_path,
                field_path=field_path
            ))
        
        # Перевandрка максимуму
        max_val = rules.get("max")
        if max_val is not None and value > max_val:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Value for {field_path} is above maximum: {value} > {max_val}",
                config_path=config_path,
                field_path=field_path
            ))
        
        # Перевandрка патерну
        pattern = rules.get("pattern")
        if pattern and isinstance(value, str):
            import re
            if not re.search(pattern, value):
                self.results.append(ValidationResult(
                    level=ValidationLevel.ERROR,
                    message=f"Value for {field_path} doesn't match pattern: {pattern}",
                    config_path=config_path,
                    field_path=field_path
                ))
        
        # Перевandрка мandнandмальної кandлькостand елементandв
        min_items = rules.get("min_items")
        if min_items is not None and isinstance(value, (list, dict)) and len(value) < min_items:
            self.results.append(ValidationResult(
                level=ValidationLevel.ERROR,
                message=f"Field {field_path} has too few items: {len(value)} < {min_items}",
                config_path=config_path,
                field_path=field_path
            ))
        
        # Перевandрка ключandв
        keys = rules.get("keys")
        if keys and isinstance(value, dict):
            for key in keys:
                if key not in value:
                    self.results.append(ValidationResult(
                        level=ValidationLevel.ERROR,
                        message=f"Required key '{key}' missing in {field_path}",
                        config_path=config_path,
                        field_path=f"{field_path}.{key}"
                    ))
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """
        Отримання пandдсумку валandдацandї
        
        Returns:
            Dict[str, Any]: Пandдсумок валandдацandї
        """
        errors = [r for r in self.results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in self.results if r.level == ValidationLevel.WARNING]
        info = [r for r in self.results if r.level == ValidationLevel.INFO]
        
        return {
            "total_results": len(self.results),
            "errors": len(errors),
            "warnings": len(warnings),
            "info": len(info),
            "is_valid": len(errors) == 0,
            "error_details": [{"message": r.message, "config_path": r.config_path, "field_path": r.field_path} for r in errors],
            "warning_details": [{"message": r.message, "config_path": r.config_path, "field_path": r.field_path} for r in warnings],
            "info_details": [{"message": r.message, "config_path": r.config_path, "field_path": r.field_path} for r in info]
        }
    
    def print_results(self):
        """Вивеwhereння реwithульandтandв валandдацandї"""
        summary = self.get_validation_summary()
        
        print(f"\n[SEARCH] CONFIG VALIDATION RESULTS")
        print(f"{'='*50}")
        print(f"Total: {summary['total_results']} | Errors: {summary['errors']} | Warnings: {summary['warnings']} | Info: {summary['info']}")
        print(f"Status: {'[OK] VALID' if summary['is_valid'] else '[ERROR] INVALID'}")
        print(f"{'='*50}")
        
        if summary['errors'] > 0:
            print(f"\n ERRORS ({summary['errors']}):")
            for error in summary['error_details']:
                print(f"  [ERROR] {error['message']}")
                if error['config_path']:
                    print(f"      {error['config_path']}")
                if error['field_path']:
                    print(f"     [TOOL] {error['field_path']}")
        
        if summary['warnings'] > 0:
            print(f"\n[WARN]  WARNINGS ({summary['warnings']}):")
            for warning in summary['warning_details']:
                print(f"  [WARN]  {warning['message']}")
                if warning['config_path']:
                    print(f"      {warning['config_path']}")
                if warning['field_path']:
                    print(f"     [TOOL] {warning['field_path']}")
        
        if summary['info'] > 0:
            print(f"\n  INFO ({summary['info']}):")
            for info_item in summary['info_details']:
                print(f"    {info_item['message']}")
                if info_item['config_path']:
                    print(f"      {info_item['config_path']}")
        
        print(f"\n{'='*50}")


# Глобальнand функцandї for withручностand
def validate_configs() -> List[ValidationResult]:
    """
    Валandдацandя allх конфandгурацandй
    
    Returns:
        List[ValidationResult]: Реwithульandти валandдацandї
    """
    validator = ConfigValidator()
    return validator.validate_all_configs()


def validate_and_print():
    """Валandдацandя and вивеwhereння реwithульandтandв"""
    validator = ConfigValidator()
    validator.validate_all_configs()
    validator.print_results()
    return validator.get_validation_summary()


if __name__ == "__main__":
    validate_and_print()
