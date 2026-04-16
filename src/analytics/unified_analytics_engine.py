import pandas as pd
import logging
import importlib
import hashlib
import json
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.analytics.interfaces import IAnalyzer
from src.config.unified_config_manager import UnifiedConfigManager
from src.analytics.data_managers.model_results_manager import ModelResultsManager

logger = logging.getLogger(__name__)

class UnifiedAnalyticsEngine:
    """
    Головний аналітичний рушій, що оркеструє виконання різноманітних аналітичних модулів.

    Ключові обов'язки:
    - Динамічно завантажує та реєструє аналізатори (IAnalyzer) на основі конфігурації.
    - Виконує аналізатори паралельно за допомогою пулу потоків для максимальної ефективності.
    - Керує передачею даних кожному аналізатору згідно з конфігурацією 'data_mapping'.
    - Інтегрує механізм кешування для уникнення повторних обчислень для однакових вхідних даних.
    - Зберігає та керує результатами аналізу за допомогою ModelResultsManager.
    """

    def __init__(self, config_manager: UnifiedConfigManager):
        """
        Ініціалізує рушій.

        Args:
            config_manager: Екземпляр UnifiedConfigManager для завантаження конфігурацій,
                            зокрема 'analysis.engine' та налаштувань аналізаторів.
        """
        self.config_manager = config_manager
        self.analyzers: Dict[str, IAnalyzer] = {}
        self.analyzer_data_map: Dict[str, List[str]] = {}
        self._load_config()
        
        self.results_manager = ModelResultsManager()
        
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self._register_analyzers_from_config()
        logger.info(f"UnifiedAnalyticsEngine ініціалізовано з {len(self.analyzers)} аналізаторами.")

    def _load_config(self):
        """Завантажує конфігурацію рушія з UnifiedConfigManager."""
        engine_config = self.config_manager.get('analysis.engine', {})
        self.max_workers = engine_config.get('max_workers', 4)
        self.analyzer_configs = engine_config.get('analyzers', [])
        logger.debug(f"Конфігурацію аналітичного рушія завантажено: max_workers={self.max_workers}")

    def _register_analyzers_from_config(self):
        """Динамічно імпортує та реєструє аналізатори на основі конфігурації."""
        for config in self.analyzer_configs:
            try:
                module_path = config['module']
                class_name = config['class']
                analyzer_name = config.get('name', class_name.lower())
                params = config.get('params', {})

                module = importlib.import_module(module_path)
                analyzer_class = getattr(module, class_name)
                analyzer_instance = analyzer_class(**params)

                if isinstance(analyzer_instance, IAnalyzer):
                    self.register_analyzer(analyzer_instance, name=analyzer_name)
                    self.analyzer_data_map[analyzer_name] = config.get('data_mapping', [])
                else:
                    logger.warning(f"Клас {class_name} з {module_path} не є валідним IAnalyzer.")
            except (ImportError, AttributeError, KeyError, TypeError) as e:
                logger.error(f"Помилка реєстрації аналізатора з конфігурації: {config}. Помилка: {e}", exc_info=True)

    def register_analyzer(self, analyzer: IAnalyzer, name: str):
        """Реєструє один екземпляр аналізатора."""
        if not isinstance(analyzer, IAnalyzer):
            raise TypeError("Наданий об'єкт не є валідним IAnalyzer.")
        self.analyzers[name] = analyzer
        logger.info(f"Зареєстровано аналізатор: {name}")

    def _generate_data_hash(self, data_map: Dict[str, Any]) -> str:
        """
        Генерує стабільний хеш для вхідних даних, що використовується як ключ для кешу.
        """
        try:
            stable_repr = {}
            for key in sorted(data_map.keys()):
                value = data_map[key]
                if isinstance(value, pd.DataFrame):
                    sample = value.head(10).tail(5)
                    stable_repr[key] = {
                        'shape': value.shape,
                        'columns': list(value.columns),
                        'sample_hash': hashlib.md5(sample.to_json(date_format='iso', orient='split').encode()).hexdigest()
                    }
                elif isinstance(value, pd.Series):
                    sample = value.head(10)
                    stable_repr[key] = {
                        'shape': value.shape,
                        'name': value.name,
                        'sample_hash': hashlib.md5(sample.to_json(date_format='iso', orient='split').encode()).hexdigest()
                    }
                else:
                    stable_repr[key] = str(value)
            deterministic_json = json.dumps(stable_repr, sort_keys=True)
            return hashlib.md5(deterministic_json.encode()).hexdigest()
        except Exception as e:
            hash_input = ""
            for key, value in sorted(data_map.items()):
                if isinstance(value, pd.DataFrame):
                    sample = value.head(3)
                    hash_input += f"{key}_{value.shape}_{hash(sample.to_json(date_format='iso', orient='split'))}"
                elif isinstance(value, pd.Series):
                    sample = value.head(3)
                    hash_input += f"{key}_{value.shape}_{hash(sample.to_json(date_format='iso', orient='split'))}"
                else:
                    hash_input += f"{key}_{str(value)}"
            return hashlib.md5(hash_input.encode()).hexdigest()

    def run_full_analysis(self, data_map: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Запускає всі зареєстровані аналізатори паралельно.

        Процес:
        1. Генерує хеш вхідних даних.
        2. Перевіряє, чи є результат у кеші. Якщо так, повертає його.
        3. Якщо ні, запускає аналізатори в окремих потоках.
        4. Кожному аналізатору передається тільки той зріз даних, який вказано в його 'data_mapping'.
        5. Збирає результати, кешує їх та повертає.

        Args:
            data_map: Словник, де ключі - це ідентифікатори даних (напр., 'price_data'),
                      а значення - самі дані (напр., pd.DataFrame).
            **kwargs: Додаткові параметри, що будуть передані в метод `analyze` кожного аналізатора.

        Returns:
            Словник, що містить результати від усіх аналізаторів.
        """
        data_hash = self._generate_data_hash(data_map)
        
        cached_results = self.results_manager.get_cached_analysis(data_hash)
        if cached_results:
            logger.info("Результати аналізу отримано з кешу.")
            return cached_results

        logger.info(f"Запуск повного паралельного аналізу з {len(self.analyzers)} аналізаторами.")
        futures = {}
        for name, analyzer in self.analyzers.items():
            input_data = self._get_data_for_analyzer(name, data_map)
            if input_data is not None:
                # Передаємо input_data як перший аргумент в analyze
                futures[name] = self.thread_pool.submit(analyzer.analyze, input_data, **kwargs)
            else:
                logger.warning(f"Пропуск аналізатора '{name}', оскільки необхідні дані відсутні.")

        results = {}
        for name, future in futures.items():
            try:
                results[name] = future.result(timeout=120)  # 120-секундний таймаут
            except Exception as e:
                logger.error(f"Паралельний аналіз для '{name}' зазнав невдачі: {e}", exc_info=True)
                results[name] = {"error": str(e)}
        
        self.results_manager.cache_analysis(data_hash, results)
        
        return results

    def _get_data_for_analyzer(self, analyzer_name: str, data_map: Dict[str, Any]) -> Optional[Any]:
        """
        Формує вхідні дані для конкретного аналізатора на основі його 'data_mapping'.
        Якщо вказано один ключ - повертає дані напряму.
        Якщо декілька - повертає словник з потрібними даними.
        """
        required_keys = self.analyzer_data_map.get(analyzer_name, [])
        if not required_keys:
            logger.warning(f"Для аналізатора '{analyzer_name}' не вказано 'data_mapping'. Пропуск.")
            return None

        if not all(key in data_map for key in required_keys):
            logger.warning(f"Відсутні дані для аналізатора '{analyzer_name}'. Потрібно: {required_keys}, Доступно: {list(data_map.keys())}")
            return None
        
        if len(required_keys) == 1:
            return data_map[required_keys[0]]
        
        return {key: data_map[key] for key in required_keys}

    def get_contextual_report(self) -> Dict[str, Any]:
        """
        Збирає та повертає звіт про поточний стан та конфігурацію рушія.
        """
        report = {
            "engine_status": "active",
            "registered_analyzers": list(self.analyzers.keys()),
            "max_workers": self.max_workers,
            "analyzer_configurations": self.analyzer_configs,
            "data_flow_map": self.analyzer_data_map
        }
        return report

    def get_registered_components(self) -> Dict[str, List[str]]:
        """Повертає словник із зареєстрованими компонентами."""
        return {'analyzers': list(self.analyzers.keys())}
