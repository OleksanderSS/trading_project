import tempfile
from pathlib import Path

import pytest
from src.core.exceptions import ConfigurationError, DataLoadError
from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
from src.analytics.data_managers.model_results_manager import ModelResultsManager
from unittest.mock import MagicMock

from typing import Any
from src.analytics.interfaces import IAnalyzer

class MockAnalyzer(IAnalyzer):
    def analyze(self, data: Any, **kwargs) -> Any:
        return {}

def test_unified_analytics_engine_exception():
    """Перевірка підняття ConfigurationError при відсутності даних."""
    config_manager = MagicMock()
    config_manager.get.return_value = {'analyzers': []}
    
    engine = UnifiedAnalyticsEngine(config_manager)
    # Створюємо екземпляр, що імплементує IAnalyzer
    analyzer = MockAnalyzer()
    engine.register_analyzer(analyzer, 'test_analyzer')
    engine.analyzer_data_map['test_analyzer'] = ['required_key']
    
    data_map = {'other_key': 123}
    
    with pytest.raises(ConfigurationError, match="Insufficient data for 'test_analyzer'"):
        engine._get_data_for_analyzer('test_analyzer', data_map)

def test_model_results_manager_exception():
    """Перевірка підняття DataLoadError при пошкодженому файлі."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        # Створюємо фіктивний файл, який не є коректним parquet
        corrupt_file = tmp_path / "corrupt.parquet"
        corrupt_file.write_text("not a parquet file")
        
        manager = ModelResultsManager(base_path=str(tmp_path))
        
        # Використовуємо наш підроблений шлях
        with pytest.raises(DataLoadError, match="Data corruption or I/O error"):
            manager._load_file(corrupt_file)
