from pathlib import Path
from typing import Any
from src.pipeline.hybrid.data_manager import HybridDataManager
from src.pipeline.hybrid.storage_manager import StorageManager
from src.pipeline.hybrid.data_cache_manager import DataCacheManager
from src.pipeline.hybrid.data_processor import DataProcessor
from src.pipeline.hybrid.data_utils import DataUtils

class DataComponentsContext:
    """Context for data-related components."""
    def __init__(self, config_manager: Any, output_dir: Path, models_dir: Path):
        self.data_manager = HybridDataManager(config_manager)
        self.storage_manager = StorageManager(output_dir=output_dir, models_dir=models_dir)
        self.data_cache_manager = DataCacheManager()
        self.data_utils = DataUtils()
        self.data_processor = DataProcessor(data_utils=self.data_utils)
