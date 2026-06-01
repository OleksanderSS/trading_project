
from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager

config = UnifiedConfigManager()
dm = DataManager(config)
schema = dm.get_table_schema("newsapi_articles")
print(f"Schema for newsapi_articles: {schema}")
