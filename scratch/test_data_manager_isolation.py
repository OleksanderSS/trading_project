import sys
from unittest.mock import MagicMock

# Додаємо шлях до проекту
sys.path.append("D:\\trading_project")

# Mocking external dependencies for isolated test
mock_config = MagicMock()
mock_config.get.return_value = ":memory:"  # Use in-memory DB for test

try:
    from src.config.unified_config_manager import UnifiedConfigManager
    from src.data.management.data_manager import DataManager
    
    # Init config with mock
    cm = UnifiedConfigManager()
    cm.get = MagicMock(return_value=":memory:")
    
    print("Testing DataManager initialization...")
    dm = DataManager(cm)
    
    print("Testing table existence check...")
    exists = dm.table_exists("non_existent_table")
    print(f"Table exists check passed (result: {exists})")
    
    print("Testing connection status...")
    if dm.con:
        print("Connection established successfully.")
    
    print("DataManager validation passed.")
    sys.exit(0)

except Exception as e:
    print(f"DataManager validation FAILED: {e}")
    sys.exit(1)
