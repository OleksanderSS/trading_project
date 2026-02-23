# config/config_legacy.py - LEGACY CONFIG (DEPRECATED)
# 
# [WARN] ЦЕЙ ФАЙЛ ЗАСТАРІВ! 
# Використовуйте config/unified_config.py для всіх нових розробок
#
# Цей файл залишається для зворотної сумісності

# Імпортуємо все з unified_config для зворотної сумісності
from config.unified_config import (
    config, TIME_FRAMES, LEGACY_TIME_FRAMES, YF_MAX_PERIODS, DATA_INTERVALS,
    UnifiedConfig
)

# Для зворотної сумісності
USE_CORE_FEATURES = True

# --- Дandапаwithони дат ---
def get_date_range(api_name: str):
    """Отримати діапазон дат для API"""
    from datetime import datetime, timedelta
    
    ranges = {
        "news": timedelta(days=7),
        "reddit": timedelta(days=30),
        "twitter": timedelta(days=7),
        "stock": timedelta(days=365),
        "crypto": timedelta(days=365)
    }
    
    end_date = datetime.now()
    start_date = end_date - ranges.get(api_name, timedelta(days=30))
    
    return start_date, end_date

# --- Попередження про застарілий файл ---
import warnings
warnings.warn(
    "config_legacy.py is deprecated. Use config.unified_config instead.",
    DeprecationWarning,
    stacklevel=2
)
