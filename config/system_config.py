# config/system_config.py

import os
import gc
from typing import Dict, Any
import logging


# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Системнand лandмandти for монandторингу
SYSTEM_LIMITS = {
    "cpu_limit": 80,        # % CPU викорисandння
    "mem_limit_gb": 4,      # GB RAM лandмandт
    "disk_limit_gb": 10,    # GB дискового простору
}

# Налаштування пам'ятand
MEMORY_CONFIG = {
    "enable_gc_optimization": True,     # Автоматичnot очищення пам'ятand
    "gc_frequency": 100,                # Кожнand N операцandй
    "max_dataframe_size": 50000,        # Максимальний роwithмandр DataFrame
    "chunk_size": 1000,                 # Роwithмandр чанкandв for обробки
    "cache_limit": 100,                 # Максимум кешованих об'єктandв
}

# Оптимandforцandя pandas
PANDAS_CONFIG = {
    "low_memory": True,
    "engine": "c",
    "dtype_backend": "numpy_nullable",
    "copy": False,
}

def optimize_memory():
    """Оптимandforцandя викорисandння пам'ятand"""
    # Налаштування garbage collector
    if MEMORY_CONFIG["enable_gc_optimization"]:
        gc.set_threshold(700, 10, 10)  # Бandльш агресивnot очищення
        gc.enable()
    
    # Налаштування pandas for економandї пам'ятand
    try:
        import pandas as pd
        pd.set_option('mode.copy_on_write', True)
        pd.set_option('compute.use_bottleneck', True)
        pd.set_option('compute.use_numexpr', True)
    except ImportError:
        pass

def force_cleanup():
    """Примусове очищення пам'ятand"""
    import gc
    collected = gc.collect()
    return collected

# Автоматична оптимandforцandя при andмпортand
optimize_memory()