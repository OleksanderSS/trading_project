import json
import pickle
import time
import hashlib
import threading
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, Union, Optional, Callable
from functools import wraps
from src.data.management.data_manager import DataManager
from src.config.unified_config_manager import get_current_config, UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

class CacheManager:
    """Централізоване кешування з підтримкою DuckDB метаданих, стиснення та просторів імен."""

    def __init__(self, cache_dir: Union[str, Path] = None, data_manager: Optional[DataManager] = None, config_manager: Optional[UnifiedConfigManager] = None):
        self.config = get_current_config()
        self.logger = ProjectLogger.get_logger("CacheManager")
        
        # Використовуємо system.temp_path з конфігурації як базу для cache_dir
        temp_path = self.config.get('system.temp_path', 'cache')
        self.cache_dir = Path(cache_dir) if cache_dir else Path(temp_path) / "unified_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.Lock()
        
        # Integration with DataManager for Meta-Storage
        if data_manager is None:
            if config_manager:
                self.db = DataManager(config_manager)
            else:
                self.db = None
        else:
            self.db = data_manager
        self._init_db()
        
        # Salt based on table states
        self.db_salt = self._get_db_salt() if self.db else None

    def _get_db_salt(self) -> str:
        """
        Генерує "сіль" на основі стану ключових таблиць, а не часу модифікації файлу.
        Це робить кеш більш стабільним, коли до бази даних додаються нові дані,
        але існуючі дані не змінюються.
        """
        try:
            # Використовуємо список таблиць, які відслідковуємо для змін
            tracked_tables = self.config.get('cache.tracked_tables', ['news', 'market_data'])
            
            table_states = []
            
            for table_name in tracked_tables:
                if self.db.table_exists(table_name):
                    # Отримуємо кількість записів та останню дату оновлення
                    count_query = f"SELECT COUNT(*) as count FROM {table_name}"
                    count_result = self.db.fetch_one(count_query)
                    count = count_result['count'] if count_result else 0
                    
                    # Спробуємо знайти колонку з датою/часом
                    schema = self.db.get_table_schema(table_name)
                    date_col = next((col for col in ['created_at', 'timestamp', 'date'] if col in schema), None)
                    
                    max_date = 'no_date_col'
                    if date_col:
                        date_query = f"SELECT MAX(\"{date_col}\") as max_date FROM {table_name}"
                        date_result = self.db.fetch_one(date_query)
                        max_date = str(date_result['max_date']) if date_result and date_result['max_date'] is not None else 'null'

                    table_states.append(f"{table_name}:{count}:{max_date}")
                else:
                    table_states.append(f"{table_name}:missing")

            state_string = "_".join(table_states)
            self.logger.info(f"Generated DB salt based on table states: {state_string}")
            return hashlib.sha256(state_string.encode()).hexdigest()

        except Exception as e:
            self.logger.warning(f"Could not generate DB salt from table states, falling back to static salt: {e}")
            return "static_fallback_salt"


    def _init_db(self):
        """Ініціалізація таблиці метаданих у DuckDB."""
        if not self.db:
            self.logger.warning("Database not initialized. Skipping cache metadata table creation.")
            return
        
        query = """
        CREATE TABLE IF NOT EXISTS cache_metadata (
            key_hash VARCHAR PRIMARY KEY,
            original_key VARCHAR,
            namespace VARCHAR,
            timestamp DOUBLE,
            ttl INTEGER,
            size_bytes BIGINT
        )
        """
        self.db.execute_query(query)

    def _get_cache_key(self, key: str, params: Any = None, namespace: str = "default", use_salt: bool = True) -> str:
        """Генерація стабільного ключа кешу за допомогою SHA-256 з додаванням DB salt."""
        actual_salt = self.db_salt if (use_salt and namespace != "collectors") else ""
        if params:
            param_str = json.dumps(params, sort_keys=True, default=str)
            full_key = f"{key}_{param_str}_{actual_salt}"
        else:
            full_key = f"{key}_{actual_salt}"
        return hashlib.sha256(full_key.encode()).hexdigest()

    def get(self, key: str, params: Any = None, namespace: str = "default", use_salt: bool = True) -> Any:
        """Отримання даних з кешу (Пам'ять -> Parquet/Pickle з валідацією через DuckDB)."""
        cache_key = self._get_cache_key(key, params, namespace=namespace, use_salt=use_salt)

        with self.lock:
            # 1. Перевірка в пам'яті
            if cache_key in self.memory_cache:
                cache_data = self.memory_cache[cache_key]
                if self._is_cache_valid(cache_data):
                    return cache_data['value']
                else:
                    del self.memory_cache[cache_key]

        # 2. Перевірка через DuckDB метадані
        query = f"SELECT timestamp, ttl FROM cache_metadata WHERE key_hash = '{cache_key}'"
        results = self.db.fetch_all(query)
        meta_df = pd.DataFrame(results, columns=['timestamp', 'ttl'])
        
        if meta_df.empty:
            return None

        meta = meta_df.iloc[0].to_dict()
        if not self._is_cache_valid(meta):
            self._delete_cache_entry(cache_key)
            return None

        # 3. Завантаження з файлової системи
        pq_file = self.cache_dir / f"{cache_key}.parquet"
        pkl_file = self.cache_dir / f"{cache_key}.pkl"

        try:
            value = None
            if pq_file.exists():
                value = pd.read_parquet(pq_file)
            elif pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    value = pickle.load(f)

            if value is not None:
                with self.lock:
                    self.memory_cache[cache_key] = {'value': value, 'timestamp': meta['timestamp'], 'ttl': meta['ttl']}
                return value

        except Exception as e:
            self.logger.warning(f"Failed to load cache file for {key}: {e}")
        
        return None

    def set(self, key: str, value: Any, params: Any = None, ttl: int = 3600, namespace: str = "default", use_salt: bool = True) -> None:
        """Збереження даних (DataFrames -> Parquet з zstd, Others -> Pickle)."""
        cache_key = self._get_cache_key(key, params, namespace=namespace, use_salt=use_salt)
        timestamp = time.time()
        
        # Оцінка розміру
        import sys
        size_bytes = sys.getsizeof(value)
        if hasattr(value, 'memory_usage'): # For DataFrames
            size_bytes = value.memory_usage(deep=True).sum()
            
        if size_bytes > 100 * 1024 * 1024: # 100MB Warning
            self.logger.warning(f"Caching large object (>100MB): {key} in namespace {namespace}. Size: {size_bytes / (1024*1024):.2f} MB")

        with self.lock:
            self.memory_cache[cache_key] = {'value': value, 'timestamp': timestamp, 'ttl': ttl}

        try:
            # Збереження контенту
            if isinstance(value, pd.DataFrame):
                value.to_parquet(self.cache_dir / f"{cache_key}.parquet", compression='zstd')
                self._remove_file_if_exists(self.cache_dir / f"{cache_key}.pkl")
            else:
                with open(self.cache_dir / f"{cache_key}.pkl", 'wb') as f:
                    pickle.dump(value, f)
                self._remove_file_if_exists(self.cache_dir / f"{cache_key}.parquet")

            # Оновлення метаданих у DuckDB
            meta_df = pd.DataFrame([{
                'key_hash': cache_key,
                'original_key': key,
                'namespace': namespace,
                'timestamp': timestamp,
                'ttl': ttl,
                'size_bytes': size_bytes
            }])
            self.db.upsert('cache_metadata', meta_df, unique_on=['key_hash'])

        except Exception as e:
            self.logger.error(f"Failed to save cache for {key}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Повертає статистику кешу: загальний розмір, кількість об'єктів у пам'яті та на диску."""
        try:
            query = "SELECT COUNT(*) as count, SUM(size_bytes) as total_size FROM cache_metadata"
            results = self.db.fetch_all(query)
            db_stats = pd.DataFrame(results, columns=['count', 'total_size'])
            
            disk_count = db_stats.iloc[0]['count'] if not db_stats.empty else 0
            disk_size = db_stats.iloc[0]['total_size'] if not db_stats.empty else 0
            
            return {
                "memory_objects": len(self.memory_cache),
                "disk_objects": int(disk_count),
                "total_disk_size_mb": round((disk_size or 0) / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir),
                "db_salt": self.db_salt
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}

    def invalidate_namespace(self, namespace: str) -> None:
        """Видалення всіх записів у конкретному просторі імен."""
        query = f"SELECT key_hash FROM cache_metadata WHERE namespace = '{namespace}'"
        results = self.db.fetch_all(query)
        hashes_df = pd.DataFrame(results, columns=['key_hash'])
        if not hashes_df.empty:
            for h in hashes_df['key_hash']:
                self._delete_cache_entry(h)
        self.logger.info(f"Invalidated cache namespace: {namespace}")

    def _is_cache_valid(self, data: Dict[str, Any]) -> bool:
        """Перевірка терміну дії кешу за TTL."""
        return time.time() - data.get('timestamp', 0) < data.get('ttl', 0)

    def _delete_cache_entry(self, cache_key: str):
        """Видалення метаданих та файлів для ключа."""
        with self.lock:
            self.memory_cache.pop(cache_key, None)
        self._remove_file_if_exists(self.cache_dir / f"{cache_key}.parquet")
        self._remove_file_if_exists(self.cache_dir / f"{cache_key}.pkl")
        self.db.execute_query(f"DELETE FROM cache_metadata WHERE key_hash = '{cache_key}'")

    def _remove_file_if_exists(self, path: Path):
        if path.exists():
            path.unlink()

    def cached(self, ttl: int = 3600, namespace: str = "default", use_salt: bool = True):
        """Декоратор для автоматичного кешування результатів функцій з підтримкою namespace."""
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__module__}.{func.__name__}"
                params = {'args': args[1:] if args and args[0] is self else args, 'kwargs': kwargs}
                
                result = self.get(key, params, namespace=namespace, use_salt=use_salt)
                if result is not None:
                    return result
                
                result = func(*args, **kwargs)
                self.set(key, result, params, ttl, namespace=namespace, use_salt=use_salt)
                return result
            return wrapper
        return decorator

    def auto_cleanup(self, max_disk_mb: int = 5000):
        """Видалення найстаріших 10% записів, якщо кеш перевищує ліміт диска."""
        total_size_query = "SELECT SUM(size_bytes) as total FROM cache_metadata"
        res_list = self.db.fetch_all(total_size_query)
        res = pd.DataFrame(res_list, columns=['total'])
        current_size = res.iloc[0]['total'] if not res.empty and res.iloc[0]['total'] else 0
        
        if current_size > max_disk_mb * 1024 * 1024:
            self.logger.info("Cache limit exceeded. Starting cleanup...")
            count_query = "SELECT COUNT(*) as total FROM cache_metadata"
            total_count_list = self.db.fetch_all(count_query)
            total_count = pd.DataFrame(total_count_list, columns=['total']).iloc[0]['total']
            limit = max(1, int(total_count * 0.1))
            
            old_hashes_query = f"SELECT key_hash FROM cache_metadata ORDER BY timestamp ASC LIMIT {limit}"
            old_hashes_list = self.db.fetch_all(old_hashes_query)
            old_hashes = pd.DataFrame(old_hashes_list, columns=['key_hash'])
            
            for h in old_hashes['key_hash']:
                self._delete_cache_entry(h)
            self.logger.info(f"Auto-cleanup removed {limit} oldest cache entries.")

    def clear(self) -> None:
        """Повне очищення кешу та метаданих."""
        with self.lock:
            self.memory_cache.clear()
        for f in self.cache_dir.iterdir():
            if f.suffix in ['.pkl', '.parquet']:
                try:
                    f.unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to delete {f}: {e}")
        self.db.execute_query("DELETE FROM cache_metadata")