# src/data/collectors/huggingface_collector.py

import pandas as pd
import hashlib
from typing import List, Dict, Any, Optional

from .base_collector import BaseCollector
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager
from src.core.cache.cache_manager import CacheManager


class HuggingfaceCollector(BaseCollector):
    """Колектор для завантаження фінансових даних з HuggingFace."""
    collector_type = "huggingface"
    data_type = "alternative"

    def __init__(
        self,
        configs: Dict[str, Any],
        http_client_factory: HttpClientFactory,
        db_manager: DataManager,
        cache_manager: Optional[CacheManager] = None,
        **kwargs,
    ):
        super().__init__(configs, http_client_factory, db_manager, cache_manager, **kwargs)
        self.dataset_name = self.configs.get("dataset_name", "financial_news")
        self.subset_name = self.configs.get("subset_name")
        self.split = self.configs.get("split", "train")
        self.hash_keys = self.configs.get("hash_keys", ["text", "timestamp"])

    async def run(self, tickers: Optional[List[str]] = None, **kwargs) -> Optional[pd.DataFrame]:
        """Завантажує дані з HuggingFace, фільтрує нові, зберігає в БД."""
        table_name = self.configs.get("table_name", "huggingface_data")
        
        cache_key = f"{self.__class__.__name__}_run"
        cache_params = {
            "dataset": self.dataset_name,
            "split": self.split,
        }

        # 1. Кеш
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params, namespace="collectors")
            if cached is not None:
                self.logger.info("[HuggingFace] Cache hit — нових записів немає.")
                return None

        # 2. Збір
        self.logger.info(f"[HuggingFace] Loading dataset '{self.dataset_name}'...")
        try:
            raw_data = await self._fetch_from_huggingface()
        except Exception as e:
            self.logger.error(f"[HuggingFace] Помилка при завантаженні: {e}")
            return None

        if not raw_data:
            self.logger.info("[HuggingFace] Записів не знайдено.")
            return None

        self.logger.info(f"[HuggingFace] Завантажено {len(raw_data)} записів. Обробка...")
        df = pd.DataFrame(raw_data)

        # 3. Оптимізований Hash (vectorized)
        self.logger.info("[HuggingFace] Розраховуємо хеші...")
        df["hash"] = df[self.hash_keys].astype(str).agg("|".join, axis=1).apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()
        )

        # 4. Фільтрація через БД (більш ефективна)
        self.logger.info("[HuggingFace] Фільтруємо нові записи...")
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info("[HuggingFace] Нових записів не знайдено в БД.")
            if self.cache_manager:
                self.cache_manager.set(
                    cache_key, True, cache_params, namespace="collectors", ttl=604800
                )
            return None

        # 5. Збереження
        self.logger.info(f"[HuggingFace] Зберігаємо {len(new_df)} нових записів...")
        self.db_manager.upsert(table_name, new_df, unique_on=["hash"])

        if self.cache_manager:
            self.cache_manager.set(
                cache_key, True, cache_params, namespace="collectors", ttl=604800
            )

        self.logger.info(f"[HuggingFace] ✅ Збережено {len(new_df)} нових записів.")
        return new_df

    async def _fetch_from_huggingface(self) -> List[Dict[str, Any]]:
        """Завантажує дані з HuggingFace Datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            self.logger.error("[HuggingFace] Бібліотека 'datasets' не встановлена. Встановіть: pip install datasets")
            return []

        try:
            # Завантажуємо датасет
            if self.subset_name:
                dataset = load_dataset(self.dataset_name, self.subset_name, split=self.split)
            else:
                dataset = load_dataset(self.dataset_name, split=self.split)

            # Оптимізована конвертація: використовуємо to_pandas() якщо доступно
            self.logger.info(f"[HuggingFace] Конвертуємо датасет в DataFrame...")
            try:
                # Спроба використати to_pandas() - набагато швидше
                df = dataset.to_pandas()
                records = df.to_dict('records')
            except Exception:
                # Fallback на ручну конвертацію
                self.logger.warning("[HuggingFace] to_pandas() не доступна, використовуємо ручну конвертацію...")
                records = [dict(item) for item in dataset]

            self.logger.info(f"[HuggingFace] ✅ Завантажено {len(records)} записів.")
            return records

        except Exception as e:
            self.logger.error(f"[HuggingFace] Помилка при завантаженні датасету: {e}")
            return []
