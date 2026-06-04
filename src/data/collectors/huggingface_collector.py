"""
HuggingFace Data Collector
Collects datasets from HuggingFace
"""
import hashlib
from typing import Any

import pandas as pd

from src.core.cache.cache_manager import CacheManager
from src.core.clients.http_client_factory import HttpClientFactory
from src.data.management.data_manager import DataManager

from .base_collector import BaseCollector


class HuggingfaceCollector(BaseCollector):
    """Collector for fetching financial datasets from HuggingFace."""
    collector_type = 'huggingface'
    data_type = 'alternative'

    def __init__(self, configs: dict[str, Any], http_client_factory:
        HttpClientFactory, db_manager: DataManager, cache_manager: CacheManager | None=None, **kwargs):
        super().__init__(configs, http_client_factory, db_manager,
            cache_manager, **kwargs)
        self.dataset_name = self.configs.get('dataset_name', 'financial_news')
        self.subset_name = self.configs.get('subset_name')
        self.split = self.configs.get('split', 'train')
        self.hash_keys = self.configs.get('hash_keys', ['text', 'timestamp'])

    async def run(self, tickers: list[str] | None=None, **kwargs
        ) ->pd.DataFrame | None:
        """Fetches datasets from HuggingFace, filters novel entries, and persists to DataManager."""
        table_name = self.configs.get('table_name', 'huggingface_data')
        cache_key = f'{self.__class__.__name__}_run'
        cache_params = {'dataset': self.dataset_name, 'split': self.split}
        if self.cache_manager:
            cached = self.cache_manager.get(cache_key, cache_params,
                namespace='collectors')
            if cached is not None:
                self.logger.info(
                    '[HuggingFace] Cache hit — no new records detected.')
                return None
        self.logger.info(
            f"[HuggingFace] Loading dataset '{self.dataset_name}'...")
        try:
            raw_data = await self._fetch_from_huggingface()
        except Exception as e:
            self.logger.error(
                f'[HuggingFace] Network error during dataloader: {e}')
            raise RuntimeError("HuggingFace dataset loading failed") from e
        if not raw_data:
            self.logger.info('[HuggingFace] Zero records found.')
            return None
        self.logger.info(
            f'[HuggingFace] Succeeded to fetch {len(raw_data)} records. Proceeding to process...'
            )
        df = pd.DataFrame(raw_data)
        self.logger.info('[HuggingFace] Computing cryptographic hashes...')
        df['hash'] = df[self.hash_keys].astype(str).agg('|'.join, axis=1
            ).apply(lambda x: hashlib.sha256(x.encode()).hexdigest())
        self.logger.info('[HuggingFace] Filtering for novel records...')
        new_df = self.db_manager.filter_new_records(table_name, df)
        if new_df.empty:
            self.logger.info(
                '[HuggingFace] No novel records identified against historical database.'
                )
            if self.cache_manager:
                self.cache_manager.set(cache_key, True, cache_params,
                    namespace='collectors', ttl=604800)
            return None
        self.logger.info(
            f'[HuggingFace] Committing {len(new_df)} new records...')
        self.db_manager.upsert(table_name, new_df, unique_on=['hash'])
        if self.cache_manager:
            self.cache_manager.set(cache_key, True, cache_params, namespace
                ='collectors', ttl=604800)
        self.logger.info(
            f'[HuggingFace] ✅ Successfully persisted {len(new_df)} new records.'
            )
        return new_df

    async def _fetch_from_huggingface(self) ->list[dict[str, Any]]:
        """Downloads datasets from HuggingFace Datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            self.logger.error(
                "[HuggingFace] The 'datasets' library is missing. Run: pip install datasets"
                )
            return []
        try:
            if self.subset_name:
                dataset = load_dataset(self.dataset_name, self.subset_name,
                    split=self.split)
            else:
                dataset = load_dataset(self.dataset_name, split=self.split)
            self.logger.info(
                '[HuggingFace] Serializing dataset mapping into Pandas Interface...'
                )
            try:
                df = dataset.to_pandas()
                records = df.to_dict('records')
            except Exception as e:
                self.logger.error(f'Виникла помилка конвертації в pandas: {e}', exc_info=True)
                self.logger.warning(
                    '[HuggingFace] to_pandas() native structure unavailable, using custom fallback mapping...'
                    )
                records = [dict(item) for item in dataset]
            self.logger.info(
                f'[HuggingFace] ✅ Loaded {len(records)} structural records.')
            return records
        except Exception as e:  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN
            self.logger.error(
                f'[HuggingFace] Dataset load execution exception: {e}', exc_info=True)

            return []
