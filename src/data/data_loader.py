"""
Завантаження та обробка даних
"""
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('DataLoader')


class DataLoaderConfig:
    """Конфігурація для завантажувача даних"""

    def __init__(self, batch_dir: str, config_loader: object):
        self.batch_dir = batch_dir
        self.config_loader = config_loader
        self.cache_dir = Path(batch_dir) / 'cache'
        self.cache_dir.mkdir(exist_ok=True)


class ColabDataLoader:
    """Завантажувач даних для Colab"""

    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.features_df = None
        self.targets_df = None
        self.cache_signature = None

    def load_data(self) ->Tuple[object, object]:
        """Завантажити дані"""
        import pandas as pd
        batch_dir = Path(self.config.batch_dir)
        features_file = batch_dir / 'enriched_features.parquet'
        targets_file = batch_dir / 'targets.parquet'
        if not features_file.exists() or not targets_file.exists():
            raise FileNotFoundError(f'Дані не знайдені в {batch_dir}')
        self.features_df = pd.read_parquet(features_file)
        self.targets_df = pd.read_parquet(targets_file)
        self._normalize_timezones()
        return self.features_df, self.targets_df

    def _normalize_timezones(self) ->None:
        """Нормалізувати часові зони"""
        if self.features_df is not None and hasattr(self.features_df.index,
            'tz'):
            self.features_df.index = self.features_df.index.tz_localize(None)
        if self.targets_df is not None and hasattr(self.targets_df.index, 'tz'
            ):
            self.targets_df.index = self.targets_df.index.tz_localize(None)

    def check_cache(self) ->bool:
        """Перевірити кеш"""
        cache_file = self.config.cache_dir / 'cache_signature.json'
        if not cache_file.exists():
            return False
        try:
            with open(cache_file, 'r') as f:
                cached = json.load(f)
            current_sig = self._compute_signature()
            return cached.get('signature') == current_sig
        except Exception as e:
            logger.error(f'Виникла помилка при перевірці кешу: {e}',
                exc_info=True)
            return False

    def save_cache_signature(self) ->None:
        """Зберегти сигнатуру кешу"""
        cache_file = self.config.cache_dir / 'cache_signature.json'
        signature_data = {'signature': self._compute_signature(),
            'timestamp': datetime.now().isoformat()}
        with open(cache_file, 'w') as f:
            json.dump(signature_data, f, indent=2)

    def _compute_signature(self) ->str:
        """Розрахувати сигнатуру даних"""
        import hashlib
        if self.features_df is None or self.targets_df is None:
            return ''
        feat_hash = hashlib.sha256(str(self.features_df.shape).encode()
            ).hexdigest()
        targ_hash = hashlib.sha256(str(self.targets_df.shape).encode()
            ).hexdigest()
        return f'{feat_hash}_{targ_hash}'
