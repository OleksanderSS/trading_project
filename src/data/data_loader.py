"""
Завантаження та обробка даних
"""
import json
from datetime import datetime
from pathlib import Path

from src.core.logging.logger import ProjectLogger
from src.core.security.path_validator import PathValidationError, validate_safe_path

logger = ProjectLogger.get_logger('DataLoader')


class DataLoaderConfig:
    """Конфігурація для завантажувача даних"""

    def __init__(self, batch_dir: str, config_loader: object):
        self.batch_dir = Path(batch_dir).resolve()
        self.config_loader = config_loader
        self.cache_dir = self.batch_dir / 'cache'
        self.cache_dir.mkdir(exist_ok=True)


class ColabDataLoader:
    """Завантажувач даних для Colab"""

    def __init__(self, config: DataLoaderConfig):
        self.config = config
        self.features_df = None
        self.targets_df = None
        self.cache_signature = None

    def load_data(self) ->tuple[object, object]:
        """Завантажити дані"""
        import pandas as pd

        try:
            features_file = validate_safe_path(self.config.batch_dir / 'enriched_features.parquet', base_dir=self.config.batch_dir)
            targets_file = validate_safe_path(self.config.batch_dir / 'targets.parquet', base_dir=self.config.batch_dir)

            if not features_file.exists() or not targets_file.exists():
                raise FileNotFoundError(f'Дані не знайдені в {self.config.batch_dir}')

            self.features_df = pd.read_parquet(features_file)
            self.targets_df = pd.read_parquet(targets_file)
            self._normalize_timezones()
            return self.features_df, self.targets_df
        except PathValidationError as e:
            logger.exception(f"Security violation: {e}")
            raise FileNotFoundError(f"Access denied to data files in {self.config.batch_dir}") from e

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
        try:
            cache_file = validate_safe_path(self.config.cache_dir / 'cache_signature.json', base_dir=self.config.cache_dir)
            if not cache_file.exists():
                return False
            with open(cache_file) as f:
                cached = json.load(f)
            current_sig = self._compute_signature()
            return cached.get('signature') == current_sig
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка при перевірці кешу: {e}',
                exc_info=True)
            return False

    def save_cache_signature(self) ->None:
        """Зберегти сигнатуру кешу"""
        try:
            cache_file = validate_safe_path(self.config.cache_dir / 'cache_signature.json', base_dir=self.config.cache_dir)
            signature_data = {'signature': self._compute_signature(),
                'timestamp': datetime.now().isoformat()}
            with open(cache_file, 'w') as f:
                json.dump(signature_data, f, indent=2)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f'Виникла помилка при збереженні кешу: {e}',
                exc_info=True)
            raise
