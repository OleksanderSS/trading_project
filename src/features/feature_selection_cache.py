"""
Feature Selection Cache
Кешує результати SmartFeatureSelector для пришвидшення повторних запусків
"""
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('FeatureSelectionCache')


class FeatureSelectionCache:
    """
    Кешує результати feature selection для кожної комбінації:
    - model_type (lgbm, xgboost, mlp, lstm, etc.)
    - target_name (target_return_1d, target_up_1d, etc.)
    - market_regime (normal, volatile, trending, crisis)
    - feature_set_hash (хеш списку доступних фіч)
    """

    def __init__(self, cache_dir: str='data/cache/feature_selection'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / 'feature_selection_cache.json'
        self.cache = self._load_cache()
        logger.info(f'FeatureSelectionCache initialized: {self.cache_file}')

    def _load_cache(self) ->dict[str, Any]:
        """Завантажити кеш з диску"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    cache = json.load(f)
                logger.info(
                    f'✅ Loaded feature selection cache: {len(cache)} entries')
                return cache
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.warning(f'Failed to load cache: {e}')
                raise RuntimeError(
                    f"Failed to load feature selection cache from {self.cache_file}"
                ) from e
        return {}

    def _save_cache(self):
        """Зберегти кеш на диск"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f'💾 Saved feature selection cache: {len(self.cache)} entries')
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.exception(f'Failed to save cache: {e}')

    def _compute_cache_key(self, model_type: str, target_name: str,
        market_regime: str, available_features: list[str]) ->str:
        """
        Обчислити унікальний ключ для кешу

        Ключ включає:
        - model_type
        - target_name
        - market_regime
        - hash(available_features) - щоб інвалідувати кеш при зміні фіч
        """
        features_sorted = sorted(available_features)
        features_str = ','.join(features_sorted)
        features_hash = hashlib.sha256(features_str.encode()).hexdigest()[:8]
        cache_key = (
            f'{model_type}_{target_name}_{market_regime}_{features_hash}')
        return cache_key

    def get_selection(self, model_type: str, target_name: str,
        market_regime: str, available_features: list[str]) ->dict[str, Any] | None:
        """
        Отримати закешовані результати feature selection

        Returns:
            Dict з:
            - selected_features: List[str]
            - feature_importance: Dict[str, float]
            - selection_metadata: Dict
            або None якщо немає в кеші
        """
        cache_key = self._compute_cache_key(model_type, target_name,
            market_regime, available_features)
        if cache_key in self.cache:
            cached_data = self.cache[cache_key]
            logger.info(f'🚀 Feature selection cache HIT: {cache_key}')
            logger.info(
                f"   Selected {len(cached_data['selected_features'])} features"
                )
            return cached_data
        else:
            logger.info(f'🔄 Feature selection cache MISS: {cache_key}')
            return None

    def save_selection(self, model_type: str, target_name: str,
        market_regime: str, available_features: list[str],
        selected_features: list[str], feature_importance: dict[str, float],
        selection_metadata: dict[str, Any] | None=None):
        """
        Зберегти результати feature selection в кеш

        Args:
            model_type: Тип моделі (lgbm, xgboost, mlp, etc.)
            target_name: Назва таргету
            market_regime: Ринковий режим
            available_features: Список всіх доступних фіч
            selected_features: Список вибраних фіч
            feature_importance: Важливість кожної фічі
            selection_metadata: Додаткова інформація
        """
        cache_key = self._compute_cache_key(model_type, target_name,
            market_regime, available_features)
        cached_data = {'model_type': model_type, 'target_name': target_name,
            'market_regime': market_regime, 'selected_features':
            selected_features, 'feature_importance': feature_importance,
            'selection_metadata': selection_metadata or {}, 'timestamp':
            datetime.now().isoformat(), 'n_available_features': len(
            available_features), 'n_selected_features': len(
            selected_features), 'selection_ratio': len(selected_features) /
            len(available_features) if available_features else 0}
        self.cache[cache_key] = cached_data
        self._save_cache()
        logger.info(f'💾 Saved feature selection to cache: {cache_key}')
        logger.info(
            f'   Selected {len(selected_features)}/{len(available_features)} features'
            )

    def is_cached(self, model_type: str, target_name: str, market_regime:
        str, available_features: list[str]) ->bool:
        """Перевірити, чи є результати в кеші"""
        cache_key = self._compute_cache_key(model_type, target_name,
            market_regime, available_features)
        return cache_key in self.cache

    def invalidate(self, model_type: str | None=None, target_name:
        str | None=None, market_regime: str | None=None):
        """
        Інвалідувати кеш для певних параметрів

        Якщо параметри не вказані, інвалідує весь кеш
        """
        if (model_type is None and target_name is None and market_regime is
            None):
            self.cache = {}
            self._save_cache()
            logger.info('🗑️ Invalidated entire feature selection cache')
            return
        keys_to_remove = []
        for cache_key, cached_data in self.cache.items():
            match = True
            if model_type and cached_data.get('model_type') != model_type:
                match = False
            if target_name and cached_data.get('target_name') != target_name:
                match = False
            if market_regime and cached_data.get('market_regime'
                ) != market_regime:
                match = False
            if match:
                keys_to_remove.append(cache_key)
        for key in keys_to_remove:
            del self.cache[key]
        self._save_cache()
        logger.info(f'🗑️ Invalidated {len(keys_to_remove)} cache entries')

    def get_statistics(self) ->dict[str, Any]:
        """Отримати статистику кешу"""
        if not self.cache:
            return {'total_entries': 0, 'cache_size_mb': 0}
        cache_size_bytes = self.cache_file.stat(
            ).st_size if self.cache_file.exists() else 0
        cache_size_mb = cache_size_bytes / (1024 * 1024)
        model_types = {}
        target_names = {}
        market_regimes = {}
        for cached_data in self.cache.values():
            model_type = cached_data.get('model_type', 'unknown')
            target_name = cached_data.get('target_name', 'unknown')
            market_regime = cached_data.get('market_regime', 'unknown')
            model_types[model_type] = model_types.get(model_type, 0) + 1
            target_names[target_name] = target_names.get(target_name, 0) + 1
            market_regimes[market_regime] = market_regimes.get(market_regime, 0
                ) + 1
        return {'total_entries': len(self.cache), 'cache_size_mb': round(
            cache_size_mb, 2), 'model_types': model_types, 'target_names':
            target_names, 'market_regimes': market_regimes, 'cache_file':
            str(self.cache_file)}

    def print_statistics(self):
        """Вивести статистику кешу"""
        stats = self.get_statistics()
        logger.info('=' * 60)
        logger.info('FEATURE SELECTION CACHE STATISTICS')
        logger.info('=' * 60)
        logger.info(f"Total entries: {stats['total_entries']}")
        logger.info(f"Cache size: {stats['cache_size_mb']} MB")
        logger.info(f"Cache file: {stats['cache_file']}")
        if stats['total_entries'] > 0:
            logger.info('\nBy model type:')
            for model_type, count in stats['model_types'].items():
                logger.info(f'  {model_type}: {count}')
            logger.info('\nBy target:')
            for target_name, count in stats['target_names'].items():
                logger.info(f'  {target_name}: {count}')
            logger.info('\nBy market regime:')
            for regime, count in stats['market_regimes'].items():
                logger.info(f'  {regime}: {count}')
        logger.info('=' * 60)


_global_cache: FeatureSelectionCache | None = None


def get_feature_selection_cache(cache_dir: str | None=None
    ) ->FeatureSelectionCache:
    """Отримати глобальний інстанс кешу"""
    global _global_cache
    if _global_cache is None:
        _global_cache = FeatureSelectionCache(cache_dir=cache_dir or
            'data/cache/feature_selection')
    return _global_cache
