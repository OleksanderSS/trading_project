from src.core.cache.cache_manager import CacheManager
from src.data.management.data_manager import DataManager


def test_cache_manager_default_constructs_data_manager_with_config_not_a_path_string(tmp_path):
    """CacheManager() with no explicit data_manager used to pass a raw
    path string (self.config.get('paths.raw_db', ...)) into
    DataManager.__init__, which expects a config_manager object and
    immediately calls .get(...) on it - AttributeError on any caller
    that doesn't pass an explicit data_manager."""
    manager = CacheManager(cache_dir=tmp_path)

    assert isinstance(manager.db, DataManager)
