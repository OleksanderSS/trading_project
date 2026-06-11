
from pathlib import Path

from src.core.cache.cache_manager import CacheManager
from src.core.error_handling.error_handler import get_error_handler
from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger
from src.core.security.secure_secrets_manager import SecretsManager

logger = ProjectLogger.get_logger("CoreValidation")

def validate_core():
    logger.info("--- Starting Deep Core Audit ---")

    # 1. Logging
    logger.info("✅ Logger is functional.")

    # 2. Security / Secrets
    try:
        secrets = SecretsManager()
        logger.info(f"✅ SecretsManager loaded. Available keys: {len(secrets.as_dict())}")
    except Exception as e:
        logger.error(f"❌ SecretsManager failed: {e}")

    # 3. Cache Manager
    try:
        cache = CacheManager(cache_dir="data/cache/test")
        cache.set("test_key", "test_value")
        val = cache.get("test_key")
        if val == "test_value":
            logger.info("✅ CacheManager functional.")
        else:
            logger.error("❌ CacheManager returned incorrect value.")
    except Exception as e:
        logger.error(f"❌ CacheManager failed: {e}")

    # 4. Error Handler
    try:
        eh = get_error_handler()
        logger.info("✅ ErrorHandler initialized.")
    except Exception as e:
        logger.error(f"❌ ErrorHandler failed: {e}")

    # 5. File Manager
    try:
        fm = FileManager(base_dir=Path("."))
        logger.info("✅ FileManager initialized.")
    except Exception as e:
        logger.error(f"❌ FileManager failed: {e}")

    logger.info("--- Core Audit Complete ---")

if __name__ == "__main__":
    validate_core()
