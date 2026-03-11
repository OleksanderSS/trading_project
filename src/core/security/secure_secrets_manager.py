"""
🔐 SECURE SECRETS MANAGER
Безпечне управління секретами та API ключами
"""

import os
import re
import hashlib
from typing import Optional, Dict, List
from pathlib import Path

# Використовуємо наш централізований логгер
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class SecurityError(Exception):
    """Виключення для критичних помилок безпеки."""
    pass

def load_dotenv(dotenv_path: str = '.env'):
    """
    Вручну завантажує змінні з .env файлу в оточення os.environ.
    Це робиться, оскільки в середовищі може бути відсутня бібліотека python-dotenv.
    """
    if not os.path.exists(dotenv_path):
        logger.warning(f"Файл .env не знайдено за шляхом: {dotenv_path}. Секрети не будуть завантажені з файлу.")
        return

    logger.debug(f"Завантаження змінних оточення з файлу {dotenv_path}...")
    try:
        loaded_keys = []
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                
                # Завжди перезаписуємо змінні з .env файлу
                os.environ[key] = value
                loaded_keys.append(key)
        
        logger.info(f"Змінні з .env ({len(loaded_keys)} шт.) успішно завантажені в оточення.")
        return loaded_keys
    except Exception as e:
        logger.error(f"Помилка під час читання або обробки файлу {dotenv_path}: {e}", exc_info=True)
        return []


class SecretsManager:
    """Безпечний менеджер секретів виробничого рівня."""

    # Регулярні вирази для валідації форматів відомих API ключів
    FORMAT_PATTERNS = {
        'FRED_API_KEY': r'^[a-f0-9]{32}$',
        'NEWS_API_KEY': r'^[a-f0-9]{32}$',
        'TELEGRAM_TOKEN': r'^\d+:[A-Za-z0-9_-]{35}$',
        'HF_TOKEN': r'^hf_[A-Za-z0-9]{34,}$'
    }

    def __init__(self, dotenv_path: str = '.env', encrypted_path: str = '.env.enc'):
        """
        Ініціалізація менеджера секретів.
        """
        self.dotenv_keys = load_dotenv(dotenv_path)
        self._secrets_cache: Dict[str, str] = {}
        
        # Опціональне завантаження зашифрованих секретів (якщо є ключ шифрування в оточенні)
        self._load_encrypted_secrets(encrypted_path)

    def _load_encrypted_secrets(self, path: str):
        """Спроба завантаження секретів із зашифрованого файлу (потребує CRYPTO_KEY)."""
        crypto_key = os.getenv('CRYPTO_KEY')
        if not crypto_key or not os.path.exists(path):
            return
            
        try:
            # Тут могла б бути реальна логіка дешифрування через cryptography.fernet
            # Зараз це просто скелет для майбутньої реалізації
            logger.info(f"Виявлено зашифрований файл {path}. Використовуйте 'cryptography' для повної реалізації.")
        except Exception as e:
            logger.error(f"Не вдалося завантажити зашифровані секрети: {e}")

    def validate_format(self, key_name: str, value: str) -> bool:
        """Перевіряє чи відповідає секрет очікуваному формату через Regex."""
        if not value:
            return False
            
        pattern = self.FORMAT_PATTERNS.get(key_name)
        if not pattern:
            return True # Якщо патерн не визначений, вважаємо валідним за замовчуванням
            
        if not re.match(pattern, value):
            logger.warning(f"Секрет '{key_name}' не відповідає очікуваному формату Regex.")
            return False
        return True

    def get_secret(self, key_name: str, default: Optional[str] = None, critical: bool = False) -> Optional[str]:
        """
        Безпечне отримання секрету. Пріоритет: os.environ -> .env.
        """
        value = os.getenv(key_name)

        if not value:
            if critical:
                logger.critical(f"КРИТИЧНА ПОМИЛКА СЕКРЕТІВ: Ключ '{key_name}' відсутній!")
                raise SecurityError(f"Critical secret '{key_name}' is missing.")
            return default

        # Перевірка на плейсхолдери та формат
        if f"your_{key_name.lower()}_here" in value.lower() or value == "":
            if critical:
                raise SecurityError(f"Secret '{key_name}' contains a placeholder value.")
            return default
            
        # Валідація формату
        if not self.validate_format(key_name, value) and critical:
             raise SecurityError(f"Secret '{key_name}' has an invalid format.")

        return value

    def as_dict(self) -> Dict[str, str]:
        """
        Повертає обмежений набір секретів. 
        Тільки ті, що були в .env або містять ознаки безпекових ключів.
        """
        result = {}
        target_patterns = ['API', 'KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'URL', 'DATABASE']
        
        for key, value in os.environ.items():
            is_from_dotenv = self.dotenv_keys and key in self.dotenv_keys
            is_security_related = any(p in key.upper() for p in target_patterns)
            
            if is_from_dotenv or is_security_related:
                result[key] = value
                
        return result

    @staticmethod
    def mask_secret(secret: Optional[str]) -> str:
        """Маскує секрет для безпечного логування (наприклад, 'AKIA...XXXX')."""
        if not secret:
            return "None"
        if len(secret) <= 8:
            return "****"
        return f"{secret[:4]}...{secret[-4:]}"

    def validate_secrets(self, expected_keys: List[str]) -> Dict[str, bool]:
        """Перевіряє наявність списку очікуваних ключів."""
        return {key: (self.get_secret(key) is not None and self.validate_format(key, self.get_secret(key))) 
                for key in expected_keys}

    def log_secrets_status(self, keys_to_check: List[str]):
        """Виводить статус завантаження секретів у лог без розкриття значень."""
        logger.info("--- СТАТУС КОНФІГУРАЦІЇ СЕКРЕТІВ (VALIDATION) ---")
        for key in keys_to_check:
            value = self.get_secret(key)
            is_present = value is not None
            is_valid = self.validate_format(key, value) if is_present else False
            
            status = f"[OK]" if is_valid else ("[FORMAT ERR]" if is_present else "[MISSING]")
            masked = self.mask_secret(value) if is_present else "N/A"
            logger.info(f"- {key:20} {status:12} {masked}")
        logger.info("--------------------------------------------------")


_secrets_manager_instance = SecretsManager()

def get_secret(key_name: str, default: Optional[str] = None, critical: bool = False) -> Optional[str]:
    """Глобальна функція доступу до секретів."""
    return _secrets_manager_instance.get_secret(key_name, default=default, critical=critical)

def mask_secret(secret: Optional[str]) -> str:
    """Глобальна функція маскування."""
    return SecretsManager.mask_secret(secret)

if __name__ == "__main__":
    test_keys = ['NEWS_API_KEY', 'FRED_API_KEY', 'HF_TOKEN', 'TELEGRAM_TOKEN', 'CRYPTO_KEY']
    _secrets_manager_instance.log_secrets_status(test_keys)