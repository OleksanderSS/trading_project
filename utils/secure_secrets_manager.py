#!/usr/bin/env python3
"""
🔐 SECURE SECRETS MANAGER
Безпечне управління секретами та API ключами
"""

import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class SecureSecretsManager:
    """Безпечний менеджер секретів"""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Ініціалізація менеджера секретів
        
        Args:
            env_file: Шлях до .env файлу
        """
        self.env_file = env_file or os.path.join(os.path.dirname(__file__), '.env')
        self._secrets = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Завантажує секрети з environment variables"""
        # Завантажуємо .env файл
        if os.path.exists(self.env_file):
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment from: {self.env_file}")
        else:
            logger.warning(f".env file not found: {self.env_file}")
        
        # Завантажуємо секрети
        self._secrets = {
            'fred_api_key': os.getenv('FRED_API_KEY'),
            'news_api_key': os.getenv('NEWS_API_KEY'),
            'hf_token': os.getenv('HF_TOKEN'),
            'google_credentials': os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
            'telegram_token': os.getenv('TELEGRAM_TOKEN'),
            'telegram_api_id': os.getenv('TELEGRAM_API_ID'),
            'telegram_api_hash': os.getenv('TELEGRAM_API_HASH'),
            'telegram_chat_id': os.getenv('TELEGRAM_CHAT_ID'),
        }
    
    def get_secret(self, secret_name: str) -> Optional[str]:
        """
        Безпечне отримання секрету
        
        Args:
            secret_name: Назва секрету
            
        Returns:
            Значення секрету або None
        """
        secret = self._secrets.get(secret_name.lower())
        
        if not secret:
            logger.warning(f"Secret not found: {secret_name}")
            return None
        
        if secret == f"your_{secret_name}_here" or secret == "":
            logger.warning(f"Secret not configured: {secret_name}")
            return None
        
        return secret
    
    def get_fred_api_key(self) -> Optional[str]:
        """Отримати FRED API ключ"""
        return self.get_secret('fred_api_key')
    
    def get_news_api_key(self) -> Optional[str]:
        """Отримати News API ключ"""
        return self.get_secret('news_api_key')
    
    def get_telegram_config(self) -> Dict[str, Optional[str]]:
        """Отримати Telegram конфігурацію"""
        return {
            'token': self.get_secret('telegram_token'),
            'api_id': self.get_secret('telegram_api_id'),
            'api_hash': self.get_secret('telegram_api_hash'),
            'chat_id': self.get_secret('telegram_chat_id'),
        }
    
    def validate_secrets(self) -> Dict[str, bool]:
        """
        Валідує наявність секретів
        
        Returns:
            Словник з статусом валідації
        """
        validation_results = {}
        
        for secret_name, secret_value in self._secrets.items():
            if secret_value and secret_value != f"your_{secret_name}_here":
                validation_results[secret_name] = True
            else:
                validation_results[secret_name] = False
        
        return validation_results
    
    def mask_secret(self, secret: str) -> str:
        """
        Маскує секрет для логування
        
        Args:
            secret: Секрет для маскування
            
        Returns:
            Замаскований секрет
        """
        if not secret or len(secret) < 8:
            return "***"
        
        return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"
    
    def log_secrets_status(self):
        """Логує статус секретів"""
        validation = self.validate_secrets()
        
        logger.info("=== SECRETS STATUS ===")
        for secret_name, is_valid in validation.items():
            status = "[OK] Configured" if is_valid else "[ERROR] Missing"
            logger.info(f"{secret_name}: {status}")

# Global instance
secrets_manager = SecureSecretsManager()

# Convenience functions
def get_fred_api_key() -> Optional[str]:
    """Отримати FRED API ключ"""
    return secrets_manager.get_fred_api_key()

def get_news_api_key() -> Optional[str]:
    """Отримати News API ключ"""
    return secrets_manager.get_news_api_key()

def get_telegram_config() -> Dict[str, Optional[str]]:
    """Отримати Telegram конфігурацію"""
    return secrets_manager.get_telegram_config()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    secrets_manager.log_secrets_status()
