# utils/config_manager.py

"""
Centralized configuration management utilities
"""

import logging
import pandas as pd
from typing import Any, Dict, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigManager:
    """Centralized configuration manager with safe access"""
    
    def __init__(self):
        self._config_cache = {}
        self._access_log = []
    
    def get_path(self, path_key: str, default: Optional[str] = None) -> str:
        """
        Safely get path configuration
        
        Args:
            path_key: Path key (e.g., 'data', 'models', 'db')
            default: Default value if key not found
            
        Returns:
            Path string
            
        Raises:
            KeyError: If path not found and no default provided
        """
        try:
            from config.config import PATHS
            path = PATHS.get(path_key, default)
            if path is None:
                raise KeyError(f"Path key '{path_key}' not found")
            
            # Ensure path exists
            Path(path).mkdir(parents=True, exist_ok=True)
            
            self._log_access(f"PATHS[{path_key}]", path)
            return path
            
        except Exception as e:
            logger.error(f"Failed to get path {path_key}: {e}")
            if default is not None:
                return default
            raise
    
    def get_ticker(self, ticker_key: str, default: Optional[str] = None) -> str:
        """
        Safely get ticker configuration
        
        Args:
            ticker_key: Ticker key
            default: Default value if key not found
            
        Returns:
            Ticker symbol
        """
        try:
            from config.config import TICKERS
            ticker = TICKERS.get(ticker_key, default)
            if ticker is None:
                raise KeyError(f"Ticker key '{ticker_key}' not found")
            
            self._log_access(f"TICKERS[{ticker_key}]", ticker)
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get ticker {ticker_key}: {e}")
            if default is not None:
                return default
            raise
    
    def get_timeframe(self, timeframe_key: str, default: Optional[Dict] = None) -> Dict:
        """
        Safely get timeframe configuration
        
        Args:
            timeframe_key: Timeframe key
            default: Default value if key not found
            
        Returns:
            Timeframe configuration dictionary
        """
        try:
            from config.config import TIME_FRAMES
            timeframe = TIME_FRAMES.get(timeframe_key, default)
            if timeframe is None:
                raise KeyError(f"Timeframe key '{timeframe_key}' not found")
            
            self._log_access(f"TIME_FRAMES[{timeframe_key}]", timeframe)
            return timeframe
            
        except Exception as e:
            logger.error(f"Failed to get timeframe {timeframe_key}: {e}")
            if default is not None:
                return default
            raise
    
    def get_secret(self, secret_key: str, default: Optional[str] = None) -> str:
        """
        Safely get secret configuration
        
        Args:
            secret_key: Secret key
            default: Default value if key not found
            
        Returns:
            Secret value
        """
        try:
            from config import config
            secret = getattr(config, secret_key, default)
            if secret is None:
                raise KeyError(f"Secret key '{secret_key}' not found")
            
            # Don't log secret values
            self._log_access(f"SECRETS[{secret_key}]", "***HIDDEN***")
            return secret
            
        except Exception as e:
            logger.error(f"Failed to get secret {secret_key}: {e}")
            if default is not None:
                return default
            raise
    
    def get_config_value(self, config_dict: Dict, key: str, default: Any = None) -> Any:
        """
        Safely get value from any configuration dictionary
        
        Args:
            config_dict: Configuration dictionary
            key: Key to retrieve
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        try:
            value = config_dict.get(key, default)
            if value is None and default is None:
                raise KeyError(f"Key '{key}' not found in configuration")
            
            self._log_access(f"CONFIG[{key}]", value if not self._is_secret(key) else "***HIDDEN***")
            return value
            
        except Exception as e:
            logger.error(f"Failed to get config value {key}: {e}")
            if default is not None:
                return default
            raise
    
    def validate_config(self) -> Dict[str, bool]:
        """
        Validate all configuration sections
        
        Returns:
            Dictionary with validation results
        """
        results = {}
        
        # Validate PATHS
        try:
            from config.config import PATHS
            required_paths = ['data', 'models', 'db']
            for path_key in required_paths:
                if path_key not in PATHS:
                    results[f'paths.{path_key}'] = False
                else:
                    Path(PATHS[path_key]).mkdir(parents=True, exist_ok=True)
                    results[f'paths.{path_key}'] = True
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            results['paths'] = False
        
        # Validate TICKERS
        try:
            from config.config import TICKERS
            results['tickers'] = len(TICKERS) > 0
        except Exception as e:
            logger.error(f"Tickers validation failed: {e}")
            results['tickers'] = False
        
        # Validate TIME_FRAMES
        try:
            from config.config import TIME_FRAMES
            results['timeframes'] = len(TIME_FRAMES) > 0
        except Exception as e:
            logger.error(f"Timeframes validation failed: {e}")
            results['timeframes'] = False
        
        # Validate secrets
        try:
            from config import config
            critical_secrets = ['FRED_API_KEY']
            for secret_key in critical_secrets:
                secret_value = getattr(config, secret_key, None)
                results[f'secrets.{secret_key}'] = secret_value is not None and len(secret_value) > 0
        except Exception as e:
            logger.error(f"Secrets validation failed: {e}")
            results['secrets'] = False
        
        return results
    
    def _log_access(self, key: str, value: Any) -> None:
        """Log configuration access for debugging"""
        self._access_log.append({
            'key': key,
            'value': value,
            'timestamp': pd.Timestamp.now()
        })
        
        # Keep only last 1000 accesses
        if len(self._access_log) > 1000:
            self._access_log = self._access_log[-1000:]
    
    def _is_secret(self, key: str) -> bool:
        """Check if key contains sensitive information"""
        secret_keywords = ['password', 'token', 'key', 'secret', 'api']
        return any(keyword in key.lower() for keyword in secret_keywords)
    
    def get_access_stats(self) -> Dict[str, Any]:
        """Get configuration access statistics"""
        if not self._access_log:
            return {'total_accesses': 0}
        
        from collections import Counter
        key_counts = Counter([access['key'] for access in self._access_log])
        
        return {
            'total_accesses': len(self._access_log),
            'most_accessed': key_counts.most_common(5),
            'last_access': self._access_log[-1]['timestamp'] if self._access_log else None
        }


# Global instance
config_manager = ConfigManager()


# Convenience functions
def get_path(path_key: str, default: Optional[str] = None) -> str:
    """Get path configuration safely"""
    return config_manager.get_path(path_key, default)


def get_ticker(ticker_key: str, default: Optional[str] = None) -> str:
    """Get ticker configuration safely"""
    return config_manager.get_ticker(ticker_key, default)


def get_timeframe(timeframe_key: str, default: Optional[Dict] = None) -> Dict:
    """Get timeframe configuration safely"""
    return config_manager.get_timeframe(timeframe_key, default)


def get_secret(secret_key: str, default: Optional[str] = None) -> str:
    """Get secret configuration safely"""
    return config_manager.get_secret(secret_key, default)


def validate_all_configs() -> Dict[str, bool]:
    """Validate all configurations"""
    return config_manager.validate_config()


# Decorator for safe configuration access
def safe_config_access(config_type: str = 'value'):
    """
    Decorator for safe configuration access
    
    Args:
        config_type: Type of configuration ('path', 'ticker', 'timeframe', 'secret', 'value')
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (KeyError, AttributeError) as e:
                logger.error(f"Configuration access error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator
