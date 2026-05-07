# src/core/security/secure_secrets_manager.py
"""
🔐 SECURE SECRETS MANAGER
Production-grade management of system secrets, API keys, and environment variables.
"""

import os
import re
from pathlib import Path

# Utilize centralized project-wide logger
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class SecurityError(Exception):
    """Exception raised for critical security violations or missing required credentials."""
    pass

def load_dotenv(dotenv_path: str = '.env'):
    """
    Manually parses a .env file and injects keys into os.environ.
    This provides robustness in environments where the 'python-dotenv' library is not available.

    Search Protocol:
    1. Specified parameter path (default: .env)
    2. Google Drive mount point (Colab support): /content/drive/MyDrive/trading_project/.env
    3. Parent directory lookup: ../.env
    4. User home directory

    CodeScene: Complex Method (cc=9) is acceptable for security-critical code with multiple
    validation checks and fallback paths. Bumpy Road (2 bumps) is natural for hierarchical
    search protocol with multiple validation points.
    """
    # Hierarchical list of potential .env locations
    search_paths: list[str | Path] = [
        dotenv_path,
        '/content/drive/MyDrive/trading_project/.env',
        '/content/drive/MyDrive/.env',
        '/content/.env',
        '../.env',
        Path.home() / '.env',
    ]

    found_path: str | None = None
    for path in search_paths:
        if os.path.exists(str(path)):
            found_path = str(path)
            logger.info(f"Environment configuration identified: {found_path}")
            break

    if not found_path:
        logger.warning(
            f"No .env configuration file found across search vectors: {search_paths}. Utilizing existing environment variables."
        )
        return

    try:
        loaded_keys: list[str] = []
        with open(found_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines, comments, and malformed lines
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")

                # Overwrite environment variable with the file value
                os.environ[key] = value
                loaded_keys.append(key)

        logger.info(f"Successfully loaded {len(loaded_keys)} variables into the active environment.")
        return loaded_keys
    except Exception as e:
        logger.error(f"Critical failure reading environment file {found_path}: {e}")
        return []


class SecretsManager:
    """Secure Secrets Manager designed for production stability and observability."""

    # Predefined validation patterns for known API providers
    FORMAT_PATTERNS: dict[str, str] = {
        'FRED_API_KEY': r'^[a-f0-9]{32}$',
        'NEWS_API_KEY': r'^[a-f0-9]{32}$',
        'TELEGRAM_TOKEN': r'^\d+:[A-Za-z0-9_-]{35}$',
        'HF_TOKEN': r'^hf_[A-Za-z0-9]{34,}$'
    }

    def __init__(self, dotenv_path: str = ".env", encrypted_path: str = ".env.enc"):
        """
        Initializes the secrets manager and synchronizes local environment variables.
        """
        self.dotenv_keys = load_dotenv(dotenv_path)
        self._secrets_cache: dict[str, str] = {}

        # Optional: Decrypt and load persistent encrypted secrets
        self._load_encrypted_secrets(encrypted_path)

    def _load_encrypted_secrets(self, path: str):
        """Attempts to load secrets from an encrypted payload (requires CRYPTO_KEY)."""
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key or not os.path.exists(path):
            return

        try:
            # Placeholder for Fernet-based decryption logic
            logger.info(
                f"Encrypted payload identified: {path}. Logic integration pending cryptography implementation."
            )
        except Exception as e:
            logger.error(f"Failed to synchronize encrypted secrets: {e}")

    def validate_format(self, key_name: str, value: str) -> bool:
        """Validates if a specific secret aligns with its expected provider format."""
        if not value:
            return False

        pattern = self.FORMAT_PATTERNS.get(key_name)
        if not pattern:
            return True  # Pass by default if no pattern is defined

        if not re.match(pattern, value):
            logger.warning(
                f"Format Validation Failure: Secret '{key_name}' does not match expected Regex pattern."
            )
            return False
        return True

    def get_secret(
        self, key_name: str, default: str | None = None, critical: bool = False
    ) -> str | None:
        """
        Safely retrieves a configuration secret.
        Hierarchy: os.environ -> Local Cache.
        """
        value = os.getenv(key_name)

        if not value:
            if critical:
                logger.critical(
                    f"AUTHENTICATION PROTOCOL BREACH: Required key '{key_name}' is missing or undefined!"
                )
                raise SecurityError(f"Critical secret '{key_name}' is missing.")
            return default

        # Block placeholder values from development templates
        if f"your_{key_name.lower()}_here" in value.lower() or value == "":
            if critical:
                logger.critical(
                    f"SECURITY PROTOCOL BREACH: Key '{key_name}' contains a template placeholder."
                )
                raise SecurityError(f"Secret '{key_name}' contains a placeholder value.")
            return default

        # Enforce format validation for critical assets
        if not self.validate_format(key_name, value) and critical:
            raise SecurityError(f"Secret '{key_name}' failed hierarchical format validation.")

        return value

    def as_dict(self) -> dict[str, str]:
        """
        Exports a filtered dictionary of sensitive keys identified in the environment.
        """
        result: dict[str, str] = {}
        target_patterns = ["API", "KEY", "TOKEN", "SECRET", "PASSWORD", "URL", "DATABASE"]

        for key, value in os.environ.items():
            is_from_dotenv = self.dotenv_keys and key in self.dotenv_keys
            is_security_related = any(p in key.upper() for p in target_patterns)

            if is_from_dotenv or is_security_related:
                result[key] = value

        return result

    @staticmethod
    def mask_secret(secret: str | None) -> str:
        """Masks sensitive content for safe observability (e.g., 'APIK...XXXX')."""
        if not secret:
            return "None"
        if len(secret) <= 8:
            return "****"
        return f"{secret[:4]}...{secret[-4:]}"

    def validate_secrets(self, expected_keys: list[str]) -> dict[str, bool]:
        """Validates the presence and format of multiple required keys."""
        res: dict[str, bool] = {}
        for key in expected_keys:
            val = self.get_secret(key)
            res[key] = val is not None and self.validate_format(key, val)
        return res

    def log_secrets_status(self, keys_to_check: list[str]):
        """Logs the readiness status of critical infrastructure keys without exposure."""
        logger.info("--- SECRETS CONFIGURATION AUDIT (VALIDATION) ---")
        for key in keys_to_check:
            value = self.get_secret(key)
            is_present = value is not None
            is_valid = self.validate_format(key, value) if value is not None else False

            if is_valid:
                status = "[OK]"
            elif is_present:
                status = "[FORMAT ERR]"
            else:
                status = "[MISSING]"
            masked = self.mask_secret(value) if is_present else "N/A"
            logger.info(f"- {key:25} {status:15} {masked}")
        logger.info("--------------------------------------------------")


# Singleton instance for global state management
_secrets_manager_instance = SecretsManager()


def get_secret(key_name: str, default: str | None = None, critical: bool = False) -> str | None:
    """Global interface for secure secret retrieval."""
    return _secrets_manager_instance.get_secret(key_name, default=default, critical=critical)


def mask_secret(secret: str | None) -> str:
    """Global interface for secret masking."""
    return SecretsManager.mask_secret(secret)


if __name__ == "__main__":
    # Internal validation logic
    test_keys = ["NEWS_API_KEY", "FRED_API_KEY", "HF_TOKEN", "TELEGRAM_TOKEN", "CRYPTO_KEY"]
    _secrets_manager_instance.log_secrets_status(test_keys)
