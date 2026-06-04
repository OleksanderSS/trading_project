# src/core/security/secure_secrets_manager.py
"""
🔐 SECURE SECRETS MANAGER
Production-grade management of system secrets, API keys, and environment variables.
"""

import base64
import os
import re
import sys
from pathlib import Path

from cryptography.fernet import Fernet

# Utilize centralized project-wide logger
from src.core.logging.logger import ProjectLogger
from src.core.security.path_validator import PathValidationError, validate_safe_path

logger = ProjectLogger.get_logger(__name__)

class SecurityError(Exception):
    """Exception raised for critical security violations or missing required credentials."""
    pass


def _get_configured_env_search_paths() -> list[str | Path]:
    """Read optional env search paths without re-entering config initialization."""
    module = sys.modules.get("src.config.unified_config_manager")
    try:
        if module is not None:
            manager_cls = getattr(module, "UnifiedConfigManager", None)
            get_current_config = getattr(module, "get_current_config", None)
            if manager_cls is None or get_current_config is None:
                logger.debug("Skipping config env paths while UnifiedConfigManager is loading.")
                return []
        else:
            from src.config.unified_config_manager import UnifiedConfigManager as manager_cls
            from src.config.unified_config_manager import get_current_config

        if getattr(manager_cls, "_initializing", False):
            logger.debug("Skipping config env paths during UnifiedConfigManager initialization.")
            return []

        config = get_current_config()
        configured_paths = config.get('security.env_search_paths', [])
        return configured_paths if isinstance(configured_paths, list) else []
    except (ImportError, AttributeError, RuntimeError, RecursionError) as e:
        logger.warning(f"Skipping config search paths loading: {e}", exc_info=True)
        return []  # audit-ignore: BROAD_EXCEPTION_SILENT_RETURN

def load_dotenv(dotenv_path: str = '.env'):
    """
    Manually parses a .env file and injects keys into os.environ.
    This provides robustness in environments where the 'python-dotenv' library is not available.

    Search Protocol:
    1. Specified parameter path (default: .env)
    2. Configured search paths from config (if available)

    Security Note: Only searches within project-local directories to prevent
    unauthorized environment variable injection.
    """
    config_paths = _get_configured_env_search_paths()

    # Hierarchical list of potential .env locations
    # Restricted to local project context
    potential_paths: list[str | Path] = [
        dotenv_path,
    ]

    # Add configured paths if available
    if config_paths:
        potential_paths.extend(config_paths)

    found_path: Path | None = None
    for path in potential_paths:
        try:
            # Validate against current working directory to keep it local
            validated_path = validate_safe_path(path, base_dir=Path.cwd())
            if validated_path.exists():
                found_path = validated_path
                logger.info(f"Environment configuration identified: {found_path}")
                break
        except PathValidationError:
            continue

    if not found_path:
        logger.warning(
            f"No .env configuration file found in project local paths: {potential_paths}. Utilizing existing environment variables."
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
        logger.error(f"Critical failure reading environment file {found_path}: {e}", exc_info=True)
        raise SecurityError(f"Critical failure reading environment file {found_path}") from e


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
        """Loads and decrypts secrets from an encrypted payload using Fernet (requires CRYPTO_KEY)."""
        crypto_key = os.getenv("CRYPTO_KEY")
        # Validate path before checking existence
        try:
            safe_path = validate_safe_path(path, base_dir=Path.cwd())
        except PathValidationError:
            logger.warning(f"Invalid path for encrypted secrets: {path}")
            return

        if not crypto_key or not os.path.exists(safe_path):
            return

        try:
            # Derive a proper Fernet key from CRYPTO_KEY
            # Fernet requires a 32-byte base64-encoded key
            key_bytes = crypto_key.encode()
            if len(key_bytes) < 32:
                key_bytes = key_bytes.ljust(32, b'\0')
            elif len(key_bytes) > 32:
                key_bytes = key_bytes[:32]

            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)

            # Read and decrypt the encrypted secrets file
            with open(safe_path, 'rb') as f:
                encrypted_data = f.read()

            decrypted_data = fernet.decrypt(encrypted_data)

            # Parse the decrypted data (assumes JSON format)
            import json
            secrets_dict = json.loads(decrypted_data.decode('utf-8'))

            # Load decrypted secrets into environment
            for key, value in secrets_dict.items():
                os.environ[key] = value
                self._secrets_cache[key] = value

            logger.info(f"Successfully loaded {len(secrets_dict)} encrypted secrets from {safe_path}")

        except (ValueError, TypeError, Exception) as e:
            logger.error(f"Failed to load encrypted secrets from {safe_path}: {e}", exc_info=True)
            raise SecurityError(f"Failed to load/decrypt encrypted secrets from {safe_path}: {e}") from e

    def encrypt_secrets(self, secrets: dict[str, str], output_path: str = ".env.enc"):
        """Encrypts a dictionary of secrets and saves to file using Fernet."""
        crypto_key = os.getenv("CRYPTO_KEY")
        if not crypto_key:
            logger.error("CRYPTO_KEY environment variable is required for encryption")
            raise SecurityError("CRYPTO_KEY environment variable is required for encryption")

        try:
            # Validate output path before writing
            safe_output_path = validate_safe_path(output_path, base_dir=Path.cwd())

            # Derive a proper Fernet key from CRYPTO_KEY
            key_bytes = crypto_key.encode()
            if len(key_bytes) < 32:
                key_bytes = key_bytes.ljust(32, b'\0')
            elif len(key_bytes) > 32:
                key_bytes = key_bytes[:32]

            fernet_key = base64.urlsafe_b64encode(key_bytes)
            fernet = Fernet(fernet_key)

            # Convert secrets to JSON and encrypt
            import json
            secrets_json = json.dumps(secrets)
            encrypted_data = fernet.encrypt(secrets_json.encode('utf-8'))

            # Save encrypted data to file
            with open(safe_output_path, 'wb') as f:
                f.write(encrypted_data)

            logger.info(f"Successfully encrypted {len(secrets)} secrets to {safe_output_path}")

        except (ValueError, TypeError, Exception) as e:
            logger.error(f"Failed to encrypt secrets: {e}", exc_info=True)
            raise SecurityError(f"Failed to encrypt secrets: {e}") from e

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
