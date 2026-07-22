"""
Path Safety Utilities

Provides centralized path validation and safe path resolution to prevent
path traversal attacks and ensure consistent path handling across the codebase.
"""

from pathlib import Path
from typing import ClassVar

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger('PathSafety')


class PathSafetyError(Exception):
    """Raised when a path validation fails."""
    pass


class PathSafety:
    """
    Centralized path safety utilities for secure file operations.
    
    Provides:
    - Safe path resolution preventing path traversal
    - Config-driven base directory management
    - Path validation against allowed roots
    """

    # Default allowed root directories for the project
    DEFAULT_ALLOWED_ROOTS: ClassVar[list[Path]] = [
        Path.cwd().resolve(),  # Project root
        Path.cwd().resolve() / 'data',
        Path.cwd().resolve() / 'data' / 'models',
        Path.cwd().resolve() / 'data' / 'cache',
        Path.cwd().resolve() / 'logs',
        Path.cwd().resolve() / 'config',
    ]

    def __init__(self, allowed_roots: list[Path] | None = None):
        """
        Initialize path safety with allowed root directories.
        
        Args:
            allowed_roots: List of allowed root directories. If None, uses DEFAULT_ALLOWED_ROOTS.
        """
        self.allowed_roots = [root.resolve() for root in (allowed_roots or self.DEFAULT_ALLOWED_ROOTS)]
        logger.info(f"PathSafety initialized with {len(self.allowed_roots)} allowed roots")

    def get_project_root(self) -> Path:
        """
        Get the project root directory in a safe manner.
        
        Returns:
            Resolved project root path
        """
        return Path.cwd().resolve()

    def get_data_dir(self, base_dir: Path | None = None) -> Path:
        """
        Get the data directory path.
        
        Args:
            base_dir: Optional base directory override
            
        Returns:
            Resolved data directory path
        """
        root = base_dir.resolve() if base_dir else self.get_project_root()
        data_dir = root / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir.resolve()

    def get_models_dir(self, base_dir: Path | None = None) -> Path:
        """
        Get the models directory path.
        
        Args:
            base_dir: Optional base directory override
            
        Returns:
            Resolved models directory path
        """
        data_dir = self.get_data_dir(base_dir)
        models_dir = data_dir / 'models'
        models_dir.mkdir(parents=True, exist_ok=True)
        return models_dir.resolve()

    def get_cache_dir(self, base_dir: Path | None = None) -> Path:
        """
        Get the cache directory path.
        
        Args:
            base_dir: Optional base directory override
            
        Returns:
            Resolved cache directory path
        """
        data_dir = self.get_data_dir(base_dir)
        cache_dir = data_dir / 'cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir.resolve()

    def get_logs_dir(self, base_dir: Path | None = None) -> Path:
        """
        Get the logs directory path.
        
        Args:
            base_dir: Optional base directory override
            
        Returns:
            Resolved logs directory path
        """
        root = base_dir.resolve() if base_dir else self.get_project_root()
        logs_dir = root / 'logs'
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir.resolve()

    def get_config_dir(self, base_dir: Path | None = None) -> Path:
        """
        Get the config directory path.
        
        Args:
            base_dir: Optional base directory override
            
        Returns:
            Resolved config directory path
        """
        root = base_dir.resolve() if base_dir else self.get_project_root()
        config_dir = root / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir.resolve()

    def validate_path(self, path: Path | str, allow_absolute: bool = False) -> Path:
        """
        Validate a path to prevent path traversal attacks.
        
        Args:
            path: Path to validate
            allow_absolute: Whether to allow absolute paths
            
        Returns:
            Resolved and validated path
            
        Raises:
            PathSafetyError: If path validation fails
        """
        path_obj = Path(path)

        # Resolve to absolute path to normalize
        resolved_path = path_obj.resolve()

        # Check for path traversal attempts
        if '..' in path_obj.parts:
            raise PathSafetyError(f"Path traversal detected in path: {path}")

        # If not allowing absolute paths, ensure it's relative to project root
        if not allow_absolute and path_obj.is_absolute():
            # Check if it's within allowed roots
            is_allowed = any(
                str(resolved_path).startswith(str(root))
                for root in self.allowed_roots
            )
            if not is_allowed:
                raise PathSafetyError(
                    f"Absolute path not in allowed roots: {resolved_path}. "
                    f"Allowed roots: {self.allowed_roots}"
                )

        return resolved_path

    def safe_join(self, base: Path | str, *paths: str | Path) -> Path:
        """
        Safely join paths preventing path traversal.
        
        Args:
            base: Base directory
            *paths: Path components to join
            
        Returns:
            Safely joined and resolved path
            
        Raises:
            PathSafetyError: If path traversal is detected
        """
        base_path = Path(base).resolve()
        result = base_path

        for path_part in paths:
            part = Path(path_part)

            # Check for path traversal
            if '..' in part.parts:
                raise PathSafetyError(f"Path traversal detected in component: {path_part}")

            result = result / part

        # Resolve and validate
        resolved = result.resolve()

        # Ensure result is within base
        try:
            resolved.relative_to(base_path)
        except ValueError as e:
            raise PathSafetyError(
                f"Resolved path escapes base directory: {resolved} not in {base_path}"
            ) from e

        return resolved


# Global singleton instance
_global_path_safety: PathSafety | None = None


def get_path_safety(allowed_roots: list[Path] | None = None) -> PathSafety:
    """
    Get the global PathSafety instance.
    
    Args:
        allowed_roots: Optional allowed roots for initialization
        
    Returns:
        PathSafety instance
    """
    global _global_path_safety
    if _global_path_safety is None:
        _global_path_safety = PathSafety(allowed_roots)
    return _global_path_safety


def safe_resolve_path(path: Path | str, base_dir: Path | None = None) -> Path:
    """
    Safely resolve a path against a base directory.
    
    Args:
        path: Path to resolve
        base_dir: Base directory (defaults to project root)
        
    Returns:
        Resolved safe path
    """
    path_safety = get_path_safety()
    root = base_dir.resolve() if base_dir else path_safety.get_project_root()
    return path_safety.safe_join(root, path)
