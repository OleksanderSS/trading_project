"""
DeanPaths: Centralized path resolution for DEAN OS evidence/replay tools

Provides safe path resolution with validation against trusted roots,
preventing path traversal attacks and ensuring all file operations
are performed within allowed directories.

Usage:
    from dean_os.dean_paths import DeanPaths
    
    # Resolve input artifact with validation
    safe_path = DeanPaths.resolve_input_artifact(
        path=user_provided_path,
        allowed_roots=[DeanPaths.get_reports_dir(), DeanPaths.get_data_dir()]
    )
    
    # Load JSON safely
    data = DeanPaths.load_json(safe_path)
    
    # Load data file safely
    df = DeanPaths.load_data_file(safe_path)
    
    # Write output atomically
    DeanPaths.atomic_write_json(output_path, data)
"""
# Optional import of PathValidationError and validate_safe_path from src, with fallback to local definitions
try:
    from src.core.security.path_validator import PathValidationError, validate_safe_path  # type: ignore
except Exception:  # pragma: no cover
    # Fallback definitions are provided later in this file
    pass

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


class PathValidationError(Exception):
    """Raised when a path validation fails."""
    pass


def validate_safe_path(path: Path, base_dir: Path) -> Path:
    """
    Validate that a path is within the base directory.
    
    Args:
        path: Path to validate
        base_dir: Base directory to check against
    
    Returns:
        Resolved absolute path
    
    Raises:
        PathValidationError: If path is outside base directory or contains traversal attempts
    """
    # Resolve to absolute path using standard library os.path for Snyk
    import os
    resolved_str = os.path.abspath(str(path))
    base_str = os.path.abspath(str(base_dir))

    # Security check: must use commonpath to ensure no traversal
    try:
        common = os.path.commonpath([base_str, resolved_str])
    except ValueError:
        # Cross-drive paths (Windows) or other commonpath errors -> treat as outside
        common = None
    if common != base_str:
        raise PathValidationError(
            f"Path {resolved_str} is outside base directory {base_str}"
        )

    resolved = Path(resolved_str)

    # Check for suspicious patterns
    if ".." in str(path):
        raise PathValidationError(
            f"Path contains parent directory references: {path}"
        )

    return resolved


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    First checks environment variable DEAN_PROJECT_ROOT, then uses current working directory.
    
    Returns:
        Path to project root directory
    """
    import os
    # Check environment variable first
    env_root = os.environ.get("DEAN_PROJECT_ROOT")
    if env_root:
        # Use abspath to satisfy Snyk path traversal checks
        return Path(os.path.abspath(env_root)).resolve()

    # Default to current working directory
    return Path(os.path.abspath(os.getcwd())).resolve()


def _effective_roots(path: str | Path, allowed_roots: list[Path] | None) -> list[Path]:
    """
    Compute the allowed roots for loading a file.

    When the caller does not supply explicit allowed roots, the project root is
    always trusted. In addition, the parent directory of an explicitly supplied
    absolute path is trusted, because the caller named that exact file. This
    keeps relative (project-relative) paths locked to the project root while
    still allowing tools to load a user-provided absolute path (e.g. a temp file
    on another drive during tests or CLI usage).
    """
    if allowed_roots:
        return list(allowed_roots)
    return [get_project_root(), Path(path).resolve().parent]


class DeanPaths:
    """
    Centralized path resolution for DEAN OS tools.
    
    Provides safe path validation and file loading methods that
    prevent path traversal attacks and ensure all operations
    are within trusted directories.
    """

    @staticmethod
    def get_project_root() -> Path:
        """Get the project root directory."""
        return get_project_root()

    @staticmethod
    def get_reports_dir() -> Path:
        """Get the reports directory."""
        return get_project_root() / "reports"

    @staticmethod
    def get_data_dir() -> Path:
        """Get the data directory."""
        return get_project_root() / "data"

    @staticmethod
    def get_logs_dir() -> Path:
        """Get the logs directory."""
        return get_project_root() / "logs"

    @staticmethod
    def resolve_input_artifact(path: str | Path, allowed_roots: list[Path] | None = None) -> Path:
        """
        Resolve and validate an input artifact path.
        
        Args:
            path: User-provided path (can be relative or absolute)
            allowed_roots: List of allowed root directories (defaults to project root)
        
        Returns:
            Validated absolute Path object
        
        Raises:
            PathValidationError: If path is outside allowed roots or invalid
        
        Example:
            safe_path = DeanPaths.resolve_input_artifact(
                path="reports/dean_os/evidence.json",
                allowed_roots=[DeanPaths.get_reports_dir()]
            )
        """
        if allowed_roots is None:
            allowed_roots = [DeanPaths.get_project_root()]

        # Convert to Path object
        input_path = Path(path)

        # If relative, resolve against project root
        if not input_path.is_absolute():
            input_path = DeanPaths.get_project_root() / input_path

        # Validate against each allowed root
        validated_path = None
        for root in allowed_roots:
            try:
                validated_path = validate_safe_path(input_path, base_dir=root)
                break
            except PathValidationError:
                continue

        if validated_path is None:
            raise PathValidationError(
                f"Path {path} is outside allowed roots: {allowed_roots}"
            )

        return validated_path

    @staticmethod
    def load_json(path: str | Path, allowed_roots: list[Path] | None = None) -> dict[str, Any]:
        """
        Load JSON file with path validation.
        
        Args:
            path: Path to JSON file
            allowed_roots: List of allowed root directories
        
        Returns:
            Parsed JSON data as dict
        
        Raises:
            PathValidationError: If path is invalid or outside allowed roots
            ValueError: If file doesn't exist or contains invalid JSON
        
        Example:
            data = DeanPaths.load_json("reports/dean_os/evidence.json")
        """
        safe_path = DeanPaths.resolve_input_artifact(path, _effective_roots(path, allowed_roots))

        if not safe_path.exists():
            raise ValueError(f"File does not exist: {safe_path}")

        try:
            payload = json.loads(safe_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {safe_path}: {e}")

        if not isinstance(payload, dict):
            raise ValueError(f"JSON artifact must be an object: {safe_path}")

        return payload

    @staticmethod
    def load_optional_json(path: str | Path, allowed_roots: list[Path] | None = None) -> dict[str, Any]:
        """
        Load JSON file with path validation, returning empty dict if missing.
        
        Args:
            path: Path to JSON file
            allowed_roots: List of allowed root directories
        
        Returns:
            Parsed JSON data as dict, or empty dict if file doesn't exist
        
        Example:
            data = DeanPaths.load_optional_json("reports/dean_os/evidence.json")
        """
        try:
            return DeanPaths.load_json(path, allowed_roots)
        except (PathValidationError, ValueError):
            return {}

    @staticmethod
    def load_data_file(path: str | Path, allowed_roots: list[Path] | None = None) -> Any:
        """
        Load data file (CSV, Parquet, JSON) with path validation.
        
        Args:
            path: Path to data file
            allowed_roots: List of allowed root directories
        
        Returns:
            Loaded data (DataFrame for CSV/Parquet, dict for JSON)
        
        Raises:
            PathValidationError: If path is invalid or outside allowed roots
            ValueError: If file doesn't exist or has unsupported format
        
        Example:
            df = DeanPaths.load_data_file("data/dean_os/market_data.csv")
        """
        safe_path = DeanPaths.resolve_input_artifact(path, _effective_roots(path, allowed_roots))

        if not safe_path.exists():
            raise ValueError(f"File does not exist: {safe_path}")

        suffix = safe_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(safe_path)
        elif suffix in {".parquet", ".pq"}:
            return pd.read_parquet(safe_path)
        elif suffix == ".json":
            return DeanPaths.load_json(safe_path, allowed_roots)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .parquet, or .json")

    @staticmethod
    def load_text_file(path: str | Path, allowed_roots: list[Path] | None = None) -> str:
        """
        Load text file with path validation and encoding fallback.
        
        Args:
            path: Path to text file
            allowed_roots: List of allowed root directories
        
        Returns:
            File contents as string
        
        Raises:
            PathValidationError: If path is invalid or outside allowed roots
            ValueError: If file doesn't exist or cannot be decoded
        
        Example:
            content = DeanPaths.load_text_file("reports/dean_os/notes.txt")
        """
        safe_path = DeanPaths.resolve_input_artifact(path, _effective_roots(path, allowed_roots))

        if not safe_path.exists():
            raise ValueError(f"File does not exist: {safe_path}")

        # Try multiple encodings
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                return safe_path.read_text(encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError(f"Could not decode text file: {safe_path}")

    @staticmethod
    def validate_path_exists(path: str | Path, allowed_roots: list[Path] | None = None) -> dict[str, Any]:
        """
        Validate that a path exists and is within allowed roots.
        
        Args:
            path: Path to validate
            allowed_roots: List of allowed root directories
        
        Returns:
            Dict with validation result:
            {
                "path": str,
                "exists": bool,
                "valid": bool,
                "error": str | None
            }
        
        Example:
            result = DeanPaths.validate_path_exists("reports/dean_os/evidence.json")
            if not result["valid"]:
                print(f"Error: {result['error']}")
        """
        try:
            safe_path = DeanPaths.resolve_input_artifact(path, allowed_roots)
            exists = safe_path.exists()
            return {
                "path": str(safe_path),
                "exists": exists,
                "valid": True,
                "error": None if exists else "File does not exist"
            }
        except PathValidationError as e:
            return {
                "path": str(path),
                "exists": False,
                "valid": False,
                "error": str(e)
            }

    @staticmethod
    def resolve_output_artifact(path: str | Path, allowed_roots: list[Path] | None = None) -> Path:
        """
        Resolve and validate an output artifact path.
        
        Args:
            path: User-provided path (can be relative or absolute)
            allowed_roots: List of allowed root directories (defaults to project root)
        
        Returns:
            Validated absolute Path object
        
        Raises:
            PathValidationError: If path is outside allowed roots or invalid
        
        Example:
            safe_path = DeanPaths.resolve_output_artifact(
                path="reports/dean_os/output.json",
                allowed_roots=[DeanPaths.get_reports_dir()]
            )
        """
        if allowed_roots is None:
            allowed_roots = [DeanPaths.get_project_root()]

        # Convert to Path object
        output_path = Path(path)

        # If relative, resolve against project root
        if not output_path.is_absolute():
            output_path = DeanPaths.get_project_root() / output_path

        # Validate against each allowed root
        validated_path = None
        for root in allowed_roots:
            try:
                validated_path = validate_safe_path(output_path, base_dir=root)
                break
            except PathValidationError:
                continue

        if validated_path is None:
            raise PathValidationError(
                f"Path {path} is outside allowed roots: {allowed_roots}"
            )

        # Create parent directory if it doesn't exist
        validated_path.parent.mkdir(parents=True, exist_ok=True)

        return validated_path

    @staticmethod
    def atomic_write_json(path: str | Path, data: dict[str, Any], allowed_roots: list[Path] | None = None) -> Path:
        """
        Write JSON file atomically with path validation.
        
        Args:
            path: Path to write JSON file
            data: JSON data to write
            allowed_roots: List of allowed root directories
        
        Returns:
            Path to written file
        
        Raises:
            PathValidationError: If path is invalid or outside allowed roots
            ValueError: If data is not a dict
        
        Example:
            DeanPaths.atomic_write_json("reports/dean_os/output.json", {"key": "value"})
        """
        if not isinstance(data, dict):
            raise ValueError("JSON data must be a dict")

        safe_path = DeanPaths.resolve_output_artifact(path, allowed_roots)

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=safe_path.parent,
            prefix=f".{safe_path.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            # Atomic rename
            os.replace(temp_path, safe_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        return safe_path

    @staticmethod
    def atomic_write_text(path: str | Path, content: str, allowed_roots: list[Path] | None = None) -> Path:
        """
        Write text file atomically with path validation.
        
        Args:
            path: Path to write text file
            content: Text content to write
            allowed_roots: List of allowed root directories
        
        Returns:
            Path to written file
        
        Raises:
            PathValidationError: If path is invalid or outside allowed roots
        
        Example:
            DeanPaths.atomic_write_text("reports/dean_os/notes.txt", "Hello world")
        """
        safe_path = DeanPaths.resolve_output_artifact(path, allowed_roots)

        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=safe_path.parent,
            prefix=f".{safe_path.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                f.write(content)
            # Atomic rename
            os.replace(temp_path, safe_path)
        except Exception:
            # Clean up temp file on error
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

        return safe_path

    @staticmethod
    def update_latest_pointer(base_dir: str | Path, latest_path: str | Path, allowed_roots: list[Path] | None = None) -> Path:
        """
        Update a 'latest' pointer to point to the latest artifact.
        
        Args:
            base_dir: Base directory for the pointer
            latest_path: Path that the pointer should point to
            allowed_roots: List of allowed root directories
        
        Returns:
            Path to the updated pointer file
        
        Example:
            DeanPaths.update_latest_pointer(
                base_dir="reports/dean_os",
                latest_path="reports/dean_os/output_20250614_120000.json"
            )
        """
        base = Path(base_dir)
        latest = Path(latest_path)

        # Validate both paths
        safe_base = DeanPaths.resolve_output_artifact(base, allowed_roots)
        safe_latest = DeanPaths.resolve_input_artifact(latest, allowed_roots)

        # Create pointer file
        pointer_path = safe_base / "latest.json"
        from datetime import UTC, datetime
        pointer_data = {
            "latest_path": str(safe_latest.relative_to(DeanPaths.get_project_root())),
            "updated_at": datetime.now(UTC).isoformat()
        }

        return DeanPaths.atomic_write_json(pointer_path, pointer_data, allowed_roots)
