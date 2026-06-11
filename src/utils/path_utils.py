"""
Path utilities for secure file operations.
Prevents path traversal attacks and ensures safe file handling.
"""

import re
from pathlib import Path


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename safe for file system
    """
    # Remove path separators and dangerous characters
    safe_chars = re.sub(r'[^\w\-_\.]', '_', filename)
    # Remove consecutive underscores
    safe_chars = re.sub(r'_+', '_', safe_chars)
    # Remove leading/trailing underscores
    safe_chars = safe_chars.strip('_')
    return safe_chars or 'unnamed'


def validate_path_within_directory(file_path: str | Path, base_dir: str | Path) -> Path:
    """
    Validate that file_path is within base_dir to prevent path traversal.

    Args:
        file_path: File path to validate
        base_dir: Base directory that file_path must be within

    Returns:
        Resolved absolute path if safe

    Raises:
        ValueError: If path traversal is detected
    """
    file_path = Path(file_path).resolve()
    base_dir = Path(base_dir).resolve()

    try:
        # Check if the file path is within the base directory
        file_path.relative_to(base_dir)
        return file_path
    except ValueError:
        raise ValueError(f"Path traversal detected: {file_path} is not within {base_dir}")


def safe_file_operation(base_dir: str | Path, filename: str, operation: str = 'create') -> Path:
    """
    Create a safe file path within base_dir.

    Args:
        base_dir: Base directory for file operations
        filename: Desired filename (will be sanitized)
        operation: Type of operation ('create', 'read', 'write', 'delete')

    Returns:
        Safe file path

    Raises:
        ValueError: If path traversal is detected or filename is invalid
    """
    safe_name = sanitize_filename(filename)
    safe_path = Path(base_dir) / safe_name

    # Validate the path is within base directory
    return validate_path_within_directory(safe_path, base_dir)
