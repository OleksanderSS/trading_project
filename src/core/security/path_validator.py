from pathlib import Path


class PathValidationError(Exception):
    """Exception raised for path validation errors."""
    pass

def validate_safe_path(
    path: str | Path,
    base_dir: str | Path,
    allow_symlinks: bool = False
) -> Path:
    """
    Validates that the given path is contained within the base_dir
    and prevents Path Traversal attacks.

    Args:
        path: The path to validate.
        base_dir: The authorized base directory.
        allow_symlinks: Whether to allow symlinks.

    Returns:
        The validated, absolute Path object.

    Raises:
        PathValidationError: If the path is outside base_dir or invalid.
    """
    try:
        # Resolve to absolute, canonicalized paths
        base_path = Path(base_dir).resolve()
        target_path = Path(path).resolve()

        if not allow_symlinks and target_path.is_symlink():
            raise PathValidationError("Symlinks are not allowed.")

        # Check if the target is within the base directory
        try:
            target_path.relative_to(base_path)
        except ValueError:
            raise PathValidationError(
                f"Path '{target_path}' is outside authorized base directory '{base_path}'"
            ) from None

        return target_path

    except (ValueError, RuntimeError) as e:
        raise PathValidationError(f"Invalid path encountered: {e}") from e
