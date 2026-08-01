from pathlib import Path


class PathValidationError(Exception):
    """Exception raised for path validation errors."""
    pass


def _reject_symlinks_within(base_path: Path, path: Path) -> None:
    """Raise if `path` or any component between it and `base_path` is a link.

    The check used to read

        target_path = Path(path).resolve()
        if not allow_symlinks and target_path.is_symlink():

    but resolve() follows symlinks, so it asked whether the ALREADY RESOLVED
    path is a link -- which by construction it is not. The flag never fired.

    Containment was never at risk from this: resolve() turns a link pointing
    outside the project into the outside path, and the relative_to check in
    the caller then rejects it. What the flag failed to do is reject a link
    that stays inside base_dir. So this was a promise the function did not
    keep, not a way out of the directory.
    """
    probe = path if path.is_absolute() else base_path / path
    while True:
        if probe.is_symlink():
            raise PathValidationError(f"Symlinks are not allowed: '{probe}'")
        if probe == base_path or probe.parent == probe:
            return
        probe = probe.parent


def validate_safe_path(
    path: str | Path,
    base_dir: str | Path,
    allow_symlinks: bool = False
) -> Path:
    """
    Validates that the given path is contained within the base_dir
    and prevents Path Traversal attacks.

    Args:
        path: The path to validate. A relative path is interpreted against
            base_dir, not the process working directory -- plain resolve()
            would silently depend on where the program was started from.
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
        given = Path(path)
        target_path = (given if given.is_absolute() else base_path / given).resolve()

        # Check if the target is within the base directory. This is the actual
        # containment guarantee, and it holds for symlinked paths too, because
        # resolve() has already followed them.
        try:
            target_path.relative_to(base_path)
        except ValueError:
            raise PathValidationError(
                f"Path '{target_path}' is outside authorized base directory '{base_path}'"
            ) from None

        if not allow_symlinks:
            _reject_symlinks_within(base_path, given)

        return target_path

    except (OSError, ValueError, RuntimeError) as e:
        # OSError included: on Windows, resolve() and is_symlink() raise it for
        # names the filesystem rejects, and it used to escape as itself rather
        # than as the PathValidationError this function documents.
        raise PathValidationError(f"Invalid path encountered: {e}") from e
