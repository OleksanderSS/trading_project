import os
import re


def sanitize_path_input(path_input: str) -> str:
    """
    Sanitize path input to prevent path traversal attacks.

    Args:
        path_input: Input string that will be used in file paths

    Returns:
        Sanitized string safe for path construction

    Raises:
        ValueError: If path traversal or null byte is detected.
    """
    if not path_input:
        return ""

    # Check for null bytes
    if "\0" in path_input:
        raise ValueError("Null byte detected in path.")

    # Check for path traversal attempts
    if ".." in path_input:
        raise ValueError("Path traversal detected.")

    # Check for absolute paths
    if os.path.isabs(path_input):
        raise ValueError("Absolute paths not allowed.")

    # Remove dangerous characters, but keep path separators (/ and \)
    sanitized = re.sub(r"[^a-zA-Z0-9./\\]", "_", path_input)

    return sanitized
