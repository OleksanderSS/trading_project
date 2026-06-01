import pytest

from src.training.security.path_security_validator import PathSecurityValidator


def test_path_security_validator_does_not_truncate_long_safe_paths():
    validator = PathSecurityValidator()
    safe_path = f"safe/{'a' * 150}/checkpoint.json"

    sanitized = validator.sanitize_path_input(safe_path)

    assert len(sanitized) > 100
    assert sanitized.endswith("checkpoint.json")


def test_path_security_validator_rejects_traversal_after_full_normalization(tmp_path):
    validator = PathSecurityValidator()
    malicious_path = f"safe/{'a' * 90}/../../../escape.txt"

    with pytest.raises(ValueError, match="traversal|outside"):
        validator.sanitize_path_input(malicious_path, base_dir=str(tmp_path))


def test_path_security_validator_allows_names_containing_two_dots():
    validator = PathSecurityValidator()

    assert validator.sanitize_path_input("model..v2/checkpoint.json")
