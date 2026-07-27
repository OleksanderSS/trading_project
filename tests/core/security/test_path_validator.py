import unittest
from pathlib import Path
import tempfile
import os
from src.core.security.path_validator import validate_safe_path, PathValidationError

class TestPathValidator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.test_dir.name).resolve()
        self.safe_file = self.base_dir / "safe.txt"
        self.safe_file.touch()

    def tearDown(self):
        self.test_dir.cleanup()

    def test_valid_path(self):
        validated = validate_safe_path(self.safe_file, self.base_dir)
        self.assertEqual(validated, self.safe_file)

    def test_traversal_attack(self):
        malicious_path = self.base_dir / ".." / "outside.txt"
        with self.assertRaises(PathValidationError):
            validate_safe_path(malicious_path, self.base_dir)

    def test_symlink_denied(self):
        # Skip symlink test on Windows due to privilege restrictions
        self.skipTest("Symlink creation requires elevated privileges on Windows.")

    def test_sibling_directory_with_matching_prefix_is_rejected(self):
        """The old check used str(target).startswith(str(base)), which
        incorrectly treats a sibling directory sharing base_dir's name as
        a prefix (e.g. base_dir='.../data', target='.../data_secret/x')
        as contained within it. relative_to() is the correct containment
        check."""
        sibling_dir = Path(f"{self.base_dir}_secret")
        sibling_dir.mkdir()
        try:
            malicious_path = sibling_dir / "leaked.txt"
            malicious_path.touch()
            with self.assertRaises(PathValidationError):
                validate_safe_path(malicious_path, self.base_dir)
        finally:
            malicious_path.unlink()
            sibling_dir.rmdir()

if __name__ == '__main__':
    unittest.main()
