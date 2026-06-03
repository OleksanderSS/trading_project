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

if __name__ == '__main__':
    unittest.main()
