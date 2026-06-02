import unittest
from pathlib import Path
import tempfile
import os
from src.core.file_management.file_manager import FileManager
from src.core.security.path_validator import PathValidationError

class TestFileManagerSecurity(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.test_dir.name).resolve()
        # Set base_dir to a subdirectory to test path traversal
        self.fm_base = self.base_dir / "app_data"
        self.fm_base.mkdir()
        self.fm = FileManager(base_dir=self.fm_base)

    def tearDown(self):
        self.test_dir.cleanup()

    def test_safe_file_access(self):
        safe_file = self.fm_base / "test.json"
        self.fm.save_json({"key": "value"}, safe_file)
        self.assertTrue(safe_file.exists())
        data = self.fm.load_json(safe_file)
        self.assertEqual(data, {"key": "value"})

    def test_traversal_attack_blocked(self):
        # Attempt to save outside base_dir using ../
        malicious_file = "../../outside.json"
        
        with self.assertRaises(PathValidationError):
            self.fm.save_json({"key": "value"}, malicious_file)

if __name__ == '__main__':
    unittest.main()
