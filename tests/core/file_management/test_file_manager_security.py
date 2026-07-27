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

    def test_atomic_write_cleans_up_temp_file_on_integrity_check_failure(self):
        """_atomic_write raises its own OSError when validate_func fails,
        clearly intending its except block (cleanup temp file + log) to
        catch it - but OSError wasn't in the except tuple, so the temp
        .tmp file was never cleaned up and the intended error log never
        fired; the exception just propagated raw instead."""
        target_file = self.fm_base / "validated.json"

        with self.assertRaises(OSError):
            self.fm._atomic_write(
                target_file,
                write_func=lambda p: p.write_text("{}"),
                validate_func=lambda p: False,
            )

        temp_path = target_file.with_suffix(target_file.suffix + ".tmp")
        self.assertFalse(temp_path.exists())
        self.assertFalse(target_file.exists())

if __name__ == '__main__':
    unittest.main()
