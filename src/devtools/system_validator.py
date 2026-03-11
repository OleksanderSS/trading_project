# src/devtools/system_validator.py

import importlib
import os
import psutil
from pathlib import Path
from typing import Dict, List, Any, Tuple

from src.core.file_management.file_manager import FileManager
from src.core.logging.logger import ProjectLogger
from src.core.security.secure_secrets_manager import SecretsManager

# Initialize logger for the module
logger = ProjectLogger.get_logger("SystemValidator")

class SystemValidator:
    """
    Performs a health check on the project to ensure key components are in place.
    Validates directory structure, core files, and essential library imports.
    """

    def __init__(self, file_manager: FileManager, root_path: str = "."):
        self.fm = file_manager
        self.root = Path(root_path)
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        self.secrets_manager = SecretsManager()
        logger.info("SystemValidator initialized.")

    def run_all_checks(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes all validation checks and returns the overall status and detailed results.

        Returns:
            Tuple[bool, Dict[str, Any]]: (Success status, detailed results dictionary)
        """
        self.results = {}
        self.errors = []
        logger.info("Starting comprehensive system validation...")

        self._check_directories([
            "src/analytics/context",
            "src/analytics/reporting",
            "src/features/enrichers",
            "src/data/collectors",
            "src/pipeline/stages",
            "src/devtools",
            "src/core/logging"
        ])
        self._check_files([
            "src/core/file_management/file_manager.py",
            "src/utils/trading_calendar.py",
            "src/utils/rate_limiter.py",
            "src/core/logging/logger.py",
            "src/pipeline/pipeline_orchestrator.py",
            "src/config/unified_config_manager.py",
            "src/devtools/task_manager.py"
        ])
        self._check_python_libraries([
            "pandas", "numpy", "yfinance", "holidays", "duckdb"
        ])
        self._check_system_resources()
        self._check_secrets(['NEWS_API_KEY', 'FRED_API_KEY', 'TELEGRAM_TOKEN'])
        # self._check_database_availability("data/main.duckdb") # DISABLED: Causes lock conflicts on startup.

        self._summarize_results()
        self.print_report()
        
        is_success = not self.errors
        if not is_success:
            logger.error(f"System validation failed with {len(self.errors)} errors.")
        else:
            logger.info("System validation passed successfully.")

        return is_success, self.results

    def _check_directories(self, dir_paths: List[str]):
        """Checks for the existence of essential directories."""
        results = {}
        for dir_path in dir_paths:
            path = self.root / dir_path
            if os.path.isdir(path):
                results[dir_path] = {"status": "PASSED"}
            else:
                results[dir_path] = {"status": "FAILED", "error": "Directory not found."}
                self.errors.append(f"Missing directory: {dir_path}")
        self.results["directories"] = results
        logger.info("Directory structure validation complete.")

    def _check_files(self, file_paths: List[str]):
        """Checks for the existence of critical files."""
        results = {}
        for file_path in file_paths:
            path = self.root / file_path
            if os.path.isfile(path):
                results[file_path] = {"status": "PASSED"}
            else:
                results[file_path] = {"status": "FAILED", "error": "File not found."}
                self.errors.append(f"Missing file: {file_path}")
        self.results["core_files"] = results
        logger.info("Core file validation complete.")

    def _check_python_libraries(self, libraries: List[str]):
        """Checks if essential third-party libraries can be imported."""
        results = {}
        for lib in libraries:
            try:
                importlib.import_module(lib)
                results[lib] = {"status": "PASSED"}
            except ImportError:
                results[lib] = {"status": "FAILED", "error": "Library not installed."}
                self.errors.append(f"Missing Python library: {lib}. Please install it.")
        self.results["python_libraries"] = results
        logger.info("Python library validation complete.")

    def _check_system_resources(self):
        """Checks for minimum system requirements: RAM and optional GPU."""
        results = {}
        # RAM Check
        mem = psutil.virtual_memory()
        free_gb = mem.available / (1024 ** 3)
        if free_gb < 2.0:
            results["ram"] = {"status": "WARNING", "details": f"Low RAM: {free_gb:.2f}GB available."}
            logger.warning(f"Low system memory detected: {free_gb:.2f}GB")
        else:
            results["ram"] = {"status": "PASSED", "details": f"{free_gb:.2f}GB available."}

        # GPU Check (PyTorch)
        gpu_status = "NOT_FOUND"
        try:
            torch = importlib.import_module("torch")
            if torch.cuda.is_available():
                gpu_status = f"FOUND (CUDA: {torch.cuda.get_device_name(0)})"
            else:
                gpu_status = "NOT_AVAILABLE (CUDA not detected)"
        except ImportError:
            gpu_status = "NOT_CHECKED (torch not installed)"
        
        results["gpu"] = {"status": "INFO", "details": gpu_status}
        self.results["system_resources"] = results
        logger.info("System resource check complete.")

    def _check_secrets(self, critical_keys: List[str]):
        """Validates that critical API keys and secrets are present."""
        results = {}
        for key in critical_keys:
            secret = self.secrets_manager.get_secret(key)
            if secret:
                results[key] = {"status": "PASSED", "masked": SecretsManager.mask_secret(secret)}
            else:
                results[key] = {"status": "FAILED", "error": "Secret missing from .env or environment."}
                self.errors.append(f"Missing critical secret: {key}")
        
        self.results["secrets"] = results
        logger.info("Secrets validation complete.")

    # def _check_database_availability(self, db_path: str):
    #     """Checks if the DuckDB database file is accessible."""
    #     results = {}
    #     path = self.root / db_path
    #     try:
    #         import duckdb
    #         # Try connecting to ensure the file is valid and not locked exclusively
    #         conn = duckdb.connect(database=str(path), read_only=True)
    #         conn.close()
    #         results[db_path] = {"status": "PASSED"}
    #     except Exception as e:
    #         results[db_path] = {"status": "FAILED", "error": str(e)}
    #         self.errors.append(f"Database error at {db_path}: {str(e)}")
    #     
    #     self.results["database"] = results
    #     logger.info("Database availability check complete.")

    def _summarize_results(self):
        """Generates a summary of the validation checks."""
        total_checks = 0
        passed_checks = 0
        for category in self.results.values():
            if isinstance(category, dict):
                for item in category.values():
                    if isinstance(item, dict):
                        total_checks += 1
                        if item.get("status") in ["PASSED", "INFO"]:
                            passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 100
        
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": len(self.errors),
            "success_rate": f"{success_rate:.1f}%",
            "overall_status": "HEALTHY" if not self.errors else "NEEDS ATTENTION"
        }

    def print_report(self):
        """Prints a formatted report of the validation results."""
        summary = self.results.get("summary", {})
        print("\n--- System Validation Report ---")
        print(f"Overall Status: {summary.get('overall_status', 'UNKNOWN')}")
        print(f"Checks Passed: {summary.get('passed_checks', 0)}/{summary.get('total_checks', 0)} ({summary.get('success_rate', 'N/A')})\n")

        if self.errors:
            print("--- Issues Found ---")
            for error in self.errors:
                print(f"[ERROR] {error}")
            print("--------------------\n")
        else:
            print("System appears to be configured correctly.\n")
