# src/core/version_checker.py
"""
Version Checker - Ensures runtime compatibility between Python environment and required dependencies.
"""

import sys
import importlib.metadata
from packaging import version
from typing import Dict, List, Tuple, Any
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("VersionChecker")


class VersionChecker:
    """
    Verifies Python runtime environment and package dependency version compliance.
    """
    
    def __init__(self, config_manager: Any):
        """
        Initializes the checker with the global configuration.
        """
        self.config_manager = config_manager
        # Attempt to retrieve specific versioning metadata from configuration
        self.version_config = config_manager.get_config('version', {})
    
    def check_python_version(self) -> Tuple[bool, str]:
        """
        Validates the current Python interpreter version against configured boundaries.
        """
        min_python = self.version_config.get('compatibility', {}).get('min_python', '3.8')
        max_python = self.version_config.get('compatibility', {}).get('max_python', '3.12')
        
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        # Enforce minimum version requirement
        if version.parse(current_version) < version.parse(min_python):
            msg = f"Incompatibility: Python {current_version} is below the required minimum of {min_python}."
            logger.error(msg)
            return False, msg
        
        # Issue warning for unverified higher versions
        if version.parse(current_version) > version.parse(max_python):
            msg = f"Warning: Python {current_version} exceeds the verified maximum of {max_python}. Compatibility issues may arise."
            logger.warning(msg)
            return True, msg
        
        msg = f"Python Environment Verified: {current_version} coincides with system requirements."
        logger.info(msg)
        return True, msg
    
    def check_package_versions(self) -> Tuple[bool, List[str]]:
        """
        Audits installed packages against requirement specifications.
        """
        required_packages = self.version_config.get('compatibility', {}).get('required_packages', {})
        issues = []
        
        for package_name, version_spec in required_packages.items():
            try:
                installed_version = importlib.metadata.version(package_name)
                
                # Parse requirement specification (e.g., ">=1.5.0")
                if ">=" in version_spec:
                    min_version = version_spec.replace(">=", "")
                    if version.parse(installed_version) < version.parse(min_version):
                        msg = f"Dependency Error: {package_name} {installed_version} is below required {min_version}."
                        logger.error(msg)
                        issues.append(msg)
                    else:
                        logger.debug(f"Package compliant: {package_name} {installed_version} OK")
                else:
                    logger.debug(f"Package check skipped (no specification): {package_name} {installed_version}")
            
            except importlib.metadata.PackageNotFoundError:
                msg = f"Dependency Missing: {package_name} is not installed in the current environment."
                logger.error(msg)
                issues.append(msg)
        
        return len(issues) == 0, issues
    
    def check_all(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Executes a comprehensive runtime environment and dependency audit.
        """
        logger.info("🔍 Initiating runtime compatibility and dependency audit...")
        
        python_ok, python_msg = self.check_python_version()
        packages_ok, package_issues = self.check_package_versions()
        
        all_ok = python_ok and packages_ok
        
        result = {
            'python_ok': python_ok,
            'python_msg': python_msg,
            'packages_ok': packages_ok,
            'package_issues': package_issues,
            'all_ok': all_ok
        }
        
        if all_ok:
            logger.info("✅ All runtime and dependency requirements are satisfied.")
        else:
            logger.error("❌ Environment configuration audit failed. Resolve dependency issues before continuing.")
        
        return all_ok, result
