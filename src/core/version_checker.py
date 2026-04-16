# src/core/version_checker.py

import sys
import importlib.metadata
from packaging import version
from typing import Dict, List, Tuple
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("VersionChecker")


class VersionChecker:
    """
    ✅ Перевіряє сумісність версій Python та залежностей.
    """
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.version_config = config_manager.get_config('version', {})
    
    def check_python_version(self) -> Tuple[bool, str]:
        """Перевіряє версію Python."""
        min_python = self.version_config.get('compatibility', {}).get('min_python', '3.8')
        max_python = self.version_config.get('compatibility', {}).get('max_python', '3.12')
        
        current_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        if version.parse(current_version) < version.parse(min_python):
            msg = f"❌ Python {current_version} < {min_python} (мінімум)"
            logger.error(msg)
            return False, msg
        
        if version.parse(current_version) > version.parse(max_python):
            msg = f"⚠️ Python {current_version} > {max_python} (максимум, може бути несумісно)"
            logger.warning(msg)
            return True, msg
        
        msg = f"✅ Python {current_version} OK"
        logger.info(msg)
        return True, msg
    
    def check_package_versions(self) -> Tuple[bool, List[str]]:
        """Перевіряє версії залежностей."""
        required_packages = self.version_config.get('compatibility', {}).get('required_packages', {})
        issues = []
        
        for package_name, version_spec in required_packages.items():
            try:
                installed_version = importlib.metadata.version(package_name)
                
                # Парсимо версійну специфікацію (наприклад, ">=1.5.0")
                if ">=" in version_spec:
                    min_version = version_spec.replace(">=", "")
                    if version.parse(installed_version) < version.parse(min_version):
                        msg = f"❌ {package_name} {installed_version} < {min_version}"
                        logger.error(msg)
                        issues.append(msg)
                    else:
                        logger.info(f"✅ {package_name} {installed_version} OK")
                else:
                    logger.info(f"✅ {package_name} {installed_version} (не перевіряється)")
            
            except importlib.metadata.PackageNotFoundError:
                msg = f"❌ {package_name} не встановлено"
                logger.error(msg)
                issues.append(msg)
        
        return len(issues) == 0, issues
    
    def check_all(self) -> Tuple[bool, Dict[str, any]]:
        """Перевіряє всі версійні вимоги."""
        logger.info("🔍 Перевірка версійної сумісності...")
        
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
            logger.info("✅ Всі версійні вимоги задоволені")
        else:
            logger.error("❌ Деякі версійні вимоги не задоволені")
        
        return all_ok, result
