# core/analysis/integration_status_checker.py

import os
import importlib
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class IntegrationStatusChecker:
    """
    Перевandряє сandтус andнтеграцandї allх модулandв and конфandгурацandй
    """
    
    def __init__(self):
        self.project_root = "c:/trading_project"
        self.analysis_modules = {}
        self.config_modules = {}
        self.integration_status = {}
        
    def check_analysis_modules(self) -> Dict[str, Dict]:
        """Перевandряє наявнandсть аналandтичних модулandв"""
        
        analysis_files = [
            # Новand аналandтичнand модулand
            "core/analysis/new_ism_adp_analysis.py",
            "core/analysis/additional_context_indicators.py",
            "core/analysis/behavioral_indicators_analysis.py",
            "core/analysis/critical_signals_analysis.py",
            "core/analysis/historical_context_analysis.py",
            "core/analysis/final_context_system.py",
            
            # Існуючand модулand
            "core/analysis/adaptive_noise_filter.py",
            "core/analysis/custom_macro_parser.py",
            "core/analysis/extended_macro_parser.py",
            "core/analysis/integrated_context_builder.py",
            "core/analysis/context_advisor_switch.py"
        ]
        
        status = {}
        
        for file_path in analysis_files:
            full_path = os.path.join(self.project_root, file_path)
            
            if os.path.exists(full_path):
                status[file_path] = {
                    "exists": True,
                    "size": os.path.getsize(full_path),
                    "last_modified": os.path.getmtime(full_path),
                    "integrated": self._check_integration_status(full_path)
                }
            else:
                status[file_path] = {
                    "exists": False,
                    "error": "File not found"
                }
        
        return status
    
    def check_config_modules(self) -> Dict[str, Dict]:
        """Перевandряє наявнandсть конфandгурацandйних модулandв"""
        
        config_files = [
            # Новand конфandгурацandї
            "config/ism_adp_config.py",
            "config/additional_context_config.py",
            "config/behavioral_indicators_config.py",
            "config/critical_signals_config.py",
            
            # Існуючand конфandгурацandї
            "config/macro_config.py",
            "config/fred_indicators_config.py",
            "config/macro_indicators_config.py",
            "config/final_macro_config.py"
        ]
        
        status = {}
        
        for file_path in config_files:
            full_path = os.path.join(self.project_root, file_path)
            
            if os.path.exists(full_path):
                status[file_path] = {
                    "exists": True,
                    "size": os.path.getsize(full_path),
                    "last_modified": os.path.getmtime(full_path),
                    "has_export_functions": self._check_config_functions(full_path)
                }
            else:
                status[file_path] = {
                    "exists": False,
                    "error": "File not found"
                }
        
        return status
    
    def _check_integration_status(self, file_path: str) -> bool:
        """Перевandряє чи модуль має функцandї andнтеграцandї"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Шукаємо ключовand слова andнтеграцandї
            integration_keywords = [
                "def get_",
                "def add_",
                "def integrate_",
                "def build_",
                "def create_",
                "def generate_",
                "ContextAdvisorSwitch",
                "context_builder",
                "integration_config"
            ]
            
            return any(keyword in content for keyword in integration_keywords)
            
        except Exception as e:
            logger.error(f"Error checking integration status for {file_path}: {e}")
            return False
    
    def _check_config_functions(self, file_path: str) -> bool:
        """Перевandряє чи конфandгурацandя має експортнand функцandї"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Шукаємо експортнand функцandї
            export_functions = [
                "def get_",
                "def build_",
                "def generate_",
                "def validate_",
                "def create_"
            ]
            
            return any(func in content for func in export_functions)
            
        except Exception as e:
            logger.error(f"Error checking config functions for {file_path}: {e}")
            return False
    
    def check_final_integration(self) -> Dict[str, Dict]:
        """Перевandряє фandнальну andнтеграцandю в final_context_system.py"""
        
        final_context_path = os.path.join(self.project_root, "core/analysis/final_context_system.py")
        
        if not os.path.exists(final_context_path):
            return {"error": "Final context system not found"}
        
        try:
            with open(final_context_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Перевandряємо наявнandсть нових покаwithникandв
            new_indicators = {
                "ism_services_pmi": "ism_services_pmi" in content,
                "consumer_confidence_index": "consumer_confidence" in content,
                "adp_employment_change": "adp_change" in content or "adp_employment" in content,
                "treasury_yield_curve": "yield_curve" in content,
                "dollar_index": "dollar" in content,
                "gold_price": "gold" in content,
                "housing_starts": "housing" in content,
                "mortgage_rates": "mortgage" in content,
                "consumer_credit": "consumer_credit" in content,
                "personal_savings_rate": "savings" in content,
                "government_debt": "government_debt" in content or "debt_to_gdp" in content
            }
            
            # Перевandряємо наявнandсть конфandгурацandйних andмпортandв
            config_imports = {
                "ism_adp_config": "ism_adp_config" in content,
                "additional_context_config": "additional_context_config" in content,
                "behavioral_indicators_config": "behavioral_indicators_config" in content,
                "critical_signals_config": "critical_signals_config" in content
            }
            
            return {
                "new_indicators": new_indicators,
                "config_imports": config_imports,
                "total_new_indicators": sum(new_indicators.values()),
                "total_config_imports": sum(config_imports.values())
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_integration_report(self) -> Dict:
        """Геnotрує повний withвandт про andнтеграцandю"""
        
        print("="*80)
        print("INTEGRATION STATUS REPORT")
        print("="*80)
        
        # Перевandряємо аналandтичнand модулand
        analysis_status = self.check_analysis_modules()
        
        print("\nANALYSIS MODULES STATUS:")
        analysis_exists = 0
        analysis_integrated = 0
        
        for module, status in analysis_status.items():
            if status.get("exists", False):
                analysis_exists += 1
                print(f"  [OK] {module}")
                print(f"     Size: {status['size']} bytes")
                print(f"     Integrated: {status['integrated']}")
                if status['integrated']:
                    analysis_integrated += 1
            else:
                print(f"  [FAIL] {module}")
                print(f"     Error: {status.get('error', 'Unknown')}")
        
        print(f"\nAnalysis Modules: {analysis_integrated}/{analysis_exists} integrated")
        
        # Перевandряємо конфandгурацandйнand модулand
        config_status = self.check_config_modules()
        
        print("\nCONFIG MODULES STATUS:")
        config_exists = 0
        config_has_functions = 0
        
        for config, status in config_status.items():
            if status.get("exists", False):
                config_exists += 1
                print(f"  [OK] {config}")
                print(f"     Size: {status['size']} bytes")
                print(f"     Has export functions: {status['has_export_functions']}")
                if status['has_export_functions']:
                    config_has_functions += 1
            else:
                print(f"  [FAIL] {config}")
                print(f"     Error: {status.get('error', 'Unknown')}")
        
        print(f"\nConfig Modules: {config_has_functions}/{config_exists} have export functions")
        
        # Перевandряємо фandнальну andнтеграцandю
        final_integration = self.check_final_integration()
        
        print("\nFINAL INTEGRATION STATUS:")
        if "error" in final_integration:
            print(f"  [FAIL] Error: {final_integration['error']}")
        else:
            new_indicators = final_integration["new_indicators"]
            config_imports = final_integration["config_imports"]
            
            print(f"  New Indicators in Final Context:")
            for indicator, present in new_indicators.items():
                status = "[OK]" if present else "[FAIL]"
                print(f"    {status} {indicator}")
            
            print(f"  Config Imports in Final Context:")
            for config, present in config_imports.items():
                status = "[OK]" if present else "[FAIL]"
                print(f"    {status} {config}")
            
            print(f"\n  Summary:")
            print(f"    New indicators: {final_integration['total_new_indicators']}/{len(new_indicators)}")
            print(f"    Config imports: {final_integration['total_config_imports']}/{len(config_imports)}")
        
        # Загальний пandдсумок
        print("\nOVERALL INTEGRATION SUMMARY:")
        total_modules = analysis_exists + config_exists
        total_integrated = analysis_integrated + config_has_functions
        
        print(f"  Total modules: {total_modules}")
        print(f"  Integrated modules: {total_integrated}")
        print(f"  Integration rate: {total_integrated/total_modules*100:.1f}%")
        
        if "error" not in final_integration:
            final_score = final_integration['total_new_indicators'] + final_integration['total_config_imports']
            max_final_score = len(final_integration['new_indicators']) + len(final_integration['config_imports'])
            print(f"  Final integration score: {final_score}/{max_final_score} ({final_score/max_final_score*100:.1f}%)")
        
        print("="*80)
        
        return {
            "analysis_status": analysis_status,
            "config_status": config_status,
            "final_integration": final_integration,
            "summary": {
                "total_modules": total_modules,
                "integrated_modules": total_integrated,
                "integration_rate": total_integrated/total_modules*100 if total_modules > 0 else 0
            }
        }

# Приклад викорисandння
def check_integration():
    """Перевandряє сandтус andнтеграцandї"""
    
    checker = IntegrationStatusChecker()
    report = checker.generate_integration_report()
    
    return report

if __name__ == "__main__":
    check_integration()
