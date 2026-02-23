# core/analysis/pipeline_integration_checker.py

import os
import importlib
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class PipelineIntegrationChecker:
    """
    Перевandряє andнтеграцandю allх модулandв в пайплайни
    """
    
    def __init__(self):
        self.project_root = "c:/trading_project"
        self.pipeline_files = {}
        self.integration_status = {}
        
    def check_pipeline_files(self) -> Dict[str, Dict]:
        """Перевandряє наявнandсть пайплайн fileandв"""
        
        pipeline_files = [
            # Основнand пайплайни
            "run_complete_pipeline.py",
            "run_full_pipeline.py", 
            "run_progressive_pipeline.py",
            "simple_optimized_pipeline.py",
            
            # Core пайплайни
            "core/pipeline/context_aware_pipeline.py",
            "core/pipeline/dual_pipeline_orchestrator.py",
            "core/pipeline/enhanced_pipeline.py",
            "core/pipeline/news_pipeline.py",
            
            # Stage пайплайни
            "core/stages/incremental_pipeline.py",
            "core/stages/stage_5_pipeline_fixed.py",
            "core/stages/unified_pipeline.py"
        ]
        
        status = {}
        
        for file_path in pipeline_files:
            full_path = os.path.join(self.project_root, file_path)
            
            if os.path.exists(full_path):
                status[file_path] = {
                    "exists": True,
                    "size": os.path.getsize(full_path),
                    "last_modified": os.path.getmtime(full_path),
                    "has_context_integration": self._check_context_integration(full_path),
                    "has_new_indicators": self._check_new_indicators(full_path)
                }
            else:
                status[file_path] = {
                    "exists": False,
                    "error": "File not found"
                }
        
        return status
    
    def _check_context_integration(self, file_path: str) -> bool:
        """Перевandряє чи пайплайн andнтегрує контекст"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Шукаємо ключовand слова контекстної andнтеграцandї
            context_keywords = [
                "ContextAdvisorSwitch",
                "context_advisor_switch",
                "context_aware",
                "context_builder",
                "integrated_context",
                "final_context",
                "context_selection",
                "context_driven"
            ]
            
            return any(keyword in content for keyword in context_keywords)
            
        except Exception as e:
            logger.error(f"Error checking context integration for {file_path}: {e}")
            return False
    
    def _check_new_indicators(self, file_path: str) -> bool:
        """Перевandряє чи пайплайн використовує новand покаwithники"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Шукаємо новand покаwithники
            new_indicators = [
                "ism_services_pmi",
                "consumer_confidence_index",
                "adp_employment_change",
                "treasury_yield_curve",
                "dollar_index",
                "gold_price",
                "housing_starts",
                "mortgage_rates",
                "consumer_credit",
                "personal_savings_rate",
                "government_debt"
            ]
            
            return any(indicator in content for indicator in new_indicators)
            
        except Exception as e:
            logger.error(f"Error checking new indicators for {file_path}: {e}")
            return False
    
    def check_main_entry_points(self) -> Dict[str, Dict]:
        """Перевandряє основнand точки входу"""
        
        entry_points = [
            "run_complete_pipeline.py",
            "run_full_pipeline.py",
            "run_progressive_pipeline.py",
            "simple_optimized_pipeline.py"
        ]
        
        status = {}
        
        for entry_point in entry_points:
            full_path = os.path.join(self.project_root, entry_point)
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                status[entry_point] = {
                    "exists": True,
                    "has_main_function": "def main(" in content or "if __name__" in content,
                    "has_context_imports": any(x in content for x in [
                        "from core.analysis.context_advisor_switch",
                        "from core.analysis.final_context_system",
                        "from core.analysis.integrated_context_builder"
                    ]),
                    "has_new_config_imports": any(x in content for x in [
                        "from config.ism_adp_config",
                        "from config.additional_context_config",
                        "from config.behavioral_indicators_config",
                        "from config.critical_signals_config"
                    ])
                }
            else:
                status[entry_point] = {
                    "exists": False,
                    "error": "File not found"
                }
        
        return status
    
    def check_core_pipeline_integration(self) -> Dict[str, Dict]:
        """Перевandряє andнтеграцandю в core пайплайни"""
        
        core_pipelines = [
            "core/pipeline/context_aware_pipeline.py",
            "core/pipeline/dual_pipeline_orchestrator.py",
            "core/pipeline/enhanced_pipeline.py"
        ]
        
        status = {}
        
        for pipeline in core_pipelines:
            full_path = os.path.join(self.project_root, pipeline)
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                status[pipeline] = {
                    "exists": True,
                    "uses_context_advisor": "ContextAdvisorSwitch" in content,
                    "uses_final_context": "FinalContextSystem" in content,
                    "uses_adaptive_filter": "AdaptiveNoiseFilter" in content,
                    "has_new_indicators": self._check_new_indicators(full_path),
                    "has_config_imports": any(x in content for x in [
                        "ism_adp_config", "additional_context_config", 
                        "behavioral_indicators_config", "critical_signals_config"
                    ])
                }
            else:
                status[pipeline] = {
                    "exists": False,
                    "error": "File not found"
                }
        
        return status
    
    def generate_pipeline_report(self) -> Dict:
        """Геnotрує повний withвandт про andнтеграцandю в пайплайни"""
        
        print("="*80)
        print("PIPELINE INTEGRATION STATUS REPORT")
        print("="*80)
        
        # Перевandряємо пайплайн fileи
        pipeline_status = self.check_pipeline_files()
        
        print("\nPIPELINE FILES STATUS:")
        pipeline_exists = 0
        pipeline_with_context = 0
        pipeline_with_new_indicators = 0
        
        for pipeline, status in pipeline_status.items():
            if status.get("exists", False):
                pipeline_exists += 1
                print(f"  [OK] {pipeline}")
                print(f"     Size: {status['size']} bytes")
                print(f"     Context integration: {status['has_context_integration']}")
                print(f"     New indicators: {status['has_new_indicators']}")
                
                if status['has_context_integration']:
                    pipeline_with_context += 1
                if status['has_new_indicators']:
                    pipeline_with_new_indicators += 1
            else:
                print(f"  [FAIL] {pipeline}")
                print(f"     Error: {status.get('error', 'Unknown')}")
        
        print(f"\nPipeline Files: {pipeline_exists} exist")
        print(f"Context Integration: {pipeline_with_context}/{pipeline_exists}")
        print(f"New Indicators: {pipeline_with_new_indicators}/{pipeline_exists}")
        
        # Перевandряємо точки входу
        entry_points = self.check_main_entry_points()
        
        print("\nENTRY POINTS STATUS:")
        entry_exists = 0
        entry_with_main = 0
        entry_with_context = 0
        entry_with_configs = 0
        
        for entry_point, status in entry_points.items():
            if status.get("exists", False):
                entry_exists += 1
                print(f"  [OK] {entry_point}")
                print(f"     Has main function: {status['has_main_function']}")
                print(f"     Has context imports: {status['has_context_imports']}")
                print(f"     Has new config imports: {status['has_new_config_imports']}")
                
                if status['has_main_function']:
                    entry_with_main += 1
                if status['has_context_imports']:
                    entry_with_context += 1
                if status['has_new_config_imports']:
                    entry_with_configs += 1
            else:
                print(f"  [FAIL] {entry_point}")
                print(f"     Error: {status.get('error', 'Unknown')}")
        
        print(f"\nEntry Points: {entry_with_main}/{entry_exists} have main function")
        print(f"Context Imports: {entry_with_context}/{entry_exists}")
        print(f"New Config Imports: {entry_with_configs}/{entry_exists}")
        
        # Перевandряємо core пайплайни
        core_pipelines = self.check_core_pipeline_integration()
        
        print("\nCORE PIPELINES STATUS:")
        core_exists = 0
        core_with_advisor = 0
        core_with_final_context = 0
        core_with_new_indicators = 0
        
        for pipeline, status in core_pipelines.items():
            if status.get("exists", False):
                core_exists += 1
                print(f"  [OK] {pipeline}")
                print(f"     Uses ContextAdvisorSwitch: {status['uses_context_advisor']}")
                print(f"     Uses FinalContextSystem: {status['uses_final_context']}")
                print(f"     Uses AdaptiveNoiseFilter: {status['uses_adaptive_filter']}")
                print(f"     Has new indicators: {status['has_new_indicators']}")
                print(f"     Has config imports: {status['has_config_imports']}")
                
                if status['uses_context_advisor']:
                    core_with_advisor += 1
                if status['uses_final_context']:
                    core_with_final_context += 1
                if status['has_new_indicators']:
                    core_with_new_indicators += 1
            else:
                print(f"  [FAIL] {pipeline}")
                print(f"     Error: {status.get('error', 'Unknown')}")
        
        print(f"\nCore Pipelines: {core_exists} exist")
        print(f"ContextAdvisorSwitch: {core_with_advisor}/{core_exists}")
        print(f"FinalContextSystem: {core_with_final_context}/{core_exists}")
        print(f"New Indicators: {core_with_new_indicators}/{core_exists}")
        
        # Загальний пandдсумок
        print("\nOVERALL PIPELINE INTEGRATION SUMMARY:")
        total_files = pipeline_exists + entry_exists + core_exists
        total_context = pipeline_with_context + entry_with_context + core_with_advisor
        total_new_indicators = pipeline_with_new_indicators + entry_with_configs + core_with_new_indicators
        
        print(f"  Total pipeline files: {total_files}")
        print(f"  Files with context integration: {total_context}")
        print(f"  Files with new indicators: {total_new_indicators}")
        print(f"  Context integration rate: {total_context/total_files*100:.1f}%")
        print(f"  New indicators integration rate: {total_new_indicators/total_files*100:.1f}%")
        
        print("="*80)
        
        return {
            "pipeline_status": pipeline_status,
            "entry_points": entry_points,
            "core_pipelines": core_pipelines,
            "summary": {
                "total_files": total_files,
                "context_integration": total_context,
                "new_indicators": total_new_indicators,
                "context_rate": total_context/total_files*100 if total_files > 0 else 0,
                "indicators_rate": total_new_indicators/total_files*100 if total_files > 0 else 0
            }
        }

# Приклад викорисandння
def check_pipeline_integration():
    """Перевandряє andнтеграцandю в пайплайни"""
    
    checker = PipelineIntegrationChecker()
    report = checker.generate_pipeline_report()
    
    return report

if __name__ == "__main__":
    check_pipeline_integration()
