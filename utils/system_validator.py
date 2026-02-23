"""
SYSTEM VALIDATOR
Інструмент for перевandрки правильностand, практичностand and повноти system
"""

import os
import sys
import importlib
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

# Додаємо шлях до проекту
sys.path.append('c:/trading_project')

logger = logging.getLogger(__name__)

class SystemValidator:
    """
    Валandдатор system for перевandрки правильностand, практичностand and повноти
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.results = {
            'imports': {},
            'files': {},
            'configurations': {},
            'integrations': {},
            'data_flow': {},
            'models': {},
            'summary': {}
        }
        
    def validate_all(self) -> Dict[str, Any]:
        """
        Повна перевandрка system
        
        Returns:
            Dict[str, Any]: Реwithульandти перевandрки
        """
        self.logger.info("[SEARCH] Starting comprehensive system validation...")
        
        # 1. Перевandрка andмпортandв
        self._validate_imports()
        
        # 2. Перевandрка fileandв
        self._validate_files()
        
        # 3. Перевandрка конфandгурацandй
        self._validate_configurations()
        
        # 4. Перевandрка andнтеграцandй
        self._validate_integrations()
        
        # 5. Перевandрка потоку data
        self._validate_data_flow()
        
        # 6. Перевandрка моwhereлей
        self._validate_models()
        
        # 7. Фandнальний withвandт
        self._generate_summary()
        
        return self.results
    
    def _validate_imports(self) -> None:
        """Перевandрка критичних andмпортandв"""
        self.logger.info(" Validating imports...")
        
        critical_imports = {
            'main': [
                'config.pipeline_config',
                'utils.model_registry',
                'utils.feature_engineering',
                'utils.data_versioning',
                'core.stages.stage_2_enrichment',
                'core.stages.stage_3_features',
                'core.stages.stage_4_modeling',
                'core.analysis.feature_optimizer',
                'core.analysis.model_comparison_engine',
                'core.analysis.context_aware_model_selector',
                'core.analysis.time_series_validator',
                'optimal_feature_selection_integration'
            ],
            'models': [
                'models.tree_models',
                'models.linear_model',
                'models.mlp_model',
                'models.cnn_model',
                'models.lstm_model',
                'models.transformer_model',
                'models.autoencoder_model'
            ],
            'utils': [
                'utils.data_storage',
                'utils.feature_selector',
                'utils.trading_days',
                'utils.advanced_features'
            ],
            'config': [
                'config.config',
                'config.pipeline_config',
                'config.feature_config',
                'config.model_features'
            ]
        }
        
        import_results = {}
        
        for category, modules in critical_imports.items():
            import_results[category] = {}
            for module_name in modules:
                try:
                    importlib.import_module(module_name)
                    import_results[category][module_name] = {
                        'status': '[OK]',
                        'error': None
                    }
                    self.logger.debug(f"[OK] {module_name}")
                except ImportError as e:
                    import_results[category][module_name] = {
                        'status': '[ERROR]',
                        'error': str(e)
                    }
                    self.logger.error(f"[ERROR] {module_name}: {e}")
        
        self.results['imports'] = import_results
    
    def _validate_files(self) -> None:
        """Перевandрка наявностand fileandв"""
        self.logger.info(" Validating files...")
        
        critical_files = {
            'main': [
                'main.py',
                'config/pipeline_config.py',
                'utils/model_registry.py',
                'utils/feature_engineering.py',
                'utils/data_versioning.py'
            ],
            'stages': [
                'core/stages/stage_2_enrichment.py',
                'core/stages/stage_3_features.py',
                'core/stages/stage_4_modeling.py'
            ],
            'models': [
                'models/tree_models.py',
                'models/linear_model.py',
                'models/mlp_model.py',
                'models/cnn_model.py',
                'models/lstm_model.py',
                'models/transformer_model.py',
                'models/autoencoder_model.py'
            ],
            'analysis': [
                'core/analysis/feature_optimizer.py',
                'core/analysis/model_comparison_engine.py',
                'core/analysis/context_aware_model_selector.py',
                'core/analysis/time_series_validator.py'
            ]
        }
        
        file_results = {}
        
        for category, files in critical_files.items():
            file_results[category] = {}
            for file_path in files:
                full_path = f'c:/trading_project/{file_path}'
                if os.path.exists(full_path):
                    file_results[category][file_path] = {
                        'status': '[OK]',
                        'size': os.path.getsize(full_path),
                        'error': None
                    }
                    self.logger.debug(f"[OK] {file_path}")
                else:
                    file_results[category][file_path] = {
                        'status': '[ERROR]',
                        'size': 0,
                        'error': 'File not found'
                    }
                    self.logger.error(f"[ERROR] {file_path}: File not found")
        
        self.results['files'] = file_results
    
    def _validate_configurations(self) -> None:
        """Перевandрка конфandгурацandй"""
        self.logger.info(" Validating configurations...")
        
        config_results = {}
        
        try:
            # Перевandрка pipeline_config
            from config.pipeline_config import (
                DEFAULT_TIMEFRAMES, DEFAULT_TICKERS, MODEL_TYPES,
                MODEL_REGISTRY, MODEL_TRAINING_CONFIG,
                BATCH_TRAINING_CONFIG, FEATURE_SELECTION_CONFIG
            )
            
            config_results['pipeline_config'] = {
                'status': '[OK]',
                'timeframes': len(DEFAULT_TIMEFRAMES),
                'tickers': len(DEFAULT_TICKERS),
                'model_types': len(MODEL_TYPES),
                'model_registry': len(MODEL_REGISTRY),
                'error': None
            }
            
            # Перевandрка структури MODEL_REGISTRY
            registry_validation = self._validate_model_registry_structure(MODEL_REGISTRY)
            config_results['model_registry_structure'] = registry_validation
            
        except ImportError as e:
            config_results['pipeline_config'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        try:
            # Перевandрка feature_config
            from config.feature_config import CORE_FEATURES, ALL_FEATURES, USE_CORE_FEATURES
            
            config_results['feature_config'] = {
                'status': '[OK]',
                'core_features': len(CORE_FEATURES),
                'all_features': len(ALL_FEATURES),
                'use_core_features': USE_CORE_FEATURES,
                'error': None
            }
            
        except ImportError as e:
            config_results['feature_config'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        self.results['configurations'] = config_results
    
    def _validate_model_registry_structure(self, registry: Dict) -> Dict[str, Any]:
        """Перевandрка структури реєстру моwhereлей"""
        validation = {
            'status': '[OK]',
            'categories': 0,
            'total_models': 0,
            'invalid_entries': [],
            'missing_fields': []
        }
        
        for category, models in registry.items():
            validation['categories'] += 1
            for model_name, model_info in models.items():
                validation['total_models'] += 1
                
                # Перевandряємо обов'яwithковand поля
                required_fields = ['module', 'function', 'task_type', 'description']
                missing_fields = [field for field in required_fields if field not in model_info]
                
                if missing_fields:
                    validation['invalid_entries'].append(f"{category}.{model_name}")
                    validation['missing_fields'].extend([f"{category}.{model_name}:{field}" for field in missing_fields])
        
        if validation['invalid_entries']:
            validation['status'] = '[ERROR]'
        
        return validation
    
    def _validate_integrations(self) -> None:
        """Перевandрка andнтеграцandй мandж компоnotнandми"""
        self.logger.info(" Validating integrations...")
        
        integration_results = {}
        
        # Перевandрка andнтеграцandї feature_engineering
        try:
            from utils.feature_engineering import create_feature_engineering_utils
            feature_utils = create_feature_engineering_utils()
            
            # Перевandряємо наявнandсть методandв
            required_methods = [
                'calculate_sma', 'calculate_ema', 'calculate_rsi', 'calculate_macd',
                'create_direction_targets', 'create_heavy_targets',
                'create_all_features_and_targets', 'validate_features'
            ]
            
            missing_methods = [method for method in required_methods if not hasattr(feature_utils, method)]
            
            integration_results['feature_engineering'] = {
                'status': '[OK]' if not missing_methods else '[ERROR]',
                'methods_found': len(required_methods) - len(missing_methods),
                'methods_missing': missing_methods,
                'error': None if not missing_methods else f"Missing methods: {missing_methods}"
            }
            
        except Exception as e:
            integration_results['feature_engineering'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        # Перевandрка andнтеграцandї model_registry
        try:
            from utils.model_registry import get_model_registry, validate_all_models
            
            registry = get_model_registry()
            validation = validate_all_models()
            
            integration_results['model_registry'] = {
                'status': '[OK]' if validation['valid_models'] > 0 else '[ERROR]',
                'valid_models': validation['valid_models'],
                'total_models': validation['total_models'],
                'invalid_models': validation['invalid_models'],
                'error': None
            }
            
        except Exception as e:
            integration_results['model_registry'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        # Перевandрка andнтеграцandї data_versioning
        try:
            from utils.data_versioning import create_data_versioning
            
            versioning = create_data_versioning()
            
            integration_results['data_versioning'] = {
                'status': '[OK]',
                'max_age_days': len(versioning.max_age_days),
                'error': None
            }
            
        except Exception as e:
            integration_results['data_versioning'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        self.results['integrations'] = integration_results
    
    def _validate_data_flow(self) -> None:
        """Перевandрка потоку data мandж еandпами"""
        self.logger.info(" Validating data flow...")
        
        data_flow_results = {}
        
        # Перевandрка Stage 2 -> Stage 3 потоку
        try:
            from core.stages.stage_2_enrichment import run_stage_2_enrich_optimized
            from core.stages.stage_3_features import prepare_stage3_datasets
            
            # Створюємо тестовand данand
            test_data = {
                'all_news': pd.DataFrame({'title': ['test'], 'published_at': ['2024-01-01']}),
                'prices': pd.DataFrame({'ticker': ['SPY'], 'close': [100.0], 'interval': ['1d']}),
                'macro': pd.DataFrame(),
                'insider': pd.DataFrame()
            }
            
            # Перевandряємо чи Stage 2 поверandє правильний формат
            stage2_result = run_stage_2_enrich_optimized(test_data, {})
            
            if isinstance(stage2_result, tuple) and len(stage2_result) >= 3:
                # Перевandряємо чи Stage 3 may прийняти данand with Stage 2
                raw_news, enhanced_data, metadata = stage2_result
                stage3_result = prepare_stage3_datasets(enhanced_data)
                
                data_flow_results['stage2_to_stage3'] = {
                    'status': '[OK]',
                    'stage2_output_type': type(stage2_result).__name__,
                    'stage2_output_length': len(stage2_result),
                    'stage3_input_accepted': True,
                    'error': None
                }
            else:
                data_flow_results['stage2_to_stage3'] = {
                    'status': '[ERROR]',
                    'stage2_output_type': type(stage2_result).__name__,
                    'stage2_output_length': len(stage2_result) if hasattr(stage2_result, '__len__') else 'N/A',
                    'stage3_input_accepted': False,
                    'error': 'Stage 2 does not return expected tuple format'
                }
                
        except Exception as e:
            data_flow_results['stage2_to_stage3'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        # Перевandрка лandнandйного потоку (беwith дублювання forванandження)
        try:
            # Перевandряємо чи prepare_stage3_datasets приймає merged_df параметр
            import inspect
            sig = inspect.signature(prepare_stage3_datasets)
            
            has_merged_df_param = 'merged_df' in sig.parameters
            
            data_flow_results['linear_flow'] = {
                'status': '[OK]' if has_merged_df_param else '[ERROR]',
                'accepts_merged_df': has_merged_df_param,
                'parameters': list(sig.parameters.keys()),
                'error': None if has_merged_df_param else 'prepare_stage3_datasets does not accept merged_df parameter'
            }
            
        except Exception as e:
            data_flow_results['linear_flow'] = {
                'status': '[ERROR]',
                'error': str(e)
            }
        
        self.results['data_flow'] = data_flow_results
    
    def _validate_models(self) -> None:
        """Перевandрка моwhereлей"""
        self.logger.info(" Validating models...")
        
        model_results = {}
        
        try:
            from utils.model_registry import get_model_registry
            
            registry = get_model_registry()
            
            # Перевandряємо легкand моwhereлand
            light_models = registry.list_models('light')
            light_validation = self._validate_model_functions(light_models.get('light', {}), 'light')
            
            model_results['light_models'] = {
                'status': '[OK]' if light_validation['valid_count'] > 0 else '[ERROR]',
                'total': light_validation['total_count'],
                'valid': light_validation['valid_count'],
                'invalid': light_validation['invalid_count'],
                'details': light_validation['details']
            }
            
            # Перевandряємо важкand моwhereлand
            heavy_models = registry.list_models('heavy')
            heavy_validation = self._validate_model_functions(heavy_models.get('heavy', {}), 'heavy')
            
            model_results['heavy_models'] = {
                'status': '[OK]' if heavy_validation['valid_count'] > 0 else '[ERROR]',
                'total': heavy_validation['total_count'],
                'valid': heavy_validation['valid_count'],
                'invalid': heavy_validation['invalid_count'],
                'details': heavy_validation['details']
            }
            
        except Exception as e:
            model_results['error'] = str(e)
        
        self.results['models'] = model_results
    
    def _validate_model_functions(self, models: Dict[str, str], category: str) -> Dict[str, Any]:
        """Перевandрка функцandй моwhereлей"""
        validation = {
            'total_count': len(models),
            'valid_count': 0,
            'invalid_count': 0,
            'details': {}
        }
        
        for model_name in models:
            try:
                from utils.model_registry import get_model_registry
                registry = get_model_registry()
                model_function = registry.get_model_function(model_name, category)
                
                if model_function:
                    validation['valid_count'] += 1
                    validation['details'][model_name] = {
                        'status': '[OK]',
                        'function': model_function.__name__ if hasattr(model_function, '__name__') else str(model_function)
                    }
                else:
                    validation['invalid_count'] += 1
                    validation['details'][model_name] = {
                        'status': '[ERROR]',
                        'error': 'Function not found'
                    }
                    
            except Exception as e:
                validation['invalid_count'] += 1
                validation['details'][model_name] = {
                    'status': '[ERROR]',
                    'error': str(e)
                }
        
        return validation
    
    def _generate_summary(self) -> None:
        """Геnotрацandя фandнального withвandту"""
        self.logger.info("[DATA] Generating summary...")
        
        # Пandдрахунок forгальних реwithульandтandв
        total_checks = 0
        passed_checks = 0
        
        # Імпорти
        for category, imports in self.results['imports'].items():
            for module, result in imports.items():
                total_checks += 1
                if result['status'] == '[OK]':
                    passed_checks += 1
        
        # Файли
        for category, files in self.results['files'].items():
            for file_path, result in files.items():
                total_checks += 1
                if result['status'] == '[OK]':
                    passed_checks += 1
        
        # Конфandгурацandї
        for config_name, result in self.results['configurations'].items():
            total_checks += 1
            if result.get('status') == '[OK]':
                passed_checks += 1
        
        # Інтеграцandї
        for integration_name, result in self.results['integrations'].items():
            total_checks += 1
            if result.get('status') == '[OK]':
                passed_checks += 1
        
        # Потandк data
        for flow_name, result in self.results['data_flow'].items():
            total_checks += 1
            if result.get('status') == '[OK]':
                passed_checks += 1
        
        # Моwhereлand
        for model_type, result in self.results['models'].items():
            if model_type != 'error':
                total_checks += 1
                if result.get('status') == '[OK]':
                    passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        self.results['summary'] = {
            'total_checks': total_checks,
            'passed_checks': passed_checks,
            'failed_checks': total_checks - passed_checks,
            'success_rate': f"{success_rate:.1f}%",
            'overall_status': '[OK] HEALTHY' if success_rate >= 80 else '[WARN] NEEDS ATTENTION' if success_rate >= 60 else '[ERROR] CRITICAL',
            'critical_issues': self._get_critical_issues(),
            'recommendations': self._get_recommendations()
        }
    
    def _get_critical_issues(self) -> List[str]:
        """Отримання критичних problems"""
        critical_issues = []
        
        # Перевandрка критичних andмпортandв
        critical_modules = [
            'config.pipeline_config',
            'utils.model_registry',
            'utils.feature_engineering',
            'core.stages.stage_2_enrichment',
            'core.stages.stage_3_features',
            'core.stages.stage_4_modeling'
        ]
        
        for module in critical_modules:
            found = False
            for category, imports in self.results['imports'].items():
                if module in imports and imports[module]['status'] == '[OK]':
                    found = True
                    break
            if not found:
                critical_issues.append(f"Critical module missing: {module}")
        
        # Перевandрка потоку data
        if self.results['data_flow'].get('stage2_to_stage3', {}).get('status') != '[OK]':
            critical_issues.append("Data flow broken between Stage 2 and Stage 3")
        
        if self.results['data_flow'].get('linear_flow', {}).get('status') != '[OK]':
            critical_issues.append("Linear data flow not working (duplicate loading)")
        
        return critical_issues
    
    def _get_recommendations(self) -> List[str]:
        """Отримання рекомендацandй"""
        recommendations = []
        
        # Рекомендацandї по andмпорandх
        failed_imports = []
        for category, imports in self.results['imports'].items():
            for module, result in imports.items():
                if result['status'] == '[ERROR]':
                    failed_imports.append(module)
        
        if failed_imports:
            recommendations.append(f"Fix missing imports: {', '.join(failed_imports[:3])}...")
        
        # Рекомендацandї по моwhereлях
        if 'models' in self.results:
            for model_type, result in self.results['models'].items():
                if model_type != 'error' and result.get('status') == '[ERROR]':
                    recommendations.append(f"Fix {model_type} model configuration")
        
        # Рекомендацandї по andнтеграцandях
        if self.results['integrations'].get('feature_engineering', {}).get('status') == '[ERROR]':
            recommendations.append("Fix feature engineering integration")
        
        if self.results['integrations'].get('model_registry', {}).get('status') == '[ERROR]':
            recommendations.append("Fix model registry integration")
        
        return recommendations
    
    def print_report(self) -> None:
        """Друк withвandту"""
        print("\n" + "="*80)
        print("[SEARCH] SYSTEM VALIDATION REPORT")
        print("="*80)
        
        # Summary
        summary = self.results['summary']
        print(f"\n[DATA] OVERALL STATUS: {summary['overall_status']}")
        print(f"[OK] Passed: {summary['passed_checks']}/{summary['total_checks']} ({summary['success_rate']})")
        
        # Critical Issues
        if summary['critical_issues']:
            print(f"\n CRITICAL ISSUES:")
            for issue in summary['critical_issues']:
                print(f"   [ERROR] {issue}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n[IDEA] RECOMMENDATIONS:")
            for rec in summary['recommendations']:
                print(f"   [TOOL] {rec}")
        
        # Detailed results
        print(f"\n IMPORTS:")
        for category, imports in self.results['imports'].items():
            passed = sum(1 for imp in imports.values() if imp['status'] == '[OK]')
            total = len(imports)
            print(f"   {category}: {passed}/{total} [OK]")
        
        print(f"\n FILES:")
        for category, files in self.results['files'].items():
            passed = sum(1 for f in files.values() if f['status'] == '[OK]')
            total = len(files)
            print(f"   {category}: {passed}/{total} [OK]")
        
        print(f"\n CONFIGURATIONS:")
        for config_name, result in self.results['configurations'].items():
            status = result.get('status', '[ERROR]')
            print(f"   {config_name}: {status}")
        
        print(f"\n INTEGRATIONS:")
        for integration_name, result in self.results['integrations'].items():
            status = result.get('status', '[ERROR]')
            print(f"   {integration_name}: {status}")
        
        print(f"\n DATA FLOW:")
        for flow_name, result in self.results['data_flow'].items():
            status = result.get('status', '[ERROR]')
            print(f"   {flow_name}: {status}")
        
        print(f"\n MODELS:")
        for model_type, result in self.results['models'].items():
            if model_type != 'error':
                status = result.get('status', '[ERROR]')
                valid = result.get('valid', 0)
                total = result.get('total', 0)
                print(f"   {model_type}: {status} ({valid}/{total})")
        
        print("\n" + "="*80)


def validate_system() -> Dict[str, Any]:
    """
    Функцandя for forпуску валandдацandї system
    
    Returns:
        Dict[str, Any]: Реwithульandти перевandрки
    """
    validator = SystemValidator()
    results = validator.validate_all()
    validator.print_report()
    return results


def quick_check() -> bool:
    """
    Швидка перевandрка критичних компоnotнтandв
    
    Returns:
        bool: True якщо все добре
    """
    try:
        validator = SystemValidator()
        
        # Перевandряємо тandльки найважливandше
        validator._validate_imports()
        validator._validate_files()
        validator._validate_data_flow()
        
        # Пandдраховуємо реwithульandти
        total_checks = 0
        passed_checks = 0
        
        for category in ['imports', 'files']:
            for group, items in validator.results[category].items():
                for item, result in items.items():
                    total_checks += 1
                    if result.get('status') == '[OK]':
                        passed_checks += 1
        
        for flow, result in validator.results['data_flow'].items():
            total_checks += 1
            if result.get('status') == '[OK]':
                passed_checks += 1
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        print(f"[SEARCH] Quick Check: {passed_checks}/{total_checks} ({success_rate:.1f}%)")
        
        return success_rate >= 80
        
    except Exception as e:
        print(f"[ERROR] Quick check failed: {e}")
        return False


if __name__ == "__main__":
    # Запуск повної валandдацandї
    validate_system()
