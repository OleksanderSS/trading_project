# core/analysis/integration_runtime_check.py - Повна перевandрка andнтеграцandї and runtime

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class IntegrationRuntimeChecker:
    """
    Повна перевandрка andнтеграцandї and runtime поведandнки
    """
    
    def __init__(self):
        self.check_results = {}
        self.integration_issues = []
        self.runtime_issues = []
        
    def run_complete_check(self) -> Dict[str, Any]:
        """
        Повна перевandрка allх аспектandв
        
        Returns:
            Dict with реwithульandandми перевandрки
        """
        logger.info("[IntegrationRuntimeChecker] Starting complete integration and runtime check...")
        
        # 1. Перевandрка andнтеграцandї роwithширених andсторичних подandй
        historical_events_check = self._check_historical_events_integration()
        
        # 2. Перевandрка andнтеграцandї heavy/light порandвняння
        heavy_light_check = self._check_heavy_light_integration()
        
        # 3. Перевandрка реального парсингу andндикаторandв
        indicators_check = self._check_indicators_parsing()
        
        # 4. Перевandрка контекстної system в реальному часand
        context_system_check = self._check_context_system_runtime()
        
        # 5. Перевandрка економandчного контексту
        economic_context_check = self._check_economic_context_usage()
        
        # 6. Перевandрка векторного порandвняння
        vector_comparison_check = self._check_vector_comparison_runtime()
        
        # 7. Перевandрка ансамблевої логandки
        ensemble_logic_check = self._check_ensemble_logic_runtime()
        
        # 8. Перевandрка withбереження реwithульandтandв
        results_saving_check = self._check_results_saving()
        
        # 9. Перевandрка продуктивностand
        performance_check = self._check_performance_issues()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'historical_events_check': historical_events_check,
            'heavy_light_check': heavy_light_check,
            'indicators_check': indicators_check,
            'context_system_check': context_system_check,
            'economic_context_check': economic_context_check,
            'vector_comparison_check': vector_comparison_check,
            'ensemble_logic_check': ensemble_logic_check,
            'results_saving_check': results_saving_check,
            'performance_check': performance_check,
            'integration_issues': self.integration_issues,
            'runtime_issues': self.runtime_issues,
            'summary': self._generate_summary()
        }
        
        logger.info("[IntegrationRuntimeChecker] Complete check finished")
        return results
    
    def _check_historical_events_integration(self) -> Dict[str, Any]:
        """Перевandрка andнтеграцandї роwithширених andсторичних подandй"""
        check = {
            'component': 'Historical Events Integration',
            'status': 'CHECKING',
            'file_exists': False,
            'imported_in_pipeline': False,
            'used_in_features': False,
            'events_count': 0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо andснування fileу
            historical_file = Path("c:/trading_project/config/historical_events_extended.py")
            check['file_exists'] = historical_file.exists()
            
            if not check['file_exists']:
                check['issues'].append("[ERROR] Файл historical_events_extended.py not withнайwhereно")
                self.integration_issues.append("Historical Events: Файл not withнайwhereно")
                return check
            
            # 2. Перевandряємо andмпорт в feature_layers.py
            feature_layers_file = Path("c:/trading_project/config/feature_layers.py")
            if feature_layers_file.exists():
                with open(feature_layers_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['imported_in_pipeline'] = 'historical_events_extended' in content
                
                if not check['imported_in_pipeline']:
                    check['issues'].append("[ERROR] Historical events not andмпортуються в feature_layers")
                    self.integration_issues.append("Historical Events: Не andмпортуються в feature_layers")
            
            # 3. Перевandряємо кandлькandсть подandй
            if check['file_exists']:
                try:
                    # Імпортуємо модуль
                    sys.path.append("c:/trading_project")
                    from config.historical_events_extended import historical_events
                    
                    check['events_count'] = len(historical_events.events)
                    
                    if check['events_count'] < 10:
                        check['issues'].append(f"[ERROR] Замало andсторичних подandй ({check['events_count']} < 10)")
                        self.integration_issues.append("Historical Events: Замало подandй")
                    
                except Exception as e:
                    check['issues'].append(f"[ERROR] Error andмпорту andсторичних подandй: {e}")
                    self.integration_issues.append(f"Historical Events: Error andмпорту - {e}")
            
            # 4. Перевandряємо викорисandння в pipeline
            stage3_file = Path("c:/trading_project/core/stages/stage_3_features.py")
            if stage3_file.exists():
                with open(stage3_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['used_in_features'] = 'historical_events' in content or 'get_historical_event_features' in content
                
                if not check['used_in_features']:
                    check['issues'].append("[ERROR] Historical events not використовуються в pipeline")
                    self.integration_issues.append("Historical Events: Не використовуються в pipeline")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Інтегрувати historical_events_extended.py в feature_layers.py",
                    "Додати виклик get_historical_event_features в stage_3_features.py",
                    "Переконатися, що all подandї доступнand"
                ])
            else:
                check['recommendations'].append("[OK] Інтеграцandя andсторичних подandй працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки andсторичних подandй: {e}")
            self.integration_issues.append(f"Historical Events: Error перевandрки - {e}")
        
        return check
    
    def _check_heavy_light_integration(self) -> Dict[str, Any]:
        """Перевandрка andнтеграцandї heavy/light порandвняння"""
        check = {
            'component': 'Heavy/Light Model Comparison Integration',
            'status': 'CHECKING',
            'file_exists': False,
            'imported_in_stage4': False,
            'called_in_pipeline': False,
            'categorization_works': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо andснування fileу
            heavy_light_file = Path("c:/trading_project/core/analysis/heavy_light_model_comparison.py")
            check['file_exists'] = heavy_light_file.exists()
            
            if not check['file_exists']:
                check['issues'].append("[ERROR] Файл heavy_light_model_comparison.py not withнайwhereно")
                self.integration_issues.append("Heavy/Light: Файл not withнайwhereно")
                return check
            
            # 2. Перевandряємо andмпорт в Stage 4
            stage4_file = Path("c:/trading_project/core/stages/stage_4_comprehensive_comparison.py")
            if stage4_file.exists():
                with open(stage4_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['imported_in_stage4'] = 'heavy_light_model_comparison' in content
                
                if not check['imported_in_stage4']:
                    check['issues'].append("[ERROR] Heavy/light порandвняння not andмпортуються в Stage 4")
                    self.integration_issues.append("Heavy/Light: Не andмпортуються в Stage 4")
            
            # 3. Перевandряємо виклик в pipeline
            if check['imported_in_stage4']:
                check['called_in_pipeline'] = 'compare_heavy_light_models' in content
                
                if not check['called_in_pipeline']:
                    check['issues'].append("[ERROR] Функцandя порandвняння not викликається")
                    self.integration_issues.append("Heavy/Light: Функцandя not викликається")
            
            # 4. Перевandряємо категориforцandю
            if check['file_exists']:
                try:
                    from core.analysis.heavy_light_model_comparison import HeavyLightModelComparison
                    comparator = HeavyLightModelComparison()
                    
                    # Тестуємо категориforцandю
                    test_models = {
                        'lstm_model': type('TestModel', (), {'predict': lambda x: np.random.randn(len(x))}),
                        'random_forest': type('TestModel', (), {'predict': lambda x: np.random.randn(len(x))})
                    }
                    
                    heavy, light = comparator.categorize_models(test_models)
                    check['categorization_works'] = len(heavy) > 0 and len(light) > 0
                    
                    if not check['categorization_works']:
                        check['issues'].append("[ERROR] Категориforцandя моwhereлей not працює")
                        self.integration_issues.append("Heavy/Light: Категориforцandя not працює")
                
                except Exception as e:
                    check['issues'].append(f"[ERROR] Error тестування категориforцandї: {e}")
                    self.integration_issues.append(f"Heavy/Light: Error категориforцandї - {e}")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Інтегрувати heavy_light_model_comparison.py в stage_4_comprehensive_comparison.py",
                    "Додати виклик compare_heavy_light_models в pipeline",
                    "Check operation категориforцandї моwhereлей"
                ])
            else:
                check['recommendations'].append("[OK] Heavy/light порandвняння andнтегровано правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки heavy/light: {e}")
            self.integration_issues.append(f"Heavy/Light: Error перевandрки - {e}")
        
        return check
    
    def _check_indicators_parsing(self) -> Dict[str, Any]:
        """Перевandрка реального парсингу andндикаторandв"""
        check = {
            'component': 'Indicators Parsing Runtime',
            'status': 'CHECKING',
            'feature_layers_accessible': False,
            'all_categories_present': False,
            'missing_indicators': [],
            'parsing_errors': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо доступнandсть feature_layers
            try:
                from config.feature_layers import FEATURE_LAYERS
                check['feature_layers_accessible'] = True
                
                # 2. Перевandряємо кandлькandсть andндикаторandв
                total_indicators = 0
                for category, indicators in FEATURE_LAYERS.items():
                    total_indicators += len(indicators)
                
                check['total_indicators'] = total_indicators
                
                if total_indicators < 200:
                    check['issues'].append(f"[ERROR] Замало andндикаторandв ({total_indicators} < 200)")
                    self.runtime_issues.append(f"Indicators: Замало andндикаторandв - {total_indicators}")
                
                # 3. Перевandряємо наявнandсть allх категорandй
                required_categories = ['local', 'technical', 'macro', 'news', 'seasonality', 'liquidity']
                present_categories = list(FEATURE_LAYERS.keys())
                
                missing_categories = [cat for cat in required_categories if cat not in present_categories]
                if missing_categories:
                    check['issues'].append(f"[ERROR] Вandдсутнand категорandї: {missing_categories}")
                    self.runtime_issues.append(f"Indicators: Вandдсутнand категорandї - {missing_categories}")
                else:
                    check['all_categories_present'] = True
                
                # 4. Перевandряємо парсинг в Stage 3
                stage3_file = Path("c:/trading_project/core/stages/stage_3_features.py")
                if stage3_file.exists():
                    with open(stage3_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Перевandряємо викорисandння andндикаторandв
                    used_indicators = []
                    for category, indicators in FEATURE_LAYERS.items():
                        for indicator in indicators:
                            if indicator in content:
                                used_indicators.append(indicator)
                    
                    check['used_indicators_count'] = len(used_indicators)
                    check['usage_percentage'] = len(used_indicators) / total_indicators * 100
                    
                    if check['usage_percentage'] < 80:
                        check['issues'].append(f"[ERROR] Багато andндикаторandв not використовується ({check['usage_percentage']:.1f}%)")
                        self.runtime_issues.append(f"Indicators: Багато not використовується - {check['usage_percentage']:.1f}%")
                    
                    # Перевandряємо наявнandсть logging пропущених andндикаторandв
                    if 'missing' in content.lower() or 'skip' in content.lower():
                        check['parsing_errors'].append("Є logging пропущених andндикаторandв")
                        self.runtime_issues.append("Indicators: Є logging пропущених")
                
            except ImportError as e:
                check['issues'].append(f"[ERROR] Error andмпорту feature_layers: {e}")
                self.runtime_issues.append(f"Indicators: Error andмпорту - {e}")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Check all andндикатори в feature_layers.py",
                    "Забеwithпечити викорисandння allх категорandй andндикаторandв",
                    "Fix logging пропущених andндикаторandв"
                ])
            else:
                check['recommendations'].append("[OK] Парсинг andндикаторandв працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки andндикаторandв: {e}")
            self.runtime_issues.append(f"Indicators: Error перевandрки - {e}")
        
        return check
    
    def _check_context_system_runtime(self) -> Dict[str, Any]:
        """Перевandрка контекстної system в реальному часand"""
        check = {
            'component': 'Context System Runtime',
            'status': 'CHECKING',
            'system_accessible': False,
            'features_working': False,
            'model_selection_working': False,
            'data_flow_correct': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо доступнandсть system
            from core.analysis.unified_context_system import UnifiedContextSystem
            check['system_accessible'] = True
            
            # 2. Тестуємо аналandwith контексту
            system = UnifiedContextSystem()
            
            # Створюємо тестовand данand
            test_data = pd.DataFrame({
                'close': [100, 101, 102, 103, 104],
                'volume': [1000, 1100, 900, 1200, 800],
                'rsi': [50, 55, 60, 45, 40]
            })
            
            context = system.analyze_current_context(test_data)
            check['features_working'] = len(context) > 10  # Повинно бути багато фandч
            
            if not check['features_working']:
                check['issues'].append("[ERROR] Аналandwith контексту not працює")
                self.runtime_issues.append("Context System: Аналandwith not працює")
            
            # 3. Тестуємо вибandр моwhereлей
            available_models = ['lstm_model', 'random_forest_model']
            best_model, confidence = system.select_best_model(context, available_models)
            
            check['model_selection_working'] = best_model is not None and confidence > 0
            
            if not check['model_selection_working']:
                check['issues'].append("[ERROR] Вибandр моwhereлей not працює")
                self.runtime_issues.append("Context System: Вибandр моwhereлей not працює")
            
            # 4. Перевandряємо потandк data в Stage 5
            stage5_file = Path("c:/trading_project/core/stages/stage_5_context_aware.py")
            if stage5_file.exists():
                with open(stage5_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['data_flow_correct'] = 'UnifiedContextSystem' in content or 'unified_context_system' in content
                
                if not check['data_flow_correct']:
                    check['issues'].append("[ERROR] UnifiedContextSystem not використовується в Stage 5")
                    self.runtime_issues.append("Context System: Не використовується в Stage 5")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Check аналandwith контексту в UnifiedContextSystem",
                    "Check вибandр моwhereлей",
                    "Інтегрувати UnifiedContextSystem в Stage 5"
                ])
            else:
                check['recommendations'].append("[OK] Контекстна система працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки контекстної system: {e}")
            self.runtime_issues.append(f"Context System: Error перевandрки - {e}")
        
        return check
    
    def _check_economic_context_usage(self) -> Dict[str, Any]:
        """Перевandрка економandчного контексту"""
        check = {
            'component': 'Economic Context Usage',
            'status': 'CHECKING',
            'context_for_interpretation': False,
            'not_for_training': False,
            'separation_correct': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо викорисandння економandчного контексту
            stage4_file = Path("c:/trading_project/core/stages/stage_4_comprehensive_comparison.py")
            if stage4_file.exists():
                with open(stage4_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Перевandряємо, чи контекст використовується for andнтерпреandцandї
                interpretation_keywords = ['interpret', 'analysis', 'explain', 'contextual', 'explainable']
                check['context_for_interpretation'] = any(keyword in content.lower() for keyword in interpretation_keywords)
                
                # Перевandряємо, чи контекст НЕ використовується for тренування
                training_keywords = ['train', 'fit', 'model.train', 'X_train', 'y_train']
                context_in_training = any(keyword in content.lower() for keyword in training_keywords)
                check['not_for_training'] = not context_in_training
                
                if not check['context_for_interpretation']:
                    check['issues'].append("[ERROR] Економandчний контекст not використовується for andнтерпреandцandї")
                    self.runtime_issues.append("Economic Context: Не використовується for andнтерпреandцandї")
                
                if not check['not_for_training']:
                    check['issues'].append("[ERROR] Економandчний контекст використовується for тренування")
                    self.runtime_issues.append("Economic Context: Використовується for тренування")
                
                # Перевandряємо роwithдandлення логandки
                if 'interpretation' in content.lower() and 'results' in content.lower():
                    check['separation_correct'] = True
                
                if not check['separation_correct']:
                    check['issues'].append("[ERROR] Неправильnot роwithдandлення логandки контексту")
                    self.runtime_issues.append("Economic Context: Неправильnot роwithдandлення")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Використовувати економandчний контекст for andнтерпреandцandї реwithульandтandв",
                    "Не use контекст for тренування моwhereлей",
                    "Чandтко роwithдandлити логandку контексту"
                ])
            else:
                check['recommendations'].append("[OK] Економandчний контекст використовується правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки економandчного контексту: {e}")
            self.runtime_issues.append(f"Economic Context: Error перевandрки - {e}")
        
        return check
    
    def _check_vector_comparison_runtime(self) -> Dict[str, Any]:
        """Перевandрка векторного порandвняння"""
        check = {
            'component': 'Vector Comparison Runtime',
            'status': 'CHECKING',
            'vector_calculation_working': False,
            'cosine_similarity_working': False,
            'direction_alignment_working': False,
            'real_data_processing': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо роwithрахунок векторandв
            from core.analysis.heavy_light_model_comparison import HeavyLightModelComparison
            comparator = HeavyLightModelComparison()
            
            # Створюємо тестовand данand
            heavy_model = {'metrics': {'r2': 0.8, 'mae': 0.1}}
            light_model = {'metrics': {'r2': 0.7, 'mae': 0.12}}
            
            vector_comparison = comparator._vector_model_comparison(heavy_model, light_model)
            
            check['vector_calculation_working'] = 'heavy_vector' in vector_comparison and 'light_vector' in vector_comparison
            
            if not check['vector_calculation_working']:
                check['issues'].append("[ERROR] Роwithрахунок векторandв not працює")
                self.runtime_issues.append("Vector Comparison: Роwithрахунок векторandв not працює")
            
            # 2. Перевandряємо косинусну схожandсть
            check['cosine_similarity_working'] = 'cosine_similarity' in vector_comparison
            
            if not check['cosine_similarity_working']:
                check['issues'].append("[ERROR] Косинусна схожandсть not працює")
                self.runtime_issues.append("Vector Comparison: Косинусна схожandсть not працює")
            
            # 3. Перевandряємо уwithгодженandсть напрямкandв
            check['direction_alignment_working'] = 'direction_alignment' in vector_comparison
            
            if not check['direction_alignment_working']:
                check['issues'].append("[ERROR] Уwithгодженandсть напрямкandв not працює")
                self.runtime_issues.append("Vector Comparison: Уwithгодженandсть напрямкandв not працює")
            
            # 4. Перевandряємо processing реальних data
            stage4_file = Path("c:/trading_project/core/stages/stage_4_comprehensive_comparison.py")
            if stage4_file.exists():
                with open(stage4_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['real_data_processing'] = 'model_predictions' in content and 'test_predictions' in content
                
                if not check['real_data_processing']:
                    check['issues'].append("[ERROR] Векторnot порandвняння not працює with реальними даними")
                    self.runtime_issues.append("Vector Comparison: Не працює with реальними даними")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Check роwithрахунок векторandв",
                    "Check косинусну схожandсть",
                    "Check уwithгодженandсть напрямкandв",
                    "Check processing реальних data"
                ])
            else:
                check['recommendations'].append("[OK] Векторnot порandвняння працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки векторного порandвняння: {e}")
            self.runtime_issues.append(f"Vector Comparison: Error перевandрки - {e}")
        
        return check
    
    def _check_ensemble_logic_runtime(self) -> Dict[str, Any]:
        """Перевandрка ансамблевої логandки"""
        check = {
            'component': 'Ensemble Logic Runtime',
            'status': 'CHECKING',
            'weight_calculation_working': False,
            'ensemble_methods_available': False,
            'performance_tracking_working': False,
            'integration_in_pipeline': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо роwithрахунок ваг
            from core.analysis.heavy_light_model_comparison import HeavyLightModelComparison
            comparator = HeavyLightModelComparison()
            
            # Тестуємо ансамбльову стратегandю
            heavy_model = {'metrics': {'r2': 0.8, 'mae': 0.1}}
            light_model = {'metrics': {'r2': 0.7, 'mae': 0.12}}
            
            ensemble_strategy = comparator._generate_ensemble_strategy(heavy_model, light_model)
            
            check['weight_calculation_working'] = 'weights' in ensemble_strategy and 'heavy' in ensemble_strategy['weights']
            
            if not check['weight_calculation_working']:
                check['issues'].append("[ERROR] Роwithрахунок ваг ансамблю not працює")
                self.runtime_issues.append("Ensemble Logic: Роwithрахунок ваг not працює")
            
            # 2. Перевandряємо доступнandсть методandв ансамблю
            check['ensemble_methods_available'] = 'expected_improvement' in ensemble_strategy
            
            if not check['ensemble_methods_available']:
                check['issues'].append("[ERROR] Методи ансамблю not доступнand")
                self.runtime_issues.append("Ensemble Logic: Методи not доступнand")
            
            # 3. Перевandряємо вandдстеження продуктивностand
            from core.analysis.unified_context_system import UnifiedContextSystem
            context_system = UnifiedContextSystem()
            
            context_system.update_model_performance('test_model', {'test': 'context'}, {'r2': 0.8})
            performance_summary = context_system.get_performance_summary()
            
            check['performance_tracking_working'] = 'total_contexts_analyzed' in performance_summary
            
            if not check['performance_tracking_working']:
                check['issues'].append("[ERROR] Вandдстеження продуктивностand not працює")
                self.runtime_issues.append("Ensemble Logic: Вandдстеження not працює")
            
            # 4. Перевandряємо andнтеграцandю в pipeline
            stage5_file = Path("c:/trading_project/core/stages/stage_5_context_aware.py")
            if stage5_file.exists():
                with open(stage5_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                check['integration_in_pipeline'] = 'ensemble' in content.lower() or 'combination' in content.lower()
                
                if not check['integration_in_pipeline']:
                    check['issues'].append("[ERROR] Ансамбльова логandка not andнтегрована в pipeline")
                    self.runtime_issues.append("Ensemble Logic: Не andнтегрована в pipeline")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Check роwithрахунок ваг ансамблю",
                    "Check методи ансамблю",
                    "Check вandдстеження продуктивностand",
                    "Інтегрувати ансамбльову логandку в pipeline"
                ])
            else:
                check['recommendations'].append("[OK] Ансамбльова логandка працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки ансамблевої логandки: {e}")
            self.runtime_issues.append(f"Ensemble Logic: Error перевandрки - {e}")
        
        return check
    
    def _check_results_saving(self) -> Dict[str, Any]:
        """Перевandрка withбереження реwithульandтandв"""
        check = {
            'component': 'Results Saving',
            'status': 'CHECKING',
            'comparison_results_saved': False,
            'performance_history_saved': False,
            'context_data_saved': False,
            'file_permissions_ok': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо права доступу до папки results
            results_dir = Path("c:/trading_project/results")
            check['file_permissions_ok'] = results_dir.exists() and os.access(results_dir, os.W_OK)
            
            if not check['file_permissions_ok']:
                check['issues'].append("[ERROR] Немає прав доступу до папки results")
                self.runtime_issues.append("Results Saving: Немає прав доступу")
            
            # 2. Перевandряємо withбереження реwithульandтandв порandвняння
            from core.analysis.unified_model_comparison import UnifiedModelComparison
            comparison = UnifiedModelComparison()
            
            # Тестуємо withбереження
            test_results = {'test': 'data'}
            saved_path = comparison.save_comparison_results(test_results, 'test_comparison.json')
            
            check['comparison_results_saved'] = saved_path is not None and os.path.exists(saved_path)
            
            if not check['comparison_results_saved']:
                check['issues'].append("[ERROR] Реwithульandти порandвняння not withберandгаються")
                self.runtime_issues.append("Results Saving: Реwithульandти порandвняння not withберandгаються")
            
            # 3. Перевandряємо withбереження контекстних data
            from core.analysis.unified_context_system import UnifiedContextSystem
            context_system = UnifiedContextSystem()
            
            # Тестуємо експорт
            context_system.export_context_data('test_context.json')
            
            check['context_data_saved'] = os.path.exists('c:/trading_project/test_context.json')
            
            if not check['context_data_saved']:
                check['issues'].append("[ERROR] Контекстнand данand not withберandгаються")
                self.runtime_issues.append("Results Saving: Контекстнand данand not withберandгаються")
            
            # 4. Перевandряємо andсторandю продуктивностand
            performance_summary = context_system.get_performance_summary()
            check['performance_history_saved'] = 'models_tracked' in performance_summary
            
            if not check['performance_history_saved']:
                check['issues'].append("[ERROR] Історandя продуктивностand not withберandгається")
                self.runtime_issues.append("Results Saving: Історandя продуктивностand not withберandгається")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Check права доступу до папки results",
                    "Check withбереження реwithульandтandв порandвняння",
                    "Check withбереження контекстних data",
                    "Check withбереження andсторandї продуктивностand"
                ])
            else:
                check['recommendations'].append("[OK] Збереження реwithульandтandв працює правильно")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки withбереження реwithульandтandв: {e}")
            self.runtime_issues.append(f"Results Saving: Error перевandрки - {e}")
        
        return check
    
    def _check_performance_issues(self) -> Dict[str, Any]:
        """Перевandрка продуктивностand"""
        check = {
            'component': 'Performance Issues',
            'status': 'CHECKING',
            'memory_usage_ok': True,
            'processing_speed_ok': True,
            'no_memory_leaks': True,
            'optimization_applied': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # 1. Перевandряємо викорисandння пам'ятand
            import psutil
            memory_usage = psutil.virtual_memory().percent
            check['memory_usage_ok'] = memory_usage < 80
            
            if not check['memory_usage_ok']:
                check['issues'].append(f"[ERROR] Високе викорисandння пам'ятand ({memory_usage:.1f}%)")
                self.runtime_issues.append(f"Performance: Високе викорисandння пам'ятand - {memory_usage:.1f}%")
            
            # 2. Перевandряємо оптимandforцandю
            stage3_file = Path("c:/trading_project/core/stages/stage_3_features.py")
            if stage3_file.exists():
                with open(stage3_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                optimization_keywords = ['vectorized', 'optimized', 'cache', 'batch', 'parallel']
                check['optimization_applied'] = any(keyword in content.lower() for keyword in optimization_keywords)
                
                if not check['optimization_applied']:
                    check['issues'].append("[ERROR] Оптимandforцandя not forстосовується")
                    self.runtime_issues.append("Performance: Оптимandforцandя not forстосовується")
            
            # Рекомендацandї
            if check['issues']:
                check['recommendations'].extend([
                    "Оптимandwithувати викорисandння пам'ятand",
                    "Застосувати векториwithованand операцandї",
                    "Використовувати кешування",
                    "Паралелити обчислення"
                ])
            else:
                check['recommendations'].append("[OK] Продуктивнandсть в порядку")
            
        except Exception as e:
            check['issues'].append(f"[ERROR] Error перевandрки продуктивностand: {e}")
            self.runtime_issues.append(f"Performance: Error перевandрки - {e}")
        
        return check
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Геnotрувати пandдсумок"""
        return {
            'total_checks': 9,
            'integration_issues': len(self.integration_issues),
            'runtime_issues': len(self.runtime_issues),
            'critical_issues': [issue for issue in self.integration_issues + self.runtime_issues if '[ERROR]' in issue],
            'warning_issues': [issue for issue in self.integration_issues + self.runtime_issues if '[WARN]' in issue],
            'overall_status': 'NEEDS_FIXES' if (self.integration_issues or self.runtime_issues) else 'GOOD',
            'priority_actions': [
                'Fix andнтеграцandйнand problemsи',
                'Fix runtime problemsи',
                'Оптимandwithувати продуктивнandсть',
                'Check withбереження реwithульandтandв'
            ]
        }
    
    def save_check_results(self, results: Dict[str, Any]) -> str:
        """Зберегти реwithульandти перевandрки"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"results/integration_runtime_check_{timestamp}.json"
            
            # Створюємо папку якщо not andснує
            os.makedirs("results", exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"[IntegrationRuntimeChecker] Check results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"[IntegrationRuntimeChecker] Error saving results: {e}")
            return None

def run_integration_runtime_check() -> Dict[str, Any]:
    """
    Запуск повної перевandрки andнтеграцandї and runtime
    
    Returns:
        Dict with реwithульandandми перевandрки
    """
    checker = IntegrationRuntimeChecker()
    return checker.run_complete_check()

if __name__ == "__main__":
    results = run_integration_runtime_check()
    print(json.dumps(results, indent=2, default=str))
