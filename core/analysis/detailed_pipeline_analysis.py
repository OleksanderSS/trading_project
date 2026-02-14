# core/analysis/detailed_pipeline_analysis.py - Деandльний аналandwith pipeline по еandпах

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DetailedPipelineAnalyzer:
    """
    Деandльний аналandwith pipeline with перевandркою allх аспектandв
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.issues_found = []
        self.recommendations = []
        
    def analyze_complete_pipeline(self) -> Dict[str, Any]:
        """
        Повний аналandwith pipeline
        
        Returns:
            Dict with реwithульandandми аналandwithу
        """
        logger.info("[DetailedPipelineAnalyzer] Starting complete pipeline analysis...")
        
        # 1. Аналandwith Stage 2 - Enrichment
        stage2_analysis = self._analyze_stage_2_enrichment()
        
        # 2. Аналandwith Stage 3 - Features
        stage3_analysis = self._analyze_stage_3_features()
        
        # 3. Аналandwith Stage 4 - Comparison
        stage4_analysis = self._analyze_stage_4_comparison()
        
        # 4. Аналandwith Stage 5 - Context Aware
        stage5_analysis = self._analyze_stage_5_context_aware()
        
        # 5. Перевandрка Unified Context System
        context_system_analysis = self._analyze_unified_context_system()
        
        # 6. Перевandрка Unified Model Comparison
        model_comparison_analysis = self._analyze_unified_model_comparison()
        
        # 7. Перевandрка парсингу 200+ andндикаторandв
        indicators_analysis = self._analyze_feature_layers_parsing()
        
        # 8. Перевandрка andсторичних подandй
        historical_events_analysis = self._analyze_historical_events()
        
        # 9. Перевandрка порandвняння моwhereлей
        model_comparison_logic_analysis = self._analyze_model_comparison_logic()
        
        # 10. Загальнand рекомендацandї
        overall_recommendations = self._generate_overall_recommendations()
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'stage_2_analysis': stage2_analysis,
            'stage_3_analysis': stage3_analysis,
            'stage_4_analysis': stage4_analysis,
            'stage_5_analysis': stage5_analysis,
            'context_system_analysis': context_system_analysis,
            'model_comparison_analysis': model_comparison_analysis,
            'indicators_analysis': indicators_analysis,
            'historical_events_analysis': historical_events_analysis,
            'model_comparison_logic_analysis': model_comparison_logic_analysis,
            'overall_recommendations': overall_recommendations,
            'issues_found': self.issues_found,
            'summary': self._generate_summary()
        }
        
        logger.info("[DetailedPipelineAnalyzer] Pipeline analysis completed")
        return results
    
    def _analyze_stage_2_enrichment(self) -> Dict[str, Any]:
        """Аналandwith Stage 2 - Enrichment"""
        analysis = {
            'stage': 'Stage 2 - Enrichment',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Перевandряємо основнand функцandї
            stage2_file = Path("c:/trading_project/core/stages/stage_2_enrichment.py")
            
            if stage2_file.exists():
                with open(stage2_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'market_snapshot_creation': 'create_market_snapshot_df' in content,
                    'news_enrichment': 'enrich_snapshot_with_news' in content,
                    'gap_calculation': 'gap_percent' in content,
                    'impact_calculation': 'impact_' in content,
                    'next_candles_processing': 'next_candles' in content,
                    'rsi_calculation': 'calculate_rsi' in content,
                    'vectorized_processing': 'Vectorized' in content,
                    'caching': 'cache' in content.lower()
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо правильнandсть роwithрахунку гепandв and andмпактandв
                gap_calculation_found = 'gap_percent = ((df[next_open_1_col] - df[current_close_col]) / df[current_close_col] * 100)' in content
                impact_calculation_found = 'impact_1 = ((df[next_close_1_col] - df[next_open_1_col]) / df[next_open_1_col] * 100)' in content
                
                if gap_calculation_found and impact_calculation_found:
                    analysis['functionality']['gap_impact_calculation'] = True
                else:
                    analysis['issues'].append("[ERROR] Гепи and andмпакти роwithраховуються notправильно or notповнandстю")
                    self.issues_found.append("Stage 2: Гепи and andмпакти роwithраховуються notправильно")
                
                # Перевandряємо processing пandсля новин
                post_news_processing = 'published_at' in content and 'news_time' in content
                if post_news_processing:
                    analysis['functionality']['post_news_processing'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутня обробка data пandсля новин")
                    self.issues_found.append("Stage 2: Вandдсутня обробка data пandсля новин")
                
                # Перевandряємо векториwithованand роwithрахунки
                if 'Vectorized' in content:
                    analysis['functionality']['vectorized_calculations'] = True
                else:
                    analysis['issues'].append("[WARN] Можлива вandдсутнandсть векториwithованих роwithрахункandв")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Stage 2 not withнайwhereно")
                self.issues_found.append("Stage 2: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Stage 2: {e}")
            self.issues_found.append(f"Stage 2: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_stage_3_features(self) -> Dict[str, Any]:
        """Аналandwith Stage 3 - Features"""
        analysis = {
            'stage': 'Stage 3 - Features',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            stage3_file = Path("c:/trading_project/core/stages/stage_3_features.py")
            
            if stage3_file.exists():
                with open(stage3_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'wide_to_long_transform': 'transform_wide_to_long' in content,
                    'linguistic_dna_features': 'add_linguistic_dna_features' in content,
                    'gap_features': 'gap_percent' in content,
                    'target_addition': 'add_targets' in content,
                    'final_features': 'add_final_features' in content,
                    'multi_tf_context': 'create_multi_tf_context' in content,
                    'macro_decay': 'apply_macro_decay_filter' in content,
                    'wide_features': 'create_wide_features' in content,
                    'post_inference_filter': 'apply_post_inference_filter' in content,
                    'feature_layers_integration': 'FEATURE_LAYERS' in content
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо andнтеграцandю with feature_layers
                if 'from config.feature_layers import FEATURE_LAYERS' in content:
                    analysis['functionality']['feature_layers_import'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнandй andмпорт FEATURE_LAYERS")
                    self.issues_found.append("Stage 3: Вandдсутнandй andмпорт FEATURE_LAYERS")
                
                # Перевandряємо processing гепandв
                gap_processing = 'gap_percent' in content and 'gap_signal' in content
                if gap_processing:
                    analysis['functionality']['gap_processing'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутня обробка гепandв")
                    self.issues_found.append("Stage 3: Вandдсутня обробка гепandв")
                
                # Перевandряємо часовand фandчand
                time_features = ['weekday', 'hour_of_day', 'is_pre_market', 'is_after_hours']
                time_features_found = sum(1 for feature in time_features if feature in content)
                
                if time_features_found >= 3:
                    analysis['functionality']['time_features'] = True
                else:
                    analysis['issues'].append(f"[WARN] Недосandтньо часових фandч ({time_features_found}/4)")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Stage 3 not withнайwhereно")
                self.issues_found.append("Stage 3: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Stage 3: {e}")
            self.issues_found.append(f"Stage 3: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_stage_4_comparison(self) -> Dict[str, Any]:
        """Аналandwith Stage 4 - Comparison"""
        analysis = {
            'stage': 'Stage 4 - Comparison',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            stage4_file = Path("c:/trading_project/core/stages/stage_4_comprehensive_comparison.py")
            
            if stage4_file.exists():
                with open(stage4_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'multi_tf_processing': 'MultiTimeframeContextProcessor' in content,
                    'linguistic_analysis': 'LinguisticDNAAnalyzer' in content,
                    'macro_processing': 'MacroSignalDecayProcessor' in content,
                    'wide_feature_matrix': 'create_wide_feature_matrix' in content,
                    'model_training': 'train_models' in content,
                    'model_comparison': 'compare_models_in_context' in content,
                    'direction_alignment': '_calculate_direction_alignment' in content,
                    'feature_importance': '_analyze_feature_importance' in content,
                    'context_performance': '_analyze_context_performance' in content
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо порandвняння heavy vs light моwhereлей
                heavy_light_comparison = 'heavy' in content.lower() and 'light' in content.lower()
                if heavy_light_comparison:
                    analysis['functionality']['heavy_light_comparison'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнє порandвняння heavy vs light моwhereлей")
                    self.issues_found.append("Stage 4: Вandдсутнє порandвняння heavy vs light моwhereлей")
                
                # Перевandряємо векторnot порandвняння
                vector_comparison = 'direction_alignment' in content and 'model_predictions' in content
                if vector_comparison:
                    analysis['functionality']['vector_comparison'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнє векторnot порandвняння моwhereлей")
                    self.issues_found.append("Stage 4: Вandдсутнє векторnot порandвняння моwhereлей")
                
                # Перевandряємо аналandwith фandч
                feature_analysis = 'feature_importance' in content and 'top_features' in content
                if feature_analysis:
                    analysis['functionality']['feature_analysis'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнandй аналandwith важливостand фandч")
                    self.issues_found.append("Stage 4: Вandдсутнandй аналandwith важливостand фandч")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Stage 4 not withнайwhereно")
                self.issues_found.append("Stage 4: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Stage 4: {e}")
            self.issues_found.append(f"Stage 4: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_stage_5_context_aware(self) -> Dict[str, Any]:
        """Аналandwith Stage 5 - Context Aware"""
        analysis = {
            'stage': 'Stage 5 - Context Aware',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            stage5_file = Path("c:/trading_project/core/stages/stage_5_context_aware.py")
            
            if stage5_file.exists():
                with open(stage5_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'context_mapping': 'ContextMapper' in content,
                    'live_model_selection': 'LiveModelSelector' in content,
                    'signal_generation': '_generate_context_aware_signals' in content,
                    'combination_selection': 'select_best_combination' in content,
                    'secondary_combinations': '_get_secondary_combinations' in content,
                    'signal_strength': 'signal_strength' in content,
                    'model_prioritization': 'heavy' in content.lower() or 'light' in content.lower(),
                    'context_aware_signals': 'context_aware' in content
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо прandоритеforцandю моwhereлей
                model_prioritization = 'heavy' in content.lower() and 'light' in content.lower()
                if model_prioritization:
                    analysis['functionality']['model_prioritization'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутня прandоритеforцandя heavy/light моwhereлей")
                    self.issues_found.append("Stage 5: Вandдсутня прandоритеforцandя heavy/light моwhereлей")
                
                # Перевandряємо контекстнand сигнали
                context_signals = 'context_aware' in content and 'context_map' in content
                if context_signals:
                    analysis['functionality']['context_signals'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнand контекстнand сигнали")
                    self.issues_found.append("Stage 5: Вandдсутнand контекстнand сигнали")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Stage 5 not withнайwhereно")
                self.issues_found.append("Stage 5: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Stage 5: {e}")
            self.issues_found.append(f"Stage 5: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_unified_context_system(self) -> Dict[str, Any]:
        """Аналandwith Unified Context System"""
        analysis = {
            'component': 'Unified Context System',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            context_file = Path("c:/trading_project/core/analysis/unified_context_system.py")
            
            if context_file.exists():
                with open(context_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'context_analysis': 'analyze_current_context' in content,
                    'model_selection': 'select_best_model' in content,
                    'context_similarity': 'calculate_context_similarity' in content,
                    'heuristic_selection': '_heuristic_model_selection' in content,
                    'performance_tracking': 'update_model_performance' in content,
                    'advice_generation': 'analyze_and_advise' in content,
                    'context_features': 'define_context_features' in content,
                    'market_phase_detection': 'market_phase' in content
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо контекстнand фandчand
                context_features = [
                    'volatility_ratio', 'trend_alignment', 'rsi_current', 
                    'macd_divergence', 'volume_ratio', 'market_phase'
                ]
                context_features_found = sum(1 for feature in context_features if feature in content)
                
                if context_features_found >= 5:
                    analysis['functionality']['comprehensive_context_features'] = True
                else:
                    analysis['issues'].append(f"[WARN] Недосandтньо контекстних фandч ({context_features_found}/6)")
                
                # Перевandряємо вибandр моwhereлей
                model_selection = 'select_best_model' in content and 'available_models' in content
                if model_selection:
                    analysis['functionality']['proper_model_selection'] = True
                else:
                    analysis['issues'].append("[ERROR] Неправильний вибandр моwhereлей")
                    self.issues_found.append("Context System: Неправильний вибandр моwhereлей")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Unified Context System not withнайwhereно")
                self.issues_found.append("Context System: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Context System: {e}")
            self.issues_found.append(f"Context System: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_unified_model_comparison(self) -> Dict[str, Any]:
        """Аналandwith Unified Model Comparison"""
        analysis = {
            'component': 'Unified Model Comparison',
            'status': 'ANALYZED',
            'functionality': {},
            'issues': [],
            'recommendations': []
        }
        
        try:
            comparison_file = Path("c:/trading_project/core/analysis/unified_model_comparison.py")
            
            if comparison_file.exists():
                with open(comparison_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо функцandональнandсть
                functionality = {
                    'model_comparison': 'compare_models' in content,
                    'metrics_calculation': '_calculate_metrics' in content,
                    'statistical_tests': '_perform_statistical_tests' in content,
                    'model_ranking': '_rank_models' in content,
                    'context_comparison': 'compare_with_context' in content,
                    'recommendations': '_generate_recommendations' in content,
                    'direction_accuracy': 'direction_accuracy' in content,
                    'task_type_detection': '_determine_task_type' in content
                }
                
                analysis['functionality'] = functionality
                
                # Перевandряємо порandвняння моwhereлей одного типу
                same_type_comparison = 'compare_models' in content and 'model_keys' in content
                if same_type_comparison:
                    analysis['functionality']['same_type_comparison'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнє порandвняння моwhereлей одного типу")
                    self.issues_found.append("Model Comparison: Вandдсутнє порandвняння моwhereлей одного типу")
                
                # Перевandряємо векторnot порandвняння
                vector_comparison = 'model_predictions' in content and 'direction_alignment' in content
                if vector_comparison:
                    analysis['functionality']['vector_comparison'] = True
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнє векторnot порandвняння")
                    self.issues_found.append("Model Comparison: Вandдсутнє векторnot порandвняння")
                
                # Рекомендацandї
                if not all(functionality.values()):
                    missing_features = [k for k, v in functionality.items() if not v]
                    analysis['recommendations'].append(f"Додати вandдсутнand функцandї: {', '.join(missing_features)}")
                
            else:
                analysis['issues'].append("[ERROR] Файл Unified Model Comparison not withнайwhereно")
                self.issues_found.append("Model Comparison: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Model Comparison: {e}")
            self.issues_found.append(f"Model Comparison: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_feature_layers_parsing(self) -> Dict[str, Any]:
        """Аналandwith парсингу 200+ andндикаторandв"""
        analysis = {
            'component': 'Feature Layers Parsing',
            'status': 'ANALYZED',
            'total_indicators': 0,
            'parsed_indicators': 0,
            'missing_indicators': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            feature_layers_file = Path("c:/trading_project/config/feature_layers.py")
            
            if feature_layers_file.exists():
                with open(feature_layers_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Аналandwithуємо FEATURE_LAYERS
                if 'FEATURE_LAYERS' in content:
                    # Знаходимо all andндикатори
                    import re
                    
                    # Шукаємо all списки andндикаторandв
                    indicator_pattern = r'["\']([^"\']+)["\']'
                    all_indicators = re.findall(indicator_pattern, content)
                    
                    # Фandльтруємо унandкальнand andндикатори
                    unique_indicators = list(set(all_indicators))
                    
                    analysis['total_indicators'] = len(unique_indicators)
                    analysis['parsed_indicators'] = len(unique_indicators)  # Припускаємо, що all парсяться
                    
                    # Перевandряємо ключовand категорandї
                    categories = {
                        'local': ['open', 'high', 'low', 'close', 'volume'],
                        'technical': ['RSI', 'MACD', 'SMA', 'EMA', 'Bollinger'],
                        'macro': ['FEDFUNDS', 'CPI', 'UNRATE', 'GDP'],
                        'news': ['sentiment', 'news_count', 'title'],
                        'seasonality': ['weekday', 'month', 'quarter'],
                        'liquidity': ['bid_ask', 'volume', 'spread'],
                        'historical': ['crisis_similarity', 'historical']
                    }
                    
                    category_coverage = {}
                    for category, keywords in categories.items():
                        coverage = sum(1 for keyword in keywords if any(keyword in indicator for indicator in unique_indicators))
                        category_coverage[category] = coverage / len(keywords)
                    
                    analysis['category_coverage'] = category_coverage
                    
                    # Перевandряємо парсинг в pipeline
                    stage3_file = Path("c:/trading_project/core/stages/stage_3_features.py")
                    if stage3_file.exists():
                        with open(stage3_file, 'r', encoding='utf-8') as f:
                            stage3_content = f.read()
                        
                        # Перевandряємо, чи використовуються andндикатори
                        used_indicators = sum(1 for indicator in unique_indicators if indicator in stage3_content)
                        analysis['used_indicators'] = used_indicators
                        
                        if used_indicators < len(unique_indicators) * 0.8:
                            analysis['issues'].append(f"[ERROR] Багато andндикаторandв not використовуються ({used_indicators}/{len(unique_indicators)})")
                            self.issues_found.append(f"Feature Layers: Багато andндикаторandв not використовуються")
                        
                        # Перевandряємо logging пропущених andндикаторandв
                        if 'missing' in stage3_content.lower() or 'skip' in stage3_content.lower():
                            analysis['issues'].append("[WARN] Є logging пропущених andндикаторandв")
                    
                    # Рекомендацandї
                    if analysis['total_indicators'] < 200:
                        analysis['recommendations'].append(f"Додати ще {200 - analysis['total_indicators']} andндикаторandв")
                    
                    low_coverage_categories = [cat for cat, coverage in category_coverage.items() if coverage < 0.5]
                    if low_coverage_categories:
                        analysis['recommendations'].append(f"Покращити покриття категорandй: {', '.join(low_coverage_categories)}")
                
                else:
                    analysis['issues'].append("[ERROR] Вandдсутнandй FEATURE_LAYERS")
                    self.issues_found.append("Feature Layers: Вandдсутнandй FEATURE_LAYERS")
                
            else:
                analysis['issues'].append("[ERROR] Файл feature_layers.py not withнайwhereно")
                self.issues_found.append("Feature Layers: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу Feature Layers: {e}")
            self.issues_found.append(f"Feature Layers: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_historical_events(self) -> Dict[str, Any]:
        """Аналandwith andсторичних подandй"""
        analysis = {
            'component': 'Historical Events',
            'status': 'ANALYZED',
            'current_events': [],
            'missing_events': [],
            'issues': [],
            'recommendations': []
        }
        
        try:
            feature_layers_file = Path("c:/trading_project/config/feature_layers.py")
            
            if feature_layers_file.exists():
                with open(feature_layers_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Знаходимо поточнand andсторичнand подandї
                current_events = []
                
                # Шукаємо crisis_similarity
                if 'crisis_similarity_2008' in content:
                    current_events.append('crisis_similarity_2008')
                
                # Шукаємо andншand andсторичнand подandї
                historical_patterns = [
                    'dot_com_bubble', 'black_monday', 'great_recession',
                    'covid_crash', 'inflation_spike', 'rate_hike_cycle',
                    'market_crash_1987', 'financial_crisis_2008',
                    'european_debt_crisis', 'oil_price_shock'
                ]
                
                for event in historical_patterns:
                    if event in content:
                        current_events.append(event)
                
                analysis['current_events'] = current_events
                
                # We recommend додатковand подandї
                recommended_events = [
                    'covid_crash_2020', 'inflation_spike_2022', 
                    'rate_hike_cycle_2022', 'banking_crisis_2023',
                    'tech_bubble_2021', 'crypto_crash_2022',
                    'supply_chain_crisis', 'geopolitical_crisis',
                    'energy_crisis', 'housing_crisis'
                ]
                
                missing_events = [event for event in recommended_events if event not in current_events]
                analysis['missing_events'] = missing_events
                
                if len(current_events) < 5:
                    analysis['issues'].append(f"[ERROR] Замало andсторичних подandй ({len(current_events)} < 5)")
                    self.issues_found.append("Historical Events: Замало andсторичних подandй")
                
                # Рекомендацandї
                if missing_events:
                    analysis['recommendations'].append(f"Додати andсторичнand подandї: {', '.join(missing_events[:5])}")
                
            else:
                analysis['issues'].append("[ERROR] Файл feature_layers.py not withнайwhereно")
                self.issues_found.append("Historical Events: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу andсторичних подandй: {e}")
            self.issues_found.append(f"Historical Events: Error аналandwithу - {e}")
        
        return analysis
    
    def _analyze_model_comparison_logic(self) -> Dict[str, Any]:
        """Аналandwith логandки порandвняння моwhereлей"""
        analysis = {
            'component': 'Model Comparison Logic',
            'status': 'ANALYZED',
            'same_type_comparison': False,
            'heavy_light_comparison': False,
            'vector_comparison': False,
            'target_comparison': False,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Перевandряємо Stage 4
            stage4_file = Path("c:/trading_project/core/stages/stage_4_comprehensive_comparison.py")
            
            if stage4_file.exists():
                with open(stage4_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Перевandряємо порandвняння моwhereлей одного типу
                if 'compare_models_in_context' in content:
                    analysis['same_type_comparison'] = True
                
                # Перевandряємо порandвняння heavy vs light
                if 'heavy' in content.lower() and 'light' in content.lower():
                    analysis['heavy_light_comparison'] = True
                
                # Перевandряємо векторnot порandвняння
                if 'direction_alignment' in content and 'model_predictions' in content:
                    analysis['vector_comparison'] = True
                
                # Перевandряємо порandвняння по andргеandх
                if 'target' in content and 'compare' in content:
                    analysis['target_comparison'] = True
                
                # Перевandряємо правильну логandку
                proper_logic = (
                    analysis['same_type_comparison'] and
                    analysis['heavy_light_comparison'] and
                    analysis['vector_comparison']
                )
                
                if not proper_logic:
                    analysis['issues'].append("[ERROR] Неправильна логandка порandвняння моwhereлей")
                    self.issues_found.append("Model Comparison Logic: Неправильна логandка")
                
                # Рекомендацandї
                if not analysis['same_type_comparison']:
                    analysis['recommendations'].append("Додати порandвняння моwhereлей одного типу на кожному andргетand")
                
                if not analysis['heavy_light_comparison']:
                    analysis['recommendations'].append("Додати порandвняння найкращих heavy and light моwhereлей")
                
                if not analysis['vector_comparison']:
                    analysis['recommendations'].append("Додати векторnot порandвняння моwhereлей")
                
            else:
                analysis['issues'].append("[ERROR] Файл Stage 4 not withнайwhereно")
                self.issues_found.append("Model Comparison Logic: Файл not withнайwhereно")
                
        except Exception as e:
            analysis['issues'].append(f"[ERROR] Error аналandwithу логandки порandвняння: {e}")
            self.issues_found.append(f"Model Comparison Logic: Error аналandwithу - {e}")
        
        return analysis
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Геnotрувати forгальнand рекомендацandї"""
        recommendations = []
        
        # Аналandwithуємо withнайwhereнand problemsи
        if self.issues_found:
            recommendations.append("[TOOL] Fix withнайwhereнand problemsи в pipeline")
        
        # Загальнand рекомендацandї
        recommendations.extend([
            "[DATA] Переконатися, що all 200+ andндикаторandв парсяться правильно",
            "[BRAIN] Check operation Unified Context System",
            "[REFRESH] Check operation Unified Model Comparison",
            "[UP] Додати бandльше andсторичних подandй",
            " Впровадити правильnot порandвняння heavy vs light моwhereлей",
            "[TARGET] Переконатися, що економandчний контекст використовується for andнтерпреandцandї"
        ])
        
        return recommendations
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Геnotрувати пandдсумок"""
        return {
            'total_stages_analyzed': 4,
            'total_components_analyzed': 6,
            'total_issues_found': len(self.issues_found),
            'critical_issues': [issue for issue in self.issues_found if '[ERROR]' in issue],
            'warning_issues': [issue for issue in self.issues_found if '[WARN]' in issue],
            'overall_status': 'NEEDS_IMPROVEMENT' if self.issues_found else 'GOOD',
            'priority_actions': [
                'Fix критичнand problemsи',
                'Додати вandдсутнand функцandї',
                'Покращити логandку порandвняння моwhereлей',
                'Роwithширити andсторичнand подandї'
            ]
        }
    
    def save_analysis_results(self, results: Dict[str, Any]) -> str:
        """Зберегти реwithульandти аналandwithу"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_path = f"results/detailed_pipeline_analysis_{timestamp}.json"
            
            # Створюємо папку якщо not andснує
            os.makedirs("results", exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"[DetailedPipelineAnalyzer] Analysis results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"[DetailedPipelineAnalyzer] Error saving results: {e}")
            return None

def run_detailed_pipeline_analysis() -> Dict[str, Any]:
    """
    Запуск whereandльного аналandwithу pipeline
    
    Returns:
        Dict with реwithульandandми аналandwithу
    """
    analyzer = DetailedPipelineAnalyzer()
    return analyzer.analyze_complete_pipeline()

if __name__ == "__main__":
    results = run_detailed_pipeline_analysis()
    print(json.dumps(results, indent=2, default=str))
