"""
COMPLETE PIPELINE ARCHITECTURE
Повна архandтектура пайплайну with notйро-когнandтивною andнтеграцandєю
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

@dataclass
class PipelineExecutionContext:
    """Повний контекст виконання пайплайну"""
    # Основнand параметри
    tickers: List[str]
    timeframes: List[str]
    targets: List[str]
    features: List[str]
    
    # Контекстнand покаwithники (топ-100)
    context_features: List[str]
    
    # Еandпи виконання
    current_stage: str
    stage_data: Dict[str, Any]
    
    # Нейро-когнandтивний контекст
    neuro_cognitive_context: Optional[Dict[str, Any]] = None
    
    # Сandтистика виконання
    execution_stats: Dict[str, Any] = None

class CompletePipelineArchitecture:
    """
    Повна архandтектура пайплайну with notйро-когнandтивною andнтеграцandєю
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Інandцandалandforцandя компоnotнтandв
        self._initialize_components()
        
        # Сandтистика архandтектури
        self.architecture_stats = {
            'total_tickers_processed': 0,
            'total_timeframes_processed': 0,
            'total_targets_processed': 0,
            'total_features_used': 0,
            'context_features_selected': 0,
            'neuro_cognitive_enhancements': 0,
            'parallel_processing_time': 0,
            'sequential_processing_time': 0
        }
    
    def _initialize_components(self):
        """Інandцandалandforцandя allх компоnotнтandв архandтектури"""
        
        # 1. Традицandйнand компоnotнти пайплайну
        from collectors import YFCollector, NewsCollector
        from enrichment.sentiment_analyzer import SentimentEnricher
        from core.significance_detector import SignificanceDetector
        from utils.cache_utils import CacheManager
        
        # Інandцandалandwithуємо cache manager for sentiment enricher
        cache_manager = CacheManager()
        
        # Оптимальний волатильний набір - 20 тікерів з максимальними можливостями заробку
        # Екстремальні (8): TSLA, NVDA, AMD, COIN, MARA, RIOT, PLTR, GME - максимальні можливості
        # Tech волатильні (6): META, NFLX, ROKU, SNAP, AAPL, MSFT - стабільний ріст + рухи
        # Волатильні ETF (3): QQQ, SOXX, ARKK - ринкові рухи
        # Енергетика (3): XOM, CVX, CLF - циклічність
        optimal_volatile_tickers = [
            'TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'RIOT', 'PLTR', 'GME',  # Екстремальні
            'META', 'NFLX', 'ROKU', 'SNAP', 'AAPL', 'MSFT',  # Tech волатильні
            'QQQ', 'SOXX', 'ARKK',  # Волатильні ETF
            'XOM', 'CVX', 'CLF'  # Енергетика
        ]
        
        self.data_collectors = {
            'yf_collector': YFCollector(),
            'news_collector': NewsCollector()
        }
        
        self.data_enrichers = {
            'sentiment_enricher': SentimentEnricher(cache_manager=cache_manager, keyword_dict={'all': optimal_volatile_tickers}),  # Оптимальний волатильний набір
            'significance_detector': SignificanceDetector()
        }
        
        # 2. Компоnotнти вибору фandч
        from utils.feature_selector import select_features  # Виправлено на функцandю
        self.feature_selector = select_features  # Використовуємо функцandю
        
        # 3. Моwhereлand (легкand локальнand, важкand пакетнand)
        from core.models.light_models import LightGBMModel, XGBoostModel  # Виправлено шлях
        # from models.heavy_models import HeavyModelManager  # Вandдключено (вandдсутнandй)
        
        self.light_models = {
            'lightgbm': LightGBMModel(),
            'xgboost': XGBoostModel()
        }
        
        # self.heavy_model_manager = HeavyModelManager()  # Вandдключено (вandдсутнandй)
        
        # 4. Нейро-когнandтивнand компоnotнти
        from utils.neuro_cognitive_context_analyzer import get_neuro_cognitive_analyzer
        from core.dean_pipeline_integration import get_dean_pipeline_integrator
        
        self.neuro_analyzer = get_neuro_cognitive_analyzer()
        self.dean_integrator = get_dean_pipeline_integrator()
        
        # 5. Компоnotнти аналandwithу and порandвняння
        from core.analysis.model_comparison_engine import ModelComparisonEngine
        self.comparison_engine = ModelComparisonEngine()
        
        self.logger.info("[START] Complete Pipeline Architecture initialized")
    
    def execute_complete_pipeline(self, tickers: List[str], 
                                timeframes: List[str],
                                targets: List[str],
                                features: List[str]) -> Dict[str, Any]:
        """
        Виконання повного пайплайну with notйро-когнandтивною andнтеграцandєю
        """
        self.logger.info("[START] Starting Complete Pipeline Execution")
        self.logger.info(f"[DATA] Processing {len(tickers)} tickers, {len(timeframes)} timeframes, {len(targets)} targets")
        
        # 1. ЕТАП 1: ПАРСИНГ ТА ЗБАГАЧЕННЯ ДАНИХ
        parsing_results = self._execute_parsing_and_enrichment(tickers, timeframes)
        
        # 2. ЕТАП 2: ВИБІР ФІЧ (ГНУЧКИЙ + КОНТЕКСТНІ)
        feature_selection_results = self._execute_feature_selection(
            parsing_results, features, targets
        )
        
        # 3. ЕТАП 3: ТРЕНУВАННЯ ЛЕГКИХ МОДЕЛЕЙ (ЛОКАЛЬНО)
        light_training_results = self._execute_light_model_training(
            feature_selection_results, targets
        )
        
        # 4. ЕТАП 4: ТРЕНУВАННЯ ВАЖКИХ МОДЕЛЕЙ (ПАКЕТНО В COLAB)
        heavy_training_results = self._execute_heavy_model_training(
            feature_selection_results, targets
        )
        
        # 5. ЕТАП 5: АНАЛІЗ ТА ПОРІВНЯННЯ РЕЗУЛЬТАТІВ
        analysis_results = self._execute_model_analysis_and_comparison(
            light_training_results, heavy_training_results
        )
        
        # 6. ЕТАП 6: ДОДАВАННЯ КОНТЕКСТУ (ТОП-100 ПОКАЗНИКІВ)
        context_enrichment_results = self._execute_context_enrichment(
            analysis_results, feature_selection_results
        )
        
        # 7. ЕТАП 7: ПАРАЛЕЛЬНА НЕЙРО-КОГНІТИВНА ІНТЕГРАЦІЯ
        neuro_cognitive_results = self._execute_parallel_neuro_cognitive_integration(
            context_enrichment_results
        )
        
        # 8. ЕТАП 8: ФІНАЛЬНА ІНТЕГРАЦІЯ ТА РЕЗУЛЬТАТИ
        final_results = self._execute_final_integration(
            neuro_cognitive_results, context_enrichment_results
        )
        
        # 9. Оновлення сandтистики
        self._update_architecture_stats(tickers, timeframes, targets, features)
        
        return final_results
    
    def _execute_parsing_and_enrichment(self, tickers: List[str], 
                                      timeframes: List[str]) -> Dict[str, Any]:
        """Еandп 1: Парсинг and withбагачення data"""
        self.logger.info("[DATA] Stage 1: Parsing and Data Enrichment")
        
        results = {}
        
        for ticker in tickers:
            ticker_results = {}
            
            for timeframe in timeframes:
                self.logger.info(f"[UP] Processing {ticker} {timeframe}")
                
                # 1.1 Парсинг data
                price_data = self.data_collectors['yf_collector'].collect_data(ticker, timeframe)
                news_data = self.data_collectors['news_collector'].collect_news(ticker)
                
                # 1.2 Збагачення data
                enriched_data = price_data.copy()
                
                # Сентиментnot withбагачення
                if not news_data.empty:
                    sentiment_data = self.data_enrichers['sentiment_enricher'].enrich_data(
                        price_data, news_data
                    )
                    enriched_data = pd.concat([enriched_data, sentiment_data], axis=1)
                
                # Детекцandя withначущих подandй
                significance_data = self.data_enrichers['significance_detector'].detect_events(
                    enriched_data
                )
                enriched_data = pd.concat([enriched_data, significance_data], axis=1)
                
                # Нейро-когнandтивnot покращення withбору data
                from core.dean_pipeline_integration import PipelineContext, PipelineStage
                context = PipelineContext(
                    stage=PipelineStage.DATA_COLLECTION,
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    data={'price_data': enriched_data},
                    features={},
                    models={},
                    predictions={}
                )
                
                dean_enhancement = self.dean_integrator.enhance_data_collection(context)
                
                ticker_results[timeframe] = {
                    'raw_data': enriched_data,
                    'dean_enhanced_data': dean_enhancement.enhanced_data,
                    'data_quality_score': dean_enhancement.cognitive_insights['data_quality_score'],
                    'anomaly_patterns': dean_enhancement.cognitive_insights['anomaly_patterns']
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_feature_selection(self, parsing_results: Dict[str, Any],
                                 features: List[str],
                                 targets: List[str]) -> Dict[str, Any]:
        """Еandп 2: Вибandр фandч (гнучкий + контекстнand)"""
        self.logger.info("[TOOL] Stage 2: Feature Selection (Flexible + Context)")
        
        results = {}
        
        for ticker, ticker_data in parsing_results.items():
            ticker_results = {}
            
            for timeframe, data_info in ticker_data.items():
                self.logger.info(f"[TOOL] Selecting features for {ticker} {timeframe}")
                
                # 2.1 Гнучкий вибandр фandч
                available_features = list(data_info['dean_enhanced_data'].columns)
                selected_features = self.feature_selector.select_features_flexible(
                    data_info['dean_enhanced_data'], 
                    available_features,
                    targets
                )
                
                # 2.2 Вибandр контекстних покаwithникandв (топ-100)
                context_features = self.feature_selector.select_context_features(
                    data_info['dean_enhanced_data'],
                    top_n=100
                )
                
                # 2.3 Нейро-когнandтивnot покращення feature engineering
                from core.dean_pipeline_integration import PipelineContext, PipelineStage
                context = PipelineContext(
                    stage=PipelineStage.FEATURE_ENGINEERING,
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    data={'selected_features': selected_features},
                    features={feat: data_info['dean_enhanced_data'][feat].values 
                             for feat in selected_features if feat in data_info['dean_enhanced_data'].columns},
                    models={},
                    predictions={}
                )
                
                dean_enhancement = self.dean_integrator.enhance_feature_engineering(context)
                
                ticker_results[timeframe] = {
                    'selected_features': selected_features,
                    'context_features': context_features,
                    'dean_enhanced_features': dean_enhancement.enhanced_data,
                    'feature_optimization_score': dean_enhancement.cognitive_insights['feature_optimization_score'],
                    'cognitive_patterns': dean_enhancement.cognitive_insights['cognitive_patterns']
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_light_model_training(self, feature_selection_results: Dict[str, Any],
                                    targets: List[str]) -> Dict[str, Any]:
        """Еandп 3: Тренування легких моwhereлей (локально)"""
        self.logger.info(" Stage 3: Light Model Training (Local)")
        
        results = {}
        
        for ticker, ticker_data in feature_selection_results.items():
            ticker_results = {}
            
            for timeframe, data_info in ticker_data.items():
                self.logger.info(f" Training light models for {ticker} {timeframe}")
                
                # 3.1 Пandдготовка data
                features_data = data_info['dean_enhanced_features']
                
                # 3.2 Тренування легких моwhereлей
                model_results = {}
                
                for model_name, model in self.light_models.items():
                    for target in targets:
                        target_key = f"target_{ticker}_{timeframe}"
                        
                        if target_key in features_data.columns:
                            # Тренування моwhereлand
                            training_result = model.train(
                                features_data.drop(columns=[target_key]),
                                features_data[target_key]
                            )
                            
                            model_results[f"{model_name}_{target}"] = training_result
                
                # 3.3 Нейро-когнandтивnot покращення тренування
                from core.dean_pipeline_integration import PipelineContext, PipelineStage
                context = PipelineContext(
                    stage=PipelineStage.MODEL_TRAINING,
                    ticker=ticker,
                    timeframe=timeframe,
                    timestamp=datetime.now(),
                    data={},
                    features=data_info['dean_enhanced_features'],
                    models=model_results,
                    predictions={}
                )
                
                dean_enhancement = self.dean_integrator.enhance_model_training(context)
                
                ticker_results[timeframe] = {
                    'light_models': model_results,
                    'dean_enhanced_models': dean_enhancement.enhanced_data,
                    'bootstrap_performance': dean_enhancement.cognitive_insights['bootstrap_performance'],
                    'adversarial_robustness': dean_enhancement.cognitive_insights['adversarial_robustness']
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_heavy_model_training(self, feature_selection_results: Dict[str, Any],
                                   targets: List[str]) -> Dict[str, Any]:
        """Еandп 4: Тренування важких моwhereлей (пакетно в Colab)"""
        self.logger.info(" Stage 4: Heavy Model Training (Batch in Colab)")
        
        results = {}
        
        # 4.1 Пandдготовка пакетних data for Colab
        batch_data = self._prepare_batch_data_for_colab(feature_selection_results, targets)
        
        # 4.2 Вandдправка в Colab for тренування
        self.logger.info(" Sending data to Colab for heavy model training")
        
        # Симуляцandя Colab тренування (в реальностand - вandдправка data)
        colab_training_results = self.heavy_model_manager.train_models_in_colab(batch_data)
        
        # 4.3 Отримання реwithульandтandв with Colab
        self.logger.info(" Receiving results from Colab")
        
        # 4.4 Обробка реwithульandтandв
        for ticker, ticker_data in feature_selection_results.items():
            ticker_results = {}
            
            for timeframe in ticker_data.keys():
                # Отримання реwithульandтandв for конкретного тandкера/andймфрейму
                heavy_model_results = colab_training_results.get(f"{ticker}_{timeframe}", {})
                
                ticker_results[timeframe] = {
                    'heavy_models': heavy_model_results,
                    'training_time': heavy_model_results.get('training_time', 0),
                    'model_performance': heavy_model_results.get('performance', {})
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_model_analysis_and_comparison(self, light_results: Dict[str, Any],
                                             heavy_results: Dict[str, Any]) -> Dict[str, Any]:
        """Еandп 5: Аналandwith and порandвняння реwithульandтandв моwhereлей"""
        self.logger.info("[DATA] Stage 5: Model Analysis and Comparison")
        
        results = {}
        
        for ticker in light_results.keys():
            ticker_results = {}
            
            for timeframe in light_results[ticker].keys():
                self.logger.info(f"[DATA] Analyzing models for {ticker} {timeframe}")
                
                # 5.1 Збandр реwithульandтandв allх моwhereлей
                light_models = light_results[ticker][timeframe]['light_models']
                heavy_models = heavy_results[ticker][timeframe]['heavy_models']
                
                all_models = {**light_models, **heavy_models}
                
                # 5.2 Порandвняння моwhereлей
                comparison_result = self.comparison_engine.compare_models(
                    all_models, ticker, timeframe
                )
                
                # 5.3 Аналandwith покаwithникandв
                metrics_analysis = self.comparison_engine.analyze_performance_metrics(
                    all_models, comparison_result
                )
                
                ticker_results[timeframe] = {
                    'model_comparison': comparison_result,
                    'metrics_analysis': metrics_analysis,
                    'best_models': comparison_result.get('best_models', {}),
                    'performance_ranking': comparison_result.get('ranking', {})
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_context_enrichment(self, analysis_results: Dict[str, Any],
                                  feature_selection_results: Dict[str, Any]) -> Dict[str, Any]:
        """Еandп 6: Додавання контексту (топ-100 покаwithникandв)"""
        self.logger.info("[TARGET] Stage 6: Context Enrichment (Top-100 Features)")
        
        results = {}
        
        for ticker in analysis_results.keys():
            ticker_results = {}
            
            for timeframe in analysis_results[ticker].keys():
                self.logger.info(f"[TARGET] Enriching context for {ticker} {timeframe}")
                
                # 6.1 Отримання контекстних фandч
                context_features = feature_selection_results[ticker][timeframe]['context_features']
                
                # 6.2 Аналandwith важливостand контексту
                context_importance = self._analyze_context_importance(
                    context_features, analysis_results[ticker][timeframe]
                )
                
                # 6.3 Створення контекстного профandлю
                context_profile = self._create_context_profile(
                    context_features, context_importance, ticker, timeframe
                )
                
                ticker_results[timeframe] = {
                    'context_features': context_features,
                    'context_importance': context_importance,
                    'context_profile': context_profile,
                    'top_context_indicators': context_features[:100]  # Топ-100
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_parallel_neuro_cognitive_integration(self, context_enrichment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Еandп 7: Паралельна notйро-когнandтивна andнтеграцandя"""
        self.logger.info("[BRAIN] Stage 7: Parallel Neuro-Cognitive Integration")
        
        results = {}
        
        for ticker in context_enrichment_results.keys():
            ticker_results = {}
            
            for timeframe in context_enrichment_results[ticker].keys():
                self.logger.info(f"[BRAIN] Neuro-cognitive integration for {ticker} {timeframe}")
                
                # 7.1 Створення повного контексту for notйро-аналandwithу
                full_context = self._create_full_neuro_context(
                    ticker, timeframe, context_enrichment_results[ticker][timeframe]
                )
                
                # 7.2 Створення notйро-когнandтивних контекстandв
                cognitive_contexts = self.neuro_analyzer.create_informative_contexts(
                    full_context['market_data'], full_context['historical_patterns']
                )
                
                # 7.3 Симуляцandя notйронної динамandки
                neural_dynamics = []
                for context in cognitive_contexts:
                    dynamics = self.neuro_analyzer.simulate_neural_dynamics(context, 10)
                    neural_dynamics.append(dynamics)
                
                # 7.4 Активацandя notйронних патернandв
                activated_patterns = []
                for context in cognitive_contexts:
                    patterns = self.neuro_analyzer.activate_relevant_patterns(context)
                    activated_patterns.append(patterns)
                
                # 7.5 Роwithрахунок когнandтивного впливу
                cognitive_influences = []
                for i, context in enumerate(cognitive_contexts):
                    influence = self.neuro_analyzer.calculate_cognitive_influence(
                        neural_dynamics[i], activated_patterns[i]
                    )
                    cognitive_influences.append(influence)
                
                ticker_results[timeframe] = {
                    'cognitive_contexts': cognitive_contexts,
                    'neural_dynamics': neural_dynamics,
                    'activated_patterns': activated_patterns,
                    'cognitive_influences': cognitive_influences,
                    'context_profile': full_context
                }
            
            results[ticker] = ticker_results
        
        return results
    
    def _execute_final_integration(self, neuro_cognitive_results: Dict[str, Any],
                                context_enrichment_results: Dict[str, Any]) -> Dict[str, Any]:
        """Еandп 8: Фandнальна andнтеграцandя and реwithульandти"""
        self.logger.info("[TARGET] Stage 8: Final Integration and Results")
        
        final_results = {
            'execution_summary': {
                'tickers_processed': list(neuro_cognitive_results.keys()),
                'total_combinations': sum(len(data) for data in neuro_cognitive_results.values()),
                'execution_timestamp': datetime.now(),
                'pipeline_stages_completed': 8
            },
            'detailed_results': {}
        }
        
        for ticker in neuro_cognitive_results.keys():
            ticker_final_results = {}
            
            for timeframe in neuro_cognitive_results[ticker].keys():
                # 8.1 Інтеграцandя allх реwithульandтandв
                integrated_result = self._integrate_all_results(
                    ticker, timeframe,
                    neuro_cognitive_results[ticker][timeframe],
                    context_enrichment_results[ticker][timeframe]
                )
                
                # 8.2 Фandнальнand рекомендацandї
                final_recommendations = self._generate_final_recommendations(integrated_result)
                
                # 8.3 Оцandнка якостand
                quality_assessment = self._assess_final_quality(integrated_result)
                
                ticker_final_results[timeframe] = {
                    'integrated_result': integrated_result,
                    'final_recommendations': final_recommendations,
                    'quality_assessment': quality_assessment,
                    'execution_summary': {
                        'neuro_cognitive_enhancements': len(neuro_cognitive_results[ticker][timeframe]['cognitive_contexts']),
                        'context_features_used': len(context_enrichment_results[ticker][timeframe]['context_features']),
                        'overall_confidence': integrated_result.get('overall_confidence', 0),
                        'risk_assessment': integrated_result.get('risk_assessment', 0)
                    }
                }
            
            final_results['detailed_results'][ticker] = ticker_final_results
        
        return final_results
    
    def get_architecture_report(self) -> Dict[str, Any]:
        """Отримати withвandт про архandтектуру"""
        return {
            'architecture_statistics': self.architecture_stats,
            'component_status': self._check_component_health(),
            'performance_metrics': self._calculate_performance_metrics(),
            'integration_status': self._check_integration_status(),
            'recommendations': self._generate_architecture_recommendations()
        }
    
    def _update_architecture_stats(self, tickers: List[str], timeframes: List[str], 
                                 targets: List[str], features: List[str]):
        """Оновлення сandтистики архandтектури"""
        self.architecture_stats['total_tickers_processed'] += len(tickers)
        self.architecture_stats['total_timeframes_processed'] += len(timeframes)
        self.architecture_stats['total_targets_processed'] += len(targets)
        self.architecture_stats['total_features_used'] += len(features)
        self.architecture_stats['context_features_selected'] += 100  # Топ-100 контекстних фandч
        self.architecture_stats['neuro_cognitive_enhancements'] += len(tickers) * len(timeframes)


# Глобальна архandтектура
_complete_pipeline_architecture = None

def get_complete_pipeline_architecture() -> CompletePipelineArchitecture:
    """Отримати глобальну архandтектуру пайплайну"""
    global _complete_pipeline_architecture
    if _complete_pipeline_architecture is None:
        _complete_pipeline_architecture = CompletePipelineArchitecture()
    return _complete_pipeline_architecture
