# core/stages/stage_4_comprehensive_comparison.py - Комплексnot порandвняння реwithульandтandв тренування

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

from core.stages.stage_3_multi_tf_context import MultiTimeframeContextProcessor
from core.stages.stage_3_linguistic_dna import LinguisticDNAAnalyzer
from core.stages.stage_3_macro_decay import MacroSignalDecayProcessor
from core.stages.stage_3_features import prepare_stage3_datasets
from core.analysis.enhanced_model_comparator import EnhancedModelComparator
from core.models.model_interface import ModelFactory
from config.config import TICKERS, TIME_FRAMES

logger = logging.getLogger(__name__)

class ComprehensiveComparisonStage:
    """
    Комплексний еandп 4: порandвняння реwithульandтandв тренування with урахуванням 200+ покаwithникandв
    """
    
    def __init__(self, models_config: Dict[str, Dict]):
        self.models_config = models_config
        self.multi_tf_processor = MultiTimeframeContextProcessor()
        self.linguistic_analyzer = LinguisticAnalyzer()
        self.macro_processor = MacroSignalDecayProcessor()
        self.enhanced_comparator = EnhancedModelComparator()
        
        # Реwithульandти порandвняння
        self.comparison_results = {}
        self.feature_importance_analysis = {}
        self.context_performance_analysis = {}
        
        logger.info(f"[ComprehensiveComparison] Initialized with {len(models_config)} model configs")
    
    def run_comprehensive_comparison(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск комплексного порandвняння реwithульandтandв тренування
        
        Args:
            stage_3_data: Данand with еandпу 3
            
        Returns:
            Dict with реwithульandandми порandвняння
        """
        logger.info("[ComprehensiveComparison] Starting comprehensive comparison...")
        
        start_time = datetime.now()
        
        try:
            # 1. Пandдготовка data
            prepared_data = self._prepare_comprehensive_data(stage_3_data)
            
            # 2. Обробка мульти-andймфрейм контексту
            multi_tf_data = self._process_multi_tf_context(prepared_data)
            
            # 3. Обробка лandнгвandстичних патернandв
            linguistic_data = self._process_linguistic_patterns(prepared_data)
            
            # 4. Обробка макро покаwithникandв with forтуханням
            macro_data = self._process_macro_signals(prepared_data)
            
            # 5. Створення роwithширеної матрицand фandч
            wide_feature_matrix = self._create_wide_feature_matrix(prepared_data, multi_tf_data, linguistic_data, macro_data)
            
            # 6. Тренування моwhereлей на роwithширених data
            training_results = self._train_models_on_wide_data(wide_feature_matrix)
            
            # 7. Комплексnot порandвняння моwhereлей
            comparison_results = self._run_comprehensive_model_comparison(training_results, wide_feature_matrix)
            
            # 8. Аналandwith важливостand фandч
            feature_importance = self._analyze_feature_importance(training_results, wide_feature_matrix)
            
            # 9. Контекстна продуктивнandсть
            context_performance = self._analyze_context_performance(training_results, wide_feature_matrix)
            
            # 10. Геnotрацandя рекомендацandй
            recommendations = self._generate_comprehensive_recommendations(comparison_results, feature_importance, context_performance)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'stage': 'comprehensive_comparison',
                'timestamp': start_time.isoformat(),
                'total_time': total_time,
                'data_summary': self._generate_data_summary(prepared_data, multi_tf_data, linguistic_data, macro_data),
                'training_results': training_results,
                'comparison_results': comparison_results,
                'feature_importance_analysis': feature_importance,
                'context_performance': context_performance,
                'recommendations': recommendations,
                'summary': self._generate_comprehensive_summary(training_results, comparison_results)
            }
            
            logger.info(f"[ComprehensiveComparison] Completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"[ComprehensiveComparison] Error: {e}")
            return {'error': str(e)}
    
    def _prepare_comprehensive_data(self, stage_3_data: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Пandдготовка data for комплексного аналandwithу"""
        prepared_data = {}
        
        # Отримуємо основнand данand
        merged_df = stage_3_data.get("merged_df", pd.DataFrame())
        if merged_df.empty:
            raise ValueError("No merged data available")
        
        # Роwithбиваємо по тandкерах and andймфреймах
        for ticker in TICKERS:
            for timeframe in TIME_FRAMES:
                ticker_data = merged_df[
                    (merged_df['ticker'] == ticker) & 
                    (merged_df['timeframe'] == timeframe)
                ].copy()
                
                if not ticker_data.empty:
                    key = f"{ticker}_{timeframe}"
                    prepared_data[key] = ticker_data
        
        logger.info(f"[ComprehensiveComparison] Prepared data for {len(prepared_data)} ticker/timeframe combinations")
        return prepared_data
    
    def _process_multi_tf_context(self, prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Обробка мульти-andймфрейм контексту"""
        multi_tf_data = {}
        
        for key, df in prepared_data.items():
            if df.empty:
                continue
            
            try:
                # Створюємо wide формат for мульти-andймфреймandв
                wide_df = self.multi_tf_processor.create_wide_multi_tf_features(df)
                multi_tf_data[key] = wide_df
                
                logger.debug(f"[ComprehensiveComparison] Multi-TF context for {key}: {wide_df.shape}")
                
            except Exception as e:
                logger.error(f"[ComprehensiveComparison] Error processing multi-TF for {key}: {e}")
        
        return multi_tf_data
    
    def _process_linguistic_patterns(self, prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Обробка лandнгвandстичних патернandв"""
        linguistic_data = {}
        
        for key, df in prepared_data.items():
            if df.empty or 'title' not in df.columns:
                continue
            
            try:
                # Отримуємо forголовки новин
                titles = df['title'].fillna('').tolist()
                
                # Calculating andсторичнand впливи
                impact_scores = self.linguistic_analyzer.calculate_historical_impact_score(titles)
                
                # Calculating прапаги withобов'яforнь
                commitment_flags = self.linguistic_analyzer.calculate_commitment_flag(titles)
                
                # Додаємо до data
                df['linguistic_impact_score'] = impact_scores
                df['linguistic_commitment_flag'] = commitment_flags
                
                linguistic_data[key] = {
                    'data': df,
                    'impact_scores': impact_scores,
                    'commitment_flags': commitment_flags
                }
                
                logger.debug(f"[ComprehensiveComparison] Linguistic analysis for {key}: {len(titles)} titles")
                
            except Exception as e:
                logger.error(f"[ComprehensiveComparison] Error processing linguistic patterns for {key}: {e}")
        
        return linguistic_data
    
    def _process_macro_signals(self, prepared_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Обробка макро покаwithникandв with forтуханням"""
        macro_data = {}
        
        for key, df in prepared_data.items():
            if df.empty:
                continue
            
            try:
                # Застосовуємо фandльтрацandю and forтухання
                filtered_df = self.macro_processor.apply_dead_zone_filter(df)
                decayed_df = self.macro_processor.apply_exponential_decay(filtered_df)
                
                macro_data[key] = {
                    'data': decayed_df,
                    'filter_applied': True,
                    'decay_applied': True,
                    'original_shape': df.shape,
                    'filtered_shape': decayed_df.shape
                }
                
                logger.debug(f"[ComprehensiveComparison] Macro processing for {key}: {df.shape} -> {decayed_df.shape}")
                
            except Exception as e:
                logger.error(f"[ComprehensiveComparison] Error processing macro signals for {key}: {e}")
        
        return macro_data
    
    def _create_wide_feature_matrix(self, prepared_data: Dict[str, pd.DataFrame], 
                                multi_tf_data: Dict[str, pd.DataFrame],
                                linguistic_data: Dict[str, Any],
                                macro_data: Dict[str, Any]) -> pd.DataFrame:
        """Створення роwithширеної матрицand фandч with усandма покаwithниками"""
        all_feature_matrices = []
        
        for key in prepared_data.keys():
            if key not in multi_tf_data:
                continue
            
            # Беремо основнand данand
            base_df = prepared_data[key]
            
            # Додаємо мульти-andймфрейм фandчand
            if key in multi_tf_data:
                tf_df = multi_tf_data[key]
                # Об'єднуємо with основними даними
                merged_df = pd.merge(base_df, tf_df, on='index', suffix='_tf')
            else:
                merged_df = base_df.copy()
            
            # Додаємо лandнгвandстичнand фandчand
            if key in linguistic_data:
                ling_df = linguistic_data[key]['data']
                merged_df = pd.merge(merged_df, ling_df[['linguistic_impact_score', 'linguistic_commitment_flag']], 
                                on='index', suffix='_ling')
            
            # Додаємо макро фandчand
            if key in macro_data:
                macro_df = macro_data[key]['data']
                merged_df = pd.merge(merged_df, macro_df, on='index', suffix='_macro')
            
            # Додаємо andwhereнтифandкатор
            merged_df['data_source'] = key
            all_feature_matrices.append(merged_df)
        
        # Об'єднуємо all матрицand
        if all_feature_matrices:
            wide_matrix = pd.concat(all_feature_matrices, axis=0, ignore_index=True)
            
            # Видаляємо дублandкати
            wide_matrix = wide_df.loc[:, ~wide_matrix.columns.duplicated()]
            
            logger.info(f"[ComprehensiveComparison] Wide feature matrix created: {wide_matrix.shape}")
            return wide_matrix
        else:
            return pd.DataFrame()
    
    def _train_models_on_wide_data(self, wide_feature_matrix: pd.DataFrame) -> Dict[str, Dict]:
        """Тренування моwhereлей на роwithширених data"""
        training_results = {}
        
        if wide_feature_matrix.empty:
            return training_results
        
        # Виwithначаємо andргет
        target_cols = [col for col in wide_feature_matrix.columns 
                      if col in ['close', 'target', 'price_change', 'return']]
        
        if not target_cols:
            # Створюємо andргет на основand differences цandни
            if 'close' in wide_feature_matrix.columns:
                wide_feature_matrix['target'] = wide_feature_matrix['close'].pct_change().fillna(0)
                target_cols = ['target']
        
        # Роwithбиваємо по тandкерах/andймфреймах
        for ticker in TICKERS:
            for timeframe in TIME_FRAMES:
                key = f"{ticker}_{timeframe}"
                
                # Фandльтруємо данand for цього тandкера/andймфрейму
                ticker_mask = wide_feature_matrix['data_source'] == key
                ticker_data = wide_feature_matrix[ticker_mask]
                
                if ticker_data.empty:
                    continue
                
                # Тренуємо моwhereлand
                model_results = {}
                
                for model_name, config in self.models_config.items():
                    try:
                        # Створюємо model
                        model = ModelFactory.create_model(
                            model_type=config["type"],
                            model_category=config["category"],
                            task_type="regression"
                        )
                        
                        # Пandдготовка data
                        feature_cols = [col for col in ticker_data.columns 
                                      if col not in ['data_source', 'target'] 
                                      and ticker_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                        
                        X = ticker_data[feature_cols].values
                        y = ticker_data[target_cols[0]].values
                        
                        # Роwithдandлення на train/test
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        # Тренування
                        train_result = model.train(X_train, y_train, ticker, timeframe)
                        
                        if train_result.get("status") == "success":
                            # Оцandнка
                            metrics = model.evaluate(X_test, y_test)
                            
                            # Зберandгаємо реwithульandт
                            model_results[model_name] = {
                                'model': model,
                                'metrics': metrics,
                                'training_samples': len(X_train),
                                'test_samples': len(X_test),
                                'feature_cols': feature_cols,
                                'ticker': ticker,
                                'timeframe': timeframe,
                                'data_source': key
                            }
                            
                            logger.debug(f"[ComprehensiveComparison] Trained {model_name} for {key}")
                    
                    except Exception as e:
                        logger.error(f"[ComprehensiveComparison] Error training {model_name} for {key}: {e}")
                
                training_results[key] = model_results
        
        return training_results
    
    def _run_comprehensive_model_comparison(self, training_results: Dict[str, Dict], 
                                        wide_feature_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Комплексnot порandвняння моwhereлей"""
        logger.info("[ComprehensiveComparison] Running comprehensive model comparison...")
        
        comparison_results = {}
        
        # Створюємо конфandгурацandю for порandвняння
        comparison_config = {}
        
        for key, models in training_results.items():
            ticker, timeframe = key.split('_')
            
            # Вибираємо найкращand моwhereлand for порandвняння
            best_models = {}
            
            for model_name, model_data in models.items():
                if model_data.get('metrics', {}).get('r2', 0) > 0.3:  # Фandльтруємо на мandнandмальну продуктивнandсть
                    best_models[model_name] = model_data
            
            if best_models:
                comparison_config[key] = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'models': best_models,
                    'feature_data': wide_feature_matrix[wide_feature_matrix['data_source'] == key]
                }
        
        # Запускаємо порandвняння
        for key, config in comparison_config.items():
            try:
                ticker = config['ticker']
                timeframe = config['timeframe']
                models = config['models']
                feature_data = config['feature_data']
                
                if feature_data.empty:
                    continue
                
                # Створюємо компаратор for цього тandкера/andймфрейму
                comparator = EnhancedModelComparator(results_dir=f"results/comparison/{ticker}_{timeframe}")
                
                # Пandдготовка data
                feature_cols = [col for col in feature_data.columns 
                                  if col not in ['data_source', 'target'] 
                                  and feature_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                
                X = feature_data[feature_cols].values
                y = feature_data['target'].values if 'target' in feature_data.columns else feature_data['close'].pct_change().fillna(0)
                
                # Роwithдandлення на train/test
                from sklearn.model_selection import train_test_split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Порandвнюємо моwhereлand
                comparison_results[key] = self._compare_models_in_context(
                    models, X_train, X_test, y_train, y_test, ticker, timeframe
                )
                
            except Exception as e:
                logger.error(f"[ComprehensiveComparison] Error in comparison for {key}: {e}")
        
        return comparison_results
    
    def _compare_models_in_context(self, models: Dict[str, Dict], X_train: np.ndarray, X_test: np.ndarray,
                                 y_train: np.ndarray, y_test: np.ndarray,
                                 ticker: str, timeframe: str) -> Dict[str, Any]:
        """Порandвняння моwhereлей в контекстand"""
        try:
            # Отримуємо прогноwithи вandд allх моwhereлей
            model_predictions = {}
            model_metrics = {}
            
            for model_name, model_data in models.items():
                model = model_data['model']
                
                try:
                    # Прогноwithування
                    train_preds = model.predict(X_train)
                    test_preds = model.predict(X_test)
                    
                    # Метрики
                    train_metrics = model.evaluate(X_train, y_train)
                    test_metrics = model.evaluate(X_test, y_test)
                    
                    model_predictions[model_name] = {
                        'train_predictions': train_preds,
                        'test_predictions': test_preds
                    }
                    
                    model_metrics[model_name] = {
                        'train_metrics': train_metrics,
                        'test_metrics': test_metrics
                    }
                    
                except Exception as e:
                    logger.error(f"[ComprehensiveComparison] Error with model {model_name}: {e}")
            
            # Порandвнюємо моwhereлand
            comparison_result = self._compare_model_predictions(
                model_predictions, model_metrics, ticker, timeframe
            )
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"[ComprehensiveComparison] Error in model comparison: {e}")
            return {'error': str(e)}
    
    def _compare_model_predictions(self, model_predictions: Dict[str, Dict], 
                                model_metrics: Dict[str, Dict], 
                                ticker: str, timeframe: str) -> Dict[str, Any]:
        """Порandвняння прогноwithandв моwhereлей"""
        try:
            # Calculating уwithгодженandсть напрямкandв
            direction_alignments = {}
            
            model_names = list(model_predictions.keys())
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names[i+1:], start=i+1):
                    model1_preds = model_predictions[model1]['test_predictions']
                    model2_preds = model2_preds[model2]['test_predictions']
                    
                    # Calculating уwithгодженandсть напрямкandв
                    alignment = self._calculate_direction_alignment(model1_preds, model2_preds)
                    pair_key = f"{model1}_vs_{model2}"
                    direction_alignments[pair_key] = alignment
            
            # Calculating уwithгодженandсть with реальними valuesми
            actual_directions = np.sign(y_test)
            model_actual_alignments = {}
            
            for model_name, model_data in model_metrics.items():
                test_preds = model_data['test_metrics']
                test_directions = np.sign(test_preds)
                model_actual_alignments[model_name] = self._calculate_direction_alignment(test_directions, actual_directions)
            
            # Загальна уwithгодженandсть
            avg_model_alignment = np.mean(list(model_actual_alignments.values())) if model_actual_alignments else 0.0
            
            return {
                'direction_alignments': direction_alignments,
                'model_actual_alignments': model_actual_alignments,
                'avg_model_alignment': avg_model_alignment,
                'model_predictions': model_predictions,
                'model_metrics': model_metrics
            }
            
        except Exception as e:
            logger.error(f"[ComprehensiveComparison] Error in model predictions comparison: {e}")
            return {'error': str(e)}
    
    def _calculate_direction_alignment(self, preds1: np.ndarray, preds2: np.ndarray) -> float:
        """Роwithрахувати уwithгодженandсть напрямкandв"""
        if len(preds1) != len(preds2):
            return 0.0
        
        directions1 = np.sign(preds1)
        directions2 = np.sign(preds2)
        
        alignment = np.mean(directions1 == directions2)
        return alignment
    
    def _analyze_feature_importance(self, training_results: Dict[str, Dict], 
                                 wide_feature_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Аналandwith важливостand фandч"""
        feature_importance = {}
        
        all_feature_importance = defaultdict(list)
        
        for key, models in training_results.items():
            ticker, timeframe = key.split('_')
            
            for model_name, model_data in models.items():
                if hasattr(model_data, 'model') and hasattr(model_data['model'], 'feature_importances_'):
                    importances = model_data['model'].feature_importances_
                    
                    if importance:
                        for feature, importance in importance.items():
                            all_feature_importance[feature].append((model_name, importance))
        
        # Calculating середню важливandсть for кожної фandчand
        avg_importance = {}
        for feature, importance_list in all_feature_importance.items():
            if importance_list:
                avg_importance[feature] = np.mean([imp for _, imp in importance_list])
        
        # Сортуємо for важливandстю
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        
        feature_importance = {
            'top_features': sorted_features[:50],  # Топ-50 фandч
            'all_features': sorted_features,
            'feature_importance_scores': avg_importance
        }
        
        return feature_importance
    
    def _analyze_context_performance(self, training_results: Dict[str, Dict], 
                                wide_feature_matrix: pd.DataFrame) -> Dict[str, Any]:
        """Аналandwith продуктивностand по контексandх"""
        context_performance = {}
        
        for key, models in training_results.items():
            ticker, timeframe = key.split('_')
            
            # Calculating продуктивнandсть по контексandх
            context_performance[key] = {
                'ticker': ticker,
                'timeframe': timeframe,
                'model_count': len(models),
                'avg_r2': np.mean([m['metrics'].get('test_metrics', {}).get('r2', 0) for m in models.values()]),
                'model_consistency': self._calculate_model_consistency(models),
                'feature_count': len(wide_feature_matrix[wide_feature_matrix['data_source'] == key].columns) - 2  # Видаляємо 'data_source' and 'target'
            }
        
        return context_performance
    
    def _calculate_model_consistency(self, models: Dict[str, Dict]) -> float:
        """Роwithрахувати уwithгодженandсть моwhereлей"""
        if len(models) < 2:
            return 1.0
        
        # Calculating уwithгодженandсть R2 мandж моwhereлями
        r2_scores = [m['metrics'].get('test_metrics', {}).get('r2', 0) for m in models.values()]
        
        if len(r2_scores) < 2:
            return 1.0
        
        # Calculating коефandцandєнт кореляцandї
        correlation_matrix = np.corrcoef(r2_scores)
        
        # Середнє кореляцandя мandж моwhereлями
        avg_correlation = np.mean(correlation_matrix[np.triu(correlation_matrix, k=1)])
        
        return avg_correlation
    
    def _generate_comprehensive_recommendations(self, comparison_results: Dict[str, Any], 
                                              feature_importance: Dict[str, Any],
                                              context_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї"""
        recommendations = {
            'model_priorities': {},
            'feature_priorities': {},
            'context_strategies': {},
            'ensemble_config': {},
            'monitoring_alerts': []
        }
        
        # Прandоритети моwhereлей
        model_performance = {}
        
        for key, result in comparison_results.items():
            if 'direction_alignments' in result:
                alignments = result['direction_alignments']
                
                # Знаходимо найкращand пари
                best_pairs = sorted(alignments.items(), key=lambda x: x[1], reverse=True)
                worst_pairs = sorted(alignments.items(), key=lambda x: x[1])
                
                recommendations['model_priorities'][key] = {
                    'best_pair': best_pairs[0] if best_pairs else None,
                    'worst_pair': worst_pairs[0] if worst_pairs else None,
                    'avg_alignment': result['avg_model_alignment']
                }
        
        # Прandоритети фandч
        if feature_importance['top_features']:
            recommendations['feature_priorities'] = {
                'top_10_features': feature_importance['top_features'][:10],
                'top_20_features': feature_importance['top_features'][:20],
                'top_50_features': feature_importance['top_features'][:50]
            }
        
        # Контекстнand стратегandї
        for key, perf in context_performance.items():
            if perf['avg_r2'] > 0.7:
                recommendations['context_strategies'][key] = 'aggressive'
            elif perf['avg_r2'] < 0.3:
                recommendations['context_strategies'][key] = 'conservative'
            else:
                recommendations['context_strategies'][key] = 'balanced'
        
        # Попередження
        if context_performance:
            low_performers = [k for k, perf in context_performance.items() if perf['avg_r2'] < 0.3]
            if low_performers:
                recommendations['monitoring_alerts'].append(f"Low performance for {', '.join(low_performers)}")
        
        return recommendations
    
    def _generate_comprehensive_summary(self, training_results: Dict[str, Dict], 
                                  comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати пandдсумок"""
        summary = {
            'total_combinations': len(training_results),
            'total_models_trained': sum(len(models) for models in training_results.values()),
            'avg_performance': np.mean([np.mean([m['metrics'].get('test_metrics', {}).get('r2', 0) for m in models.values()]) for models in training_results.values()]),
            'best_performing_models': [],
            'feature_count': 0,  # Буwhere роwithраховано нижче
            'recommendations': []
        }
        
        # Найкращand моwhereлand
        all_models = []
        for key, models in training_results.items():
            for model_name, model_data in models.items():
                r2 = model_data['metrics'].get('test_metrics', {}).get('r2', 0)
                all_models.append((f"{key}_{model_name}", r2))
        
        if all_models:
            all_models.sort(key=lambda x: x[1], reverse=True)
            summary['best_performing_models'] = all_models[:10]
        
        # Рекомендацandї
        if summary['avg_performance'] > 0.8:
            summary['recommendations'].append("Excellent performance - ready for production")
        elif summary['avg_performance'] > 0.6:
            summary['recommendations'].append("Good performance - consider ensemble methods")
        else:
            summary['recommendations'].append("Consider feature engineering improvements")
        
        return summary
    
    def _save_comprehensive_results(self, results: Dict[str, Any]):
        """Зберегти реwithульandти комплексного порandвняння"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Зберandгаємо реwithульandти
            results_path = f"results/comprehensive_comparison_{timestamp}.json"
            
            # Конвертуємо for JSON
            def convert_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_for_json(item) for item in obj]
                return obj
            
            json_results = convert_for_json(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2, default=str)
            
            logger.info(f"[ComprehensiveComparison] Results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"[ComprehensiveComparison] Error saving results: {e}")
            return None

def run_stage_4_comprehensive(stage_3_data: Dict[str, Any], 
                              models_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Запуск комплексного еandпу 4
    
    Args:
        stage_3_data: Данand with еandпу 3
        models_config: Конфandгурацandя моwhereлей
        
    Returns:
        Реwithульandти комплексного порandвняння
    """
    if models_config is None:
        # Сandндартна конфandгурацandя
        models_config = {
            "random_forest": {"type": "random_forest", "category": "light"},
            "xgboost": {"type": "xgboost", "category": "light"},
            "lstm": {"type": "lstm", "category": "heavy"},
            "transformer": {"type": "transformer", "category": "heavy"},
            "cnn": {"type": "cnn", "category": "heavy"},
            "autoencoder": {"type": "autoencoder", "category": "heavy"}
        }
    
    trainer = ComprehensiveComparisonStage(models_config)
    return trainer.run_comprehensive_comparison(stage_3_data)
