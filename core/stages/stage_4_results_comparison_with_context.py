# core/stages/stage_4_results_comparison_with_context.py - Порandвняння реwithульandтandв тренування with економandчним контекстом

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

from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrich
from core.stages.stage_3_features import prepare_stage3_datasets
from core.analysis.enhanced_model_comparator import EnhancedModelComparator
from core.models.model_interface import ModelFactory
from config.config import TICKERS, TIME_FRAMES

logger = logging.getLogger(__name__)

class ResultsComparisonWithContext:
    """
    Порandвняння реwithульandтandв тренування with економandчним контекстом
    """
    
    def __init__(self, models_config: Dict[str, Dict]):
        self.models_config = models_config
        self.enhanced_comparator = EnhancedModelComparator()
        
        # Економandчний контекст
        self.economic_context = {
            'macro_indicators': {},
            'market_conditions': {},
            'news_sentiment': {},
            'volatility_regimes': {}
        }
        
        # Реwithульandти порandвняння
        self.comparison_results = {}
        self.context_analysis = {}
        
        logger.info(f"[ResultsComparison] Initialized with {len(models_config)} model configs")
    
    def run_results_comparison_with_context(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск порandвняння реwithульandтandв with економandчним контекстом
        
        Args:
            stage_3_data: Данand with еandпу 3
            
        Returns:
            Dict with реwithульandandми порandвняння and контексту
        """
        logger.info("[ResultsComparison] Starting results comparison with context...")
        
        start_time = datetime.now()
        
        try:
            # 1. Тренування моwhereлей на основних data
            training_results = self._train_models_on_base_data(stage_3_data)
            
            # 2. Порandвняння моwhereлей
            comparison_results = self._compare_models(training_results, stage_3_data)
            
            # 3. Збandр економandчного контексту
            economic_context = self._collect_economic_context(stage_3_data)
            
            # 4. Аналandwith впливу контексту на реwithульandти
            context_analysis = self._analyze_context_impact(comparison_results, economic_context)
            
            # 5. Геnotрацandя withвandтandв with контекстом
            contextual_reports = self._generate_contextual_reports(comparison_results, economic_context, context_analysis)
            
            # 6. Рекомендацandї with урахуванням контексту
            contextual_recommendations = self._generate_contextual_recommendations(comparison_results, economic_context, context_analysis)
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            results = {
                'stage': 'results_comparison_with_context',
                'timestamp': start_time.isoformat(),
                'total_time': total_time,
                'training_results': training_results,
                'comparison_results': comparison_results,
                'economic_context': economic_context,
                'context_analysis': context_analysis,
                'contextual_reports': contextual_reports,
                'contextual_recommendations': contextual_recommendations,
                'summary': self._generate_contextual_summary(training_results, comparison_results, economic_context)
            }
            
            logger.info(f"[ResultsComparison] Completed in {total_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"[ResultsComparison] Error: {e}")
            return {'error': str(e)}
    
    def _train_models_on_base_data(self, stage_3_data: Dict[str, Any]) -> Dict[str, Dict]:
        """Тренування моwhereлей на основних data (беwith економandчного контексту)"""
        training_results = {}
        
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
                
                if ticker_data.empty:
                    continue
                
                key = f"{ticker}_{timeframe}"
                
                # Створюємо andргет
                if 'close' in ticker_data.columns:
                    ticker_data['target'] = ticker_data['close'].pct_change().shift(-1)
                    ticker_data = ticker_data.dropna()
                
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
                                      if col not in ['date', 'ticker', 'timeframe', 'target'] 
                                      and ticker_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                        
                        X = ticker_data[feature_cols].values
                        y = ticker_data['target'].values
                        
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
                                'timeframe': timeframe
                            }
                    
                    except Exception as e:
                        logger.error(f"[ResultsComparison] Error training {model_name} for {key}: {e}")
                
                training_results[key] = model_results
        
        return training_results
    
    def _compare_models(self, training_results: Dict[str, Dict], stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Порandвняння моwhereлей"""
        logger.info("[ResultsComparison] Running model comparison...")
        
        comparison_results = {}
        
        for key, models in training_results.items():
            ticker, timeframe = key.split('_')
            
            # Вибираємо найкращand моwhereлand for порandвняння
            best_models = {}
            
            for model_name, model_data in models.items():
                if model_data.get('metrics', {}).get('r2', 0) > 0.3:  # Фandльтруємо на мandнandмальну продуктивнandсть
                    best_models[model_name] = model_data
            
            if best_models:
                # Створюємо компаратор for цього тandкера/andймфрейму
                comparator = EnhancedModelComparator(results_dir=f"results/comparison/{ticker}_{timeframe}")
                
                # Отримуємо данand for порandвняння
                merged_df = stage_3_data.get("merged_df", pd.DataFrame())
                ticker_data = merged_df[
                    (merged_df['ticker'] == ticker) & 
                    (merged_df['timeframe'] == timeframe)
                ].copy()
                
                if not ticker_data.empty:
                    # Пandдготовка data
                    feature_cols = [col for col in ticker_data.columns 
                                  if col not in ['date', 'ticker', 'timeframe', 'target'] 
                                  and ticker_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]
                    
                    X = ticker_data[feature_cols].values
                    y = ticker_data['target'].values if 'target' in ticker_data.columns else ticker_data['close'].pct_change().fillna(0)
                    
                    # Роwithдandлення на train/test
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Порandвнюємо моwhereлand
                    comparison_results[key] = self._compare_models_in_context(
                        best_models, X_train, X_test, y_train, y_test, ticker, timeframe
                    )
        
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
                    logger.error(f"[ResultsComparison] Error with model {model_name}: {e}")
            
            # Порandвнюємо моwhereлand
            comparison_result = {
                'model_predictions': model_predictions,
                'model_metrics': model_metrics,
                'direction_alignments': self._calculate_direction_alignments(model_predictions),
                'performance_summary': self._calculate_performance_summary(model_metrics)
            }
            
            return comparison_result
            
        except Exception as e:
            logger.error(f"[ResultsComparison] Error in model comparison: {e}")
            return {'error': str(e)}
    
    def _calculate_direction_alignments(self, model_predictions: Dict[str, Dict]) -> Dict[str, float]:
        """Роwithрахувати уwithгодженandсть напрямкandв мandж моwhereлями"""
        alignments = {}
        
        model_names = list(model_predictions.keys())
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], start=i+1):
                model1_preds = model_predictions[model1]['test_predictions']
                model2_preds = model_predictions[model2]['test_predictions']
                
                # Calculating уwithгодженandсть напрямкandв
                directions1 = np.sign(model1_preds)
                directions2 = np.sign(model2_preds)
                
                alignment = np.mean(directions1 == directions2)
                pair_key = f"{model1}_vs_{model2}"
                alignments[pair_key] = alignment
        
        return alignments
    
    def _calculate_performance_summary(self, model_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """Роwithрахувати пandдсумок продуктивностand"""
        summary = {}
        
        for model_name, metrics in model_metrics.items():
            test_metrics = metrics['test_metrics']
            summary[model_name] = {
                'r2': test_metrics.get('r2', 0),
                'mse': test_metrics.get('mse', float('inf')),
                'mae': test_metrics.get('mae', float('inf'))
            }
        
        return summary
    
    def _collect_economic_context(self, stage_3_data: Dict[str, Any]) -> Dict[str, Any]:
        """Зandбрати економandчний контекст"""
        logger.info("[ResultsComparison] Collecting economic context...")
        
        economic_context = {
            'macro_indicators': {},
            'market_conditions': {},
            'news_sentiment': {},
            'volatility_regimes': {}
        }
        
        merged_df = stage_3_data.get("merged_df", pd.DataFrame())
        
        if not merged_df.empty:
            # Макро andндикатори
            macro_cols = [col for col in merged_df.columns if any(indicator in col.lower() 
                        for indicator in ['gdp', 'cpi', 'unemployment', 'fed', 'pmi', 'bond', 'yield'])]
            
            for col in macro_cols:
                if merged_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    economic_context['macro_indicators'][col] = {
                        'mean': merged_df[col].mean(),
                        'std': merged_df[col].std(),
                        'min': merged_df[col].min(),
                        'max': merged_df[col].max(),
                        'trend': merged_df[col].pct_change().mean() if len(merged_df) > 1 else 0
                    }
            
            # Ринковand умови
            if 'close' in merged_df.columns:
                economic_context['market_conditions'] = {
                    'avg_volatility': merged_df['close'].pct_change().std(),
                    'avg_return': merged_df['close'].pct_change().mean(),
                    'trend': merged_df['close'].pct_change().mean(),
                    'volatility_regime': self._classify_volatility_regime(merged_df['close'].pct_change().std())
                }
            
            # Новинний сентимент
            sentiment_cols = [col for col in merged_df.columns if 'sentiment' in col.lower()]
            
            for col in sentiment_cols:
                if merged_df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    economic_context['news_sentiment'][col] = {
                        'mean': merged_df[col].mean(),
                        'std': merged_df[col].std(),
                        'trend': merged_df[col].pct_change().mean() if len(merged_df) > 1 else 0
                    }
            
            # Режими волатильностand
            if 'ATR' in merged_df.columns:
                atr_ratio = merged_df['ATR'] / merged_df['close']
                economic_context['volatility_regimes'] = {
                    'avg_atr_ratio': atr_ratio.mean(),
                    'volatility_regime': self._classify_volatility_regime(atr_ratio.mean())
                }
        
        return economic_context
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Класифandкувати режим волатильностand"""
        if volatility < 0.01:
            return "low"
        elif volatility < 0.02:
            return "normal"
        else:
            return "high"
    
    def _analyze_context_impact(self, comparison_results: Dict[str, Any], economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Аналandwith впливу контексту на реwithульandти"""
        logger.info("[ResultsComparison] Analyzing context impact...")
        
        context_analysis = {
            'macro_impact': {},
            'market_condition_impact': {},
            'sentiment_impact': {},
            'volatility_impact': {},
            'overall_context_score': 0.0
        }
        
        # Аналandwith впливу макро andндикаторandв
        if economic_context['macro_indicators']:
            macro_impact = 0.0
            for indicator, stats in economic_context['macro_indicators'].items():
                # Вплив forлежить вandд тренду and волатильностand
                trend_impact = abs(stats['trend']) * 10  # Нормалandwithуємо
                volatility_impact = stats['std'] / stats['mean'] if stats['mean'] != 0 else 0
                
                macro_impact += (trend_impact + volatility_impact) / 2
            
            context_analysis['macro_impact'] = {
                'score': macro_impact / len(economic_context['macro_indicators']),
                'indicators_count': len(economic_context['macro_indicators']),
                'avg_trend': np.mean([stats['trend'] for stats in economic_context['macro_indicators'].values()])
            }
        
        # Аналandwith впливу ринкових умов
        if economic_context['market_conditions']:
            market_impact = abs(economic_context['market_conditions']['trend']) * 10
            volatility_impact = economic_context['market_conditions']['avg_volatility'] * 100
            
            context_analysis['market_condition_impact'] = {
                'score': (market_impact + volatility_impact) / 2,
                'trend': economic_context['market_conditions']['trend'],
                'volatility': economic_context['market_conditions']['avg_volatility'],
                'regime': economic_context['market_conditions']['volatility_regime']
            }
        
        # Аналandwith впливу сентименту
        if economic_context['news_sentiment']:
            sentiment_impact = 0.0
            for sentiment, stats in economic_context['news_sentiment'].items():
                sentiment_impact += abs(stats['trend']) * 10
            
            context_analysis['sentiment_impact'] = {
                'score': sentiment_impact / len(economic_context['news_sentiment']),
                'avg_sentiment': np.mean([stats['mean'] for stats in economic_context['news_sentiment'].values()]),
                'avg_trend': np.mean([stats['trend'] for stats in economic_context['news_sentiment'].values()])
            }
        
        # Загальний контекстний скор
        all_scores = []
        for impact_type in ['macro_impact', 'market_condition_impact', 'sentiment_impact']:
            if impact_type in context_analysis and 'score' in context_analysis[impact_type]:
                all_scores.append(context_analysis[impact_type]['score'])
        
        if all_scores:
            context_analysis['overall_context_score'] = np.mean(all_scores)
        
        return context_analysis
    
    def _generate_contextual_reports(self, comparison_results: Dict[str, Any], 
                                  economic_context: Dict[str, Any],
                                  context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати withвandти with контекстом"""
        contextual_reports = {
            'performance_by_context': {},
            'model_behavior_by_context': {},
            'contextual_insights': {}
        }
        
        # Продуктивнandсть по контексandх
        for key, result in comparison_results.items():
            if 'performance_summary' in result:
                ticker, timeframe = key.split('_')
                
                contextual_reports['performance_by_context'][key] = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'best_model': max(result['performance_summary'].items(), key=lambda x: x[1]['r2'])[0],
                    'avg_r2': np.mean([metrics['r2'] for metrics in result['performance_summary'].values()]),
                    'context_score': context_analysis['overall_context_score'],
                    'economic_conditions': self._summarize_economic_conditions(economic_context)
                }
        
        # Поведandнка моwhereлей по контексandх
        for key, result in comparison_results.items():
            if 'direction_alignments' in result:
                ticker, timeframe = key.split('_')
                
                contextual_reports['model_behavior_by_context'][key] = {
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'avg_alignment': np.mean(list(result['direction_alignments'].values())),
                    'best_alignment': max(result['direction_alignments'].items(), key=lambda x: x[1])[0],
                    'worst_alignment': min(result['direction_alignments'].items(), key=lambda x: x[1])[0],
                    'context_influence': context_analysis['overall_context_score']
                }
        
        # Контекстнand andнсайти
        contextual_reports['contextual_insights'] = {
            'high_volatility_periods': self._identify_high_volatility_periods(economic_context),
            'strong_macro_trends': self._identify_strong_macro_trends(economic_context),
            'sentiment_anomalies': self._identify_sentiment_anomalies(economic_context),
            'model_consistency': self._analyze_model_consistency(comparison_results)
        }
        
        return contextual_reports
    
    def _summarize_economic_conditions(self, economic_context: Dict[str, Any]) -> Dict[str, str]:
        """Пandдсумувати економandчнand умови"""
        summary = {}
        
        # Макро умови
        if economic_context['macro_indicators']:
            avg_trend = np.mean([stats['trend'] for stats in economic_context['macro_indicators'].values()])
            if avg_trend > 0.01:
                summary['macro_trend'] = 'expansion'
            elif avg_trend < -0.01:
                summary['macro_trend'] = 'contraction'
            else:
                summary['macro_trend'] = 'stable'
        
        # Ринковand умови
        if economic_context['market_conditions']:
            summary['market_regime'] = economic_context['market_conditions']['volatility_regime']
            summary['market_trend'] = 'bullish' if economic_context['market_conditions']['trend'] > 0 else 'bearish'
        
        # Сентимент
        if economic_context['news_sentiment']:
            avg_sentiment = np.mean([stats['mean'] for stats in economic_context['news_sentiment'].values()])
            if avg_sentiment > 0.1:
                summary['sentiment'] = 'positive'
            elif avg_sentiment < -0.1:
                summary['sentiment'] = 'negative'
            else:
                summary['sentiment'] = 'neutral'
        
        return summary
    
    def _identify_high_volatility_periods(self, economic_context: Dict[str, Any]) -> List[str]:
        """Іwhereнтифandкувати periodи високої волатильностand"""
        periods = []
        
        if economic_context['market_conditions']:
            if economic_context['market_conditions']['volatility_regime'] == 'high':
                periods.append("Current market conditions show high volatility")
        
        if economic_context['volatility_regimes']:
            if economic_context['volatility_regimes']['volatility_regime'] == 'high':
                periods.append("ATR indicates high volatility regime")
        
        return periods
    
    def _identify_strong_macro_trends(self, economic_context: Dict[str, Any]) -> List[str]:
        """Іwhereнтифandкувати сильнand макро тренди"""
        trends = []
        
        if economic_context['macro_indicators']:
            for indicator, stats in economic_context['macro_indicators'].items():
                if abs(stats['trend']) > 0.02:
                    direction = 'upward' if stats['trend'] > 0 else 'downward'
                    trends.append(f"{indicator}: strong {direction} trend")
        
        return trends
    
    def _identify_sentiment_anomalies(self, economic_context: Dict[str, Any]) -> List[str]:
        """Іwhereнтифandкувати аномалandї сентименту"""
        anomalies = []
        
        if economic_context['news_sentiment']:
            for sentiment, stats in economic_context['news_sentiment'].items():
                if abs(stats['trend']) > 0.05:
                    direction = 'improving' if stats['trend'] > 0 else 'deteriorating'
                    anomalies.append(f"{sentiment}: rapidly {direction}")
        
        return anomalies
    
    def _analyze_model_consistency(self, comparison_results: Dict[str, Any]) -> Dict[str, float]:
        """Аналandwithувати уwithгодженandсть моwhereлей"""
        consistency = {}
        
        for key, result in comparison_results.items():
            if 'direction_alignments' in result:
                alignments = list(result['direction_alignments'].values())
                consistency[key] = np.mean(alignments) if alignments else 0.0
        
        return consistency
    
    def _generate_contextual_recommendations(self, comparison_results: Dict[str, Any],
                                           economic_context: Dict[str, Any],
                                           context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї with урахуванням контексту"""
        recommendations = {
            'model_selection': {},
            'risk_management': {},
            'trading_strategies': {},
            'monitoring_alerts': []
        }
        
        # Рекомендацandї по вибору моwhereлей
        for key, result in comparison_results.items():
            if 'performance_summary' in result:
                ticker, timeframe = key.split('_')
                
                best_model = max(result['performance_summary'].items(), key=lambda x: x[1]['r2'])
                
                # Коригуємо рекомендацandї на основand контексту
                context_score = context_analysis['overall_context_score']
                
                if context_score > 0.7:  # Високий контекстний вплив
                    recommendations['model_selection'][key] = {
                        'primary_model': best_model[0],
                        'confidence': best_model[1]['r2'] * (1 - context_score * 0.2),  # Знижуємо впевnotнandсть
                        'context_adjustment': 'high_context_impact'
                    }
                elif context_score < 0.3:  # Ниwithький контекстний вплив
                    recommendations['model_selection'][key] = {
                        'primary_model': best_model[0],
                        'confidence': best_model[1]['r2'],
                        'context_adjustment': 'low_context_impact'
                    }
                else:
                    recommendations['model_selection'][key] = {
                        'primary_model': best_model[0],
                        'confidence': best_model[1]['r2'],
                        'context_adjustment': 'moderate_context_impact'
                    }
        
        # Рекомендацandї по управлandнню риwithиками
        if economic_context['market_conditions']:
            volatility_regime = economic_context['market_conditions']['volatility_regime']
            
            if volatility_regime == 'high':
                recommendations['risk_management']['position_size'] = 'reduce_by_30%'
                recommendations['risk_management']['stop_loss'] = 'tighten'
                recommendations['monitoring_alerts'].append("High volatility detected - tighten risk management")
            elif volatility_regime == 'low':
                recommendations['risk_management']['position_size'] = 'normal'
                recommendations['risk_management']['stop_loss'] = 'normal'
        
        # Торговand стратегandї
        if economic_context['news_sentiment']:
            avg_sentiment = np.mean([stats['mean'] for stats in economic_context['news_sentiment'].values()])
            
            if avg_sentiment > 0.2:
                recommendations['trading_strategies']['bias'] = 'bullish'
            elif avg_sentiment < -0.2:
                recommendations['trading_strategies']['bias'] = 'bearish'
            else:
                recommendations['trading_strategies']['bias'] = 'neutral'
        
        return recommendations
    
    def _generate_contextual_summary(self, training_results: Dict[str, Dict], 
                                  comparison_results: Dict[str, Any],
                                  economic_context: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати пandдсумок with контекстом"""
        summary = {
            'total_combinations': len(training_results),
            'total_models_trained': sum(len(models) for models in training_results.values()),
            'avg_performance': 0.0,
            'context_summary': {},
            'key_insights': [],
            'recommendations': []
        }
        
        # Середня продуктивнandсть
        all_r2_scores = []
        for result in comparison_results.values():
            if 'performance_summary' in result:
                all_r2_scores.extend([metrics['r2'] for metrics in result['performance_summary'].values()])
        
        if all_r2_scores:
            summary['avg_performance'] = np.mean(all_r2_scores)
        
        # Пandдсумок контексту
        summary['context_summary'] = {
            'macro_indicators_count': len(economic_context['macro_indicators']),
            'market_regime': economic_context['market_conditions'].get('volatility_regime', 'unknown'),
            'sentiment_trend': np.mean([stats['trend'] for stats in economic_context['news_sentiment'].values()]) if economic_context['news_sentiment'] else 0,
            'overall_context_score': economic_context.get('overall_context_score', 0)
        }
        
        # Ключовand andнсайти
        if summary['context_summary']['market_regime'] == 'high':
            summary['key_insights'].append("High volatility regime detected - models may be less reliable")
        
        if summary['context_summary']['overall_context_score'] > 0.7:
            summary['key_insights'].append("Strong economic context influence - consider context adjustments")
        
        # Рекомендацandї
        if summary['avg_performance'] > 0.7:
            summary['recommendations'].append("Good model performance - ready for production with context monitoring")
        elif summary['avg_performance'] > 0.5:
            summary['recommendations'].append("Moderate performance - consider ensemble methods with context")
        else:
            summary['recommendations'].append("Low performance - review feature engineering and context integration")
        
        return summary
    
    def save_contextual_results(self, results: Dict[str, Any]) -> str:
        """Зберегти реwithульandти with контекстом"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Зберandгаємо реwithульandти
            results_path = f"results/contextual_comparison_{timestamp}.json"
            
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
            
            logger.info(f"[ResultsComparison] Contextual results saved to {results_path}")
            return results_path
            
        except Exception as e:
            logger.error(f"[ResultsComparison] Error saving contextual results: {e}")
            return None

def run_stage_4_contextual_comparison(stage_3_data: Dict[str, Any], 
                                     models_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Запуск контекстного порandвняння реwithульandтandв
    
    Args:
        stage_3_data: Данand with еandпу 3
        models_config: Конфandгурацandя моwhereлей
    
    Returns:
        Реwithульandти порandвняння with контекстом
    """
    if models_config is None:
        # Сandндартна конфandгурацandя
        models_config = {
            "random_forest": {"type": "random_forest", "category": "light"},
            "xgboost": {"type": "xgboost", "category": "light"},
            "lstm": {"type": "lstm", "category": "heavy"},
            "transformer": {"type": "transformer", "category": "heavy"}
        }
    
    comparator = ResultsComparisonWithContext(models_config)
    return comparator.run_results_comparison_with_context(stage_3_data)
