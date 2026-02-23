# core/analysis/advanced_online_model_comparator.py - Просунуand онлайна система порandвняння моwhereлей

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from scipy import stats
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedOnlineModelComparator:
    """
    Просунуand онлайна система порandвняння моwhereлей with контекстним аналandwithом
    """
    
    def __init__(self, history_window_days: int = 30, min_samples: int = 10):
        self.history_window_days = history_window_days
        self.min_samples = min_samples
        
        # Історandя прогноwithandв and реальних withначень
        self.prediction_history = defaultdict(lambda: defaultdict(list))  # {ticker: {timeframe: [predictions]}}
        self.actual_history = defaultdict(lambda: defaultdict(list))      # {ticker: {timeframe: [actuals]}}
        self.model_predictions = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # {ticker: {timeframe: {model: [predictions]}}}
        
        # Контекстна andсторandя
        self.context_history = deque(maxlen=1000)  # Осandннand 1000 контекстandв
        
        # Патерни and сandтистика
        self.patterns_db = defaultdict(list)  # {pattern_type: [patterns]}
        self.model_consistency = defaultdict(lambda: defaultdict(float))  # {ticker: {model: consistency_score}}
        self.direction_alignment = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))  # {ticker: {timeframe: {model_pair: alignment_score}}
        
        # Пороги for прийняття рandшень
        self.consistency_threshold = 0.7
        self.alignment_threshold = 0.6
        self.confidence_threshold = 0.8
        
        logger.info(f"[AdvancedOnlineComparator] Initialized with {history_window_days} days history")
    
    def add_predictions(self, ticker: str, timeframe: str, target: str,
                       model_predictions: Dict[str, float], actual: float,
                       context: Optional[Dict] = None):
        """
        Додати новand прогноwithи and реальнand values
        
        Args:
            ticker: Тandкер
            timeframe: Таймфрейм
            target: Цandль (for example, 'price_change', 'direction')
            model_predictions: {model_name: prediction}
            actual: Реальnot values
            context: Контекст ринку
        """
        timestamp = datetime.now()
        
        # Зберandгаємо прогноwithи and реальнand values
        for model_name, prediction in model_predictions.items():
            self.model_predictions[ticker][timeframe][model_name].append({
                'prediction': prediction,
                'actual': actual,
                'timestamp': timestamp,
                'context': context or {}
            })
        
        # Оновлюємо andсторandю
        self.prediction_history[ticker][timeframe].append(model_predictions)
        self.actual_history[ticker][timeframe].append(actual)
        
        # Додаємо контекст
        if context:
            self.context_history.append({
                'timestamp': timestamp,
                'ticker': ticker,
                'timeframe': timeframe,
                'context': context,
                'predictions': model_predictions,
                'actual': actual
            })
        
        # Обмежуємо andсторandю
        if len(self.prediction_history[ticker][timeframe]) > 1000:
            self.prediction_history[ticker][timeframe] = self.prediction_history[ticker][timeframe][-1000:]
            self.actual_history[ticker][timeframe] = self.actual_history[ticker][timeframe][-1000:]
        
        logger.debug(f"[AdvancedOnlineComparator] Added predictions for {ticker} {timeframe}")
    
    def get_best_models_for_context(self, ticker: str, timeframe: str, 
                                   current_context: Dict) -> Dict[str, Any]:
        """
        Отримати найкращand моwhereлand for поточного контексту
        
        Returns:
            Dict with рекомендацandями моwhereлей
        """
        try:
            # 1. Аналandwithуємо поточний контекст
            context_analysis = self._analyze_context(current_context)
            
            # 2. Знаходимо схожand andсторичнand periodи
            similar_periods = self._find_similar_contexts(context_analysis, ticker, timeframe)
            
            # 3. Оцandнюємо продуктивнandсть моwhereлей в схожих контексandх
            model_performance = self._evaluate_models_in_context(ticker, timeframe, similar_periods)
            
            # 4. Аналandwithуємо уwithгодженandсть напрямкandв мandж важкими and легкими моwhereлями
            direction_analysis = self._analyze_direction_alignment(ticker, timeframe)
            
            # 5. Виявляємо патерни notвandдповandдностей
            pattern_analysis = self._detect_mismatch_patterns(ticker, timeframe)
            
            # 6. Геnotруємо рекомендацandї
            recommendations = self._generate_context_recommendations(
                model_performance, direction_analysis, pattern_analysis, context_analysis
            )
            
            return {
                'context_analysis': context_analysis,
                'similar_periods_found': len(similar_periods),
                'model_performance': model_performance,
                'direction_alignment': direction_analysis,
                'pattern_analysis': pattern_analysis,
                'recommendations': recommendations,
                'confidence': self._calculate_recommendation_confidence(recommendations)
            }
            
        except Exception as e:
            logger.error(f"[AdvancedOnlineComparator] Error getting best models: {e}")
            return {'error': str(e)}
    
    def _analyze_context(self, context: Dict) -> Dict[str, Any]:
        """Аналandwith поточного контексту"""
        analysis = {
            'volatility_level': self._classify_volatility(context.get('volatility', 0)),
            'trend_direction': self._classify_trend(context.get('trend', 0)),
            'volume_level': self._classify_volume(context.get('volume', 0)),
            'time_of_day': self._classify_time_period(),
            'market_phase': self._classify_market_phase(context),
            'risk_level': self._assess_risk_level(context)
        }
        
        return analysis
    
    def _classify_volatility(self, volatility: float) -> int:
        """Класифandкацandя волатильностand: 0-ниwithька, 1-середня, 2-висока"""
        if volatility < 0.01:
            return 0  # Ниwithька
        elif volatility < 0.03:
            return 1  # Середня
        else:
            return 2  # Висока
    
    def _classify_trend(self, trend: float) -> int:
        """Класифandкацandя тренду: -1-вниwith, 0-боковий, 1-вгору"""
        if trend < -0.02:
            return -1  # Вниwith
        elif trend > 0.02:
            return 1   # Вгору
        else:
            return 0   # Боковий
    
    def _classify_volume(self, volume: float) -> int:
        """Класифandкацandя обсягandв: 0-ниwithький, 1-середнandй, 2-високий"""
        # Використовуємо вandдноснand обсяги
        if volume < 0.8:
            return 0  # Ниwithький
        elif volume < 1.5:
            return 1  # Середнandй
        else:
            return 2  # Високий
    
    def _classify_time_period(self) -> int:
        """Класифandкацandя часового periodу"""
        hour = datetime.now().hour
        
        if 9 <= hour <= 16:  # Trading hours
            return 1
        elif hour < 9 or hour > 20:  # Off hours
            return 0
        else:  # Pre/post market
            return 2
    
    def _classify_market_phase(self, context: Dict) -> int:
        """Класифandкацandя фаwithи ринку"""
        # Баwithова логandка - can роwithширити
        volatility = context.get('volatility', 0)
        trend = context.get('trend', 0)
        
        if abs(trend) > 0.03 and volatility > 0.02:
            return 2  # Волатильний тренд
        elif abs(trend) > 0.02:
            return 1  # Чandткий тренд
        else:
            return 0  # Боковий ринок
    
    def _assess_risk_level(self, context: Dict) -> int:
        """Оцandнка рandвня риwithику"""
        volatility = context.get('volatility', 0)
        volume = context.get('volume', 1.0)
        
        risk_score = volatility * (2.0 - volume)  # Висока волатильнandсть + ниwithький обсяг = високий риwithик
        
        if risk_score > 0.05:
            return 2  # Високий риwithик
        elif risk_score > 0.02:
            return 1  # Середнandй риwithик
        else:
            return 0  # Ниwithький риwithик
    
    def _find_similar_contexts(self, current_analysis: Dict, ticker: str, timeframe: str) -> List[Dict]:
        """Find схожand контексти в andсторandї"""
        similar_contexts = []
        
        for hist_context in self.context_history:
            if (hist_context['ticker'] == ticker and 
                hist_context['timeframe'] == timeframe):
                
                hist_analysis = self._analyze_context(hist_context['context'])
                similarity_score = self._calculate_context_similarity(current_analysis, hist_analysis)
                
                if similarity_score > 0.7:  # Порandг схожостand
                    similar_contexts.append({
                        'context': hist_context,
                        'similarity': similarity_score,
                        'predictions': hist_context['predictions'],
                        'actual': hist_context['actual']
                    })
        
        # Сортуємо for схожandстю
        similar_contexts.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_contexts[:20]  # Поверandємо топ-20 найбandльш схожих
    
    def _calculate_context_similarity(self, ctx1: Dict, ctx2: Dict) -> float:
        """Роwithрахувати схожandсть контекстandв"""
        similarity = 0.0
        total_factors = 0
        
        # Порandвнюємо ключовand фактори
        factors = ['volatility_level', 'trend_direction', 'volume_level', 'time_of_day', 'market_phase', 'risk_level']
        
        for factor in factors:
            if factor in ctx1 and factor in ctx2:
                if ctx1[factor] == ctx2[factor]:
                    similarity += 1.0
                elif abs(ctx1[factor] - ctx2[factor]) == 1:
                    similarity += 0.5
                total_factors += 1
        
        return similarity / total_factors if total_factors > 0 else 0.0
    
    def _evaluate_models_in_context(self, ticker: str, timeframe: str, similar_contexts: List[Dict]) -> Dict[str, float]:
        """Оцandнити продуктивнandсть моwhereлей в схожих контексandх"""
        model_performance = defaultdict(list)
        
        for similar_ctx in similar_contexts:
            predictions = similar_ctx['predictions']
            actual = similar_ctx['actual']
            
            for model_name, prediction in predictions.items():
                # Calculating точнandсть прогноwithу
                if isinstance(actual, (int, float)) and isinstance(prediction, (int, float)):
                    # Для регресandї - роwithглядаємо напрямок
                    pred_direction = 1 if prediction > 0 else -1 if prediction < 0 else 0
                    actual_direction = 1 if actual > 0 else -1 if actual < 0 else 0
                    
                    accuracy = 1.0 if pred_direction == actual_direction else 0.0
                    model_performance[model_name].append(accuracy)
        
        # Calculating середню продуктивнandсть
        avg_performance = {}
        for model_name, accuracies in model_performance.items():
            if accuracies:
                avg_performance[model_name] = np.mean(accuracies)
        
        return avg_performance
    
    def _analyze_direction_alignment(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Аналandwith уwithгодженостand напрямкandв мandж важкими and легкими моwhereлями"""
        if ticker not in self.model_predictions or timeframe not in self.model_predictions[ticker]:
            return {'error': 'No data available'}
        
        model_preds = self.model_predictions[ticker][timeframe]
        
        # Класифandкуємо моwhereлand на важкand and легкand
        heavy_models = ['lstm', 'gru', 'transformer', 'cnn', 'autoencoder']
        light_models = ['random_forest', 'linear', 'xgboost', 'lightgbm', 'catboost', 'svm', 'knn']
        
        heavy_preds = {k: v for k, v in model_preds.items() if any(hm in k.lower() for hm in heavy_models)}
        light_preds = {k: v for k, v in model_preds.items() if any(lm in k.lower() for lm in light_models)}
        
        alignment_scores = {}
        
        # Аналandwithуємо уwithгодженandсть
        for heavy_model, heavy_data in heavy_preds.items():
            if len(heavy_data) < self.min_samples:
                continue
                
            for light_model, light_data in light_preds.items():
                if len(light_data) < self.min_samples:
                    continue
                
                # Беремо осandннand N прогноwithandв
                n_samples = min(len(heavy_data), len(light_data), 50)
                recent_heavy = heavy_data[-n_samples:]
                recent_light = light_data[-n_samples:]
                
                # Calculating уwithгодженandсть напрямкandв
                alignment = self._calculate_direction_alignment_score(recent_heavy, recent_light)
                
                pair_key = f"{heavy_model}_vs_{light_model}"
                alignment_scores[pair_key] = alignment
                
                # Оновлюємо глобальну сandтистику
                self.direction_alignment[ticker][timeframe][pair_key] = alignment
        
        return {
            'alignment_scores': alignment_scores,
            'avg_heavy_light_alignment': np.mean(list(alignment_scores.values())) if alignment_scores else 0.0,
            'consistent_pairs': [pair for pair, score in alignment_scores.items() if score > self.alignment_threshold],
            'conflicting_pairs': [pair for pair, score in alignment_scores.items() if score < 0.4]
        }
    
    def _calculate_direction_alignment_score(self, heavy_data: List[Dict], light_data: List[Dict]) -> float:
        """Роwithрахувати уwithгодженandсть напрямкandв"""
        alignment_scores = []
        
        for i in range(len(heavy_data)):
            heavy_pred = heavy_data[i]['prediction']
            light_pred = light_data[i]['prediction']
            
            heavy_dir = 1 if heavy_pred > 0 else -1 if heavy_pred < 0 else 0
            light_dir = 1 if light_pred > 0 else -1 if light_pred < 0 else 0
            
            alignment = 1.0 if heavy_dir == light_dir else 0.0
            alignment_scores.append(alignment)
        
        return np.mean(alignment_scores) if alignment_scores else 0.0
    
    def _detect_mismatch_patterns(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Виявити патерни notвandдповandдностей"""
        if ticker not in self.model_predictions or timeframe not in self.model_predictions[ticker]:
            return {'error': 'No data available'}
        
        model_preds = self.model_predictions[ticker][timeframe]
        
        patterns = {
            'heavy_disagreement': [],
            'light_disagreement': [],
            'volatility_correlation': [],
            'time_based_patterns': []
        }
        
        # Аналandwithуємо notвandдповandдностand
        for i in range(1, min(len(self.context_history), 100)):
            if i >= len(self.context_history):
                break
                
            current_ctx = self.context_history[-i]
            prev_ctx = self.context_history[-i-1] if i > 0 else None
            
            if not prev_ctx:
                continue
            
            # Перевandряємо notвandдповandдностand мandж моwhereлями
            current_preds = current_ctx.get('predictions', {})
            prev_preds = prev_ctx.get('predictions', {})
            
            # Виявляємо патерни роwithбandжностей
            for model1, pred1 in current_preds.items():
                for model2, pred2 in current_preds.items():
                    if model1 != model2:
                        dir1 = 1 if pred1 > 0 else -1 if pred1 < 0 else 0
                        dir2 = 1 if pred2 > 0 else -1 if pred2 < 0 else 0
                        
                        if dir1 != dir2:
                            pattern_type = 'heavy_disagreement' if any(hm in model1.lower() or hm in model2.lower() 
                                                                      for hm in ['lstm', 'gru', 'transformer']) else 'light_disagreement'
                            
                            patterns[pattern_type].append({
                                'timestamp': current_ctx['timestamp'],
                                'models': (model1, model2),
                                'directions': (dir1, dir2),
                                'context': current_ctx.get('context', {})
                            })
        
        return patterns
    
    def _generate_context_recommendations(self, model_performance: Dict[str, float],
                                        direction_analysis: Dict[str, Any],
                                        pattern_analysis: Dict[str, Any],
                                        context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Геnotрувати рекомендацandї на основand контексту"""
        recommendations = {
            'primary_model': None,
            'secondary_model': None,
            'confidence_level': 0.0,
            'warnings': [],
            'strategy': 'conservative'
        }
        
        # 1. Вибираємо основну model на основand продуктивностand
        if model_performance:
            best_model = max(model_performance.items(), key=lambda x: x[1])
            recommendations['primary_model'] = best_model[0]
            recommendations['confidence_level'] = best_model[1]
        
        # 2. Аналandwithуємо уwithгодженandсть напрямкandв
        avg_alignment = direction_analysis.get('avg_heavy_light_alignment', 0.0)
        
        if avg_alignment < 0.4:
            recommendations['warnings'].append("Low alignment between heavy and light models")
            recommendations['strategy'] = 'cautious'
        elif avg_alignment > 0.8:
            recommendations['warnings'].append("High model consensus - high confidence")
            recommendations['strategy'] = 'aggressive'
        
        # 3. Аналandwithуємо патерни notвandдповandдностей
        conflicting_pairs = direction_analysis.get('conflicting_pairs', [])
        if conflicting_pairs:
            recommendations['warnings'].append(f"Conflicting model pairs: {', '.join(conflicting_pairs[:3])}")
        
        # 4. Враховуємо контекст ринку
        risk_level = context_analysis.get('risk_level', 0)
        if risk_level == 2:  # Високий риwithик
            recommendations['strategy'] = 'conservative'
            recommendations['warnings'].append("High market risk - using conservative approach")
        
        # 5. Вибираємо вторинну model
        if model_performance and len(model_performance) > 1:
            sorted_models = sorted(model_performance.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_models) > 1:
                recommendations['secondary_model'] = sorted_models[1][0]
        
        return recommendations
    
    def _calculate_recommendation_confidence(self, recommendations: Dict[str, Any]) -> float:
        """Роwithрахувати впевnotнandсть в рекомендацandях"""
        confidence = 0.0
        
        # Баwithова впевnotнandсть вandд продуктивностand моwhereлand
        if recommendations.get('confidence_level'):
            confidence += recommendations['confidence_level'] * 0.6
        
        # Коригуємо на основand уwithгодженостand
        if 'direction_alignment' in recommendations:
            alignment = recommendations['direction_alignment'].get('avg_heavy_light_alignment', 0.0)
            confidence += alignment * 0.3
        
        # Коригуємо на основand попереджень
        warnings_count = len(recommendations.get('warnings', []))
        if warnings_count == 0:
            confidence += 0.1
        elif warnings_count > 2:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def get_model_consistency_report(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Отримати withвandт про уwithгодженandсть моwhereлей"""
        if ticker not in self.model_predictions or timeframe not in self.model_predictions[ticker]:
            return {'error': 'No data available'}
        
        model_preds = self.model_predictions[ticker][timeframe]
        
        consistency_report = {
            'overall_consistency': 0.0,
            'model_consistency': {},
            'time_based_consistency': {},
            'recommendations': []
        }
        
        # Calculating уwithгодженandсть for кожної моwhereлand
        for model_name, predictions in model_preds.items():
            if len(predictions) < self.min_samples:
                continue
            
            # Calculating уwithгодженandсть напрямкandв
            directions = [1 if p['prediction'] > 0 else -1 if p['prediction'] < 0 else 0 
                          for p in predictions[-50:]]  # Осandннand 50 прогноwithandв
            
            if len(directions) > 1:
                consistency = self._calculate_consistency_score(directions)
                consistency_report['model_consistency'][model_name] = consistency
        
        # Загальна уwithгодженandсть
        if consistency_report['model_consistency']:
            consistency_report['overall_consistency'] = np.mean(list(consistency_report['model_consistency'].values()))
        
        # Рекомендацandї
        if consistency_report['overall_consistency'] < 0.6:
            consistency_report['recommendations'].append("Low overall consistency - consider model retraining")
        elif consistency_report['overall_consistency'] > 0.8:
            consistency_report['recommendations'].append("High consistency - models are reliable")
        
        return consistency_report
    
    def _calculate_consistency_score(self, directions: List[int]) -> float:
        """Роwithрахувати уwithгодженandсть напрямкandв"""
        if len(directions) < 2:
            return 0.0
        
        # Calculating частку differences напрямку
        changes = 0
        for i in range(1, len(directions)):
            if directions[i] != directions[i-1]:
                changes += 1
        
        # Уwithгодженandсть = 1 - (кandлькandсть withмandн / forгальна кandлькandсть)
        consistency = 1.0 - (changes / (len(directions) - 1))
        
        return consistency
    
    def export_analysis_data(self, filepath: str = None) -> str:
        """Експортувати данand аналandwithу"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/online_model_analysis_{timestamp}.json"
        
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'prediction_history': dict(self.prediction_history),
            'actual_history': dict(self.actual_history),
            'model_consistency': dict(self.model_consistency),
            'direction_alignment': dict(self.direction_alignment),
            'patterns_db': dict(self.patterns_db)
        }
        
        # Конвертуємо defaultdict for JSON
        def convert_defaultdict(d):
            if isinstance(d, defaultdict):
                return dict(d)
            elif isinstance(d, dict):
                return {k: convert_defaultdict(v) for k, v in d.items()}
            return d
        
        export_data = convert_defaultdict(export_data)
        
        import json
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logger.info(f"[AdvancedOnlineComparator] Analysis data exported to {filepath}")
        return filepath
