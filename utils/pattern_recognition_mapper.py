# utils/pattern_recognition_mapper.py

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger("PatternRecognitionMapper")


class PatternRecognitionMapper:
    """
    Система порівняння поточної ситуації з історичними результатами тренувань
    для підвищення впевненості вибору моделі/таргета/таймфрейму
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PatternRecognitionMapper")
        
        # База даних історичних результатів тренувань
        self.historical_results = {}
        self.pattern_cache = {}
        self.similarity_cache = {}
        
        # Пороги схожості
        self.similarity_threshold = 0.8  # 80% схожість для високого довіри
        self.min_history_size = 10  # Мінімум історичних даних для порівняння
        
        # Ваги для різних аспектів
        self.weights = {
            'market_regime': 0.25,      # Режим ринку
            'risk_level': 0.20,         # Рівень ризику
            'technical_signals': 0.20,    # Технічні сигнали
            'economic_context': 0.15,   # Економічний контекст
            'ticker_performance': 0.20    # Перформанс тікера
        }
        
        self.logger.info("PatternRecognitionMapper initialized")
    
    def add_training_result(self, ticker: str, target: str, timeframe: str, 
                          context: Dict[str, Any], performance_metrics: Dict[str, float],
                          timestamp: datetime = None) -> None:
        """
        Додає результат тренування до історичної бази
        
        Args:
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            context: Контекст тренування
            performance_metrics: Метрики продуктивності
            timestamp: Час тренування
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Створюємо унікальний ключ
        key = f"{ticker}_{target}_{timeframe}"
        
        # Додаємо результат до бази
        if key not in self.historical_results:
            self.historical_results[key] = []
        
        result = {
            'timestamp': timestamp,
            'context': context,
            'performance_metrics': performance_metrics,
            'success_metrics': self._calculate_success_metrics(performance_metrics)
        }
        
        self.historical_results[key].append(result)
        
        # Обмежуємо розмір бази (тільки останні N результатів)
        if len(self.historical_results[key]) > 100:
            self.historical_results[key] = self.historical_results[key][-100:]
        
        self.logger.info(f"Added training result for {key}. Total results: {len(self.historical_results[key])}")
    
    def find_similar_patterns(self, current_context: Dict[str, Any], 
                             ticker: str = None, target: str = None, 
                             timeframe: str = None,
                             min_similarity: float = 0.7) -> Dict[str, Any]:
        """
        Знаходить схожі патерни в історичних результатах тренувань
        
        Args:
            current_context: Поточний контекст
            ticker: Тікер (опціонально)
            target: Таргет (опціонально)
            timeframe: Таймфрейм (опціонально)
            min_similarity: Мінімальна схожість
            
        Returns:
            Dict: Результати пошуку схожих патернів
        """
        similar_patterns = {
            'exact_matches': [],
            'high_similarity': [],
            'medium_similarity': [],
            'low_similarity': [],
            'recommendations': []
        }
        
        # Фільтруємо історичні результати
        filtered_results = self._filter_historical_results(ticker, target, timeframe)
        
        # Пошук схожих патернів
        for key, results in filtered_results.items():
            for i, historical_result in enumerate(results):
                similarity = self._calculate_pattern_similarity(
                    current_context, historical_result['context']
                )
                
                if similarity >= 0.95:
                    similar_patterns['exact_matches'].append({
                        'key': key,
                        'index': i,
                        'similarity': similarity,
                        'historical_result': historical_result,
                        'performance': historical_result['performance_metrics']
                    })
                elif similarity >= 0.85:
                    similar_patterns['high_similarity'].append({
                        'key': key,
                        'index': i,
                        'similarity': similarity,
                        'historical_result': historical_result,
                        'performance': historical_result['performance_metrics']
                    })
                elif similarity >= 0.7:
                    similar_patterns['medium_similarity'].append({
                        'key': key,
                        'index': i,
                        'similarity': similarity,
                        'historical_result': historical_result,
                        'performance': historical_result['performance_metrics']
                    })
                elif similarity >= min_similarity:
                    similar_patterns['low_similarity'].append({
                        'key': key,
                        'index': i,
                        'similarity': similarity,
                        'historical_result': historical_result,
                        'performance': historical_result['performance_metrics']
                    })
        
        # Генеруємо рекомендації
        similar_patterns['recommendations'] = self._generate_recommendations(
            similar_patterns, current_context
        )
        
        # Сортуємо за схожістю
        for category in ['exact_matches', 'high_similarity', 'medium_similarity', 'low_similarity']:
            similar_patterns[category].sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_patterns
    
    def get_pattern_confidence_boost(self, current_context: Dict[str, Any],
                                     ticker: str = None, target: str = None, 
                                     timeframe: str = None) -> Dict[str, Any]:
        """
        Отримує підвищення впевненості на основі схожих патернів
        
        Args:
            current_context: Поточний контекст
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            Dict: Підвищення впевненості
        """
        # Знаходимо схожі патерни
        similar_patterns = self.find_similar_patterns(
            current_context, ticker, target, timeframe, min_similarity=0.7
        )
        
        confidence_boost = {
            'base_confidence': 0.5,
            'pattern_confidence': 0.0,
            'boost_applied': False,
            'similar_patterns_found': len(similar_patterns['exact_matches']) + 
                                len(similar_patterns['high_similarity']) +
                                len(similar_patterns['medium_similarity']),
            'recommendations': []
        }
        
        # Розраховуємо підвищення впевненості
        if similar_patterns['exact_matches']:
            # Точні збіги - висока впевненість
            best_match = similar_patterns['exact_matches'][0]
            confidence_boost['pattern_confidence'] = 0.9
            confidence_boost['boost_applied'] = True
            confidence_boost['base_confidence'] = min(0.95, 0.5 + 0.4)
            
            # Додаємо рекомендації на основі успішних результатів
            if best_match['performance'].get('sharpe_ratio', 0) > 1.5:
                confidence_boost['base_confidence'] = min(0.98, confidence_boost['base_confidence'] + 0.03)
            
            confidence_boost['recommendations'].append(
                f"Exact pattern match found with Sharpe: {best_match['performance'].get('sharpe_ratio', 0):.2f}"
            )
            
        elif similar_patterns['high_similarity']:
            # Висока схожість - помірна впевненість
            best_match = similar_patterns['high_similarity'][0]
            confidence_boost['pattern_confidence'] = 0.7
            confidence_boost['boost_applied'] = True
            confidence_boost['base_confidence'] = min(0.85, 0.5 + 0.3)
            
            confidence_boost['recommendations'].append(
                f"High similarity pattern found with accuracy: {best_match['performance'].get('accuracy', 0):.2f}"
            )
            
        elif similar_patterns['medium_similarity']:
            # Середня схожість - невелике підвищення
            best_match = similar_patterns['medium_similarity'][0]
            confidence_boost['pattern_confidence'] = 0.5
            confidence_boost['boost_applied'] = True
            confidence_boost['base_confidence'] = min(0.75, 0.5 + 0.2)
            
            confidence_boost['recommendations'].append(
                f"Medium similarity pattern found with win rate: {best_match['performance'].get('win_rate', 0):.2f}"
            )
        
        # Додаємо статистичну інформацію
        if confidence_boost['similar_patterns_found'] > 0:
            avg_similarity = np.mean([
                p['similarity'] for p in similar_patterns['exact_matches'] + 
                similar_patterns['high_similarity'] + 
                similar_patterns['medium_similarity']
            ])
            confidence_boost['average_similarity'] = avg_similarity
            confidence_boost['pattern_count'] = confidence_boost['similar_patterns_found']
        
        return confidence_boost
    
    def get_historical_performance_analysis(self, ticker: str = None, target: str = None, 
                                           timeframe: str = None) -> Dict[str, Any]:
        """
        Аналіз історичної продуктивності
        
        Args:
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            Dict: Аналіз продуктивності
        """
        analysis = {
            'summary': {},
            'performance_trends': {},
            'best_periods': {},
            'worst_periods': {},
            'recommendations': []
        }
        
        # Фільтруємо результати
        filtered_results = self._filter_historical_results(ticker, target, timeframe)
        
        if not filtered_results:
            analysis['summary']['message'] = "No historical data found"
            return analysis
        
        # Аналізуємо продуктивність по часу
        all_results = []
        for key, results in filtered_results.items():
            for result in results:
                result['key'] = key
                all_results.append(result)
        
        # Сортуємо за часом
        all_results.sort(key=lambda x: x['timestamp'])
        
        # Загальна статистика
        if all_results:
            performance_data = [r['performance_metrics'] for r in all_results]
            
            analysis['summary'] = {
                'total_results': len(all_results),
                'date_range': {
                    'start': all_results[0]['timestamp'].isoformat(),
                    'end': all_results[-1]['timestamp'].isoformat()
                },
                'average_metrics': self._calculate_average_metrics(performance_data),
                'best_performance': max(performance_data, key=lambda x: x.get('sharpe_ratio', 0)),
                'worst_performance': min(performance_data, key=lambda x: x.get('sharpe_ratio', 0)),
                'performance_consistency': self._calculate_consistency(performance_data)
            }
            
            # Тренди продуктивності
            analysis['performance_trends'] = self._analyze_performance_trends(all_results)
            
            # Найкращі періоди
            analysis['best_periods'] = self._find_best_periods(all_results, top_n=5)
            
            # Найгірші періоди
            analysis['worst_periods'] = self._find_worst_periods(all_results, top_n=5)
            
            # Рекомендації
            analysis['recommendations'] = self._generate_performance_recommendations(all_results)
        
        return analysis
    
    def get_pattern_based_selection(self, current_context: Dict[str, Any],
                                   candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вибирає найкращих кандидатів на основі схожих патернів
        
        Args:
            current_context: Поточний контекст
            candidates: Список кандидатів (моделі/таргети/таймфрейми)
            
        Returns:
            Dict: Вибір з підвищеною впевненістю
        """
        selection_results = {
            'candidates': [],
            'pattern_analysis': {},
            'confidence_scores': {},
            'final_ranking': []
        }
        
        # Оцінюємо кожного кандидата
        for candidate in candidates:
            candidate_key = f"{candidate.get('ticker', 'unknown')}_{candidate.get('target', 'unknown')}_{candidate.get('timeframe', 'unknown')}"
            
            # Отримуємо підвищення впевненості
            confidence_boost = self.get_pattern_confidence_boost(
                current_context, 
                candidate.get('ticker'), 
                candidate.get('target'), 
                candidate.get('timeframe')
            )
            
            # Розраховуємо загальний скор
            base_score = candidate.get('confidence_score', 0.5)
            pattern_score = confidence_boost['pattern_confidence']
            
            final_score = (base_score * 0.6) + (pattern_score * 0.4)
            
            candidate_analysis = {
                'candidate': candidate,
                'base_confidence': base_score,
                'pattern_confidence': pattern_score,
                'final_score': final_score,
                'confidence_boost': confidence_boost,
                'similar_patterns': confidence_boost.get('similar_patterns_found', 0)
            }
            
            selection_results['candidates'].append(candidate_analysis)
            selection_results['pattern_analysis'][candidate_key] = candidate_analysis
            selection_results['confidence_scores'][candidate_key] = final_score
        
        # Сортуємо за фінальним скором
        selection_results['candidates'].sort(key=lambda x: x['final_score'], reverse=True)
        selection_results['final_ranking'] = [
            {
                'candidate': c['candidate'],
                'final_score': c['final_score'],
                'confidence_boost': c['confidence_boost'],
                'pattern_confidence': c['pattern_confidence']
            }
            for c in selection_results['candidates']
        ]
        
        return selection_results
    
    def _filter_historical_results(self, ticker: str = None, target: str = None, 
                                  timeframe: str = None) -> Dict[str, List]:
        """
        Фільтрує історичні результати за параметрами
        
        Args:
            ticker: Тікер
            target: Таргет
            timeframe: Таймфрейм
            
        Returns:
            Dict: Відфільтровані результати
        """
        filtered = {}
        
        for key, results in self.historical_results.items():
            ticker_match = ticker is None or key.startswith(f"{ticker}_")
            target_match = target is None or key.split('_')[1] == target
            timeframe_match = timeframe is None or key.split('_')[2] == timeframe
            
            if ticker_match and target_match and timeframe_match:
                filtered[key] = results
        
        return filtered
    
    def _calculate_pattern_similarity(self, current_context: Dict[str, Any], 
                                    historical_context: Dict[str, Any]) -> float:
        """
        Розраховує схожість між поточним та історичним контекстом
        
        Args:
            current_context: Поточний контекст
            historical_context: Історичний контекст
            
        Returns:
            float: Рівень схожості (0-1)
        """
        similarity_score = 0.0
        total_weight = 0
        
        # Порівнюємо режими ринку
        current_regime = current_context.get('market_regime', 'neutral')
        historical_regime = historical_context.get('market_regime', 'neutral')
        regime_similarity = 1.0 if current_regime == historical_regime else 0.5
        similarity_score += regime_similarity * self.weights['market_regime']
        total_weight += self.weights['market_regime']
        
        # Порівнюємо рівні ризику
        current_risk = current_context.get('risk_level', 'medium')
        historical_risk = historical_context.get('risk_level', 'medium')
        risk_similarity = 1.0 if current_risk == historical_risk else 0.7
        similarity_score += risk_similarity * self.weights['risk_level']
        total_weight += self.weights['risk_level']
        
        # Порівнюємо технічні сигнали
        current_signals = current_context.get('indicator_signals', {})
        historical_signals = historical_context.get('indicator_signals', {})
        
        if current_signals and historical_signals:
            signal_similarity = self._compare_signal_sets(current_signals, historical_signals)
            similarity_score += signal_similarity * self.weights['technical_signals']
            total_weight += self.weights['technical_signals']
        
        # Порівнюємо економічний контекст
        current_economic = current_context.get('economic_indicators', {})
        historical_economic = historical_context.get('economic_indicators', {})
        
        if current_economic and historical_economic:
            economic_similarity = self._compare_economic_contexts(current_economic, historical_economic)
            similarity_score += economic_similarity * self.weights['economic_context']
            total_weight += self.weights['economic_context']
        
        # Нормалізуємо результат
        if total_weight > 0:
            similarity_score = similarity_score / total_weight
        
        return similarity_score
    
    def _compare_signal_sets(self, current_signals: Dict, historical_signals: Dict) -> float:
        """Порівнює набори сигналів"""
        if not current_signals or not historical_signals:
            return 0.0
        
        current_bullish = len(current_signals.get('bullish_signals', []))
        current_bearish = len(current_signals.get('bearish_signals', []))
        historical_bullish = len(historical_signals.get('bullish_signals', []))
        historical_bearish = len(historical_signals.get('bearish_signals', []))
        
        # Розраховуємо схожість сигналів
        total_signals = current_bullish + current_bearish + historical_bullish + historical_bearish
        if total_signals == 0:
            return 1.0
        
        matching_signals = 0
        if current_bullish > 0 and historical_bullish > 0:
            matching_signals += min(current_bullish, historical_bullish) / max(current_bullish, historical_bullish)
        if current_bearish > 0 and historical_bearish > 0:
            matching_signals += min(current_bearish, historical_bearish) / max(current_bearish, historical_bearish)
        
        return matching_signals / total_signals
    
    def _compare_economic_contexts(self, current_economic: Dict, historical_economic: Dict) -> float:
        """Порівнює економічні контексти"""
        if not current_economic or not historical_economic:
            return 0.0
        
        # Порівнюємо загальні скори
        current_score = current_economic.get('overall_score', 0)
        historical_score = historical_economic.get('overall_score', 0)
        
        # Розраховуємо схожість скірів
        max_score = max(abs(current_score), abs(historical_score))
        if max_score == 0:
            return 1.0
        
        score_similarity = 1.0 - (abs(current_score - historical_score) / max_score)
        
        return score_similarity
    
    def _calculate_success_metrics(self, performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Розраховує успішні метрики"""
        return {
            'success_score': (
                performance_metrics.get('sharpe_ratio', 0) * 0.4 +
                performance_metrics.get('win_rate', 0) * 0.3 +
                performance_metrics.get('profit_factor', 0) * 0.3
            ),
            'risk_adjusted_return': (
                performance_metrics.get('return', 0) / max(performance_metrics.get('max_drawdown', 0.01), 0.01)
            )
        }
    
    def _calculate_average_metrics(self, performance_data: List[Dict[str, float]]) -> Dict[str, float]:
        """Розраховує середні метрики"""
        if not performance_data:
            return {}
        
        metrics = ['sharpe_ratio', 'win_rate', 'profit_factor', 'return', 'max_drawdown']
        averages = {}
        
        for metric in metrics:
            values = [p.get(metric, 0) for p in performance_data]
            if values:
                averages[metric] = np.mean(values)
        
        return averages
    
    def _calculate_consistency(self, performance_data: List[Dict[str, float]]) -> float:
        """Розраховує консистентність результатів"""
        if len(performance_data) < 2:
            return 0.0
        
        sharpe_ratios = [p.get('sharpe_ratio', 0) for p in performance_data]
        
        if not sharpe_ratios:
            return 0.0
        
        # Розраховуємо коефіцієнт варіації Sharpe ratio
        mean_sharpe = np.mean(sharpe_ratios)
        std_sharpe = np.std(sharpe_ratios)
        
        if std_sharpe == 0:
            return 1.0
        
        consistency = 1.0 - (std_sharpe / mean_sharpe)
        return max(0.0, consistency)
    
    def _analyze_performance_trends(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Аналізує тренди продуктивності"""
        if len(all_results) < 2:
            return {'trend': 'insufficient_data'}
        
        # Розбиваємо на періоди (наприклад, по місяцях)
        monthly_data = defaultdict(list)
        
        for result in all_results:
            month_key = result['timestamp'].strftime('%Y-%m')
            monthly_data[month_key].append(result['performance_metrics'])
        
        trends = {}
        for month, metrics_list in monthly_data.items():
            if len(metrics_list) > 1:
                avg_metrics = self._calculate_average_metrics(metrics_list)
                prev_month = list(monthly_data.keys()).index(month) - 1 if month in list(monthly_data.keys()) else 0
                
                if prev_month >= 0:
                    prev_month_key = list(monthly_data.keys())[prev_month]
                    prev_metrics = self._calculate_average_metrics(monthly_data[prev_month_key])
                    
                    trends[month] = {
                        'current_avg': avg_metrics,
                        'previous_avg': prev_metrics,
                        'trend': 'improving' if avg_metrics.get('sharpe_ratio', 0) > prev_metrics.get('shrank_ratio', 0) else 'declining'
                    }
        
        return trends
    
    def _find_best_periods(self, all_results: List[Dict], top_n: int = 5) -> List[Dict]:
        """Знаходить найкращі періоди"""
        sorted_results = sorted(all_results, 
                           key=lambda x: x['performance_metrics'].get('sharpe_ratio', 0), 
                           reverse=True)
        return sorted_results[:top_n]
    
    def _find_worst_periods(self, all_results: List[Dict], top_n: int = 5) -> List[Dict]:
        """Знаходить найгірші періоди"""
        sorted_results = sorted(all_results, 
                           key=lambda x: x['performance_metrics'].get('sharpe_ratio', 0))
        return sorted_results[:top_n]
    
    def _generate_performance_recommendations(self, all_results: List[Dict]) -> List[str]:
        """Генерує рекомендації на основі аналізу продуктивності"""
        recommendations = []
        
        if len(all_results) < 5:
            return ["Insufficient data for recommendations"]
        
        # Аналізуємо останні результати
        recent_results = all_results[-5:]
        recent_metrics = self._calculate_average_metrics([r['performance_metrics'] for r in recent_results])
        
        avg_sharpe = recent_metrics.get('sharpe_ratio', 0)
        avg_win_rate = recent_metrics.get('win_rate', 0)
        
        if avg_sharpe > 1.5:
            recommendations.append("Recent performance is excellent - consider increasing position sizes")
        elif avg_sharpe < 0.5:
            recommendations.append("Recent performance is poor - reduce risk exposure")
        
        if avg_win_rate > 0.6:
            recommendations.append("High win rate detected - strategy is working well")
        elif avg_win_rate < 0.4:
            recommendations.append("Low win rate - consider strategy revision")
        
        # Аналізуємо тренди
        trends = self._analyze_performance_trends(all_results)
        if trends:
            latest_trend = list(trends.values())[-1]
            if latest_trend.get('trend') == 'improving':
                recommendations.append("Performance is trending upward - continue current approach")
            elif latest_trend.get('trend') == 'declining':
                recommendations.append("Performance is declining - consider strategy adjustment")
        
        return recommendations
    
    def _generate_recommendations(self, similar_patterns: Dict[str, Any], 
                                 current_context: Dict[str, Any]) -> List[str]:
        """Генерує рекомендації на основі схожих патернів"""
        recommendations = []
        
        if similar_patterns['exact_matches']:
            best_match = similar_patterns['exact_matches'][0]
            perf = best_match['performance']
            
            recommendations.append(
                f"Exact pattern match found with Sharpe: {perf.get('sharpe_ratio', 0):.2f}, "
                f"Accuracy: {perf.get('accuracy', 0):.2f}"
            )
            
            if perf.get('sharpe_ratio', 0) > 2.0:
                recommendations.append("Excellent historical performance - high confidence recommendation")
            elif perf.get('sharpe_ratio', 0) < 1.0:
                recommendations.append("Poor historical performance - exercise caution")
        
        elif similar_patterns['high_similarity']:
            best_match = similar_patterns['high_similarity'][0]
            perf = best_match['performance']
            
            recommendations.append(
                f"High similarity pattern found with Win Rate: {perf.get('win_rate', 0):.2f}"
            )
        
        elif similar_patterns['medium_similarity']:
            recommendations.append("Medium similarity pattern found - moderate confidence")
        
        if not similar_patterns['exact_matches'] and not similar_patterns['high_similarity']:
            recommendations.append("No similar patterns found - using base confidence only")
        
        # Додаємо рекомендації на основі поточного контексту
        current_regime = current_context.get('market_regime', 'neutral')
        current_risk = current_context.get('risk_level', 'medium')
        
        if current_regime == 'bullish' and current_risk == 'low':
            recommendations.append("Favorable market conditions - consider increasing exposure")
        elif current_regime == 'bearish' and current_risk == 'high':
            recommendations.append("Unfavorable market conditions - reduce exposure")
        
        return recommendations


class PatternBasedSelector:
    """
    Селектор на основі розпізнавання патернів
    """
    
    def __init__(self):
        self.pattern_mapper = PatternRecognitionMapper()
        self.logger = logging.getLogger("PatternBasedSelector")
    
    def select_best_candidates(self, current_context: Dict[str, Any],
                              candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Вибирає найкращих кандидатів на основі схожих патернів
        
        Args:
            current_context: Поточний контекст
            candidates: Список кандидатів
            
        Returns:
            Dict: Результати вибору
        """
        return self.pattern_mapper.get_pattern_based_selection(current_context, candidates)
    
    def get_confidence_enhanced_selection(self, current_context: Dict[str, Any],
                                         base_selection: Dict[str, Any]) -> Dict[str, Any]:
        """
        Покращує базовий вибір підвищеноючи впевненість
        
        Args:
            current_context: Поточний контекст
            base_selection: Базовий вибір
            
        Returns:
            Dict: Покращений вибір
        """
        # Отримуємо підвищення впевненості
        ticker = base_selection.get('ticker')
        target = base_selection.get('target')
        timeframe = base_selection.get('timeframe')
        
        confidence_boost = self.pattern_mapper.get_pattern_confidence_boost(
            current_context, ticker, target, timeframe
        )
        
        # Покращуємо базовий вибір
        enhanced_selection = base_selection.copy()
        enhanced_selection['pattern_confidence'] = confidence_boost['pattern_confidence']
        enhanced_selection['base_confidence'] = confidence_boost['base_confidence']
        enhanced_selection['confidence_boost_applied'] = confidence_boost['boost_applied']
        enhanced_selection['similar_patterns_found'] = confidence_boost['similar_patterns_found']
        
        # Оновлюємо загальний скор
        enhanced_selection['final_confidence'] = (
            enhanced_selection.get('base_confidence', 0.5) * 0.6 +
            enhanced_selection.get('pattern_confidence', 0.0) * 0.4
        )
        
        # Додаємо рекомендації
        enhanced_selection['recommendations'] = confidence_boost['recommendations']
        
        return enhanced_selection
