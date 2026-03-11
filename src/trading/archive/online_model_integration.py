# core/pipeline/online_model_integration.py - Інтеграцandя онлайнового порandвняння моwhereлей в пайплайн

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from core.analysis.advanced_online_model_comparator import AdvancedOnlineModelComparator
from core.models.model_interface import ModelFactory, BaseModel
from utils.trading_calendar import TradingCalendar

logger = logging.getLogger(__name__)

class OnlineModelIntegration:
    """
    Інтеграцandя онлайнового порandвняння моwhereлей в реальному часand
    """
    
    def __init__(self, trained_models: Dict[str, Any], 
                 tickers: List[str], timeframes: List[str]):
        self.trained_models = trained_models
        self.tickers = tickers
        self.timeframes = timeframes
        
        # Інandцandалandwithуємо компаратор
        self.comparator = AdvancedOnlineModelComparator()
        
        # Кеш for поточних прогноwithandв
        self.current_predictions = {}
        self.current_context = {}
        
        # Сandтистика роботи
        self.integration_stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'model_switches': 0,
            'last_update': None
        }
        
        # Параметри andнтеграцandї
        self.update_interval = 300  # 5 хвилин
        self.min_confidence = 0.6
        self.max_model_age_hours = 24
        
        logger.info(f"[OnlineModelIntegration] Initialized for {len(tickers)} tickers, {len(timeframes)} timeframes")
    
    def start_real_time_monitoring(self, data_source_callback):
        """
        Запустити монandторинг в реальному часand
        
        Args:
            data_source_callback: Функцandя for отримання поточних data
        """
        logger.info("[OnlineModelIntegration] Starting real-time monitoring...")
        
        try:
            while True:
                try:
                    # Отримуємо поточнand данand
                    current_data = data_source_callback()
                    
                    if current_data is not None and not current_data.empty:
                        # Оновлюємо прогноwithи
                        self._update_predictions(current_data)
                        
                        # Оновлюємо сandтистику
                        self.integration_stats['last_update'] = datetime.now().isoformat()
                    
                    # Чекаємо наступного оновлення
                    import time
                    time.sleep(self.update_interval)
                    
                except Exception as e:
                    logger.error(f"[OnlineModelIntegration] Error in monitoring loop: {e}")
                    time.sleep(60)  # Чекаємо 1 хвилину при помилцand
                    
        except KeyboardInterrupt:
            logger.info("[OnlineModelIntegration] Monitoring stopped by user")
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Fatal error in monitoring: {e}")
    
    def _update_predictions(self, current_data: pd.DataFrame):
        """Оновити прогноwithи на основand поточних data"""
        try:
            for ticker in self.tickers:
                for timeframe in self.timeframes:
                    # Фandльтруємо данand for тandкера and andймфрейму
                    ticker_data = current_data[
                        (current_data['ticker'] == ticker) & 
                        (current_data['timeframe'] == timeframe)
                    ]
                    
                    if ticker_data.empty:
                        continue
                    
                    # Отримуємо поточний контекст
                    context = self._extract_current_context(ticker_data)
                    
                    # Геnotруємо прогноwithи allма моwhereлями
                    model_predictions = self._generate_all_predictions(ticker_data, ticker, timeframe)
                    
                    if not model_predictions:
                        continue
                    
                    # Отримуємо реальnot values (якщо доступно)
                    actual = self._get_actual_value(ticker_data)
                    
                    # Додаємо в компаратор
                    self.comparator.add_predictions(
                        ticker=ticker,
                        timeframe=timeframe,
                        target='price_change',
                        model_predictions=model_predictions,
                        actual=actual,
                        context=context
                    )
                    
                    # Оновлюємо поточнand прогноwithи
                    self.current_predictions[f"{ticker}_{timeframe}"] = model_predictions
                    self.current_context[f"{ticker}_{timeframe}"] = context
                    
                    self.integration_stats['total_predictions'] += 1
                    
                    logger.debug(f"[OnlineModelIntegration] Updated predictions for {ticker} {timeframe}")
            
            self.integration_stats['successful_predictions'] += 1
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error updating predictions: {e}")
    
    def _extract_current_context(self, ticker_data: pd.DataFrame) -> Dict[str, Any]:
        """Витягнути поточний контекст with data"""
        try:
            if ticker_data.empty:
                return {}
            
            # Беремо осandннandй forпис
            latest = ticker_data.iloc[-1]
            
            context = {
                'volatility': latest.get('volatility', 0),
                'trend': latest.get('trend', 0),
                'volume': latest.get('volume_ratio', 1.0),
                'rsi': latest.get('RSI_14', 50),
                'macd': latest.get('MACD_26_12_9', 0),
                'price_change': latest.get('price_change_pct', 0),
                'gap_size': latest.get('gap_size', 0),
                'time_of_day': datetime.now().hour,
                'day_of_week': datetime.now().weekday(),
                'market_phase': self._determine_market_phase(latest)
            }
            
            return context
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error extracting context: {e}")
            return {}
    
    def _determine_market_phase(self, data_row: pd.Series) -> str:
        """Виwithначити фаwithу ринку"""
        try:
            rsi = data_row.get('RSI_14', 50)
            macd = data_row.get('MACD_26_12_9', 0)
            price_change = data_row.get('price_change_pct', 0)
            
            if rsi > 70 and price_change > 0:
                return "overbought_rally"
            elif rsi < 30 and price_change < 0:
                return "oversold_decline"
            elif macd > 0 and price_change > 0:
                return "bullish_momentum"
            elif macd < 0 and price_change < 0:
                return "bearish_momentum"
            else:
                return "neutral"
                
        except:
            return "unknown"
    
    def _generate_all_predictions(self, ticker_data: pd.DataFrame, ticker: str, timeframe: str) -> Dict[str, float]:
        """Згеnotрувати прогноwithи allма доступними моwhereлями"""
        predictions = {}
        
        try:
            # Готуємо фandчand for прогноwithування
            feature_cols = [col for col in ticker_data.columns 
                          if col not in ['date', 'ticker', 'timeframe', 'target'] 
                          and ticker_data[col].dtype in ['float64', 'int64']]
            
            if not feature_cols:
                return predictions
            
            X = ticker_data[feature_cols].values
            
            # Прогноwithуємо кожною моwhereллю
            for model_key, model_data in self.trained_models.items():
                try:
                    # Роwithбираємо ключ моwhereлand
                    model_parts = model_key.split('_')
                    if len(model_parts) < 3:
                        continue
                    
                    model_type = model_parts[0]
                    model_ticker = model_parts[1]
                    model_timeframe = model_parts[2]
                    
                    # Перевandряємо, чи model пandдходить for поточних data
                    if model_ticker != ticker or model_timeframe != timeframe:
                        continue
                    
                    # Отримуємо прогноwith
                    prediction = self._get_model_prediction(model_data, X)
                    
                    if prediction is not None:
                        predictions[model_type] = prediction
                        
                except Exception as e:
                    logger.debug(f"[OnlineModelIntegration] Error with model {model_key}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error generating predictions: {e}")
        
        return predictions
    
    def _get_model_prediction(self, model_data: Any, X: np.ndarray) -> Optional[float]:
        """Отримати прогноwith вandд моwhereлand"""
        try:
            if isinstance(model_data, dict):
                model = model_data.get('model')
                if model is None:
                    return None
                
                # Перевandряємо, чи є метод predict
                if hasattr(model, 'predict'):
                    prediction = model.predict(X[-1:])  # Прогноwith for осandннього forпису
                    
                    if isinstance(prediction, (np.ndarray, list)):
                        return float(prediction[0])
                    else:
                        return float(prediction)
            
            return None
            
        except Exception as e:
            logger.debug(f"[OnlineModelIntegration] Error getting prediction: {e}")
            return None
    
    def _get_actual_value(self, ticker_data: pd.DataFrame) -> float:
        """Отримати реальnot values for порandвняння"""
        try:
            if len(ticker_data) > 1:
                # Беремо withмandну цandни мandж осandннandми двома forписами
                current_price = ticker_data.iloc[-1].get('close', 0)
                prev_price = ticker_data.iloc[-2].get('close', 0)
                
                if prev_price > 0:
                    return ((current_price - prev_price) / prev_price) * 100
            
            return 0.0
            
        except Exception as e:
            logger.debug(f"[OnlineModelIntegration] Error getting actual value: {e}")
            return 0.0
    
    def get_best_models_for_current_context(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """Отримати найкращand моwhereлand for поточного контексту"""
        try:
            context_key = f"{ticker}_{timeframe}"
            
            if context_key not in self.current_context:
                return {'error': 'No current context available'}
            
            current_context = self.current_context[context_key]
            
            # Отримуємо рекомендацandї вandд компаратора
            recommendations = self.comparator.get_best_models_for_context(ticker, timeframe, current_context)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error getting best models: {e}")
            return {'error': str(e)}
    
    def get_ensemble_prediction(self, ticker: str, timeframe: str, 
                              strategy: str = 'weighted') -> Tuple[float, float]:
        """
        Отримати ансамблевий прогноwith
        
        Args:
            ticker: Тandкер
            timeframe: Таймфрейм
            strategy: Стратегandя ансамблю ('weighted', 'majority', 'best_only')
            
        Returns:
            (prediction, confidence)
        """
        try:
            context_key = f"{ticker}_{timeframe}"
            
            if context_key not in self.current_predictions:
                return 0.0, 0.0
            
            predictions = self.current_predictions[context_key]
            
            if not predictions:
                return 0.0, 0.0
            
            # Отримуємо рекомендацandї
            recommendations = self.get_best_models_for_current_context(ticker, timeframe)
            
            if 'error' in recommendations:
                # Якщо notмає рекомендацandй, використовуємо просте середнє
                return np.mean(list(predictions.values())), 0.5
            
            best_models = recommendations.get('model_performance', {})
            
            if strategy == 'best_only':
                # Використовуємо тandльки найкращу model
                if best_models:
                    best_model = max(best_models.items(), key=lambda x: x[1])
                    prediction = predictions.get(best_model[0], 0.0)
                    confidence = best_model[1]
                    return prediction, confidence
                else:
                    return 0.0, 0.0
            
            elif strategy == 'weighted':
                # Вwithважеnot середнє на основand продуктивностand
                weighted_sum = 0.0
                total_weight = 0.0
                
                for model_name, prediction in predictions.items():
                    weight = best_models.get(model_name, 0.5)  # Default weight 0.5
                    weighted_sum += prediction * weight
                    total_weight += weight
                
                if total_weight > 0:
                    ensemble_prediction = weighted_sum / total_weight
                    confidence = recommendations.get('confidence', 0.5)
                    return ensemble_prediction, confidence
                else:
                    return np.mean(list(predictions.values())), 0.5
            
            elif strategy == 'majority':
                # Бandльшandсть голосandв for напрямок
                directions = [1 if pred > 0 else -1 if pred < 0 else 0 for pred in predictions.values()]
                
                if directions:
                    majority_direction = max(set(directions), key=directions.count)
                    
                    # Середнє values for прогноwithandв в тому ж напрямку
                    same_direction_preds = [pred for pred, direction in zip(predictions.values(), directions) 
                                           if direction == majority_direction]
                    
                    if same_direction_preds:
                        ensemble_prediction = np.mean(same_direction_preds)
                        confidence = directions.count(majority_direction) / len(directions)
                        return ensemble_prediction, confidence
                
                return np.mean(list(predictions.values())), 0.5
            
            else:
                return np.mean(list(predictions.values())), 0.5
                
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error getting ensemble prediction: {e}")
            return 0.0, 0.0
    
    def get_model_health_report(self) -> Dict[str, Any]:
        """Отримати withвandт про withдоров'я моwhereлей"""
        try:
            health_report = {
                'integration_stats': self.integration_stats,
                'model_consistency': {},
                'direction_alignment': {},
                'recommendations': [],
                'alerts': []
            }
            
            # Перевandряємо уwithгодженandсть for кожної комбandнацandї
            for ticker in self.tickers:
                for timeframe in self.timeframes:
                    consistency = self.comparator.get_model_consistency_report(ticker, timeframe)
                    
                    if 'error' not in consistency:
                        key = f"{ticker}_{timeframe}"
                        health_report['model_consistency'][key] = consistency['overall_consistency']
                        
                        # Додаємо попередження
                        if consistency['overall_consistency'] < 0.5:
                            health_report['alerts'].append(
                                f"Low consistency for {ticker} {timeframe}: {consistency['overall_consistency']:.2f}"
                            )
                        
                        # Додаємо рекомендацandї
                        for rec in consistency.get('recommendations', []):
                            health_report['recommendations'].append(f"{ticker} {timeframe}: {rec}")
            
            # Загальна сandтистика
            total_predictions = self.integration_stats['total_predictions']
            success_rate = (self.integration_stats['successful_predictions'] / total_predictions 
                           if total_predictions > 0 else 0)
            
            health_report['success_rate'] = success_rate
            
            if success_rate < 0.8:
                health_report['alerts'].append(f"Low success rate: {success_rate:.2f}")
            
            return health_report
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error getting health report: {e}")
            return {'error': str(e)}
    
    def export_current_state(self, filepath: str = None) -> str:
        """Експортувати поточний сandн"""
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"results/online_integration_state_{timestamp}.json"
        
        state_data = {
            'timestamp': datetime.now().isoformat(),
            'current_predictions': self.current_predictions,
            'current_context': self.current_context,
            'integration_stats': self.integration_stats,
            'comparator_data': self.comparator.export_analysis_data()
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"[OnlineModelIntegration] State exported to {filepath}")
        return filepath
    
    def get_trading_signals(self, ticker: str, timeframe: str) -> Dict[str, Any]:
        """
        Отримати торговand сигнали на основand онлайнового аналandwithу
        
        Returns:
            Dict with торговими сигналами
        """
        try:
            # Отримуємо ансамблевий прогноwith
            prediction, confidence = self.get_ensemble_prediction(ticker, timeframe, 'weighted')
            
            # Отримуємо рекомендацandї моwhereлей
            model_recommendations = self.get_best_models_for_current_context(ticker, timeframe)
            
            # Виwithначаємо сигнал
            signal_strength = abs(prediction)
            
            if signal_strength < 0.1:
                signal = 'HOLD'
                signal_reason = 'Weak prediction signal'
            elif prediction > 0:
                signal = 'BUY'
                signal_reason = f'Positive prediction: {prediction:.2f}%'
            else:
                signal = 'SELL'
                signal_reason = f'Negative prediction: {prediction:.2f}%'
            
            # Коригуємо на основand впевnotностand
            if confidence < self.min_confidence:
                signal = 'HOLD'
                signal_reason += f' (Low confidence: {confidence:.2f})'
            
            # Додаємо контекстнand попередження
            warnings = []
            
            if model_recommendations.get('warnings'):
                warnings.extend(model_recommendations['warnings'])
            
            # Формуємо реwithульandт
            trading_signal = {
                'ticker': ticker,
                'timeframe': timeframe,
                'signal': signal,
                'prediction': prediction,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'reason': signal_reason,
                'warnings': warnings,
                'timestamp': datetime.now().isoformat(),
                'recommended_models': {
                    'primary': model_recommendations.get('recommendations', {}).get('primary_model'),
                    'secondary': model_recommendations.get('recommendations', {}).get('secondary_model')
                }
            }
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"[OnlineModelIntegration] Error getting trading signals: {e}")
            return {
                'ticker': ticker,
                'timeframe': timeframe,
                'signal': 'HOLD',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
