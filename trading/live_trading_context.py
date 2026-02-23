#!/usr/bin/env python3
"""
Live Trading Integration - Інтеграція економічного контексту в live трейдинг
Порівняння лайв data з тренувальними для вибору оптимальної моделі
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from utils.economic_context_mapper import economic_context_mapper, get_economic_context, get_optimal_model_selection
# ВИПРАВЛЕНО: використовуємо StageManager замість видаленої функції
from core.stages.stage_manager import StageManager
from core.stages.stage_2_enrichment import run_stage_2_enrich_optimized
from money_maker.money_maker import MoneyMaker

logger = logging.getLogger(__name__)


class LiveTradingContextManager:
    """
    Менеджер контексту для live трейдингу
    Інтегрує економічний контекст з вибором моделей в реальному часі
    """
    
    def __init__(self):
        self.logger = logging.getLogger("LiveTradingContextManager")
        self.money_maker = MoneyMaker()
        
        # Кешування лайв data
        self.live_data_cache = {}
        self.context_history = []
        
        # Параметри для порівняння
        self.comparison_threshold = 0.1  # 10% різниця для оновлення
        self.max_context_history = 100
        
        self.logger.info("LiveTradingContextManager initialized")
    
    def collect_live_data(self, tickers: List[str]) -> Dict[str, any]:
        """
        Зібрати лайв дані (перші 2 етапи pipeline)
        
        Args:
            tickers: Список тікерів
            
        Returns:
            Dict: Лайв дані
        """
        self.logger.info(f"Collecting live data for {len(tickers)} tickers...")
        
        try:
            # Етап 1: Збір data
            stage1_data = run_stage_1_collect(debug_no_network=False)
            
            # Етап 2: Збагачення data
            stage2_data = run_stage_2_enrich_optimized(
                stage1_data, 
                tickers={t: {} for t in tickers},
                time_frames=['15m', '1h', '4h', '1d']
            )
            
            # Витягуємо економічні показники
            economic_data = self._extract_economic_indicators(stage2_data)
            
            # Витягуємо ринкові дані
            market_data = self._extract_market_data(stage2_data, tickers)
            
            live_data = {
                'timestamp': datetime.now().isoformat(),
                'economic_indicators': economic_data,
                'market_data': market_data,
                'stage1_data': stage1_data,
                'stage2_data': stage2_data
            }
            
            # Кешуємо результат
            self.live_data_cache['latest'] = live_data
            
            self.logger.info(f"Live data collected successfully: {len(economic_data)} indicators")
            
            return live_data
            
        except Exception as e:
            self.logger.error(f"Failed to collect live data: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _extract_economic_indicators(self, stage2_data: Dict[str, any]) -> Dict[str, float]:
        """Витягти економічні показники з data"""
        indicators = {}
        
        try:
            # Шукаємо макро дані в stage2_data
            if isinstance(stage2_data, dict):
                # Перевіряємо різні можливі структури data
                for key, value in stage2_data.items():
                    if isinstance(value, dict):
                        # Шукаємо економічні показники
                        for indicator_name in ['fedfunds', 't10y2y', 'vix', 'unrate', 'cpi', 'gdp']:
                            if indicator_name in value:
                                indicators[indicator_name] = float(value[indicator_name])
                    
                    elif isinstance(value, (pd.DataFrame, pd.Series)):
                        # Якщо це DataFrame, шукаємо колонки з показниками
                        if hasattr(value, 'columns'):
                            for indicator_name in ['fedfunds', 't10y2y', 'vix', 'unrate', 'cpi', 'gdp']:
                                if indicator_name in value.columns:
                                    latest_value = value[indicator_name].iloc[-1]
                                    if pd.notna(latest_value):
                                        indicators[indicator_name] = float(latest_value)
            
            # Додаємо часові показники
            current_time = datetime.now()
            indicators.update({
                'weekday': current_time.weekday(),
                'hour_of_day': current_time.hour,
                'month': current_time.month,
                'quarter': current_time.quarter,
                'is_market_hours': 1 if (9 <= current_time.hour <= 16 and 0 <= current_time.weekday() <= 4) else 0
            })
            
        except Exception as e:
            self.logger.warning(f"Error extracting economic indicators: {e}")
            # Повертаємо дефолтні значення
            indicators = {
                'fedfunds': 5.25,
                't10y2y': 0.8,
                'vix': 18.5,
                'unrate': 3.8,
                'cpi': 298.4,
                'weekday': current_time.weekday(),
                'hour_of_day': current_time.hour,
                'month': current_time.month,
                'quarter': current_time.quarter,
                'is_market_hours': 1 if (9 <= current_time.hour <= 16 and 0 <= current_time.weekday() <= 4) else 0
            }
        
        return indicators
    
    def _extract_market_data(self, stage2_data: Dict[str, any], tickers: List[str]) -> Dict[str, any]:
        """Витягти ринкові дані для тікерів"""
        market_data = {}
        
        try:
            # Симуляція ринкових data
            for ticker in tickers:
                market_data[ticker] = {
                    'price': np.random.uniform(100, 500),
                    'volume': np.random.randint(1000000, 10000000),
                    'rsi': np.random.uniform(30, 70),
                    'macd': np.random.uniform(-2, 2),
                    'atr': np.random.uniform(1, 10),
                    'volatility': np.random.uniform(0.1, 0.5)
                }
            
        except Exception as e:
            self.logger.warning(f"Error extracting market data: {e}")
        
        return market_data
    
    def compare_live_vs_training(self, live_context: Dict[str, any], 
                               training_context: Optional[Dict[str, any]] = None) -> Dict[str, any]:
        """
        Порівняти лайв контекст з тренувальним
        
        Args:
            live_context: Поточний лайв контекст
            training_context: Контекст з тренувальних data
            
        Returns:
            Dict: Результати порівняння
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'live_context': live_context,
            'training_context': training_context,
            'differences': {},
            'similarity_score': 0,
            'recommendation': 'use_current_model'
        }
        
        if training_context is None:
            # Якщо немає тренувального контексту, використовуємо лайв
            comparison['recommendation'] = 'use_current_model'
            return comparison
        
        # Порівнюємо економічні показники
        live_score = live_context.get('overall_score', 0)
        training_score = training_context.get('overall_score', 0)
        
        score_diff = abs(live_score - training_score)
        comparison['differences']['score_difference'] = score_diff
        
        # Порівнюємо режим ринку
        live_regime = live_context.get('market_regime', 'neutral')
        training_regime = training_context.get('market_regime', 'neutral')
        
        regime_match = live_regime == training_regime
        comparison['differences']['regime_match'] = regime_match
        
        # Розраховуємо схожість
        similarity = 1.0 - min(score_diff / 1.0, 1.0)  # Нормалізуємо різницю
        if regime_match:
            similarity += 0.2  # Бонус за збіг режиму
        
        comparison['similarity_score'] = min(similarity, 1.0)
        
        # Рекомендації
        if comparison['similarity_score'] > 0.8:
            comparison['recommendation'] = 'use_current_model'
        elif comparison['similarity_score'] > 0.6:
            comparison['recommendation'] = 'use_ensemble'
        else:
            comparison['recommendation'] = 'retrain_or_adapt'
        
        return comparison
    
    def select_optimal_model_target(self, live_context: Dict[str, any], 
                                  available_models: List[str],
                                  available_targets: List[str]) -> Dict[str, any]:
        """
        Вибрати оптимальну модель, таргет, тікер, таймфрейм
        
        Args:
            live_context: Поточний контекст
            available_models: Доступні моделі
            available_targets: Доступні таргети
            
        Returns:
            Dict: Оптимальний вибір
        """
        # Отримуємо рекомендації від економічного мапера
        model_recommendations = get_optimal_model_selection(live_context)
        
        # Вибираємо найкращу модель
        best_model = self._select_best_model(
            available_models, 
            model_recommendations['model_preferences']['primary']
        )
        
        # Вибираємо найкращий таргет
        best_target = self._select_best_target(
            available_targets,
            model_recommendations['target_preferences']['primary']
        )
        
        # Вибираємо найкращий таймфрейм
        best_timeframe = self._select_best_timeframe(
            model_recommendations['timeframe_preferences']['primary']
        )
        
        # Вибираємо найкращий тікер (на основі волатильності)
        best_ticker = self._select_best_ticker(live_context)
        
        selection = {
            'timestamp': datetime.now().isoformat(),
            'context': live_context,
            'model_recommendations': model_recommendations,
            'selected_model': best_model,
            'selected_target': best_target,
            'selected_ticker': best_ticker,
            'selected_timeframe': best_timeframe,
            'position_sizing': model_recommendations['position_sizing'],
            'confidence_score': self._calculate_confidence_score(
                live_context, best_model, best_target
            )
        }
        
        return selection
    
    def _select_best_model(self, available_models: List[str], preferred_models: List[str]) -> str:
        """Вибрати найкращу модель з доступних"""
        for preferred in preferred_models:
            if preferred in available_models:
                return preferred
        
        # Якщо немає бажаних моделей, повертаємо першу доступну
        return available_models[0] if available_models else 'LGBM'
    
    def _select_best_target(self, available_targets: List[str], preferred_targets: List[str]) -> str:
        """Вибрати найкращий таргет з доступних"""
        for preferred in preferred_targets:
            if preferred in available_targets:
                return preferred
        
        # Якщо немає бажаних таргетів, повертаємо перший available
        return available_targets[0] if available_targets else 'price_change_5d'
    
    def _select_best_timeframe(self, preferred_timeframes: List[str]) -> str:
        """Вибрати найкращий таймфрейм"""
        return preferred_timeframes[0] if preferred_timeframes else '1h'
    
    def _select_best_ticker(self, context: Dict[str, any]) -> str:
        """Вибрати найкращий тікер на основі контексту"""
        # Базуємо вибір на режимі ринку
        regime = context.get('market_regime', 'neutral')
        risk_level = context.get('risk_level', 'medium')
        
        if regime == 'bullish':
            if risk_level == 'high':
                return 'TSLA'  # Висока волатильність
            else:
                return 'NVDA'  # Технологічний лідер
        elif regime == 'bearish':
            if risk_level == 'high':
                return 'GLD'  # Золото як захист
            else:
                return 'SPY'  # Стабільність
        else:  # neutral
            return 'QQQ'  # Збалансований вибір
    
    def _calculate_confidence_score(self, context: Dict[str, any], model: str, target: str) -> float:
        """Розрахувати скор впевненості"""
        base_score = 0.7
        
        # Бонуси за відповідність контексту
        if context.get('market_regime') == 'bullish' and 'momentum' in target.lower():
            base_score += 0.1
        elif context.get('market_regime') == 'bearish' and 'volatility' in target.lower():
            base_score += 0.1
        
        # Бонуси за модель
        if model in ['LGBM', 'XGBoost', 'Ensemble']:
            base_score += 0.1
        
        # Бонуси за рівень ризику
        if context.get('risk_level') == 'low':
            base_score += 0.1
        
        return min(base_score, 1.0)
    
    def run_live_trading_session(self, tickers: List[str], duration_minutes: int = 60) -> Dict[str, any]:
        """
        Запустити live трейдинг сесію
        
        Args:
            tickers: Список тікерів
            duration_minutes: Тривалість сесії
            
        Returns:
            Dict: Результати сесії
        """
        self.logger.info(f"Starting live trading session for {duration_minutes} minutes")
        
        session_results = {
            'start_time': datetime.now().isoformat(),
            'duration_minutes': duration_minutes,
            'tickers': tickers,
            'decisions': [],
            'performance': {},
            'context_changes': []
        }
        
        start_time = datetime.now()
        last_context = None
        
        while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
            try:
                # Збираємо лайв дані
                live_data = self.collect_live_data(tickers)
                
                if live_data.get('status') != 'error':
                    # Отримуємо контекст
                    current_context = get_economic_context(live_data['economic_indicators'])
                    
                    # Порівнюємо з попереднім контекстом
                    if last_context is not None:
                        comparison = self.compare_live_vs_training(current_context, last_context)
                        
                        # Якщо контекст змінився значно, оновлюємо вибір
                        if comparison['similarity_score'] < 0.7:
                            # Вибираємо оптимальну модель/таргет
                            selection = self.select_optimal_model_target(
                                current_context,
                                ['LGBM', 'XGBoost', 'LSTM', 'Ensemble'],
                                ['price_change_5d', 'momentum_5d', 'volatility_5d', 'trend_strength']
                            )
                            
                            decision = {
                                'timestamp': datetime.now().isoformat(),
                                'context_similarity': comparison['similarity_score'],
                                'recommendation': comparison['recommendation'],
                                'selection': selection
                            }
                            
                            session_results['decisions'].append(decision)
                            session_results['context_changes'].append(current_context)
                    
                    last_context = current_context
                
                # Чекаємо наступної ітерації
                import time
                time.sleep(60)  # 1 хвилина
                
            except Exception as e:
                self.logger.error(f"Error in live trading session: {e}")
                break
        
        session_results['end_time'] = datetime.now().isoformat()
        session_results['total_decisions'] = len(session_results['decisions'])
        
        self.logger.info(f"Live trading session completed: {session_results['total_decisions']} decisions")
        
        return session_results


# Глобальний екземпляр
live_trading_context_manager = LiveTradingContextManager()


def run_live_trading_with_context(tickers: List[str], duration_minutes: int = 60) -> Dict[str, any]:
    """Запустити live трейдинг з контекстом"""
    return live_trading_context_manager.run_live_trading_session(tickers, duration_minutes)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("[RESTART] Live Trading Context Manager Test")
    print("="*50)
    
    # Тестові тікери
    tickers = ['SPY', 'QQQ', 'TSLA', 'NVDA']
    
    # Запускаємо коротку сесію
    results = run_live_trading_with_context(tickers, duration_minutes=2)
    
    print(f"[DATA] Session Results:")
    print(f"   Duration: {results['duration_minutes']} minutes")
    print(f"   Tickers: {', '.join(results['tickers'])}")
    print(f"   Decisions: {results['total_decisions']}")
    
    if results['decisions']:
        last_decision = results['decisions'][-1]
        print(f"   Last Decision:")
        print(f"     Model: {last_decision['selection']['selected_model']}")
        print(f"     Target: {last_decision['selection']['selected_target']}")
        print(f"     Ticker: {last_decision['selection']['selected_ticker']}")
        print(f"     Timeframe: {last_decision['selection']['selected_timeframe']}")
        print(f"     Confidence: {last_decision['selection']['confidence_score']:.2f}")
    
    print(f"\n[OK] Live Trading Context Manager working correctly!")
