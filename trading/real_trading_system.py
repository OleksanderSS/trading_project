#!/usr/bin/env python3
"""
Real Trading System - Система реального трейдингу з віртуальним рахунком
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
from pathlib import Path

from trading.real_data_collector import RealDataCollector
from trading.virtual_portfolio import VirtualPortfolio
# from core.analysis.smart_switcher import SmartSwitcher  # TODO: fix missing module
# from flexible_feature_selection import FlexibleFeatureSelection  # TODO: fix missing module

logger = logging.getLogger(__name__)


class RealTradingSystem:
    """
    Система реального трейдингу з віртуальним рахунком
    Використовує реальні дані, але віртуальні гроші
    """
    
    def __init__(self, initial_balance: float = 10000.0, portfolio_name: str = "real_trader"):
        # Компоненти системи
        self.data_collector = RealDataCollector()
        self.portfolio = VirtualPortfolio(initial_balance, portfolio_name)
        self.smart_switcher = SmartSwitcher()
        self.feature_selector = FlexibleFeatureSelection()
        
        # Налаштування системи
        self.trading_hours = {
            'market_open': datetime.now().replace(hour=9, minute=30),
            'market_close': datetime.now().replace(hour=16, minute=0)
        }
        
        # Моніторинг
        self.last_analysis_time = datetime.now()
        self.analysis_interval = timedelta(minutes=15)  # Аналіз кожні 15 хвилин
        self.last_portfolio_update = datetime.now()
        self.portfolio_update_interval = timedelta(minutes=5)  # Оновлення кожні 5 хвилин
        
        # Статистика
        self.session_stats = {
            'start_time': datetime.now(),
            'analyses_performed': 0,
            'trades_executed': 0,
            'signals_generated': 0,
            'errors': 0
        }
        
        logger.info(f"[START] Real Trading System initialized with ${initial_balance:,.2f}")
    
    def is_market_open(self) -> bool:
        """
        Перевірка чи ринок відкритий
        """
        now = datetime.now()
        
        # Перевірка дня тижня (Mon-Fri)
        if now.weekday() >= 5:  # Weekend
            return False
        
        # Перевірка часу
        market_open = now.replace(hour=9, minute=30)
        market_close = now.replace(hour=16, minute=0)
        
        return market_open <= now <= market_close
    
    def run_trading_session(self, duration_hours: int = 8) -> Dict:
        """
        Запуск торгової сесії
        """
        logger.info(f"[TARGET] Starting trading session for {duration_hours} hours...")
        
        session_start = datetime.now()
        session_end = session_start + timedelta(hours=duration_hours)
        
        session_results = {
            'session_start': session_start.isoformat(),
            'session_end': session_end.isoformat(),
            'duration_hours': duration_hours,
            'trades': [],
            'analyses': [],
            'signals': [],
            'performance': {},
            'errors': []
        }
        
        try:
            while datetime.now() < session_end:
                # Перевірка чи ринок відкритий
                if not self.is_market_open():
                    logger.info("📴 Market is closed. Waiting...")
                    time.sleep(300)  # Чекаємо 5 хвилин
                    continue
                
                # Оновлення data портфеля
                if datetime.now() - self.last_portfolio_update >= self.portfolio_update_interval:
                    self._update_portfolio_performance()
                    self.last_portfolio_update = datetime.now()
                
                # Аналіз ринку та генерація сигналів
                if datetime.now() - self.last_analysis_time >= self.analysis_interval:
                    analysis_result = self._perform_market_analysis()
                    session_results['analyses'].append(analysis_result)
                    self.last_analysis_time = datetime.now()
                    
                    # Обробка сигналів
                    signals = self._process_analysis_result(analysis_result)
                    session_results['signals'].extend(signals)
                    
                    # Виконання торгів
                    for signal in signals:
                        trade_result = self._execute_signal(signal)
                        if trade_result['success']:
                            session_results['trades'].append(trade_result)
                            self.session_stats['trades_executed'] += 1
                        else:
                            session_results['errors'].append(trade_result)
                            self.session_stats['errors'] += 1
                
                # Перевірка stop loss/take profit
                self._check_exit_signals()
                
                # Невелика затримка
                time.sleep(60)  # 1 хвилина
            
            # Фінальне оновлення
            self._update_portfolio_performance()
            
            # Збір фінальних результатів
            session_results['performance'] = self._get_session_performance()
            session_results['session_stats'] = self.session_stats
            
            logger.info(f"[OK] Trading session completed successfully")
            
        except Exception as e:
            logger.error(f"[ERROR] Trading session failed: {e}")
            session_results['error'] = str(e)
        
        return session_results
    
    def _perform_market_analysis(self) -> Dict:
        """
        Виконання аналізу ринку
        """
        try:
            logger.info("[SEARCH] Performing market analysis...")
            
            # Отримання комплексних data
            market_data = self.data_collector.get_comprehensive_market_data()
            
            # Підготовка data для Smart Switcher
            analysis_data = self._prepare_analysis_data(market_data)
            
            # Використання Smart Switcher для вибору найкращих комбінацій
            best_combinations = self.smart_switcher.select_best_combinations(
                analysis_data, 
                top_k=5  # Топ-5 комбінацій
            )
            
            # Додавання контексту
            enriched_combinations = self._add_context_to_combinations(
                best_combinations, 
                market_data
            )
            
            self.session_stats['analyses_performed'] += 1
            
            return {
                'timestamp': datetime.now().isoformat(),
                'market_data_summary': {
                    'tickers_analyzed': len(market_data.get('price_data', {})),
                    'news_count': len(market_data.get('news_data', {}).get('all_news', [])),
                    'macro_indicators': len(market_data.get('macro_data', []))
                },
                'best_combinations': enriched_combinations,
                'market_sentiment': self.data_collector.get_market_sentiment()
            }
            
        except Exception as e:
            logger.error(f"[ERROR] Market analysis failed: {e}")
            return {'error': str(e)}
    
    def _prepare_analysis_data(self, market_data: Dict) -> pd.DataFrame:
        """
        Підготовка data для аналізу
        """
        analysis_rows = []
        
        price_data = market_data.get('price_data', {})
        
        for ticker, df in price_data.items():
            if df.empty:
                continue
            
            # Отримуємо останні дані
            latest_data = df.iloc[-1]
            
            # Створення рядка для аналізу
            row_data = {
                'ticker': ticker,
                'timestamp': datetime.now(),
                'price': latest_data.get('close', 0),
                'volume': latest_data.get('volume', 0),
                'rsi': latest_data.get('rsi', 50),
                'macd': latest_data.get('macd', 0),
                'price_change': latest_data.get('price_change', 0),
                'vix': market_data.get('indices_data', {}).get('^VIX', {}).iloc[-1, 0] if not market_data.get('indices_data', {}).get('^VIX', pd.DataFrame()).empty else 20
            }
            
            # Додавання технічних індикаторів
            if 'ma_20' in latest_data:
                row_data['ma_20'] = latest_data['ma_20']
            if 'ma_50' in latest_data:
                row_data['ma_50'] = latest_data['ma_50']
            if 'bb_upper' in latest_data:
                row_data['bb_upper'] = latest_data['bb_upper']
            if 'bb_lower' in latest_data:
                row_data['bb_lower'] = latest_data['bb_lower']
            
            analysis_rows.append(row_data)
        
        return pd.DataFrame(analysis_rows)
    
    def _add_context_to_combinations(self, combinations: List[Dict], 
                                   market_data: Dict) -> List[Dict]:
        """
        Додавання контексту до комбінацій
        """
        enriched_combinations = []
        
        for combo in combinations:
            enriched_combo = combo.copy()
            
            # Додавання ринкового сентименту
            sentiment = market_data.get('market_sentiment', {})
            enriched_combo['market_sentiment'] = sentiment.get('overall_sentiment', 0.5)
            
            # Додавання волатильності (VIX)
            vix_data = market_data.get('indices_data', {}).get('^VIX', pd.DataFrame())
            if not vix_data.empty:
                enriched_combo['vix'] = vix_data.iloc[-1, 0]
            
            # Додавання новинного контексту
            news_data = market_data.get('news_data', {}).get('all_news', pd.DataFrame())
            if not news_data.empty:
                enriched_combo['recent_news_count'] = len(news_data)
                
                # Сентимент новин
                if 'sentiment_score' in news_data.columns:
                    enriched_combo['news_sentiment'] = news_data['sentiment_score'].mean()
            
            # Додавання макро контексту
            macro_data = market_data.get('macro_data', pd.DataFrame())
            if not macro_data.empty:
                enriched_combo['macro_context'] = 'normal'
                # Логіка для визначення макро контексту
                if 'fedfunds' in macro_data.columns:
                    fed_funds = macro_data['fedfunds'].iloc[-1]
                    if fed_funds > 4.0:
                        enriched_combo['macro_context'] = 'tight'
                    elif fed_funds < 2.0:
                        enriched_combo['macro_context'] = 'loose'
            
            enriched_combinations.append(enriched_combo)
        
        return enriched_combinations
    
    def _process_analysis_result(self, analysis_result: Dict) -> List[Dict]:
        """
        Обробка результатів аналізу та генерація сигналів
        """
        signals = []
        
        if 'error' in analysis_result:
            return signals
        
        best_combinations = analysis_result.get('best_combinations', [])
        
        for combo in best_combinations:
            # Перевірка чи комбінація достатньо хороша
            confidence = combo.get('confidence', 0)
            accuracy = combo.get('accuracy', 0)
            
            # Мінімальні пороги
            if confidence < 0.7 or accuracy < 0.7:
                continue
            
            # Створення сигналу
            signal = {
                'timestamp': datetime.now().isoformat(),
                'ticker': combo.get('ticker'),
                'signal_type': combo.get('signal', 'HOLD'),
                'confidence': confidence,
                'accuracy': accuracy,
                'model': combo.get('model'),
                'target': combo.get('target'),
                'timeframe': combo.get('timeframe'),
                'expected_return': combo.get('expected_return', 0),
                'risk_score': combo.get('risk_score', 0.5),
                'reason': self._generate_signal_reason(combo),
                'market_sentiment': combo.get('market_sentiment', 0.5),
                'vix': combo.get('vix', 20)
            }
            
            signals.append(signal)
            self.session_stats['signals_generated'] += 1
        
        return signals
    
    def _generate_signal_reason(self, combo: Dict) -> str:
        """
        Генерація причини сигналу
        """
        reasons = []
        
        if combo.get('accuracy', 0) > 0.8:
            reasons.append(f"High accuracy ({combo.get('accuracy', 0):.1%})")
        
        if combo.get('confidence', 0) > 0.8:
            reasons.append(f"High confidence ({combo.get('confidence', 0):.1%})")
        
        if combo.get('expected_return', 0) > 0.02:
            reasons.append(f"Positive expected return ({combo.get('expected_return', 0):.1%})")
        
        if combo.get('market_sentiment', 0.5) > 0.6:
            reasons.append("Bullish market sentiment")
        elif combo.get('market_sentiment', 0.5) < 0.4:
            reasons.append("Bearish market sentiment")
        
        if combo.get('vix', 20) < 15:
            reasons.append("Low volatility")
        elif combo.get('vix', 20) > 25:
            reasons.append("High volatility")
        
        return "; ".join(reasons) if reasons else "Model recommendation"
    
    def _execute_signal(self, signal: Dict) -> Dict:
        """
        Виконання торгового сигналу
        """
        try:
            ticker = signal['ticker']
            signal_type = signal['signal_type']
            confidence = signal['confidence']
            
            # Отримання поточної ціни
            current_price = self.data_collector.get_current_price(ticker)
            if current_price is None:
                return {'success': False, 'error': f'No price data for {ticker}'}
            
            # Розрахунок розміру позиції
            position_size = self.portfolio.get_position_size(
                ticker, current_price, confidence
            )
            
            # Виконання операції
            if signal_type == 'BUY':
                result = self.portfolio.buy_stock(
                    ticker=ticker,
                    quantity=position_size,
                    price=current_price,
                    reason=signal['reason'],
                    confidence=confidence
                )
            elif signal_type == 'SELL':
                # Перевірка чи є позиція для продажу
                if ticker in self.portfolio.positions:
                    position = self.portfolio.positions[ticker]
                    quantity = min(position_size, position['quantity'])
                    
                    result = self.portfolio.sell_stock(
                        ticker=ticker,
                        quantity=quantity,
                        price=current_price,
                        reason=signal['reason']
                    )
                else:
                    result = {'success': False, 'error': f'No position to sell for {ticker}'}
            else:
                result = {'success': False, 'error': f'Unknown signal type: {signal_type}'}
            
            if result['success']:
                logger.info(f"[OK] Executed {signal_type} for {ticker}: {position_size} shares at ${current_price:.2f}")
            else:
                logger.warning(f"[WARN] Failed to execute {signal_type} for {ticker}: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            logger.error(f"[ERROR] Error executing signal: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_exit_signals(self):
        """
        Перевірка сигналів виходу (stop loss, take profit)
        """
        try:
            # Отримання поточних цін
            current_prices = {}
            
            for ticker in self.portfolio.positions.keys():
                price = self.data_collector.get_current_price(ticker)
                if price is not None:
                    current_prices[ticker] = price
            
            if not current_prices:
                return
            
            # Перевірка stop loss/take profit
            exit_signals = self.portfolio.check_stop_loss_take_profit(current_prices)
            
            for signal in exit_signals:
                ticker = signal['ticker']
                price = signal['price']
                reason = signal['reason']
                
                if ticker in self.portfolio.positions:
                    position = self.portfolio.positions[ticker]
                    quantity = position['quantity']
                    
                    result = self.portfolio.sell_stock(
                        ticker=ticker,
                        quantity=quantity,
                        price=price,
                        reason=reason
                    )
                    
                    if result['success']:
                        logger.info(f"[OK] Exit signal executed for {ticker}: {reason}")
                    else:
                        logger.warning(f"[WARN] Failed to execute exit signal for {ticker}: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error checking exit signals: {e}")
    
    def _update_portfolio_performance(self):
        """
        Оновлення продуктивності портфеля
        """
        try:
            # Отримання поточних цін
            current_prices = {}
            
            for ticker in self.portfolio.positions.keys():
                price = self.data_collector.get_current_price(ticker)
                if price is not None:
                    current_prices[ticker] = price
            
            # Оновлення продуктивності
            self.portfolio.update_performance(current_prices)
            
        except Exception as e:
            logger.error(f"[ERROR] Error updating portfolio performance: {e}")
    
    def _get_session_performance(self) -> Dict:
        """
        Отримання продуктивності сесії
        """
        try:
            # Отримання поточних цін
            current_prices = {}
            
            for ticker in self.portfolio.positions.keys():
                price = self.data_collector.get_current_price(ticker)
                if price is not None:
                    current_prices[ticker] = price
            
            # Резюме портфеля
            portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
            
            # Додавання статистики сесії
            portfolio_summary['session_stats'] = self.session_stats
            
            return portfolio_summary
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting session performance: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """
        Отримання статусу системи
        """
        try:
            # Отримання поточних цін
            current_prices = {}
            
            for ticker in self.portfolio.positions.keys():
                price = self.data_collector.get_current_price(ticker)
                if price is not None:
                    current_prices[ticker] = price
            
            # Статус портфеля
            portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
            
            # Статус системи
            system_status = {
                'system_name': 'Real Trading System',
                'version': '1.0.0',
                'market_open': self.is_market_open(),
                'last_analysis': self.last_analysis_time.isoformat(),
                'last_portfolio_update': self.last_portfolio_update.isoformat(),
                'session_stats': self.session_stats,
                'portfolio': portfolio_summary
            }
            
            return system_status
            
        except Exception as e:
            logger.error(f"[ERROR] Error getting system status: {e}")
            return {'error': str(e)}
    
    def save_session_results(self, session_results: Dict, filename: str = None):
        """
        Збереження результатів сесії
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"trading_session_{timestamp}.json"
            
            # Створення директорії
            results_dir = Path("data/trading_results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Збереження
            filepath = results_dir / filename
            with open(filepath, 'w') as f:
                json.dump(session_results, f, indent=2, default=str)
            
            logger.info(f"[SAVE] Session results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"[ERROR] Error saving session results: {e}")
    
    def load_session_results(self, filename: str) -> Dict:
        """
        Завантаження результатів сесії
        """
        try:
            filepath = Path("data/trading_results") / filename
            
            with open(filepath, 'r') as f:
                session_results = json.load(f)
            
            logger.info(f"[OK] Session results loaded from {filepath}")
            return session_results
            
        except Exception as e:
            logger.error(f"[ERROR] Error loading session results: {e}")
            return {}
