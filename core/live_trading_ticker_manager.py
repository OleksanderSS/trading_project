#!/usr/bin/env python3
"""
Live Trading Ticker Manager
Інтельектуальна система вибору тікерів для лайв трейдингу
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

# Імпортуємо існуючі системи
from config.enhanced_sector_tickers import enhanced_sector_manager
from utils.news_ticker_detector import NewsTickerDetector
from utils.pipeline_optimizer import EnhancedPipelineOptimizer

logger = logging.getLogger(__name__)


@dataclass
class MarketCondition:
    """Поточні ринкові умови"""
    volatility_level: float  # 0-1
    trend_direction: str    # 'bull', 'bear', 'sideways'
    volume_level: float      # 0-1
    news_intensity: float    # 0-1
    sector_rotation: str     # 'tech', 'finance', 'energy', 'balanced'
    market_phase: str        # 'pre_market', 'regular', 'after_hours'


@dataclass
class TickerScore:
    """Оцінка тікера"""
    ticker: str
    volatility_score: float
    momentum_score: float
    news_score: float
    sector_score: float
    liquidity_score: float
    total_score: float
    recommended_position_size: float
    optimal_timeframes: List[str]


class LiveTradingTickerManager:
    """
    Інтелектуальний менеджер тікерів для лайв трейдингу
    """
    
    def __init__(self, max_tickers: int = 25, risk_tolerance: str = "medium"):
        self.max_tickers = max_tickers
        self.risk_tolerance = risk_tolerance
        self.enhanced_manager = enhanced_sector_manager
        self.news_detector = NewsTickerDetector()
        self.pipeline_optimizer = EnhancedPipelineOptimizer()
        
        # Базові налаштування
        self.core_tickers = [
            # ETF для ринку
            'SPY', 'QQQ', 'IWM', 'DIA',
            # AI/Tech лідери
            'NVDA', 'AAPL', 'MSFT', 'GOOGL', 'META', 'TSLA',
            # Напівпровідники
            'AMD', 'INTC', 'MU',
            # Фінанси
            'JPM', 'BAC', 'GS',
            # Енергія
            'XOM', 'CVX',
            # Споживчий сектор
            'WMT', 'HD', 'PG'
        ]
        
        logger.info(f"[LiveTradingTickerManager] Initialized with max_tickers={max_tickers}, risk_tolerance={risk_tolerance}")
    
    def analyze_market_conditions(self) -> MarketCondition:
        """
        Аналіз поточних ринкових умов
        """
        try:
            # Симуляція аналізу ринку (в реальності - з data)
            current_time = datetime.now()
            
            # Визначаємо фазу ринку
            if current_time.hour < 9 or (current_time.hour == 9 and current_time.minute < 30):
                market_phase = 'pre_market'
            elif current_time.hour > 16:
                market_phase = 'after_hours'
            else:
                market_phase = 'regular'
            
            # Аналіз волатильності (симуляція)
            volatility_level = np.random.uniform(0.3, 0.8)  # В реальності - з VIX
            
            # Визначаємо тренд (симуляція)
            trend_options = ['bull', 'bear', 'sideways']
            trend_weights = [0.4, 0.2, 0.4]  # В реальності - з індикаторів
            trend_direction = np.random.choice(trend_options, p=trend_weights)
            
            # Рівень обсягів (симуляція)
            volume_level = np.random.uniform(0.4, 0.9)
            
            # Інтенсивність новин (симуляція)
            news_intensity = np.random.uniform(0.2, 0.8)
            
            # Секторна ротація (симуляція)
            sector_rotation = np.random.choice(['tech', 'finance', 'energy', 'balanced'])
            
            conditions = MarketCondition(
                volatility_level=volatility_level,
                trend_direction=trend_direction,
                volume_level=volume_level,
                news_intensity=news_intensity,
                sector_rotation=sector_rotation,
                market_phase=market_phase
            )
            
            logger.info(f"[MarketAnalysis] Volatility: {volatility_level:.2f}, Trend: {trend_direction}, Phase: {market_phase}")
            return conditions
            
        except Exception as e:
            logger.error(f"[MarketAnalysis] Error analyzing market conditions: {e}")
            # Повертаємо умови за замовчуванням
            return MarketCondition(0.5, 'sideways', 0.5, 0.5, 'balanced', 'regular')
    
    def get_base_strategy_tickers(self, conditions: MarketCondition) -> List[str]:
        """
        Отримати базові тікери на основі стратегії
        """
        # Вибір стратегії на основі умов ринку
        if conditions.volatility_level > 0.7:
            strategy = "extreme_volatility"
        elif conditions.trend_direction == 'bull' and conditions.volume_level > 0.6:
            strategy = "momentum"
        elif conditions.news_intensity > 0.6:
            strategy = "news_driven"
        elif conditions.volatility_level < 0.4:
            strategy = "conservative"
        else:
            strategy = "balanced_growth"
        
        logger.info(f"[Strategy] Selected strategy: {strategy}")
        
        # Отримуємо тікери для стратегії
        base_tickers = self.enhanced_manager.get_tickers_by_strategy(strategy, limit=20)
        
        # Додаємо базові ETF
        etf_tickers = ['SPY', 'QQQ', 'IWM']
        all_tickers = list(set(base_tickers + etf_tickers))
        
        return all_tickers[:self.max_tickers]
    
    def get_trending_tickers(self, hours: int = 24) -> List[str]:
        """
        Отримати трендові тікери з новин
        """
        try:
            # Симуляція отримання трендових тікерів
            # В реальності - з аналізу новин та соціальних мереж
            trending_candidates = [
                'COIN', 'MARA', 'RIOT', 'PLTR', 'GME', 'AMC',  # High volatility
                'SNAP', 'ROKU', 'TWTR', 'DIS', 'NFLX',          # Tech/media
                'SQ', 'PYPL', 'BKNG', 'EBAY', 'SHOP'            # Fintech/ecommerce
            ]
            
            # Вибираємо випадково (в реальності - на основі аналізу)
            trending_tickers = np.random.choice(
                trending_candidates, 
                size=min(8, len(trending_candidates)), 
                replace=False
            ).tolist()
            
            logger.info(f"[Trending] Found {len(trending_tickers)} trending tickers")
            return trending_tickers
            
        except Exception as e:
            logger.error(f"[Trending] Error getting trending tickers: {e}")
            return []
    
    def score_tickers(self, tickers: List[str], conditions: MarketCondition) -> List[TickerScore]:
        """
        Оцінити тікери за різними критеріями
        """
        scores = []
        
        for ticker in tickers:
            try:
                # Волатильність (симуляція)
                if ticker in ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'GME']:
                    volatility_score = 0.9
                elif ticker in ['SPY', 'QQQ', 'IWM']:
                    volatility_score = 0.4
                else:
                    volatility_score = np.random.uniform(0.3, 0.8)
                
                # Моментум (симуляція)
                momentum_score = np.random.uniform(0.2, 0.9)
                
                # Новинний скор (симуляція)
                news_score = np.random.uniform(0.1, 0.8)
                
                # Секторний скор (симуляція)
                sector_score = np.random.uniform(0.3, 0.8)
                
                # Ліквідність (симуляція)
                if ticker in ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL']:
                    liquidity_score = 0.9
                elif ticker in ['NVDA', 'TSLA', 'META']:
                    liquidity_score = 0.8
                else:
                    liquidity_score = np.random.uniform(0.3, 0.7)
                
                # Загальний скор
                total_score = (
                    volatility_score * 0.25 +
                    momentum_score * 0.25 +
                    news_score * 0.20 +
                    sector_score * 0.15 +
                    liquidity_score * 0.15
                )
                
                # Оптимальні таймфрейми
                if volatility_score > 0.7:
                    optimal_timeframes = ['15m', '1h', '4h']
                elif volatility_score > 0.5:
                    optimal_timeframes = ['1h', '4h', '1d']
                else:
                    optimal_timeframes = ['4h', '1d', '1w']
                
                # Рекомендований розмір позиції
                if total_score > 0.8:
                    position_size = 0.1
                elif total_score > 0.6:
                    position_size = 0.08
                else:
                    position_size = 0.05
                
                scores.append(TickerScore(
                    ticker=ticker,
                    volatility_score=volatility_score,
                    momentum_score=momentum_score,
                    news_score=news_score,
                    sector_score=sector_score,
                    liquidity_score=liquidity_score,
                    total_score=total_score,
                    recommended_position_size=position_size,
                    optimal_timeframes=optimal_timeframes
                ))
                
            except Exception as e:
                logger.error(f"[Scoring] Error scoring {ticker}: {e}")
                continue
        
        return scores
    
    def optimize_for_resources(self, scores: List[TickerScore]) -> List[TickerScore]:
        """
        Оптимізувати список тікерів для ресурсів
        """
        # Сортуємо за загальним скором
        sorted_scores = sorted(scores, key=lambda x: x.total_score, reverse=True)
        
        # Обмежуємо кількість
        optimized = sorted_scores[:self.max_tickers]
        
        # Перевіряємо різноманітність секторів
        sectors = set()
        final_tickers = []
        
        for score in optimized:
            # Проста симуляція перевірки секторів
            ticker_sector = self._get_ticker_sector(score.ticker)
            
            # Додаємо якщо сектор ще не представлений або якщо це високий скор
            if ticker_sector not in sectors or score.total_score > 0.8:
                final_tickers.append(score)
                sectors.add(ticker_sector)
            
            if len(final_tickers) >= self.max_tickers:
                break
        
        logger.info(f"[Optimization] Selected {len(final_tickers)} tickers from {len(scores)} candidates")
        return final_tickers
    
    def _get_ticker_sector(self, ticker: str) -> str:
        """
        Визначити сектор тікера
        """
        sector_mapping = {
            'SPY': 'market', 'QQQ': 'market', 'IWM': 'market', 'DIA': 'market',
            'NVDA': 'tech', 'AAPL': 'tech', 'MSFT': 'tech', 'GOOGL': 'tech', 'META': 'tech',
            'TSLA': 'tech', 'AMD': 'tech', 'INTC': 'tech', 'MU': 'tech',
            'JPM': 'finance', 'BAC': 'finance', 'GS': 'finance', 'MS': 'finance',
            'XOM': 'energy', 'CVX': 'energy', 'COP': 'energy',
            'WMT': 'consumer', 'HD': 'consumer', 'PG': 'consumer',
            'COIN': 'crypto', 'MARA': 'crypto', 'RIOT': 'crypto',
            'PLTR': 'speculative', 'GME': 'speculative', 'AMC': 'speculative'
        }
        return sector_mapping.get(ticker, 'other')
    
    def get_optimal_tickers_for_live_trading(self) -> Tuple[List[str], Dict[str, Any]]:
        """
        Основний метод - отримати оптимальні тікери для лайв трейдингу
        """
        logger.info("[LiveTrading] Starting optimal ticker selection...")
        
        # 1. Аналіз ринкових умов
        conditions = self.analyze_market_conditions()
        
        # 2. Отримання базових тікерів за стратегією
        base_tickers = self.get_base_strategy_tickers(conditions)
        
        # 3. Отримання трендових тікерів
        trending_tickers = self.get_trending_tickers(hours=24)
        
        # 4. Об'єднання та видалення дублікатів
        all_candidates = list(set(base_tickers + trending_tickers))
        
        # 5. Оцінка тікерів
        scores = self.score_tickers(all_candidates, conditions)
        
        # 6. Оптимізація для ресурсів
        optimized_scores = self.optimize_for_resources(scores)
        
        # 7. Формування результату
        final_tickers = [score.ticker for score in optimized_scores]
        
        # 8. Створення звіту
        report = {
            'market_conditions': conditions,
            'strategy_used': self._get_used_strategy(conditions),
            'total_candidates': len(all_candidates),
            'selected_count': len(final_tickers),
            'selection_details': [
                {
                    'ticker': score.ticker,
                    'total_score': score.total_score,
                    'volatility_score': score.volatility_score,
                    'momentum_score': score.momentum_score,
                    'recommended_position_size': score.recommended_position_size,
                    'optimal_timeframes': score.optimal_timeframes
                }
                for score in optimized_scores
            ]
        }
        
        logger.info(f"[LiveTrading] Selected {len(final_tickers)} optimal tickers for live trading")
        return final_tickers, report
    
    def _get_used_strategy(self, conditions: MarketCondition) -> str:
        """
        Визначити використану стратегію
        """
        if conditions.volatility_level > 0.7:
            return "extreme_volatility"
        elif conditions.trend_direction == 'bull' and conditions.volume_level > 0.6:
            return "momentum"
        elif conditions.news_intensity > 0.6:
            return "news_driven"
        elif conditions.volatility_level < 0.4:
            return "conservative"
        else:
            return "balanced_growth"
    
    def get_adaptive_timeframes(self, tickers: List[str]) -> List[str]:
        """
        Отримати адаптивні таймфрейми для тікерів
        """
        has_high_volatility = any(ticker in ['TSLA', 'NVDA', 'AMD', 'COIN', 'MARA', 'GME'] for ticker in tickers)
        
        # ВИПРАВЛЕНО: Повертаємо таймфрейми що очікує YFCollector
        if has_high_volatility:
            return ['5m', '15m', '60m', '1d']  # Високоволатильні - всі таймфрейми
        else:
            return ['15m', '60m', '1d']  # Низьковолатильні - без 5m
    
    def update_tickers_during_session(self, current_tickers: List[str]) -> Tuple[List[str], List[str]]:
        """
        Оновити тікери під час сесії
        """
        # Аналіз поточних умов
        conditions = self.analyze_market_conditions()
        
        # Отримати нові оптимальні тікери
        new_tickers, _ = self.get_optimal_tickers_for_live_trading()
        
        # Визначити зміни
        added = list(set(new_tickers) - set(current_tickers))
        removed = list(set(current_tickers) - set(new_tickers))
        
        if added or removed:
            logger.info(f"[SessionUpdate] Added: {added}, Removed: {removed}")
        
        return new_tickers, added


# Глобальний екземпляр
live_trading_manager = LiveTradingTickerManager()


def get_optimal_tickers_for_live_trading(max_tickers: int = 25, risk_tolerance: str = "medium") -> Tuple[List[str], Dict[str, Any]]:
    """
    Зручна функція для отримання оптимальних тікерів
    """
    manager = LiveTradingTickerManager(max_tickers=max_tickers, risk_tolerance=risk_tolerance)
    return manager.get_optimal_tickers_for_live_trading()


if __name__ == "__main__":
    # Тестування системи
    print("=== Testing Live Trading Ticker Manager ===")
    
    tickers, report = get_optimal_tickers_for_live_trading()
    
    print(f"\nSelected {len(tickers)} tickers:")
    for i, ticker in enumerate(tickers, 1):
        print(f"{i:2d}. {ticker}")
    
    print(f"\nMarket Conditions:")
    conditions = report['market_conditions']
    print(f"  Volatility: {conditions.volatility_level:.2f}")
    print(f"  Trend: {conditions.trend_direction}")
    print(f"  Phase: {conditions.market_phase}")
    print(f"  Strategy: {report['strategy_used']}")
    
    print(f"\nTop 5 Tickers by Score:")
    for detail in report['selection_details'][:5]:
        print(f"  {detail['ticker']}: {detail['total_score']:.3f} (vol: {detail['volatility_score']:.2f}, mom: {detail['momentum_score']:.2f})")
