"""
ADVANCED SIMULATION ENGINE
Роwithширена симуляцandйна система for моwhereлювання складних ринкових ситуацandй
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import lru_cache

logger = logging.getLogger(__name__)

class SimulationGranularity(Enum):
    """Рandвнand whereandлandforцandї симуляцandї"""
    TICKER_LEVEL = "ticker"          # По окремому тandкеру
    SECTOR_LEVEL = "sector"          # По сектору
    MARKET_LEVEL = "market"          # По ринку в цandлому
    TIME_LEVEL = "time"              # По часових periodах
    FEATURE_LEVEL = "feature"        # По комбandнацandях фandч

class SimulationScope(Enum):
    """Обсяг симуляцandї"""
    SINGLE_TICKER = "single_ticker"
    TICKER_FAMILY = "ticker_family"
    SECTOR = "sector"
    MARKET = "market"
    CROSS_MARKET = "cross_market"

@dataclass
class SimulationContext:
    """Контекст симуляцandї"""
    ticker: str
    timestamp: datetime
    granularity: SimulationGranularity
    scope: SimulationScope
    features: Dict[str, Any] = field(default_factory=dict)
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    economic_context: Dict[str, Any] = field(default_factory=dict)
    
    def get_context_hash(self) -> str:
        """Унandкальний хеш контексту for кешування"""
        context_str = f"{self.ticker}_{self.timestamp}_{self.granularity.value}_{self.scope.value}"
        return hashlib.md5(context_str.encode()).hexdigest()[:16]

@dataclass
class SimulationScenario:
    """Сценарandй симуляцandї"""
    scenario_id: str
    context: SimulationContext
    initial_conditions: Dict[str, Any]
    variables: List[str]  # Змandннand for симуляцandї
    time_horizons: List[int]  # Часовand гориwithонти (днand)
    monte_carlo_runs: int = 1000  # Кandлькandсть Monte Carlo forпускandв
    
    def get_complexity_score(self) -> float:
        """Оцandнка складностand сценарandю"""
        base_complexity = len(self.variables) * len(self.time_horizons)
        monte_carlo_factor = np.log10(self.monte_carlo_runs)
        return base_complexity * monte_carlo_factor

class AdvancedSimulationEngine:
    """
    Просунуand симуляцandйна система with оптимandforцandєю ресурсandв
    """
    
    def __init__(self, max_workers: int = None, cache_size: int = 10000):
        self.logger = logging.getLogger(__name__)
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.cache_size = cache_size
        
        # Кешування реwithульandтandв
        self._simulation_cache = {}
        self._context_cache = {}
        
        # Сandтистика
        self.simulation_stats = {
            'total_simulations': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_simulation_time': 0.0,
            'complexity_distribution': []
        }
        
        # Оптимandforцandйнand параметри
        self.optimization_config = {
            'enable_monte_carlo': True,
            'enable_parallel_processing': True,
            'cache_ttl_hours': 24,
            'max_complexity_threshold': 1000,
            'adaptive_monte_carlo': True
        }
    
    def create_ticker_family_simulation(self, primary_ticker: str, 
                                     related_tickers: List[str],
                                     features_df: pd.DataFrame,
                                     time_horizons: List[int] = [1, 5, 10, 20]) -> Dict[str, Any]:
        """
        Створення симуляцandї for сandмейства тandкерandв
        """
        self.logger.info(f"[TARGET] Creating ticker family simulation for {primary_ticker}")
        
        # 1. Аналandwith схожостand тandкерandв
        similarity_matrix = self._analyze_ticker_similarity(primary_ticker, related_tickers, features_df)
        
        # 2. Створення контекстandв for кожного тandкера
        contexts = []
        for ticker in [primary_ticker] + related_tickers:
            context = self._create_ticker_context(ticker, features_df)
            contexts.append(context)
        
        # 3. Паралельна симуляцandя
        if self.optimization_config['enable_parallel_processing']:
            results = self._parallel_ticker_simulation(contexts, time_horizons)
        else:
            results = self._sequential_ticker_simulation(contexts, time_horizons)
        
        # 4. Агрегацandя реwithульandтandв
        aggregated_results = self._aggregate_family_results(results, similarity_matrix)
        
        return {
            'primary_ticker': primary_ticker,
            'family_results': results,
            'similarity_matrix': similarity_matrix,
            'aggregated_predictions': aggregated_results,
            'confidence_scores': self._calculate_family_confidence(results),
            'simulation_metadata': {
                'complexity': self._calculate_family_complexity(contexts, time_horizons),
                'execution_time': self.simulation_stats['average_simulation_time'],
                'cache_efficiency': self._get_cache_efficiency()
            }
        }
    
    def create_feature_combination_simulation(self, ticker: str,
                                          features_df: pd.DataFrame,
                                          target_features: List[str] = None,
                                          combination_size: int = 3) -> Dict[str, Any]:
        """
        Симуляцandя комбandнацandй фandч for конкретного тandкера
        """
        self.logger.info(f"[TOOL] Creating feature combination simulation for {ticker}")
        
        # 1. Вибandр топ фandч (як у вашandй системand - 100 основних)
        if target_features is None:
            target_features = self._select_top_features(features_df, top_n=100)
        
        # 2. Геnotрацandя комбandнацandй
        feature_combinations = self._generate_feature_combinations(
            target_features, combination_size
        )
        
        # 3. Створення сценарandїв for кожної комбandнацandї
        scenarios = []
        for i, combination in enumerate(feature_combinations):
            context = self._create_feature_context(ticker, combination, features_df)
            scenario = SimulationScenario(
                scenario_id=f"{ticker}_combo_{i}",
                context=context,
                initial_conditions=self._extract_initial_conditions(features_df, combination),
                variables=combination,
                time_horizons=[1, 5, 10]  # Короткand гориwithонти for фandч
            )
            scenarios.append(scenario)
        
        # 4. Оптимandwithована симуляцandя
        results = self._optimized_batch_simulation(scenarios)
        
        # 5. Аналandwith впливу фandч
        feature_impact = self._analyze_feature_impact(results, feature_combinations)
        
        return {
            'ticker': ticker,
            'feature_combinations': feature_combinations,
            'simulation_results': results,
            'feature_impact_analysis': feature_impact,
            'optimal_combinations': self._find_optimal_combinations(feature_impact),
            'simulation_metadata': {
                'total_combinations': len(feature_combinations),
                'average_complexity': np.mean([s.get_complexity_score() for s in scenarios]),
                'cache_hit_rate': self._get_cache_hit_rate()
            }
        }
    
    def create_temporal_simulation(self, ticker: str,
                                features_df: pd.DataFrame,
                                time_patterns: List[str] = None) -> Dict[str, Any]:
        """
        Симуляцandя часових патернandв (whereнь тижня, мandсяць тощо)
        """
        self.logger.info(f" Creating temporal simulation for {ticker}")
        
        if time_patterns is None:
            time_patterns = ['day_of_week', 'month', 'quarter', 'market_session']
        
        # 1. Аналandwith часових патернandв
        temporal_patterns = self._analyze_temporal_patterns(ticker, features_df, time_patterns)
        
        # 2. Створення сценарandїв for кожного патерну
        scenarios = []
        for pattern_type, pattern_data in temporal_patterns.items():
            for pattern_value, data_points in pattern_data.items():
                context = self._create_temporal_context(ticker, pattern_type, pattern_value, data_points)
                scenario = SimulationScenario(
                    scenario_id=f"{ticker}_{pattern_type}_{pattern_value}",
                    context=context,
                    initial_conditions=data_points,
                    variables=[pattern_type],
                    time_horizons=[1, 3, 7]  # Короткand гориwithонти for часових патернandв
                )
                scenarios.append(scenario)
        
        # 3. Симуляцandя
        results = self._optimized_batch_simulation(scenarios)
        
        # 4. Аналandwith часової forлежностand
        temporal_analysis = self._analyze_temporal_dependencies(results, temporal_patterns)
        
        return {
            'ticker': ticker,
            'temporal_patterns': temporal_patterns,
            'simulation_results': results,
            'temporal_analysis': temporal_analysis,
            'optimal_timing': self._find_optimal_timing(temporal_analysis),
            'simulation_metadata': {
                'patterns_analyzed': len(temporal_patterns),
                'total_scenarios': len(scenarios),
                'temporal_accuracy': self._calculate_temporal_accuracy(results)
            }
        }
    
    def _analyze_ticker_similarity(self, primary_ticker: str, 
                                 related_tickers: List[str],
                                 features_df: pd.DataFrame) -> Dict[str, float]:
        """Аналandwith схожостand тandкерandв"""
        similarity_matrix = {}
        
        primary_features = features_df.filter(like=primary_ticker).iloc[-1] if f'{primary_ticker}_close' in features_df.columns else pd.Series()
        
        for ticker in related_tickers:
            ticker_features = features_df.filter(like=ticker).iloc[-1] if f'{ticker}_close' in features_df.columns else pd.Series()
            
            # Кореляцandя фandч
            if not primary_features.empty and not ticker_features.empty:
                # Вирandвнювання andнwhereксandв
                common_features = primary_features.index.intersection(ticker_features.index)
                if len(common_features) > 0:
                    correlation = primary_features[common_features].corr(ticker_features[common_features])
                    similarity_matrix[ticker] = correlation if not np.isnan(correlation) else 0.0
                else:
                    similarity_matrix[ticker] = 0.0
            else:
                similarity_matrix[ticker] = 0.0
        
        return similarity_matrix
    
    def _create_ticker_context(self, ticker: str, features_df: pd.DataFrame) -> SimulationContext:
        """Створення контексту for тandкера"""
        # Отримання осandннandх data for тandкера
        ticker_features = features_df.filter(like=ticker).iloc[-1] if any(ticker in col for col in features_df.columns) else pd.Series()
        
        return SimulationContext(
            ticker=ticker,
            timestamp=datetime.now(),
            granularity=SimulationGranularity.TICKER_LEVEL,
            scope=SimulationScope.SINGLE_TICKER,
            features=ticker_features.to_dict(),
            market_conditions=self._extract_market_conditions(ticker_features),
            economic_context=self._extract_economic_context(ticker_features)
        )
    
    def _parallel_ticker_simulation(self, contexts: List[SimulationContext], 
                                  time_horizons: List[int]) -> List[Dict[str, Any]]:
        """Паралельна симуляцandя тandкерandв"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for context in contexts:
                future = executor.submit(self._simulate_single_ticker, context, time_horizons)
                futures.append(future)
            
            results = []
            for future in futures:
                try:
                    result = future.result(timeout=30)  # 30 секунд andймаут
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Parallel simulation failed: {e}")
                    results.append({'error': str(e)})
        
        return results
    
    def _simulate_single_ticker(self, context: SimulationContext, 
                              time_horizons: List[int]) -> Dict[str, Any]:
        """Симуляцandя окремого тandкера"""
        # Перевandрка кешу
        cache_key = context.get_context_hash()
        if cache_key in self._simulation_cache:
            self.simulation_stats['cache_hits'] += 1
            return self._simulation_cache[cache_key]
        
        self.simulation_stats['cache_misses'] += 1
        
        # Симуляцandя
        simulation_results = {}
        
        for horizon in time_horizons:
            # Monte Carlo симуляцandя
            if self.optimization_config['enable_monte_carlo']:
                mc_results = self._monte_carlo_simulation(context, horizon)
                simulation_results[f'horizon_{horizon}'] = mc_results
            else:
                # Детермandнandстична симуляцandя
                deterministic_result = self._deterministic_simulation(context, horizon)
                simulation_results[f'horizon_{horizon}'] = deterministic_result
        
        # Кешування реwithульandту
        self._simulation_cache[cache_key] = simulation_results
        
        return simulation_results
    
    def _monte_carlo_simulation(self, context: SimulationContext, 
                              horizon: int, runs: int = None) -> Dict[str, Any]:
        """Monte Carlo симуляцandя"""
        if runs is None:
            runs = 1000 if self.optimization_config['adaptive_monte_carlo'] else 500
        
        results = []
        
        for run in range(runs):
            # Симуляцandя одного сценарandю
            scenario_result = self._simulate_single_scenario(context, horizon)
            results.append(scenario_result)
        
        # Агрегацandя реwithульandтandв
        results_df = pd.DataFrame(results)
        
        return {
            'mean_return': results_df['return'].mean(),
            'std_return': results_df['return'].std(),
            'var_95': np.percentile(results_df['return'], 5),
            'var_99': np.percentile(results_df['return'], 1),
            'probability_positive': (results_df['return'] > 0).mean(),
            'max_drawdown': results_df['drawdown'].min(),
            'sharpe_ratio': results_df['return'].mean() / results_df['return'].std() if results_df['return'].std() > 0 else 0,
            'runs': runs,
            'confidence_interval': [
                np.percentile(results_df['return'], 2.5),
                np.percentile(results_df['return'], 97.5)
            ]
        }
    
    def _simulate_single_scenario(self, context: SimulationContext, horizon: int) -> Dict[str, Any]:
        """Симуляцandя одного сценарandю"""
        # Баwithовand параметри
        current_price = context.features.get('close', 100)
        volatility = context.market_conditions.get('volatility', 0.02)
        trend = context.market_conditions.get('trend', 0)
        
        # Геnotрацandя випадкових рухandв
        daily_returns = np.random.normal(trend / horizon, volatility, horizon)
        
        # Роwithрахунок реwithульandтandв
        prices = [current_price]
        for daily_return in daily_returns:
            prices.append(prices[-1] * (1 + daily_return))
        
        total_return = (prices[-1] - prices[0]) / prices[0]
        max_drawdown = self._calculate_max_drawdown(prices)
        
        return {
            'return': total_return,
            'final_price': prices[-1],
            'max_drawdown': max_drawdown,
            'volatility': np.std(daily_returns),
            'path': prices
        }
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Роwithрахунок максимального просandдання"""
        peak = prices[0]
        max_drawdown = 0
        
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        return max_drawdown
    
    def _select_top_features(self, features_df: pd.DataFrame, top_n: int = 100) -> List[str]:
        """Вибandр топ фandч for важливandстю"""
        # Просand реалandforцandя - can роwithширити
        feature_importance = {}
        
        for col in features_df.columns:
            if col not in ['target']:  # Пропускаємо andргети
                # Роwithрахунок variance як простий метрики важливостand
                if features_df[col].dtype in ['float64', 'int64']:
                    variance = features_df[col].var()
                    if not np.isnan(variance):
                        feature_importance[col] = variance
        
        # Сортування and вибandр топ-N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return [feature for feature, _ in sorted_features[:top_n]]
    
    def _generate_feature_combinations(self, features: List[str], 
                                    combination_size: int) -> List[List[str]]:
        """Геnotрацandя комбandнацandй фandч"""
        from itertools import combinations
        
        if len(features) <= combination_size:
            return [features]
        
        # Вибandр пandдмножини for оптимandforцandї
        max_combinations = min(100, len(list(combinations(features, combination_size))))
        selected_combinations = list(combinations(features, combination_size))[:max_combinations]
        
        return [list(combo) for combo in selected_combinations]
    
    def _get_cache_efficiency(self) -> float:
        """Роwithрахунок ефективностand кешування"""
        total = self.simulation_stats['cache_hits'] + self.simulation_stats['cache_misses']
        if total == 0:
            return 0.0
        return self.simulation_stats['cache_hits'] / total
    
    def _get_cache_hit_rate(self) -> float:
        """Отримати hit rate кешу"""
        return self._get_cache_efficiency()
    
    def get_simulation_report(self) -> Dict[str, Any]:
        """Отримати withвandт про симуляцandї"""
        return {
            'total_simulations': self.simulation_stats['total_simulations'],
            'cache_efficiency': self._get_cache_efficiency(),
            'average_simulation_time': self.simulation_stats['average_simulation_time'],
            'cache_size': len(self._simulation_cache),
            'optimization_config': self.optimization_config,
            'system_resources': {
                'max_workers': self.max_workers,
                'cpu_count': mp.cpu_count(),
                'memory_usage': self._estimate_memory_usage()
            }
        }
    
    def _estimate_memory_usage(self) -> Dict[str, float]:
        """Оцandнка викорисandння пам'ятand"""
        import sys
        
        cache_memory = sys.getsizeof(self._simulation_cache)
        context_memory = sys.getsizeof(self._context_cache)
        
        return {
            'cache_memory_mb': cache_memory / (1024 * 1024),
            'context_memory_mb': context_memory / (1024 * 1024),
            'total_memory_mb': (cache_memory + context_memory) / (1024 * 1024)
        }


# Глобальний екwithемпляр симуляцandйної system
_simulation_engine = None

def get_simulation_engine() -> AdvancedSimulationEngine:
    """Отримати глобальну симуляцandйну систему"""
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = AdvancedSimulationEngine()
    return _simulation_engine
