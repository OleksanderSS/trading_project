#!/usr/bin/env python3
"""
Portfolio Optimization Module
Оптимізація портфоліо: Markowitz, Black-Litterman, Risk Parity, Hierarchical Risk Parity, Kelly Criterion
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any

from src.optimization.base import BaseOptimizer
from src.core.logging.logger import ProjectLogger
from src.metrics.calculator import MetricsCalculator
from src.analytics.calculators.fama_french_factors import FamaFrenchFactors

class PortfolioOptimizer(BaseOptimizer):
    """
    Оптимізатор портфоліо з різними методами та підтримкою мульти-таймфреймів.
    """
    
    def __init__(self, timeframe: str = '1d', transaction_cost_lambda: float = 0.001):
        self.logger = ProjectLogger.get_logger("PortfolioOptimizer")
        self.timeframe = timeframe
        self.transaction_cost_lambda = transaction_cost_lambda
        self.metrics_calculator = MetricsCalculator()
        
        # Динамічний розрахунок періодів на рік
        self.periods_per_year = self._calculate_periods_per_year(timeframe)
        
        # Методи оптимізації
        self.optimization_methods = [
            'markowitz', 'min_variance', 'max_sharpe', 'risk_parity', 
            'hrp', 'black_litterman', 'equal_weight', 'inverse_volatility', 'kelly'
        ]
        
        # Обмеження оптимізації
        self.constraints = {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'max_positions': 10,
            'turnover_limit': 0.5,
            'sector_limit': 0.3,
            'supports_fractional': True
        }
        
        self.logger.info(f"PortfolioOptimizer initialized for {timeframe} ({self.periods_per_year} periods/year)")

    @property
    def optimizer_type(self) -> str:
        return "portfolio"

    def optimize(self, returns: pd.DataFrame, method: str = 'max_sharpe', **kwargs) -> Dict[str, Any]:
        """
        Головна точка входу для оптимізації портфоліо.
        """
        try:
            if method == 'markowitz': return self.markowitz_optimization(returns, **kwargs)
            elif method == 'max_sharpe': return self.max_sharpe_optimization(returns, **kwargs)
            elif method == 'min_variance': return self.markowitz_optimization(returns, target_return=0, **kwargs)
            elif method == 'risk_parity': return self.risk_parity_optimization(returns, **kwargs)
            elif method == 'hrp': return self.hierarchical_risk_parity(returns, **kwargs)
            elif method == 'black_litterman': return self.black_litterman_optimization(returns, **kwargs)
            elif method == 'equal_weight': return self.equal_weight_portfolio(returns, **kwargs)
            elif method == 'inverse_volatility': return self.inverse_volatility_portfolio(returns, **kwargs)
            elif method == 'kelly': return self.kelly_optimization(kwargs.get('win_rate', 0.5), kwargs.get('profit_factor', 2.0), list(returns.columns))
            else: 
                self.logger.error(f"Unknown optimization method: {method}")
                return {'success': False, 'error': f'Unknown method: {method}'}
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization ({method}): {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _calculate_periods_per_year(self, timeframe: str) -> float:
        """Розраховує кількість торгових періодів у році для заданого таймфрейму."""
        base_days = 252
        if timeframe == '1d': return float(base_days)
        elif timeframe in ['1h', '60m']: return float(base_days * 6.5)
        elif timeframe == '15m': return float(base_days * 26)
        elif timeframe == '5m': return float(base_days * 78)
        elif timeframe == '1m': return float(base_days * 390)
        return float(base_days)
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Розрахувати доходності цін"""
        try:
            returns = prices.pct_change().dropna()
            self.logger.info(f"Returns calculated: {len(returns)} observations, {len(returns.columns)} assets")
            return returns
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, method: Union[str, pd.DataFrame] = 'ledoit-wolf') -> pd.DataFrame:
        """Розрахувати коваріаційну матрицю або використати надану."""
        try:
            if isinstance(method, pd.DataFrame):
                self.logger.info("Using pre-calculated covariance matrix.")
                return method

            if method == 'sample':
                cov_matrix = returns.cov()
            elif method in ['ledoit-wolf', 'shrinkage']:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf().fit(returns)
                cov_matrix = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
            else:
                cov_matrix = returns.cov()
            
            cov_matrix = self._ensure_positive_definite(cov_matrix)
            self.logger.info(f"Covariance matrix calculated: method={method}")
            return cov_matrix
        except Exception as e:
            self.logger.error(f"Error calculating covariance matrix: {e}")
            return returns.cov()
    
    def markowitz_optimization(self, returns: pd.DataFrame, 
                             risk_free_rate: float = 0.02,
                             target_return: float = None,
                             current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Markowitz mean-variance оптимізація."""
        try:
            mu = returns.mean() * self.periods_per_year
            cov_matrix = self.calculate_covariance_matrix(returns)
            n_assets = len(mu)
            if current_weights is None: current_weights = np.zeros(n_assets)

            def objective(weights):
                port_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
                turnover_penalty = self.transaction_cost_lambda * np.sum(np.abs(weights - current_weights))
                return port_variance + turnover_penalty
            
            def portfolio_return(weights): return np.dot(weights.T, mu)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            if target_return is not None:
                constraints.append({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return})
            
            bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = self._apply_fractional_constraints(pd.Series(result.x, index=mu.index))
                ret = portfolio_return(weights.values)
                vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                sharpe = (ret - risk_free_rate) / vol if vol != 0 else 0
                return {
                    'weights': weights, 'expected_return': ret, 'volatility': vol,
                    'sharpe_ratio': sharpe, 'method': 'markowitz', 'success': True
                }
            return {'success': False, 'error': result.message}
        except Exception as e:
            self.logger.error(f"Error in Markowitz optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def max_sharpe_optimization(self, returns: pd.DataFrame, 
                               risk_free_rate: float = 0.02,
                               current_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Максимізація Sharpe ratio."""
        try:
            mu = returns.mean() * self.periods_per_year
            cov_matrix = self.calculate_covariance_matrix(returns)
            n_assets = len(mu)
            if current_weights is None: current_weights = np.zeros(n_assets)

            def objective(weights):
                port_return = np.dot(weights.T, mu)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                if port_volatility == 0: return 1e10
                sharpe = (port_return - risk_free_rate) / port_volatility
                turnover_penalty = self.transaction_cost_lambda * np.sum(np.abs(weights - current_weights))
                return -sharpe + turnover_penalty
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            
            result = opt.minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = self._apply_fractional_constraints(pd.Series(result.x, index=mu.index))
                port_ret = np.dot(weights.T, mu)
                port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                return {
                    'weights': weights, 'expected_return': port_ret, 'volatility': port_vol,
                    'sharpe_ratio': (port_ret - risk_free_rate) / port_vol, 'method': 'max_sharpe', 'success': True
                }
            return {'success': False, 'error': result.message}
        except Exception as e:
            self.logger.error(f"Error in Max Sharpe optimization: {e}")
            return {'success': False, 'error': str(e)}

    def kelly_optimization(self, win_rate: float, profit_factor: float, tickers: List[str]) -> Dict[str, Any]:
        """Kelly Criterion optimization with robust constraints."""
        try:
            p, b = win_rate, profit_factor
            if b <= 0:
                self.logger.warning(f"Invalid profit factor {b} for Kelly. Setting weights to zero.")
                return {'weights': pd.Series(0.0, index=tickers), 'kelly_fraction': 0.0, 'method': 'kelly', 'success': True}
            
            q = 1 - p
            kelly_f = (p * b - q) / b
            # Use fractional Kelly (e.g., half-Kelly) and clip to avoid extreme leverage
            kelly_f = max(0, min(kelly_f * 0.5, 0.25))
            
            n_assets = len(tickers)
            if n_assets == 0:
                return {'success': False, 'error': 'No tickers provided'}
                
            weights = pd.Series(kelly_f / n_assets, index=tickers)
            return {'weights': weights, 'kelly_fraction': kelly_f, 'method': 'kelly', 'success': True}
        except Exception as e:
            self.logger.error(f"Error in Kelly optimization: {e}")
            return {'success': False, 'error': str(e)}

    def black_litterman_optimization(self, returns: pd.DataFrame,
                                   views: Dict[str, float] = None,
                                   tau: float = 0.025,
                                   risk_free_rate: float = 0.02,
                                   benchmark_ticker: str = 'SPY') -> Dict[str, Any]:
        """Black-Litterman оптимізація."""
        try:
            mu = returns.mean() * self.periods_per_year
            cov_matrix = self.calculate_covariance_matrix(returns)
            n_assets = len(mu)
            market_weights = np.array([1/n_assets] * n_assets)
            risk_aversion = 3.0
            implied_returns = risk_aversion * np.dot(cov_matrix.values, market_weights)
            
            if views is None: views = {asset: implied_returns[i] for i, asset in enumerate(mu.index)}
            P = np.eye(n_assets)
            Q = np.array([views.get(asset, implied_returns[i]) for i, asset in enumerate(mu.index)])
            tau_cov = tau * cov_matrix.values
            omega = np.diag(np.diag(tau_cov))
            tau_cov_inv, omega_inv = np.linalg.inv(tau_cov), np.linalg.inv(omega)
            posterior_returns = np.linalg.inv(tau_cov_inv + P.T @ omega_inv @ P) @ (tau_cov_inv @ implied_returns + P.T @ omega_inv @ Q)
            res = self.markowitz_optimization(returns, risk_free_rate=risk_free_rate)
            if res['success']:
                res['method'] = 'black_litterman'
                res['posterior_returns'] = pd.Series(posterior_returns, index=mu.index)
            return res
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {e}")
            return {'success': False, 'error': str(e)}

    def _apply_fractional_constraints(self, weights: pd.Series) -> pd.Series:
        """Обробка обмежень на дробові акції."""
        if self.constraints['supports_fractional']: return weights
        rounded = (weights * 100).round() / 100
        return rounded / rounded.sum() if rounded.sum() > 0 else rounded

    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Risk Parity оптимізація."""
        try:
            cov_matrix = self.calculate_covariance_matrix(returns)
            n_assets = len(returns.columns)
            def risk_budget_objective(weights):
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_volatility
                contrib = weights * marginal_contrib
                target_risk = 1.0 / n_assets
                return np.sum((contrib - target_risk) ** 2)
            
            constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
            bounds = tuple((0.01, 1.0) for _ in range(n_assets))
            x0 = np.array([1/n_assets] * n_assets)
            result = opt.minimize(risk_budget_objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = pd.Series(result.x, index=returns.columns)
                mu = returns.mean() * self.periods_per_year
                portfolio_return = np.dot(weights.T, mu)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                return {
                    'weights': weights, 'expected_return': portfolio_return, 'volatility': portfolio_volatility,
                    'sharpe_ratio': portfolio_return / portfolio_volatility, 'method': 'risk_parity', 'success': True
                }
            return {'success': False, 'error': result.message}
        except Exception as e:
            self.logger.error(f"Error in Risk Parity optimization: {e}")
            return {'success': False, 'error': str(e)}

    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Hierarchical Risk Parity (HRP)"""
        try:
            cov_matrix = self.calculate_covariance_matrix(returns)
            distance_matrix = self._calculate_distance_matrix(cov_matrix)
            linkage_matrix = self._hierarchical_clustering(distance_matrix)
            clusters = self._get_cluster_order(linkage_matrix)
            weights = self._recursive_bisection(cov_matrix, clusters)
            mu = returns.mean() * self.periods_per_year
            weights_s = pd.Series(weights, index=returns.columns)
            portfolio_return = np.dot(weights_s.T, mu)
            portfolio_volatility = np.sqrt(np.dot(weights_s.T, np.dot(cov_matrix.values, weights_s)))
            return {
                'weights': weights_s, 'expected_return': portfolio_return, 'volatility': portfolio_volatility,
                'sharpe_ratio': portfolio_return / portfolio_volatility, 'method': 'hrp', 'success': True
            }
        except Exception as e:
            self.logger.error(f"Error in HRP optimization: {e}")
            return {'success': False, 'error': str(e)}

    def equal_weight_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Рівноваговий портфоліо"""
        try:
            n_assets = len(returns.columns)
            weights = np.array([1/n_assets] * n_assets)
            mu = returns.mean() * self.periods_per_year
            cov_matrix = self.calculate_covariance_matrix(returns)
            portfolio_return = np.dot(weights.T, mu)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            return {
                'weights': pd.Series(weights, index=returns.columns), 'expected_return': portfolio_return,
                'volatility': portfolio_volatility, 'sharpe_ratio': portfolio_return / portfolio_volatility,
                'method': 'equal_weight', 'success': True
            }
        except Exception as e:
            self.logger.error(f"Error in equal weight portfolio: {e}")
            return {'success': False, 'error': str(e)}

    def inverse_volatility_portfolio(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Портфоліо з оберненою волатильністю"""
        try:
            volatilities = returns.std() * np.sqrt(self.periods_per_year)
            inv_vols = 1 / volatilities
            weights = inv_vols / inv_vols.sum()
            mu = returns.mean() * self.periods_per_year
            cov_matrix = self.calculate_covariance_matrix(returns)
            portfolio_return = np.dot(weights.T, mu)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
            return {
                'weights': pd.Series(weights, index=returns.columns), 'expected_return': portfolio_return,
                'volatility': portfolio_volatility, 'sharpe_ratio': portfolio_return / portfolio_volatility,
                'method': 'inverse_volatility', 'success': True
            }
        except Exception as e:
            self.logger.error(f"Error in inverse volatility portfolio: {e}")
            return {'success': False, 'error': str(e)}

    def compare_optimization_methods(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """Порівняти різні методи оптимізації"""
        try:
            results = {}
            for method in self.optimization_methods:
                if method == 'kelly': continue
                result = self.optimize(returns, method=method)
                results[method] = result
            comparison = self._create_comparison_table(results)
            return {'results': results, 'comparison': comparison}
        except Exception as e:
            self.logger.error(f"Error comparing optimization methods: {e}")
            return {}

    def _ensure_positive_definite(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        try:
            eigenvalues = np.linalg.eigvals(cov_matrix.values)
            if np.all(eigenvalues > 0): return cov_matrix
            noise = abs(np.min(eigenvalues)) + 1e-8
            cov_matrix_fixed = cov_matrix.copy()
            np.fill_diagonal(cov_matrix_fixed.values, np.diag(cov_matrix_fixed.values) + noise)
            return cov_matrix_fixed
        except Exception as e:
            self.logger.error(f"Error ensuring positive definiteness: {e}")
            return cov_matrix

    def _calculate_distance_matrix(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        corr_matrix = cov_matrix.corr()
        return np.sqrt(0.5 * (1 - corr_matrix)).values

    def _hierarchical_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        from scipy.cluster.hierarchy import linkage
        condensed_distance = []
        n = distance_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n): condensed_distance.append(distance_matrix[i, j])
        return linkage(condensed_distance, method='single')

    def _get_cluster_order(self, linkage_matrix: np.ndarray) -> List[int]:
        from scipy.cluster.hierarchy import dendrogram
        return dendrogram(linkage_matrix, no_plot=True)['leaves']

    def _recursive_bisection(self, cov_matrix: pd.DataFrame, clusters: List[int]) -> np.ndarray:
        n_assets = len(cov_matrix)
        weights = np.zeros(n_assets)
        def allocate_cluster(cluster_indices, cluster_weight):
            if len(cluster_indices) == 1: weights[cluster_indices[0]] = cluster_weight
            else:
                cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
                cluster_var = np.diag(cluster_cov)
                inv_var = 1 / cluster_var
                sub_weights = inv_var / inv_var.sum()
                for i, idx in enumerate(cluster_indices): weights[idx] = cluster_weight * sub_weights[i]
        allocate_cluster(clusters, 1.0)
        return weights

    def _create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        comparison_data = []
        for method, result in results.items():
            if result.get('success', False):
                comparison_data.append({
                    'Method': method, 'Expected Return': result.get('expected_return', 0),
                    'Volatility': result.get('volatility', 0), 'Sharpe Ratio': result.get('sharpe_ratio', 0), 'Success': True
                })
            else:
                comparison_data.append({'Method': method, 'Expected Return': 0, 'Volatility': 0, 'Sharpe Ratio': 0, 'Success': False})
        return pd.DataFrame(comparison_data).sort_values('Sharpe Ratio', ascending=False)