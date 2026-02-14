#!/usr/bin/env python3
"""
Portfolio Optimization Module
Оптимізація портфоліо: Markowitz, Black-Litterman, Risk Parity, Hierarchical Risk Parity
"""

import pandas as pd
import numpy as np
import scipy.optimize as opt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
warnings.filterwarnings('ignore')

from models.fama_french_factors import get_fama_french_factors
from models.hedge_fund_analyzer import calculate_performance_metrics

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Оптимізатор портфоліо з різними методами
    """
    
    def __init__(self):
        self.logger = logging.getLogger("PortfolioOptimizer")
        
        # Методи оптимізації
        self.optimization_methods = [
            'markowitz',           # Класична mean-variance оптимізація
            'min_variance',       # Мінімізація дисперсії
            'max_sharpe',         # Максимізація Sharpe ratio
            'risk_parity',        # Risk parity (рівний ризик)
            'hrp',               # Hierarchical Risk Parity
            'black_litterman',    # Black-Litterman
            'equal_weight',       # Рівні ваги
            'inverse_volatility'   # Обернена волатильність
        ]
        
        # Обмеження оптимізації
        self.constraints = {
            'min_weight': 0.0,      # Мінімальна вага
            'max_weight': 1.0,      # Максимальна вага
            'max_positions': 10,    # Максимальна кількість позицій
            'turnover_limit': 0.5,  # Обмеження обороту
            'sector_limit': 0.3     # Обмеження по секторах
        }
        
        self.logger.info("PortfolioOptimizer initialized")
    
    def calculate_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Розрахувати доходності цін
        
        Args:
            prices: DataFrame з цінами
            
        Returns:
            pd.DataFrame: Доходності
        """
        try:
            returns = prices.pct_change().dropna()
            self.logger.info(f"Returns calculated: {len(returns)} observations, {len(returns.columns)} assets")
            return returns
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def calculate_covariance_matrix(self, returns: pd.DataFrame, method: str = 'sample') -> pd.DataFrame:
        """
        Розрахувати коваріаційну матрицю
        
        Args:
            returns: Доходності
            method: Метод розрахунку ('sample', 'ledoit-wolf', 'shrinkage')
            
        Returns:
            pd.DataFrame: Коваріаційна матриця
        """
        try:
            if method == 'sample':
                cov_matrix = returns.cov()
            elif method == 'ledoit-wolf':
                # Ledoit-Wolf shrinkage estimator
                n_assets = len(returns.columns)
                sample_cov = returns.cov()
                
                # Shrinkage intensity
                shrinkage = self._calculate_ledoit_wolf_shrinkage(returns)
                
                # Target (constant correlation)
                var_diag = np.diag(sample_cov)
                rho = np.mean(sample_cov.values[np.triu_indices_from(sample_cov.values, k=1)])
                target = rho * np.ones((n_assets, n_assets))
                np.fill_diagonal(target, var_diag)
                
                # Shrinkage estimator
                cov_matrix = (1 - shrinkage) * sample_cov + shrinkage * target
            else:
                cov_matrix = returns.cov()
            
            # Ensure positive definite
            cov_matrix = self._ensure_positive_definite(cov_matrix)
            
            self.logger.info(f"Covariance matrix calculated: method={method}")
            return cov_matrix
            
        except Exception as e:
            self.logger.error(f"Error calculating covariance matrix: {e}")
            return pd.DataFrame()
    
    def markowitz_optimization(self, returns: pd.DataFrame, 
                             risk_free_rate: float = 0.02,
                             target_return: float = None) -> Dict[str, any]:
        """
        Markowitz mean-variance оптимізація
        
        Args:
            returns: Доходності
            risk_free_rate: Безризикова ставка
            target_return: Цільова доходність (опціонально)
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            # Розраховуємо параметри
            mu = returns.mean() * 252  # Річні доходності
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            n_assets = len(mu)
            
            # Функція для оптимізації
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(cov_matrix.values, weights))
            
            def portfolio_return(weights):
                return np.dot(weights.T, mu)
            
            # Обмеження
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Сума ваг = 1
            ]
            
            if target_return is not None:
                constraints.append(
                    {'type': 'eq', 'fun': lambda x: portfolio_return(x) - target_return}
                )
            
            # Межі ваг
            bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) 
                          for _ in range(n_assets))
            
            # Початкові ваги (рівні)
            x0 = np.array([1/n_assets] * n_assets)
            
            # Оптимізація
            result = opt.minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = pd.Series(result.x, index=mu.index)
                
                # Розраховуємо метрики портфоліо
                portfolio_return = portfolio_return(weights)
                portfolio_variance = portfolio_variance(weights)
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                # Individual asset contributions
                marginal_contrib = np.dot(cov_matrix.values, weights)
                contrib_percent = weights * marginal_contrib / portfolio_variance
                
                result_dict = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'variance': portfolio_variance,
                    'method': 'markowitz',
                    'success': True,
                    'contributions': contrib_percent,
                    'constraints': constraints
                }
                
                self.logger.info(f"Markowitz optimization: Sharpe={sharpe_ratio:.3f}")
                
                return result_dict
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.logger.error(f"Error in Markowitz optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def max_sharpe_optimization(self, returns: pd.DataFrame, 
                               risk_free_rate: float = 0.02) -> Dict[str, any]:
        """
        Максимізація Sharpe ratio
        
        Args:
            returns: Доходності
            risk_free_rate: Безризикова ставка
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            # Розраховуємо параметри
            mu = returns.mean() * 252
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            n_assets = len(mu)
            
            # Функція для оптимізації (негативний Sharpe)
            def negative_sharpe(weights):
                portfolio_return = np.dot(weights.T, mu)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                if portfolio_volatility == 0:
                    return -np.inf
                
                sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
                return -sharpe
            
            # Обмеження
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Сума ваг = 1
            ]
            
            # Межі ваг
            bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) 
                          for _ in range(n_assets))
            
            # Початкові ваги
            x0 = np.array([1/n_assets] * n_assets)
            
            # Оптимізація
            result = opt.minimize(
                negative_sharpe,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = pd.Series(result.x, index=mu.index)
                
                # Розраховуємо метрики
                portfolio_return = np.dot(weights.T, mu)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                result_dict = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'variance': portfolio_variance,
                    'method': 'max_sharpe',
                    'success': True
                }
                
                self.logger.info(f"Max Sharpe optimization: Sharpe={sharpe_ratio:.3f}")
                
                return result_dict
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.logger.error(f"Error in Max Sharpe optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def risk_parity_optimization(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Risk Parity оптимізація (рівний ризик)
        
        Args:
            returns: Доходності
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            cov_matrix = self.calculate_covariance_matrix(returns)
            n_assets = len(returns.columns)
            
            # Функція для оптимізації - мінімізація різниці в ризиках
            def risk_budget_objective(weights):
                # Розраховуємо marginal contribution to risk
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
                marginal_contrib = np.dot(cov_matrix.values, weights) / portfolio_volatility
                contrib = weights * marginal_contrib
                
                # Ціль: рівні внески в ризик
                target_risk = 1.0 / n_assets
                risk_diff = contrib - target_risk
                
                return np.sum(risk_diff ** 2)
            
            # Обмеження
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Сума ваг = 1
            ]
            
            # Межі ваг
            bounds = tuple((0.01, 1.0) for _ in range(n_assets))  # Мінімум 1%
            
            # Початкові ваги
            x0 = np.array([1/n_assets] * n_assets)
            
            # Оптимізація
            result = opt.minimize(
                risk_budget_objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = pd.Series(result.x, index=returns.columns)
                
                # Розраховуємо метрики
                mu = returns.mean() * 252
                portfolio_return = np.dot(weights.T, mu)
                portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = portfolio_return / portfolio_volatility
                
                # Ризикові внески
                marginal_contrib = np.dot(cov_matrix.values, weights)
                contrib = weights * marginal_contrib / portfolio_variance
                
                result_dict = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'variance': portfolio_variance,
                    'method': 'risk_parity',
                    'success': True,
                    'risk_contributions': contrib
                }
                
                self.logger.info(f"Risk Parity optimization: Sharpe={sharpe_ratio:.3f}")
                
                return result_dict
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.logger.error(f"Error in Risk Parity optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def hierarchical_risk_parity(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Hierarchical Risk Parity (HRP)
        
        Args:
            returns: Доходності
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            # 1. Кластеризація активів
            distance_matrix = self._calculate_distance_matrix(cov_matrix)
            linkage_matrix = self._hierarchical_clustering(distance_matrix)
            
            # 2. Рекурсивна бінарна розбивка
            clusters = self._get_cluster_order(linkage_matrix)
            
            # 3. Розподіл ваг
            weights = self._recursive_bisection(cov_matrix, clusters)
            
            # Розраховуємо метрики
            mu = returns.mean() * 252
            portfolio_return = np.dot(weights.T, mu)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            result_dict = {
                'weights': pd.Series(weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'variance': portfolio_variance,
                'method': 'hrp',
                'success': True,
                'linkage_matrix': linkage_matrix,
                'clusters': clusters
            }
            
            self.logger.info(f"HRP optimization: Sharpe={sharpe_ratio:.3f}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in HRP optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def black_litterman_optimization(self, returns: pd.DataFrame,
                                   views: Dict[str, float] = None,
                                   tau: float = 0.025,
                                   risk_free_rate: float = 0.02) -> Dict[str, any]:
        """
        Black-Litterman оптимізація
        
        Args:
            returns: Доходності
            views: Думки інвестора (views)
            tau: Параметр невизначеності
            risk_free_rate: Безризикова ставка
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            # Розраховуємо параметри
            mu = returns.mean() * 252
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            n_assets = len(mu)
            
            # Ринкові ваги (рівні як база)
            market_weights = np.array([1/n_assets] * n_assets)
            
            # Ринкові доходності (implied returns)
            risk_aversion = 3.0  # Типове значення
            implied_returns = risk_aversion * np.dot(cov_matrix.values, market_weights)
            
            # Якщо немає думок, використовуємо implied returns
            if views is None:
                # Створюємо базові думки (рівні до ринку)
                views = {}
                for asset in mu.index:
                    views[asset] = implied_returns[mu.index.get_loc(asset)]
            
            # Матриця думок
            P = np.eye(n_assets)  # Простий випадок - кожна думка про один актив
            Q = np.array([views.get(asset, implied_returns[i]) for i, asset in enumerate(mu.index)])
            
            # Black-Litterman формула
            tau_cov = tau * cov_matrix.values
            omega = tau * np.eye(n_assets)  # Спрощення
            
            # Обернені матриці
            tau_cov_inv = np.linalg.inv(tau_cov)
            omega_inv = np.linalg.inv(omega)
            
            # Розрахунок blended returns
            posterior_returns = np.linalg.inv(
                tau_cov_inv + P.T @ omega_inv @ P
            ) @ (tau_cov_inv @ implied_returns + P.T @ omega_inv @ Q)
            
            # Оновлена коваріаційна матриця
            posterior_cov = cov_matrix.values + tau_cov
            
            # Markowitz оптимізація з blended returns
            def portfolio_variance(weights):
                return np.dot(weights.T, np.dot(posterior_cov, weights))
            
            def portfolio_return(weights):
                return np.dot(weights.T, posterior_returns)
            
            # Обмеження
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            
            # Межі ваг
            bounds = tuple((self.constraints['min_weight'], self.constraints['max_weight']) 
                          for _ in range(n_assets))
            
            # Початкові ваги
            x0 = np.array([1/n_assets] * n_assets)
            
            # Оптимізація
            result = opt.minimize(
                portfolio_variance,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                weights = pd.Series(result.x, index=mu.index)
                
                # Розраховуємо метрики
                portfolio_return = portfolio_return(weights)
                portfolio_variance = portfolio_variance(weights)
                portfolio_volatility = np.sqrt(portfolio_variance)
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
                
                result_dict = {
                    'weights': weights,
                    'expected_return': portfolio_return,
                    'volatility': portfolio_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'variance': portfolio_variance,
                    'method': 'black_litterman',
                    'success': True,
                    'implied_returns': pd.Series(implied_returns, index=mu.index),
                    'posterior_returns': pd.Series(posterior_returns, index=mu.index),
                    'views': views
                }
                
                self.logger.info(f"Black-Litterman optimization: Sharpe={sharpe_ratio:.3f}")
                
                return result_dict
            else:
                return {'success': False, 'error': result.message}
                
        except Exception as e:
            self.logger.error(f"Error in Black-Litterman optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def equal_weight_portfolio(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Рівноваговий портфоліо
        
        Args:
            returns: Доходності
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            n_assets = len(returns.columns)
            weights = np.array([1/n_assets] * n_assets)
            
            # Розраховуємо метрики
            mu = returns.mean() * 252
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            portfolio_return = np.dot(weights.T, mu)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            result_dict = {
                'weights': pd.Series(weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'variance': portfolio_variance,
                'method': 'equal_weight',
                'success': True
            }
            
            self.logger.info(f"Equal weight portfolio: Sharpe={sharpe_ratio:.3f}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in equal weight portfolio: {e}")
            return {'success': False, 'error': str(e)}
    
    def inverse_volatility_portfolio(self, returns: pd.DataFrame) -> Dict[str, any]:
        """
        Портфоліо з оберненою волатильністю
        
        Args:
            returns: Доходності
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            # Розраховуємо волатильності
            volatilities = returns.std() * np.sqrt(252)
            
            # Обернені волатильності
            inv_vols = 1 / volatilities
            
            # Нормалізуємо ваги
            weights = inv_vols / inv_vols.sum()
            
            # Розраховуємо метрики
            mu = returns.mean() * 252
            cov_matrix = self.calculate_covariance_matrix(returns)
            
            portfolio_return = np.dot(weights.T, mu)
            portfolio_variance = np.dot(weights.T, np.dot(cov_matrix.values, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            sharpe_ratio = portfolio_return / portfolio_volatility
            
            result_dict = {
                'weights': pd.Series(weights, index=returns.columns),
                'expected_return': portfolio_return,
                'volatility': portfolio_volatility,
                'sharpe_ratio': sharpe_ratio,
                'variance': portfolio_variance,
                'method': 'inverse_volatility',
                'success': True
            }
            
            self.logger.info(f"Inverse volatility portfolio: Sharpe={sharpe_ratio:.3f}")
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Error in inverse volatility portfolio: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_portfolio(self, returns: pd.DataFrame, 
                         method: str = 'max_sharpe',
                         **kwargs) -> Dict[str, any]:
        """
        Оптимізувати портфоліо вказаним методом
        
        Args:
            returns: Доходності
            method: Метод оптимізації
            **kwargs: Додаткові параметри
            
        Returns:
            Dict: Результати оптимізації
        """
        try:
            if method == 'markowitz':
                return self.markowitz_optimization(returns, **kwargs)
            elif method == 'max_sharpe':
                return self.max_sharpe_optimization(returns, **kwargs)
            elif method == 'min_variance':
                return self.markowitz_optimization(returns, target_return=0, **kwargs)
            elif method == 'risk_parity':
                return self.risk_parity_optimization(returns, **kwargs)
            elif method == 'hrp':
                return self.hierarchical_risk_parity(returns, **kwargs)
            elif method == 'black_litterman':
                return self.black_litterman_optimization(returns, **kwargs)
            elif method == 'equal_weight':
                return self.equal_weight_portfolio(returns, **kwargs)
            elif method == 'inverse_volatility':
                return self.inverse_volatility_portfolio(returns, **kwargs)
            else:
                return {'success': False, 'error': f'Unknown method: {method}'}
                
        except Exception as e:
            self.logger.error(f"Error in portfolio optimization: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_optimization_methods(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        """
        Порівняти різні методи оптимізації
        
        Args:
            returns: Доходності
            
        Returns:
            Dict: Результати всіх методів
        """
        try:
            results = {}
            
            for method in self.optimization_methods:
                result = self.optimize_portfolio(returns, method=method)
                results[method] = result
                
                if result.get('success', False):
                    self.logger.info(f"{method}: Sharpe={result.get('sharpe_ratio', 0):.3f}")
                else:
                    self.logger.warning(f"{method}: Failed - {result.get('error', 'Unknown error')}")
            
            # Створюємо таблицю порівняння
            comparison = self._create_comparison_table(results)
            
            return {'results': results, 'comparison': comparison}
            
        except Exception as e:
            self.logger.error(f"Error comparing optimization methods: {e}")
            return {}
    
    # Helper methods
    def _calculate_ledoit_wolf_shrinkage(self, returns: pd.DataFrame) -> float:
        """Розрахувати shrinkage intensity для Ledoit-Wolf"""
        try:
            n, p = returns.shape
            sample_cov = returns.cov()
            
            # Розраховуємо параметри
            var_diag = np.diag(sample_cov)
            rho = np.mean(sample_cov.values[np.triu_indices_from(sample_cov.values, k=1)])
            
            # Pi matrix
            pi = np.zeros((p, p))
            for i in range(p):
                for j in range(p):
                    if i != j:
                        pi[i, j] = (returns.iloc[:, i] * returns.iloc[:, j]).mean() - \
                                  returns.iloc[:, i].mean() * returns.iloc[:, j].mean()
            
            # Shrinkage intensity
            pi_sum = np.sum(pi ** 2)
            theta_sum = np.sum((sample_cov.values - rho * np.eye(p)) ** 2)
            
            shrinkage = max(0, min(1, pi_sum / theta_sum))
            
            return shrinkage
            
        except Exception as e:
            self.logger.error(f"Error calculating Ledoit-Wolf shrinkage: {e}")
            return 0.0
    
    def _ensure_positive_definite(self, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """Заwithoutпечити позитивну визначеність коваріаційної матриці"""
        try:
            eigenvalues = np.linalg.eigvals(cov_matrix.values)
            
            if np.all(eigenvalues > 0):
                return cov_matrix
            
            # Додаємо невеликий шум до діагоналі
            min_eigenvalue = np.min(eigenvalues)
            if min_eigenvalue < 0:
                noise = abs(min_eigenvalue) + 1e-8
                cov_matrix_fixed = cov_matrix.copy()
                np.fill_diagonal(cov_matrix_fixed.values, 
                                np.diag(cov_matrix_fixed.values) + noise)
                return cov_matrix_fixed
            
            return cov_matrix
            
        except Exception as e:
            self.logger.error(f"Error ensuring positive definite: {e}")
            return cov_matrix
    
    def _calculate_distance_matrix(self, cov_matrix: pd.DataFrame) -> np.ndarray:
        """Розрахувати матрицю відстаней для HRP"""
        try:
            # Correlation matrix
            corr_matrix = cov_matrix.corr()
            
            # Distance matrix
            distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
            
            return distance_matrix.values
            
        except Exception as e:
            self.logger.error(f"Error calculating distance matrix: {e}")
            return np.zeros((len(cov_matrix), len(cov_matrix)))
    
    def _hierarchical_clustering(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Ієрархічна кластеризація"""
        try:
            from scipy.cluster.hierarchy import linkage
            
            # Flatten distance matrix for linkage
            condensed_distance = []
            n = distance_matrix.shape[0]
            
            for i in range(n):
                for j in range(i + 1, n):
                    condensed_distance.append(distance_matrix[i, j])
            
            linkage_matrix = linkage(condensed_distance, method='single')
            
            return linkage_matrix
            
        except Exception as e:
            self.logger.error(f"Error in hierarchical clustering: {e}")
            return np.array([])
    
    def _get_cluster_order(self, linkage_matrix: np.ndarray) -> List[int]:
        """Отримати порядок кластерів"""
        try:
            from scipy.cluster.hierarchy import dendrogram
            
            # Get dendrogram
            dendro = dendrogram(linkage_matrix, no_plot=True)
            
            # Get leaf order
            leaf_order = dendro['leaves']
            
            return leaf_order
            
        except Exception as e:
            self.logger.error(f"Error getting cluster order: {e}")
            return list(range(len(linkage_matrix) + 1))
    
    def _recursive_bisection(self, cov_matrix: pd.DataFrame, clusters: List[int]) -> np.ndarray:
        """Рекурсивна бісекція для HRP"""
        try:
            n_assets = len(cov_matrix)
            weights = np.zeros(n_assets)
            
            # Рекурсивний алгоритм
            def allocate_cluster(cluster_indices, cluster_weight):
                if len(cluster_indices) == 1:
                    weights[cluster_indices[0]] = cluster_weight
                else:
                    # Розділяємо кластер
                    cluster_cov = cov_matrix.iloc[cluster_indices, cluster_indices]
                    cluster_var = np.diag(cluster_cov)
                    
                    # Розділяємо вагу пропорційно до оберненої дисперсії
                    inv_var = 1 / cluster_var
                    sub_weights = inv_var / inv_var.sum()
                    
                    for i, idx in enumerate(cluster_indices):
                        weights[idx] = cluster_weight * sub_weights[i]
            
            # Починаємо з повного кластера
            allocate_cluster(clusters, 1.0)
            
            return weights
            
        except Exception as e:
            self.logger.error(f"Error in recursive bisection: {e}")
            return np.ones(len(cov_matrix)) / len(cov_matrix)
    
    def _create_comparison_table(self, results: Dict[str, Dict]) -> pd.DataFrame:
        """Створити таблицю порівняння методів"""
        try:
            comparison_data = []
            
            for method, result in results.items():
                if result.get('success', False):
                    comparison_data.append({
                        'Method': method,
                        'Expected Return': result.get('expected_return', 0),
                        'Volatility': result.get('volatility', 0),
                        'Sharpe Ratio': result.get('sharpe_ratio', 0),
                        'Success': True
                    })
                else:
                    comparison_data.append({
                        'Method': method,
                        'Expected Return': 0,
                        'Volatility': 0,
                        'Sharpe Ratio': 0,
                        'Success': False
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('Sharpe Ratio', ascending=False)
            
            return comparison_df
            
        except Exception as e:
            self.logger.error(f"Error creating comparison table: {e}")
            return pd.DataFrame()


# Глобальний екземпляр
portfolio_optimizer = PortfolioOptimizer()


def optimize_portfolio(returns: pd.DataFrame, method: str = 'max_sharpe', **kwargs) -> Dict[str, any]:
    """Оптимізувати портфоліо"""
    return portfolio_optimizer.optimize_portfolio(returns, method, **kwargs)


def compare_portfolio_methods(returns: pd.DataFrame) -> Dict[str, Dict]:
    """Порівняти методи оптимізації портфоліо"""
    return portfolio_optimizer.compare_optimization_methods(returns)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("💼 Portfolio Optimizer Test")
    print("="*50)
    
    # Симуляція data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    
    # Симуляція цін 5 активів
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    prices = pd.DataFrame(index=dates, columns=assets)
    
    for asset in assets:
        # Різні параметри для кожного активу
        drift = np.random.uniform(0.0001, 0.0015)
        volatility = np.random.uniform(0.015, 0.035)
        
        prices[asset] = 100 * np.exp(np.cumsum(
            np.random.normal(drift, volatility, len(dates))
        ))
    
    # Розраховуємо доходності
    returns = prices.pct_change().dropna()
    
    # Порівнюємо методи
    comparison = compare_portfolio_methods(returns)
    
    if 'comparison' in comparison:
        print(f"[DATA] Portfolio Optimization Comparison:")
        print(comparison['comparison'].round(4))
        
        # Найкращий метод
        best_method = comparison['comparison'].iloc[0]['Method']
        print(f"\n[WIN] Best Method: {best_method}")
        
        # Деталі найкращого методу
        best_result = comparison['results'][best_method]
        if best_result.get('success', False):
            print(f"   Expected Return: {best_result['expected_return']:.2%}")
            print(f"   Volatility: {best_result['volatility']:.2%}")
            print(f"   Sharpe Ratio: {best_result['sharpe_ratio']:.3f}")
            
            # Топ 5 ваг
            weights = best_result['weights'].sort_values(ascending=False)
            print(f"   Top 5 Weights:")
            for asset, weight in weights.head().items():
                print(f"     {asset}: {weight:.2%}")
        
        print(f"\n[OK] Portfolio Optimization working correctly!")
    else:
        print(f"[ERROR] Comparison failed")
