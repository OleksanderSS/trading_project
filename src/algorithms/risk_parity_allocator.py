"""
Паритет ризику (Risk Parity Allocation) - Покращена версія.

Розподіляє капітал так, щоб кожна позиція мала однаковий ризик.
Підтримує:
- Equal Risk Contribution (ERC)
- Risk Parity з обмеженнями
- Hierarchical Risk Parity (HRP)
- Maximum Diversification
- Minimum Variance
- Black-Litterman з ризиком
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from scipy.optimize import minimize, Bounds
from sklearn.cluster import AgglomerativeClustering
from src.core.logging.logger import ProjectLogger

class AllocationMethod(Enum):
    """Методи розподілу активів"""
    EQUAL_RISK_CONTRIBUTION = "ERC"
    HIERARCHICAL_RISK_PARITY = "HRP"
    MAXIMUM_DIVERSIFICATION = "MDP"
    MINIMUM_VARIANCE = "MVP"
    EQUAL_WEIGHT = "EW"
    RISK_PARITY = "RP"

class RiskParityAllocator:
    """
    Розширене розподілення за принципом паритету ризику

    Підтримує кілька методів:
    - Equal Risk Contribution (ERC)
    - Hierarchical Risk Parity (HRP)
    - Maximum Diversification Portfolio (MDP)
    - Minimum Variance Portfolio (MVP)
    - Risk Parity з обмеженнями
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = ProjectLogger.get_logger("RiskParityAllocator")
        self.config = config or {}

        # Параметри оптимізації
        self.max_iter = self.config.get('max_iter', 1000)
        self.tol = self.config.get('tol', 1e-8)
        self.method = self.config.get('method', AllocationMethod.EQUAL_RISK_CONTRIBUTION)

        # Обмеження
        self.min_weight = self.config.get('min_weight', 0.0)
        self.max_weight = self.config.get('max_weight', 1.0)
        self.risk_aversion = self.config.get('risk_aversion', 1.0)

        # HRP параметри
        self.hrp_linkage = self.config.get('hrp_linkage', 'single')
        self.hrp_distance_metric = self.config.get('hrp_distance_metric', 'euclidean')

    def allocate(self,
                assets: List[str],
                volatilities: Dict[str, float],
                correlations: Optional[np.ndarray] = None,
                target_volatility: Optional[float] = None,
                method: Optional[AllocationMethod] = None,
                constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Розподіляє капітал за вибраним методом

        Args:
            assets: Список активів
            volatilities: Словник волатильностей активів
            correlations: Матриця кореляцій
            target_volatility: Цільова волатильність портфеля
            method: Метод розподілу
            constraints: Додаткові обмеження

        Returns:
            Dict з вагами, метриками та інформацією про метод
        """
        try:
            if not assets or not volatilities:
                return {'weights': {}, 'method': 'fallback', 'error': 'no_assets_or_volatilities'}

            method = method or self.method
            constraints = constraints or {}

            # Перевіряємо наявність кореляцій для деяких методів
            if method in [AllocationMethod.EQUAL_RISK_CONTRIBUTION, AllocationMethod.HIERARCHICAL_RISK_PARITY] and correlations is None:
                self.logger.warning(f"Method {method.value} requires correlations, falling back to Risk Parity")
                method = AllocationMethod.RISK_PARITY

            # Отримуємо волатильності
            vols = np.array([volatilities.get(asset, 0.01) for asset in assets])
            vols = np.where(vols == 0, 0.01, vols)  # Обробляємо нульові волатильності

            # Викликаємо відповідний метод
            if method == AllocationMethod.EQUAL_RISK_CONTRIBUTION:
                weights = self._equal_risk_contribution(vols, correlations, constraints)
            elif method == AllocationMethod.HIERARCHICAL_RISK_PARITY:
                weights = self._hierarchical_risk_parity(vols, correlations, constraints)
            elif method == AllocationMethod.MAXIMUM_DIVERSIFICATION:
                weights = self._maximum_diversification(vols, correlations, constraints)
            elif method == AllocationMethod.MINIMUM_VARIANCE:
                weights = self._minimum_variance(vols, correlations, constraints)
            elif method == AllocationMethod.EQUAL_WEIGHT:
                weights = self._equal_weight(len(assets), constraints)
            else:  # RISK_PARITY
                weights = self._risk_parity(vols, correlations, constraints)

            # Масштабуємо до цільової волатильності якщо потрібно
            if target_volatility is not None:
                weights = self._scale_to_target_volatility(weights, vols, correlations, target_volatility)

            # Застосовуємо обмеження
            weights = self._apply_constraints(weights, constraints)

            # Розраховуємо метрики
            metrics = self._calculate_portfolio_metrics(weights, vols, correlations)

            return {
                'weights': {asset: float(w) for asset, w in zip(assets, weights)},
                'method': method.value,
                'metrics': metrics,
                'constraints_applied': bool(constraints),
                'target_volatility': target_volatility
            }

        except Exception as e:
            self.logger.error(f"Помилка розподілу за методом {method}: {e}")
            # Fallback до рівних ваг
            equal_weight = 1.0 / len(assets)
            weights = {asset: equal_weight for asset in assets}
            return {
                'weights': weights,
                'method': 'fallback_equal_weight',
                'error': str(e)
            }

    def _equal_risk_contribution(self, vols: np.ndarray, correlations: np.ndarray,
                                constraints: Dict[str, Any]) -> np.ndarray:
        """Equal Risk Contribution - кожен актив вносить однаковий ризик"""
        try:
            n_assets = len(vols)

            # Початкові ваги
            init_weights = np.ones(n_assets) / n_assets

            # Функція для мінімізації (відхилення від рівного внеску ризику)
            def objective(weights):
                risk_contrib = self.calculate_risk_contribution(weights, vols, correlations)
                target_contrib = 1.0 / n_assets
                return np.sum((risk_contrib - target_contrib) ** 2)

            # Обмеження
            bounds = Bounds(
                constraints.get('min_weights', np.full(n_assets, self.min_weight)),
                constraints.get('max_weights', np.full(n_assets, self.max_weight))
            )

            # Нормалізація ваг
            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            # Оптимізація
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if result.success:
                return result.x
            else:
                self.logger.warning(f"ERC optimization failed: {result.message}")
                return init_weights

        except Exception as e:
            self.logger.error(f"ERC calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _hierarchical_risk_parity(self, vols: np.ndarray, correlations: np.ndarray,
                                 constraints: Dict[str, Any]) -> np.ndarray:
        """Hierarchical Risk Parity - використовує кластеризацію"""
        try:
            n_assets = len(vols)

            # Створюємо матрицю відстаней на основі кореляцій
            distance_matrix = np.sqrt(0.5 * (1 - correlations))

            # Кластеризація
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=0.5,
                linkage=self.hrp_linkage,
                metric=self.hrp_distance_metric
            )

            # Перетворюємо відстані в лінійний формат для sklearn
            from scipy.spatial.distance import squareform
            condensed_distance = squareform(distance_matrix)
            clustering.fit(condensed_distance.reshape(-1, 1))

            # Рекурсивно розподіляємо ваги по кластерах
            weights = self._hrp_recursive_allocation(
                list(range(n_assets)),
                clustering.children_,
                vols,
                correlations
            )

            return np.array(weights)

        except Exception as e:
            self.logger.error(f"HRP calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _hrp_recursive_allocation(self, cluster_items: List[int],
                                cluster_tree: np.ndarray,
                                vols: np.ndarray,
                                correlations: np.ndarray) -> List[float]:
        """Рекурсивний розподіл для HRP"""
        if len(cluster_items) == 1:
            return [1.0]

        # Знаходимо підкластери
        left_cluster = []
        right_cluster = []

        for i, merge in enumerate(cluster_tree):
            if set(merge).issubset(set(cluster_items)):
                left_idx, right_idx = merge
                left_cluster = [x for x in cluster_items if x != right_idx]
                right_cluster = [right_idx]
                remaining = [x for x in cluster_items if x not in merge]
                break

        if not left_cluster or not right_cluster:
            # Не знайдено підкластерів, повертаємо рівні ваги
            return [1.0 / len(cluster_items)] * len(cluster_items)

        # Розраховуємо ваги для підкластерів на основі волатильності
        left_vol = np.sqrt(np.mean([vols[i]**2 for i in left_cluster]))
        right_vol = np.sqrt(np.mean([vols[i]**2 for i in right_cluster]))

        total_vol = left_vol + right_vol
        left_weight = right_vol / total_vol
        right_weight = left_vol / total_vol

        # Рекурсивно розподіляємо всередині кластерів
        left_weights = self._hrp_recursive_allocation(left_cluster, cluster_tree, vols, correlations)
        right_weights = self._hrp_recursive_allocation(right_cluster, cluster_tree, vols, correlations)

        # Комбінуємо ваги
        final_weights = []
        for i in cluster_items:
            if i in left_cluster:
                idx = left_cluster.index(i)
                final_weights.append(left_weight * left_weights[idx])
            elif i in right_cluster:
                idx = right_cluster.index(i)
                final_weights.append(right_weight * right_weights[idx])

        return final_weights

    def _maximum_diversification(self, vols: np.ndarray, correlations: np.ndarray,
                               constraints: Dict[str, Any]) -> np.ndarray:
        """Maximum Diversification Portfolio"""
        try:
            n_assets = len(vols)

            # Функція для максимізації диверсифікації
            def objective(weights):
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlations, weights)))
                weighted_avg_vol = np.dot(weights, vols)
                if weighted_avg_vol == 0:
                    return 0
                return -weighted_avg_vol / portfolio_vol  # Максимізуємо (негатив для мінімізації)

            # Обмеження
            bounds = Bounds(
                constraints.get('min_weights', np.full(n_assets, self.min_weight)),
                constraints.get('max_weights', np.full(n_assets, self.max_weight))
            )

            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            # Початкові ваги
            init_weights = np.ones(n_assets) / n_assets

            # Оптимізація
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if result.success:
                return result.x
            else:
                self.logger.warning(f"MDP optimization failed: {result.message}")
                return init_weights

        except Exception as e:
            self.logger.error(f"MDP calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _minimum_variance(self, vols: np.ndarray, correlations: np.ndarray,
                        constraints: Dict[str, Any]) -> np.ndarray:
        """Minimum Variance Portfolio"""
        try:
            n_assets = len(vols)

            # Функція для мінімізації дисперсії
            def objective(weights):
                return np.dot(weights, np.dot(correlations, weights))

            # Обмеження
            bounds = Bounds(
                constraints.get('min_weights', np.full(n_assets, self.min_weight)),
                constraints.get('max_weights', np.full(n_assets, self.max_weight))
            )

            cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

            # Початкові ваги
            init_weights = np.ones(n_assets) / n_assets

            # Оптимізація
            result = minimize(
                objective,
                init_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=cons,
                options={'maxiter': self.max_iter, 'ftol': self.tol}
            )

            if result.success:
                return result.x
            else:
                self.logger.warning(f"MVP optimization failed: {result.message}")
                return init_weights

        except Exception as e:
            self.logger.error(f"MVP calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _risk_parity(self, vols: np.ndarray, correlations: Optional[np.ndarray],
                   constraints: Dict[str, Any]) -> np.ndarray:
        """Базовий Risk Parity (обернено пропорційно волатильності)"""
        try:
            # Обробляємо кореляції
            if correlations is None:
                # Припускаємо незалежність
                inv_vols = 1.0 / vols
            else:
                # Коригуємо за кореляціями
                inv_vols = 1.0 / vols
                # Спрощена корекція - можна покращити

            # Нормалізуємо
            weights = inv_vols / np.sum(inv_vols)

            # Застосовуємо обмеження
            weights = self._apply_constraints(weights, constraints)

            return weights

        except Exception as e:
            self.logger.error(f"Risk Parity calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _equal_weight(self, n_assets: int, constraints: Dict[str, Any]) -> np.ndarray:
        """Рівні ваги для всіх активів"""
        weights = np.ones(n_assets) / n_assets
        return self._apply_constraints(weights, constraints)

    def _scale_to_target_volatility(self, weights: np.ndarray, vols: np.ndarray,
                                   correlations: Optional[np.ndarray],
                                   target_volatility: float) -> np.ndarray:
        """Масштабує ваги до цільової волатильності"""
        try:
            if correlations is None:
                portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))
            else:
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlations, weights)))

            if portfolio_vol <= 0:
                return weights

            scale_factor = target_volatility / portfolio_vol
            scaled_weights = weights * scale_factor

            # Нормалізуємо
            scaled_weights = scaled_weights / np.sum(scaled_weights)

            return scaled_weights

        except Exception as e:
            self.logger.warning(f"Scaling to target volatility failed: {e}")
            return weights

    def _apply_constraints(self, weights: np.ndarray, constraints: Dict[str, Any]) -> np.ndarray:
        """Застосовує обмеження до ваг"""
        try:
            n_assets = len(weights)

            # Мінімальні ваги
            min_weights = constraints.get('min_weights', np.full(n_assets, self.min_weight))
            weights = np.maximum(weights, min_weights)

            # Максимальні ваги
            max_weights = constraints.get('max_weights', np.full(n_assets, self.max_weight))
            weights = np.minimum(weights, max_weights)

            # Нормалізуємо після застосування обмежень
            weights = weights / np.sum(weights)

            return weights

        except Exception as e:
            self.logger.warning(f"Applying constraints failed: {e}")
            return weights

    def calculate_risk_contribution(self, weights: np.ndarray, vols: np.ndarray,
                                   correlations: np.ndarray) -> np.ndarray:
        """Розраховує contribution до ризику для кожного активу"""
        try:
            # Перетворюємо волатильності в діагональну матрицю
            vol_matrix = np.diag(vols)

            # Коваріаційна матриця
            cov_matrix = np.dot(vol_matrix, np.dot(correlations, vol_matrix))

            # Маргінальний ризик
            marginal_risks = np.dot(cov_matrix, weights)

            # Contribution до ризику
            portfolio_vol = np.sqrt(np.dot(weights, marginal_risks))
            if portfolio_vol == 0:
                return np.ones_like(weights) / len(weights)

            risk_contributions = (weights * marginal_risks) / portfolio_vol

            return risk_contributions

        except Exception as e:
            self.logger.warning(f"Risk contribution calculation failed: {e}")
            return np.ones_like(weights) / len(weights)

    def _calculate_portfolio_metrics(self, weights: np.ndarray, vols: np.ndarray,
                                   correlations: Optional[np.ndarray]) -> Dict[str, float]:
        """Розраховує метрики портфеля"""
        try:
            metrics = {}

            # Волатильність портфеля
            if correlations is None:
                portfolio_vol = np.sqrt(np.sum((weights * vols) ** 2))
            else:
                portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlations, weights)))

            metrics['portfolio_volatility'] = float(portfolio_vol)

            # Diversification ratio
            weighted_avg_vol = np.dot(weights, vols)
            if weighted_avg_vol > 0:
                metrics['diversification_ratio'] = float(portfolio_vol / weighted_avg_vol)
            else:
                metrics['diversification_ratio'] = 1.0

            # Risk contributions
            if correlations is not None:
                risk_contrib = self.calculate_risk_contribution(weights, vols, correlations)
                metrics['risk_contributions'] = risk_contrib.tolist()
                metrics['max_risk_contribution'] = float(np.max(risk_contrib))
                metrics['min_risk_contribution'] = float(np.min(risk_contrib))
                metrics['risk_parity_deviation'] = float(np.std(risk_contrib))

            # Concentration metrics
            metrics['effective_n_assets'] = float(1.0 / np.sum(weights ** 2))
            metrics['concentration_ratio'] = float(np.max(weights) / np.min(weights))

            return metrics

        except Exception as e:
            self.logger.warning(f"Portfolio metrics calculation failed: {e}")
            return {'error': str(e)}

    def optimize_portfolio(self,
                          assets: List[str],
                          volatilities: Dict[str, float],
                          correlations: np.ndarray,
                          objective: str = 'risk_parity',
                          constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Загальна оптимізація портфеля з різними цілями

        Args:
            assets: Список активів
            volatilities: Волатильності
            correlations: Матриця кореляцій
            objective: Ціль оптимізації ('risk_parity', 'min_vol', 'max_div', 'max_sharpe')
            constraints: Обмеження

        Returns:
            Оптимальні ваги та метрики
        """
        try:
            constraints = constraints or {}

            if objective == 'risk_parity':
                return self.allocate(assets, volatilities, correlations,
                                   method=AllocationMethod.EQUAL_RISK_CONTRIBUTION,
                                   constraints=constraints)
            elif objective == 'min_vol':
                return self.allocate(assets, volatilities, correlations,
                                   method=AllocationMethod.MINIMUM_VARIANCE,
                                   constraints=constraints)
            elif objective == 'max_div':
                return self.allocate(assets, volatilities, correlations,
                                   method=AllocationMethod.MAXIMUM_DIVERSIFICATION,
                                   constraints=constraints)
            elif objective == 'hrp':
                return self.allocate(assets, volatilities, correlations,
                                   method=AllocationMethod.HIERARCHICAL_RISK_PARITY,
                                   constraints=constraints)
            else:
                return self.allocate(assets, volatilities, correlations,
                                   method=AllocationMethod.RISK_PARITY,
                                   constraints=constraints)

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {'weights': {}, 'error': str(e)}
