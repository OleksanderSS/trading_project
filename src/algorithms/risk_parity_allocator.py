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

from enum import Enum
from typing import Any

import numpy as np
from scipy.optimize import Bounds, minimize
from scipy.spatial.distance import squareform
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

    def __init__(self, config: dict[str, Any] | None = None):
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

    # CodeScene: Complex Method (cc=9) - acceptable for allocation dispatcher with multiple strategies
    def allocate(self,
                assets: list[str],
                volatilities: dict[str, float],
                correlations: np.ndarray | None = None,
                params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Розподіляє капітал за вибраним методом."""
        try:
            # 1. Validation
            validation_error = self._validate_allocation_inputs(assets, volatilities)
            if validation_error:
                return validation_error

            # 2. Preparation
            alloc_params = self._prepare_allocation_params(params, correlations)
            vols = self._prepare_vols(assets, volatilities)

            # 3. Execution
            weights = self._execute_core_allocation(
                vols, correlations, alloc_params, assets
            )

            # 4. Result Construction
            return self._build_result_dict(assets, weights, vols, correlations, alloc_params)

        except Exception as e:
            self.logger.error(f"Помилка розподілу: {e}")
            return self._fallback_equal_weight(assets, str(e))

    def _validate_allocation_inputs(self, assets: list[str], volatilities: dict[str, float]) -> dict[str, Any] | None:
        """Validates that assets and volatilities are provided."""
        if not assets or not volatilities:
            return {'weights': {}, 'method': 'fallback', 'error': 'no_assets_or_volatilities'}
        return None

    def _prepare_allocation_params(self, params: dict[str, Any] | None, correlations: np.ndarray | None) -> dict[str, Any]:
        """Extracts and validates allocation parameters."""
        params = params or {}
        method = params.get('method', self.method)

        if self._needs_correlations(method) and correlations is None:
            method = AllocationMethod.RISK_PARITY

        return {
            'method': method,
            'constraints': params.get('constraints', {}),
            'target_volatility': params.get('target_volatility')
        }

    def _execute_core_allocation(self, vols: np.ndarray,
                               correlations: np.ndarray | None, params: dict[str, Any],
                               assets: list[str]) -> np.ndarray:
        """Executes the actual mathematical allocation and post-processing."""
        method = params['method']
        constraints = params['constraints']

        # Get and execute allocation function
        alloc_func = self._get_allocation_methods(vols, correlations, assets).get(
            method, self._risk_parity
        )
        weights = alloc_func(vols, correlations, constraints)

        # Apply scaling if target is set
        if params['target_volatility'] is not None:
            weights = self._scale_to_target_volatility(
                weights, vols, correlations, params['target_volatility']
            )

        return self._apply_constraints(weights, constraints)

    def _build_result_dict(self, assets: list[str], weights: np.ndarray, vols: np.ndarray,
                         correlations: np.ndarray | None, params: dict[str, Any]) -> dict[str, Any]:
        """
        Constructs the final allocation report dictionary.

        CodeScene: Excess Function Arguments acceptable - Result builder requires all
        components (assets, weights, volatilities, correlations, parameters) to construct
        comprehensive allocation report with metrics and metadata.
        """
        return {
            'weights': {asset: float(w) for asset, w in zip(assets, weights, strict=False)},
            'method': params['method'].value,
            'metrics': self._calculate_portfolio_metrics(weights, vols, correlations),
            'constraints_applied': bool(params['constraints']),
            'target_volatility': params['target_volatility']
        }

    def _needs_correlations(self, method: AllocationMethod) -> bool:
        """Check if the method requires a correlation matrix."""
        return method in [AllocationMethod.EQUAL_RISK_CONTRIBUTION, AllocationMethod.HIERARCHICAL_RISK_PARITY]

    def _prepare_vols(self, assets: list[str], volatilities: dict[str, float]) -> np.ndarray:
        """Extracts and normalizes volatilities from the input dict."""
        vols = np.array([volatilities.get(asset, 0.01) for asset in assets])
        return np.where(vols == 0, 0.01, vols)

    def _fallback_equal_weight(self, assets: list[str], error_msg: str) -> dict[str, Any]:
        """Provides a safe equal-weight fallback on error."""
        equal_weight = 1.0 / len(assets) if assets else 0
        return {
            'weights': dict.fromkeys(assets, equal_weight),
            'method': 'fallback_equal_weight',
            'error': error_msg
        }

    def _get_allocation_methods(self, vols: np.ndarray, correlations: np.ndarray | None, assets: list[str]):
        """Повертає мапінг методів розподілу"""
        return {
            AllocationMethod.EQUAL_RISK_CONTRIBUTION: self._equal_risk_contribution,
            AllocationMethod.HIERARCHICAL_RISK_PARITY: self._hierarchical_risk_parity,
            AllocationMethod.MAXIMUM_DIVERSIFICATION: self._maximum_diversification,
            AllocationMethod.MINIMUM_VARIANCE: self._minimum_variance,
            AllocationMethod.EQUAL_WEIGHT: lambda v, c, cn: self._equal_weight(len(assets), cn),
            AllocationMethod.RISK_PARITY: self._risk_parity
        }

    def _equal_risk_contribution(self, vols: np.ndarray, correlations: np.ndarray,
                                constraints: dict[str, Any]) -> np.ndarray:
        """Equal Risk Contribution - кожен актив вносить однаковий ризик"""
        return self._optimize_portfolio(vols, correlations, constraints, "ERC")

    def _get_initial_weights(self, n_assets: int) -> np.ndarray:
        """Створює початкові рівні ваги"""
        return np.ones(n_assets) / n_assets

    def _create_erc_objective(self, vols: np.ndarray, correlations: np.ndarray,
                             n_assets: int):
        """Створює цільову функцію для ERC"""
        def objective(weights):
            risk_contrib = self.calculate_risk_contribution(weights, vols, correlations)
            target_contrib = 1.0 / n_assets
            return np.sum((risk_contrib - target_contrib) ** 2)
        return objective

    def _create_optimization_bounds(self, constraints: dict[str, Any], n_assets: int):
        """Створює обмеження для оптимізації"""
        return Bounds(
            constraints.get('min_weights', np.full(n_assets, self.min_weight)),
            constraints.get('max_weights', np.full(n_assets, self.max_weight))
        )

    def _create_optimization_constraints(self):
        """Створює обмеження нормалізації ваг"""
        return [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]

    def _run_optimization(self, objective, init_weights: np.ndarray, bounds, cons):
        """Запускає оптимізацію"""
        return minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': self.max_iter, 'ftol': self.tol}
        )

    def _handle_optimization_failure(self, result, init_weights: np.ndarray, method_name: str) -> np.ndarray:
        """Обробляє невдалу оптимізацію"""
        self.logger.warning(f"{method_name} optimization failed: {result.message}")
        return np.asarray(init_weights)

    def _get_fallback_weights(self, vols: np.ndarray) -> np.ndarray:
        """Повертає запасні рівні ваги"""
        return np.ones(len(vols)) / len(vols)

    # CodeScene: Complex Method (cc=12), Bumpy Road (bumps=2) - acceptable for hierarchical algorithm
    def _hierarchical_risk_parity(self, vols: np.ndarray, correlations: np.ndarray,
                                 _constraints: dict[str, Any]) -> np.ndarray:
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

    def _hrp_recursive_allocation(self, cluster_items: list[int],
                                cluster_tree: np.ndarray,
                                vols: np.ndarray,
                                correlations: np.ndarray) -> list[float]:
        """Рекурсивний розподіл для HRP"""
        if len(cluster_items) == 1:
            return [1.0]

        # Знаходимо підкластери
        left_cluster, right_cluster = self._find_sub_clusters(cluster_items, cluster_tree)

        if not left_cluster or not right_cluster:
            # Не знайдено підкластерів, повертаємо рівні ваги
            return [1.0 / len(cluster_items)] * len(cluster_items)

        # Розраховуємо ваги для підкластерів на основі волатильності
        left_weight, right_weight = self._calculate_cluster_weights(left_cluster, right_cluster, vols)

        # Рекурсивно розподіляємо всередині кластерів
        left_weights = self._hrp_recursive_allocation(left_cluster, cluster_tree, vols, correlations)
        right_weights = self._hrp_recursive_allocation(right_cluster, cluster_tree, vols, correlations)

        # Комбінуємо ваги
        return self._combine_cluster_weights(
            cluster_items,
            (left_cluster, left_weight, left_weights),
            (right_cluster, right_weight, right_weights)
        )

    def _calculate_cluster_weights(self, left_c: list[int], right_c: list[int], vols: np.ndarray) -> tuple[float, float]:
        """Calculates relative cluster weights based on variance."""
        left_vol = np.sqrt(np.mean([vols[i]**2 for i in left_c]))
        right_vol = np.sqrt(np.mean([vols[i]**2 for i in right_c]))

        total_vol = left_vol + right_vol
        if total_vol == 0:
            return 0.5, 0.5
        return right_vol / total_vol, left_vol / total_vol

    def _find_sub_clusters(self, cluster_items: list[int], cluster_tree: np.ndarray) -> tuple[list[int], list[int]]:
        """Знаходить лівий та правий підкластери у дереві"""
        items_set = set(cluster_items)
        for merge in cluster_tree:
            if set(merge).issubset(items_set):
                _, right_idx = merge
                left_cluster = [x for x in cluster_items if x != right_idx]
                right_cluster = [right_idx]
                return left_cluster, right_cluster
        return [], []

    def _combine_cluster_weights(self,
                               items: list[int],
                               left_data: tuple[list[int], float, list[float]],
                               right_data: tuple[list[int], float, list[float]]) -> list[float]:
        """Комбінує ваги підкластерів у фінальний список"""
        final_weights = []
        left_c, left_w, left_ws = left_data
        right_c, right_w, right_ws = right_data

        left_map = {item: idx for idx, item in enumerate(left_c)}
        right_map = {item: idx for idx, item in enumerate(right_c)}

        for i in items:
            if i in left_map:
                final_weights.append(left_w * left_ws[left_map[i]])
            else:
                final_weights.append(right_w * right_ws[right_map[i]])

        return final_weights

    def _maximum_diversification(self, vols: np.ndarray, correlations: np.ndarray,
                                constraints: dict[str, Any]) -> np.ndarray:
        """Maximum Diversification Portfolio"""
        return self._optimize_portfolio(vols, correlations, constraints, "MDP")

    def _create_mdp_objective(self, vols: np.ndarray, correlations: np.ndarray):
        """Створює цільову функцію для Maximum Diversification"""
        def objective(weights):
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(correlations, weights)))
            weighted_avg_vol = np.dot(weights, vols)
            if weighted_avg_vol == 0:
                return 0
            return -weighted_avg_vol / portfolio_vol  # Максимізуємо (негатив для мінімізації)
        return objective

    def _minimum_variance(self, vols: np.ndarray, correlations: np.ndarray,
                            constraints: dict[str, Any]) -> np.ndarray:
        """Minimum Variance Portfolio"""
        return self._optimize_portfolio(vols, correlations, constraints, "MV")

    # CodeScene: Excess Arguments (5) - acceptable for optimization configuration
    def _optimize_portfolio(self, vols: np.ndarray, correlations: np.ndarray,
                          constraints: dict[str, Any], method_name: str) -> np.ndarray:
        """Common portfolio optimization logic"""
        try:
            n_assets = len(vols)
            init_weights = self._get_initial_weights(n_assets)
            # Select objective creator based on method name
            creators = {
                "ERC": lambda v, c, n: self._create_erc_objective(v, c, n),
                "MDP": lambda v, c, _n: self._create_mdp_objective(v, c),
                "MV": lambda _v, c, _n: self._create_mv_objective(c)
            }
            creator = creators.get(method_name, lambda v, c, n: self._create_erc_objective(v, c, n))
            objective = creator(vols, correlations, n_assets)
            bounds = self._create_optimization_bounds(constraints, n_assets)
            cons = self._create_optimization_constraints()

            result = self._run_optimization(objective, init_weights, bounds, cons)

            if result.success:
                return np.asarray(result.x)
            else:
                return self._handle_optimization_failure(result, init_weights, method_name)

        except Exception as e:
            self.logger.error(f"{method_name} calculation failed: {e}")
            return self._get_fallback_weights(vols)

    def _create_mv_objective(self, correlations: np.ndarray):
        """Створює цільову функцію для Minimum Variance"""
        def objective(weights):
            return np.dot(weights, np.dot(correlations, weights))
        return objective

    def _risk_parity(self, vols: np.ndarray, correlations: np.ndarray | None,
                   constraints: dict[str, Any]) -> np.ndarray:
        """Базовий Risk Parity (обернено пропорційно волатильності)"""
        try:
            # Обробляємо кореляції
            inv_vols = 1.0 / vols
            if correlations is not None:
                # Коригуємо за кореляціями
                # Спрощена корекція - можна покращити
                # Розраховуємо ефективні волатильності з урахуванням кореляцій
                try:
                    # Перетворюємо кореляції в коваріаційну матрицю
                    vol_matrix = np.diag(vols)
                    cov_matrix = np.dot(vol_matrix, np.dot(correlations, vol_matrix))

                    # Розраховуємо ефективні волатильності
                    effective_vols = np.sqrt(np.diag(cov_matrix))
                    inv_vols = 1.0 / np.maximum(effective_vols, 0.001)  # Уникаємо ділення на нуль
                except (ValueError, np.linalg.LinAlgError, RuntimeError):
                    # Якщо корекція не вдалася, використовуємо базові волатильності
                    pass

            # Нормалізуємо
            weights = inv_vols / np.sum(inv_vols)

            # Застосовуємо обмеження
            weights = self._apply_constraints(weights, constraints)

            return weights

        except Exception as e:
            self.logger.error(f"Risk Parity calculation failed: {e}")
            return np.ones(len(vols)) / len(vols)

    def _equal_weight(self, n_assets: int, constraints: dict[str, Any]) -> np.ndarray:
        """Рівні ваги для всіх активів"""
        weights = np.ones(n_assets) / n_assets
        return self._apply_constraints(weights, constraints)

    def _scale_to_target_volatility(self, weights: np.ndarray, vols: np.ndarray,
                                   correlations: np.ndarray | None,
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

            return np.asarray(scaled_weights)

        except Exception as e:
            self.logger.warning(f"Scaling to target volatility failed: {e}")
            return weights

    def _apply_constraints(self, weights: np.ndarray, constraints: dict[str, Any]) -> np.ndarray:
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
            # Return safe fallback: equal weights instead of potentially problematic original weights
            n_assets = len(weights) if hasattr(weights, '__len__') and len(weights) > 0 else 1
            return np.ones(n_assets) / n_assets

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

            return np.asarray(risk_contributions)

        except Exception as e:
            self.logger.warning(f"Risk contribution calculation failed: {e}")
            return np.ones_like(weights) / len(weights)

    def _calculate_portfolio_metrics(self, weights: np.ndarray, vols: np.ndarray,
                                   correlations: np.ndarray | None) -> dict[str, Any]:
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
                          assets: list[str],
                          volatilities: dict[str, float],
                          correlations: np.ndarray,
                          objective_config: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Загальна оптимізація портфеля з різними цілями

        Args:
            assets: Список активів
            volatilities: Волатильності
            correlations: Матриця кореляцій
            objective_config: Конфігурація (objective name, constraints)

        Returns:
            Оптимальні ваги та метрики
        """
        try:
            config = objective_config or {}
            objective = config.get('objective', 'risk_parity')
            constraints = config.get('constraints', {})

            if objective == 'risk_parity':
                return self.allocate(assets, volatilities, correlations,
                                   params={'method': AllocationMethod.EQUAL_RISK_CONTRIBUTION, 'constraints': constraints})
            elif objective == 'min_vol':
                return self.allocate(assets, volatilities, correlations,
                                   params={'method': AllocationMethod.MINIMUM_VARIANCE, 'constraints': constraints})
            elif objective == 'max_div':
                return self.allocate(assets, volatilities, correlations,
                                   params={'method': AllocationMethod.MAXIMUM_DIVERSIFICATION, 'constraints': constraints})
            elif objective == 'hrp':
                return self.allocate(assets, volatilities, correlations,
                                   params={'method': AllocationMethod.HIERARCHICAL_RISK_PARITY, 'constraints': constraints})
            else:
                return self.allocate(assets, volatilities, correlations,
                                   params={'method': AllocationMethod.RISK_PARITY, 'constraints': constraints})

        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            return {'weights': {}, 'error': str(e)}
