from collections.abc import Callable, Iterable
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("WalkForwardOptimizer")


class WalkForwardOptimizer:
    """Walk-forward optimization for stable parameter selection."""

    def __init__(self, config_manager: Any | None = None):
        self.config_manager = config_manager
        self.logger = logger

    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any] | Callable[[pd.DataFrame], dict[str, Any]] | None = None,
        anchor_type: str = "expanding",
        train_size: int = 252,
        test_size: int = 63,
        optimization_func: Callable[[pd.DataFrame], dict[str, Any]] | None = None,
        metric: str = "sharpe",
    ) -> dict[str, Any]:
        """Run rolling or expanding walk-forward validation."""
        if data.empty:
            return {
                "walk_forward_results": [],
                "average_oos_score": {"mean_oos_score": 0.0},
                "success": False,
                "error": "empty_data",
            }

        if callable(param_space) and optimization_func is None:
            optimization_func = param_space
            param_space = None

        results: list[dict[str, Any]] = []
        n_obs = len(data)
        current_start = 0
        train_size = max(1, int(train_size))
        test_size = max(1, int(test_size))

        self.logger.info(
            f"Starting WFO ({anchor_type}) on {n_obs} points. Train={train_size}, Test={test_size}"
        )

        while current_start + train_size + test_size <= n_obs:
            train_end = current_start + train_size
            test_end = train_end + test_size
            train_data = data.iloc[current_start:train_end]
            test_data = data.iloc[train_end:test_end]
            best_params = self._select_best_params(
                train_data,
                param_space if isinstance(param_space, dict) else {},
                metric,
                optimization_func,
            )
            performance = self._evaluate_parameters(test_data)
            results.append(
                {
                    "train_range": (current_start, train_end),
                    "test_range": (train_end, test_end),
                    "params": best_params,
                    "out_sample_performance": performance,
                    "oos_score": performance.get(metric, performance.get("sharpe", 0.0)),
                }
            )

            if anchor_type == "rolling":
                current_start += test_size
            else:
                train_size += test_size

        return {
            "walk_forward_results": results,
            "average_oos_score": self._calculate_average_performance(results),
            "success": bool(results),
        }

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any],
        metric: str = "sharpe",
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        anchor_type: str = "rolling",
    ) -> dict[str, Any]:
        """Run WFO using a fold count and return optimizer-friendly keys."""
        if data.empty:
            return {"success": False, "error": "empty_data", "best_params": {}, "fold_results": []}

        n_splits = max(1, int(n_splits))
        test_size = test_size or max(1, len(data) // (n_splits + 1))
        train_size = train_size or max(1, len(data) - n_splits * test_size)
        result = self.walk_forward_optimization(
            data=data,
            param_space=param_space,
            anchor_type=anchor_type,
            train_size=train_size,
            test_size=test_size,
            metric=metric,
        )
        folds = result.get("walk_forward_results", [])
        best_fold = max(folds, key=lambda fold: fold.get("oos_score", float("-inf")), default={})
        return {
            "success": bool(folds),
            "folds": folds,
            "fold_results": folds,
            "best_params": best_fold.get("params", {}),
            "best_performance": best_fold.get("out_sample_performance", {}),
            "out_of_sample_data": data.iloc[-test_size:] if test_size else data,
            "stability_score": self._calculate_stability_score(folds),
        }

    def _evaluate_parameters(self, data: pd.DataFrame | None = None) -> dict[str, float]:
        """Evaluate out-of-sample data using return, Sharpe, and max drawdown."""
        if data is None or len(data) < 2:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        returns = data.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if isinstance(returns, pd.DataFrame):
            returns = returns.mean(axis=1)
        if len(returns) == 0:
            return {"return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0}
        total_return = float((1 + returns).prod() - 1)
        std_val = float(returns.std())
        sharpe = (
            float(returns.mean() / std_val * np.sqrt(252))
            if np.isfinite(std_val) and std_val > 1e-12
            else 0.0
        )
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        max_drawdown = float(((cumulative - running_max) / running_max).min())
        return {"return": total_return, "sharpe": sharpe, "max_drawdown": max_drawdown}

    def _calculate_average_performance(self, results: list[dict[str, Any]]) -> dict[str, float]:
        if not results:
            return {"mean_oos_score": 0.0}
        scores = [r.get("oos_score", 0.0) for r in results]
        return {"mean_oos_score": float(np.mean(scores)), "std_oos_score": float(np.std(scores))}

    def _select_best_params(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any],
        metric: str,
        optimization_func: Callable[[pd.DataFrame], dict[str, Any]] | None,
    ) -> dict[str, Any]:
        if optimization_func is not None:
            return optimization_func(data)
        candidates = list(self._iter_param_candidates(param_space))
        if not candidates:
            return {}
        score = self._evaluate_parameters(data).get(metric, 0.0)
        return max(candidates, key=lambda _: score)

    @staticmethod
    def _iter_param_candidates(param_space: dict[str, Any]) -> Iterable[dict[str, Any]]:
        if not param_space:
            return []
        keys = list(param_space)
        values = [value if isinstance(value, (list, tuple, set)) else [value] for value in param_space.values()]
        return [dict(zip(keys, combo, strict=False)) for combo in product(*values)]

    @staticmethod
    def _calculate_stability_score(folds: list[dict[str, Any]]) -> float:
        scores = [fold.get("oos_score", 0.0) for fold in folds]
        if len(scores) < 2:
            return 0.0
        finite_scores = [float(score) for score in scores if np.isfinite(score)]
        if len(finite_scores) < 2:
            return 0.0
        mean_score = float(np.mean(finite_scores))
        if abs(mean_score) <= 1e-12:
            return 0.0
        return float(max(0.0, 1.0 - abs(np.std(finite_scores) / mean_score)))


class WalkForwardOptimizerExtended(WalkForwardOptimizer):
    """Extended compatibility class."""

    def run_walk_forward(
        self,
        data: pd.DataFrame,
        param_space: dict[str, Any],
        metric: str = "sharpe",
        n_splits: int = 5,
        train_size: int | None = None,
        test_size: int | None = None,
        anchor_type: str = "rolling",
    ) -> dict[str, Any]:
        return super().run_walk_forward(
            data,
            param_space,
            metric=metric,
            n_splits=n_splits,
            train_size=train_size,
            test_size=test_size,
            anchor_type=anchor_type,
        )
