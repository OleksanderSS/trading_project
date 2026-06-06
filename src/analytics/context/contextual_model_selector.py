
import logging
from typing import Any

import numpy as np
import pandas as pd

from ..analyzers.knn_similarity_finder import KnnSimilarityFinder
from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class ContextualModelSelector(IAnalyzer):
    """
    Selects the optimal predictive model by analyzing historical performance
    in contexts similar to the current one. This analyzer requires a pre-fitted
    KnnSimilarityFinder instance.
    """

    def __init__(self, available_models: list[str]):
        """
        Initializes the ContextualModelSelector.

        Args:
            available_models (List[str]): A list of model names that are available for selection.
        """
        if not available_models:
            raise ValueError("The list of available models cannot be empty.")
        self.available_models = available_models
        logger.info(f"ContextualModelSelector initialized for models: {self.available_models}")

    def analyze(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        Selects the best model based on the provided context and a similarity finder.

        Args:
            data (Dict[str, Any]): A dictionary containing:
                - 'current_context' (pd.Series): The feature vector of the current market context.
                - 'similarity_finder' (KnnSimilarityFinder): A pre-fitted instance.
            **kwargs: Not used in this implementation.

        Returns:
            Dict[str, Any]: A dictionary containing the name of the best model,
                          confidence level, and detailed analysis results.
        """
        validation_result = self._validate_analyze_inputs(data)
        if not validation_result['valid']:
            return self._heuristic_fallback(validation_result['error'])

        try:
            analysis_result = self._perform_knn_analysis(data)
            return analysis_result
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error during kNN model selection: {e}", exc_info=True)
            return self._heuristic_fallback(f"Exception: {str(e)}")

    def select_models(self, ticker: str, context_fingerprint: str, data: dict[str, Any] | None = None) -> list[str] | None:
        """
        Compatibility wrapper for callers that expect a simple "recommended models" API.

        This selector fundamentally needs:
          - current_context: pd.Series
          - similarity_finder: KnnSimilarityFinder (pre-fitted with historical contexts/outcomes)

        If those are not provided, we return None and let the caller fall back.
        """
        if not data:
            return None
        current_context = data.get("current_context")
        finder = data.get("similarity_finder")
        if not isinstance(current_context, pd.Series) or not isinstance(finder, KnnSimilarityFinder):
            return None

        res = self.analyze({"current_context": current_context, "similarity_finder": finder})
        selected = res.get("selected_model")
        if not selected:
            return None
        return [str(selected)]

    def _validate_analyze_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        """Validate inputs for analyze method."""
        current_context = data.get('current_context')
        finder = data.get('similarity_finder')

        if not isinstance(current_context, pd.Series) or not isinstance(finder, KnnSimilarityFinder):
            return {
                'valid': False,
                'error': "Invalid input: 'current_context' must be a pd.Series and 'similarity_finder' a KnnSimilarityFinder instance."
            }

        return {'valid': True, 'error': None}

    def _perform_knn_analysis(self, data: dict[str, Any]) -> dict[str, Any]:
        """Perform kNN-based model selection analysis."""
        try:
            analysis_context = self._prepare_analysis_context(data)
            neighbor_analysis = self._analyze_neighbors(analysis_context)

            if not neighbor_analysis['has_scores']:
                return self._heuristic_fallback("No performance data for models in neighbors")

            return self._complete_model_selection(analysis_context, neighbor_analysis)
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error during kNN analysis: {e}")
            raise

    def _prepare_analysis_context(self, data: dict[str, Any]) -> dict[str, Any]:
        """Prepare analysis context from input data."""
        return {
            'current_context': data['current_context'],
            'finder': data['similarity_finder']
        }

    def _analyze_neighbors(self, analysis_context: dict[str, Any]) -> dict[str, Any]:
        """Analyze similar neighbors and calculate model scores."""
        finder = analysis_context['finder']
        current_context = analysis_context['current_context']

        neighbor_indices = self._find_similar_neighbors(finder, current_context)
        model_scores = self._calculate_model_scores(finder, neighbor_indices)

        return {
            'neighbor_indices': neighbor_indices,
            'model_scores': model_scores,
            'has_scores': bool(model_scores)
        }

    def _complete_model_selection(self, analysis_context: dict[str, Any], neighbor_analysis: dict[str, Any]) -> dict[str, Any]:
        """Complete the model selection process."""
        finder = analysis_context['finder']
        model_scores = neighbor_analysis['model_scores']
        neighbor_indices = neighbor_analysis['neighbor_indices']

        best_model_name = self._select_best_model(model_scores)
        confidence = self._calculate_model_confidence(finder, neighbor_indices, best_model_name)

        return self._build_analysis_result(best_model_name, confidence, model_scores, neighbor_indices)

    def _select_best_model(self, model_scores: dict[str, float]) -> str:
        """Select the best model from scores."""
        return max(model_scores, key=model_scores.get)

    def _find_similar_neighbors(self, finder: KnnSimilarityFinder, current_context: pd.Series) -> np.ndarray:
        """Find similar historical situations."""
        neighbor_indices, _ = finder.find_similar_situations(current_context)

        if not neighbor_indices.any():
            logger.warning("KnnSimilarityFinder returned no neighbors. Falling back to heuristic.")
            raise ValueError("No similar neighbors found")

        return neighbor_indices

    def _calculate_model_scores(self, finder: KnnSimilarityFinder, neighbor_indices: np.ndarray) -> dict[str, float]:
        """Calculate performance scores for all models."""
        neighbor_outcomes = finder.historical_outcomes.iloc[neighbor_indices]
        model_scores = {}

        for model_name in self.available_models:
            target_col = f"target_{model_name}"
            if target_col in neighbor_outcomes.columns:
                model_scores[model_name] = neighbor_outcomes[target_col].mean()

        return model_scores

    def _calculate_model_confidence(self, finder: KnnSimilarityFinder, neighbor_indices: np.ndarray, best_model_name: str) -> float:
        """Calculate confidence based on win rate in neighboring instances."""
        win_count = self._count_model_wins(finder, neighbor_indices, best_model_name)
        return win_count / len(neighbor_indices) if len(neighbor_indices) > 0 else 0.0

    def _count_model_wins(self, finder: KnnSimilarityFinder, neighbor_indices: np.ndarray, best_model_name: str) -> int:
        """Count wins for the best model across neighbors."""
        win_count = 0

        for idx in neighbor_indices:
            row = finder.historical_outcomes.iloc[idx]
            if self._is_best_model_in_row(row, best_model_name):
                win_count += 1

        return win_count

    def _is_best_model_in_row(self, row: pd.Series, best_model_name: str) -> bool:
        """Check if the best model is the top performer in this row."""
        best_in_row = self._find_best_model_in_row(row)
        return best_in_row == best_model_name

    def _find_best_model_in_row(self, row: pd.Series) -> str:
        """Find the best performing model in a specific historical row."""
        performances = self._extract_model_performances(row)
        return self._get_top_performing_model(performances)

    def _extract_model_performances(self, row: pd.Series) -> list[tuple[str, float]]:
        """Extract model performances from a row."""
        return [
            (model, row.get(f"target_{model}", -np.inf))
            for model in self.available_models
        ]

    def _get_top_performing_model(self, performances: list[tuple[str, float]]) -> str:
        """Get the top performing model from performances list."""
        return max(performances, key=lambda item: item[1])[0]

    def _build_analysis_result(self, best_model_name: str, confidence: float, model_scores: dict[str, float], neighbor_indices: np.ndarray) -> dict[str, Any]:
        """Build the final analysis result."""
        analysis_details = {
            "selected_model": best_model_name,
            "confidence": confidence,
            "avg_performance_in_neighbors": model_scores,
            "num_neighbors_analyzed": len(neighbor_indices),
            "status": "Success"
        }
        logger.info(f"Model selection complete. Best model: {best_model_name} with confidence {confidence:.2f}")
        return analysis_details

    def _heuristic_fallback(self, reason: str) -> dict[str, Any]:
        """Provides a simple fallback model selection if the primary logic fails."""
        logger.error(f"CRITICAL: Falling back to heuristic model selection due to: {reason}")

        # Simple heuristic: prefer 'LSTM' if available, otherwise take the first model
        best_model = 'LSTM' if 'LSTM' in self.available_models else self.available_models[0]

        return {
            "selected_model": best_model,
            "confidence": 0.0,  # Explicitly 0.0 to signal invalid/heuristic selection
            "status": "Fallback",
            "reason": reason
        }
