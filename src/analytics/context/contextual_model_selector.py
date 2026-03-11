
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

from ..interfaces import IAnalyzer
from ..knn_similarity_finder import KnnSimilarityFinder

logger = logging.getLogger(__name__)

class ContextualModelSelector(IAnalyzer):
    """
    Selects the optimal predictive model by analyzing historical performance 
    in contexts similar to the current one. This analyzer requires a pre-fitted 
    KnnSimilarityFinder instance.
    """

    def __init__(self, available_models: List[str]):
        """
        Initializes the ContextualModelSelector.

        Args:
            available_models (List[str]): A list of model names that are available for selection.
        """
        if not available_models:
            raise ValueError("The list of available models cannot be empty.")
        self.available_models = available_models
        logger.info(f"ContextualModelSelector initialized for models: {self.available_models}")

    def analyze(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
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
        current_context = data.get('current_context')
        finder = data.get('similarity_finder')

        if not isinstance(current_context, pd.Series) or not isinstance(finder, KnnSimilarityFinder):
            logger.error("Invalid input: 'current_context' must be a pd.Series and 'similarity_finder' a KnnSimilarityFinder instance.")
            return self._heuristic_fallback("Invalid input data")

        try:
            # 1. Find similar historical situations using the provided finder
            neighbor_indices, _ = finder.find_similar_situations(current_context)

            if not neighbor_indices.any():
                logger.warning("KnnSimilarityFinder returned no neighbors. Falling back to heuristic.")
                return self._heuristic_fallback("No similar neighbors found")

            # 2. Analyze which models were successful in those situations
            neighbor_outcomes = finder.historical_outcomes.iloc[neighbor_indices]

            model_scores = {}
            for model_name in self.available_models:
                target_col = f"target_{model_name}"
                if target_col in neighbor_outcomes.columns:
                    model_scores[model_name] = neighbor_outcomes[target_col].mean()

            if not model_scores:
                logger.warning("Could not calculate model performance for any available models. Using heuristic.")
                return self._heuristic_fallback("No performance data for models in neighbors")

            # 3. Select the best model
            best_model_name = max(model_scores, key=model_scores.get)
            
            # 4. Calculate confidence based on win rate in neighboring historical instances
            best_model_wins = 0
            for idx in neighbor_indices:
                row = finder.historical_outcomes.iloc[idx]
                # Find the best performing model in this specific historical row
                best_in_row = max([(m, row.get(f"target_{m}", -np.inf)) for m in self.available_models], key=lambda item: item[1])
                if best_in_row[0] == best_model_name:
                    best_model_wins += 1
            
            confidence = best_model_wins / len(neighbor_indices)

            analysis_details = {
                "selected_model": best_model_name,
                "confidence": confidence,
                "avg_performance_in_neighbors": model_scores,
                "num_neighbors_analyzed": len(neighbor_indices),
                "status": "Success"
            }
            logger.info(f"Model selection complete. Best model: {best_model_name} with confidence {confidence:.2f}")
            return analysis_details

        except Exception as e:
            logger.error(f"Error during kNN model selection: {e}", exc_info=True)
            return self._heuristic_fallback(f"Exception: {str(e)}")

    def _heuristic_fallback(self, reason: str) -> Dict[str, Any]:
        """Provides a simple fallback model selection if the primary logic fails."""
        logger.warning(f"Falling back to heuristic model selection. Reason: {reason}")
        
        # Simple heuristic: prefer 'LSTM' if available, otherwise take the first model
        best_model = 'LSTM' if 'LSTM' in self.available_models else self.available_models[0]

        return {
            "selected_model": best_model,
            "confidence": 0.3,  # Low confidence for heuristic choice
            "status": "Fallback",
            "reason": reason
        }
