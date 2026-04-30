import pandas as pd
from typing import Dict, Any, List

from ..interfaces import IAnalyzer
from ..utils.knn_model_wrapper import KnnModelWrapper  # Corrected import path
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class KnnSimilarityFinder(IAnalyzer):
    """
    A stateless analyzer that uses a KNN model to find similar items.
    It creates, fits, and uses a KNN model within a single 'analyze' call.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initializes the finder with configuration.

        Args:
            config (Dict[str, Any]): Configuration dictionary,
                                   expecting 'n_neighbors'.
        """
        self.config = config or {}
        self.n_neighbors = self.config.get('n_neighbors', 5)
        if self.n_neighbors <= 0:
            raise ValueError(
                "n_neighbors must be a positive integer."
            )
        logger.info(
            f"KnnSimilarityFinder initialized "
            f"with n_neighbors={self.n_neighbors}."
        )

    def analyze(
        self, data: Dict[str, pd.DataFrame], **kwargs
    ) -> Dict[str, Any]:
        """
        Finds similar items in a stateless manner.

        This method expects a dictionary with 'historical_features'
        and 'target_features'. It will instantiate a KNN model,
        fit it on historical data, and find neighbors for target data
        all in one go.

        Args:
            data (Dict[str, pd.DataFrame]): A dictionary containing:
                - 'historical_features': DataFrame to build KNN model
                  from.
                - 'target_features': DataFrame to find similarities
                  for.

        Returns:
            Dict[str, Any]: A dictionary containing the similarity results,
                            structured by the index of the target features.
        """
        historical_df = data.get('historical_features')
        target_df = data.get('target_features')

        if (not isinstance(historical_df, pd.DataFrame) or
                not isinstance(target_df, pd.DataFrame)):
            logger.error(
                "Input data must be a dict with "
                "'historical_features' and 'target_features' DataFrames."
            )
            return {"error": "Invalid input format."}

        try:
            # 1. Instantiate the stateful wrapper
            n_neighbors_override = kwargs.get('n_neighbors', self.n_neighbors)
            knn_model = KnnModelWrapper(n_neighbors=n_neighbors_override)

            # 2. Fit on historical data
            knn_model.fit(historical_df)

            # 3. Find neighbors for the target data
            distances, indices = knn_model.find_neighbors(target_df)

            if len(indices) == 0:
                return {"similarities": {}}

            # 4. Process and format results
            results = self._format_results(
                distances, indices, target_df.index,
                knn_model.fitted_data_index
            )

            return {"similarities": results}

        except (ValueError, RuntimeError) as e:
            logger.error(f"KNN analysis failed: {e}", exc_info=True)
            return {"error": str(e)}
        except Exception as e:
            logger.error(
                f"An unexpected error occurred during KNN analysis: {e}",
                exc_info=True
            )
            return {"error": "An unexpected error occurred."}

    def _format_results(
        self,
        distances: List,
        indices: List,
        target_index: pd.Index,
        historical_index: pd.Index
    ) -> Dict[Any, List[Dict[str, Any]]]:
        """
        Formats the raw output from the KNN model
        into a structured dictionary.
        """
        results = {}
        for i, target_id in enumerate(target_index):
            similar_items = []
            if i >= len(indices):
                continue

            for j, neighbor_idx in enumerate(indices[i]):
                neighbor_id = historical_index[neighbor_idx]

                # Calculate a normalized similarity score (0 to 1)
                # Add a small epsilon to avoid division by zero
                similarity_score = 1 / (1 + distances[i][j])

                similar_items.append({
                    "id": neighbor_id,
                    "distance": round(float(distances[i][j]), 5),
                    "similarity_score": round(similarity_score, 4)
                })

            # Sort by similarity score in descending order
            results[target_id] = sorted(
                similar_items, key=lambda x: x['similarity_score'],
                reverse=True
            )

        return results
