import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class KnnModelWrapper:
    """
    A stateful wrapper around the scikit-learn NearestNeighbors model.
    It handles feature scaling, fitting, and finding neighbors.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = 'minkowski', p: int = 2):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors must be a positive integer.")
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric, p=p)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.fitted_data_index = None

    def fit(self, features_df: pd.DataFrame):
        """
        Fits the KNN model on the provided historical feature data.
        It scales the data and stores the index for later reference.

        Note: To prevent scaler leakage, ensure features_df contains only
        historical/training data, not future/test data.
        """
        if features_df.empty:
            logger.exception("Cannot fit on an empty DataFrame.")
            raise ValueError("Input DataFrame for fitting cannot be empty.")

        # Ensure we only use numeric, non-null data
        numeric_df = features_df.select_dtypes(include=['number']).dropna()
        if numeric_df.empty:
            logger.exception("No numeric, non-null data available for fitting after cleaning.")
            raise ValueError("Cleaned DataFrame has no data to fit.")

        dropped = len(features_df) - len(numeric_df)
        if dropped:
            # kneighbors returns POSITIONS into this cleaned array, and a
            # caller that maps them back through the ORIGINAL index gets a
            # different row for every neighbour after the first gap --
            # silently, with plausible-looking ids. Use neighbor_labels()
            # instead of indexing the input frame.
            logger.warning(
                "KnnModelWrapper dropped %d of %d rows while cleaning; "
                "neighbour positions no longer line up with the input "
                "frame's index. Map them with neighbor_labels().",
                dropped, len(features_df),
            )
        self.fitted_data_index = numeric_df.index
        scaled_features = self.scaler.fit_transform(numeric_df)
        self.model.fit(scaled_features)
        self.is_fitted = True
        logger.info(f"KnnModelWrapper fitted on data with shape {scaled_features.shape}.")

    def find_neighbors(self, target_features_df: pd.DataFrame) -> tuple:
        """
        Finds the k-most similar items for each row in the target DataFrame.

        Returns:
            A tuple containing (distances, indices) of the neighbors.
        """
        if not self.is_fitted:
            logger.exception("The model must be fitted before finding neighbors.")
            raise RuntimeError("Model is not fitted. Call .fit() first.")

        # Ensure columns and data types match what the scaler expects
        numeric_target_df = target_features_df.select_dtypes(include=['number']).dropna()
        if numeric_target_df.empty:
            logger.warning("Target DataFrame is empty or has no numeric data after cleaning.")
            return ([], [])

        try:
            # Use the already fitted scaler to transform the target data
            scaled_target = self.scaler.transform(numeric_target_df)
        except ValueError as e:
            logger.exception(f"Error transforming target data. Ensure columns match the fitted data. Details: {e}")
            raise

        distances, indices = self.model.kneighbors(scaled_target)
        return distances, indices

    def neighbor_labels(self, indices) -> list[list]:
        """Translate neighbour POSITIONS into the fitted frame's index labels.

        fit() drops non-numeric and null rows, so the positions kneighbors
        returns address the cleaned array, not the frame the caller passed
        in. fitted_data_index was already being stored for this and nothing
        used it; indexing the original frame instead works only as long as
        nothing was dropped, which is a property of the caller's data rather
        than of this class.
        """
        if not self.is_fitted or self.fitted_data_index is None:
            raise RuntimeError("Model is not fitted. Call .fit() first.")

        return [
            [self.fitted_data_index[int(position)] for position in row]
            for row in indices
        ]
