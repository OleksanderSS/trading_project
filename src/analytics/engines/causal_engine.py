import numpy as np
import pandas as pd
from dowhy import CausalModel

from src.core.exceptions import DataProcessingError
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class CausalEngine:
    """
    Performs causal inference to estimate the effect of a specific treatment
    (e.g., a detected event) on an outcome (e.g., future returns).
    """

    def __init__(self, data: pd.DataFrame, treatment: str, outcome: str, common_causes: list = None):
        """
        Initializes the CausalEngine with data and model specifications.

        Args:
            data (pd.DataFrame): The dataset containing treatment, outcome, and common causes.
            treatment (str): The name of the column representing the treatment (binary or continuous).
            outcome (str): The name of the column representing the outcome.
            common_causes (list, optional): A list of column names to be used as common causes (confounders).
        """
        if not all(col in data.columns for col in [treatment, outcome] + (common_causes or [])):
            raise DataProcessingError("Data must contain treatment, outcome, and all common cause columns.")

        self.data = data
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes
        self._model = self._create_model()

    def _create_model(self) -> CausalModel:
        """
        Creates a CausalModel instance from the provided data and specifications.
        """
        try:
            model = CausalModel(
                data=self.data,
                treatment=self.treatment,
                outcome=self.outcome,
                common_causes=self.common_causes
            )
            logger.info("Causal model created successfully.")
            return model
        except Exception as e:
            logger.error(f"Error creating CausalModel: {e}", exc_info=True)
            raise DataProcessingError(f"Error creating CausalModel: {e}") from e

    def identify_effect(self):
        """
        Identifies the causal estimand (the query to be answered).
        """
        if not self._model:
            raise DataProcessingError("Model has not been created.")

        self.identified_estimand = self._model.identify_effect(proceed_when_unidentifiable=True)
        logger.info(f"Causal estimand identified: {self.identified_estimand}")

    def estimate_effect(self, method_name="backdoor.linear_regression", **kwargs) -> float:
        """
        Estimates the causal effect using a specified method.

        Args:
            method_name (str): The name of the estimation method to use.
            **kwargs: Additional arguments for the estimation method.

        Returns:
            float: The estimated causal effect.
        """
        if not hasattr(self, 'identified_estimand'):
            logger.warning("Estimand not identified. Identifying first.")
            self.identify_effect()

        try:
            estimate = self._model.estimate_effect(
                self.identified_estimand,
                method_name=method_name,
                **kwargs
            )
            effect_value = estimate.value
            logger.info(f"Causal effect estimated using {method_name}: {effect_value}")
            return float(effect_value) if not np.isnan(effect_value) else 0.0
        except Exception as e:
            logger.error(f"Error during causal effect estimation: {e}", exc_info=True)
            raise DataProcessingError(f"Error during causal effect estimation: {e}") from e

    def run_refutation_tests(self, **kwargs) -> dict:
        """
        Runs refutation tests to check the robustness of the causal estimate.

        Returns:
            dict: A summary of the refutation test results.
        """
        if not hasattr(self, 'identified_estimand'):
            raise DataProcessingError("Cannot run refutation without an identified estimand.")

        refutation_results = {}

        # Example: Random Common Cause
        try:
            res_random = self._model.refute_estimate(self.identified_estimand, self._model.latest_estimate, method_name="random_common_cause")
            refutation_results['random_common_cause'] = str(res_random)
            logger.info(f"Refutation (Random Common Cause): {res_random.new_effect} (p-value: {res_random.p_value})")
        except Exception as e:
            logger.error(f"Could not run random_common_cause refutation: {e}", exc_info=True)
            raise DataProcessingError(f"Could not run random_common_cause refutation: {e}") from e

        # Example: Data Subset Refuter
        try:
            res_subset = self._model.refute_estimate(self.identified_estimand, self._model.latest_estimate, method_name="data_subset_refuter", subset_fraction=0.8)
            refutation_results['data_subset_refuter'] = str(res_subset)
            logger.info(f"Refutation (Data Subset): {res_subset.new_effect} (p-value: {res_subset.p_value})")
        except Exception as e:
            logger.error(f"Could not run data_subset_refuter refutation: {e}", exc_info=True)
            raise DataProcessingError(f"Could not run data_subset_refuter refutation: {e}") from e

        return refutation_results
