import pandas as pd
import numpy as np
import logging
from typing import Dict, Any
from dowhy import CausalModel

from ..interfaces import IAnalyzer

logger = logging.getLogger(__name__)

class CausalEventFinder(IAnalyzer):
    """
    Wrapper for CausalEngine that implements IAnalyzer interface.
    Detects causal events and estimates their effects on outcomes.
    """
    
    def __init__(self, treatment: str = "event_detected", outcome: str = "future_return", 
                 common_causes: list = None):
        """
        Initialize the CausalEventFinder.
        
        Args:
            treatment: Column name for treatment variable
            outcome: Column name for outcome variable
            common_causes: List of confounding variables
        """
        self.treatment = treatment
        self.outcome = outcome
        self.common_causes = common_causes or []
        logger.info(f"CausalEventFinder initialized: treatment={treatment}, outcome={outcome}")
    
    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Analyze data to find causal effects.
        
        Args:
            data: Can be DataFrame or dict with 'macro_data' and 'price_data'
            **kwargs: Additional parameters
            
        Returns:
            Dict with causal analysis results
        """
        try:
            # Handle dict input (multiple data sources)
            if isinstance(data, dict):
                # Merge macro and price data if both provided
                if 'price_data' in data and 'macro_data' in data:
                    df = pd.merge(data['price_data'], data['macro_data'], 
                                left_index=True, right_index=True, how='left')
                elif 'price_data' in data:
                    df = data['price_data']
                else:
                    df = data.get('macro_data', pd.DataFrame())
            else:
                df = data
            
            if df.empty or len(df) < 10:
                logger.warning("Insufficient data for causal analysis")
                return {"causal_effect": 0.0, "status": "insufficient_data"}
            
            # ⚠️ CRITICAL FIX: Check if required columns exist
            # Do NOT create synthetic dummy columns - instead skip analysis if missing
            if self.treatment not in df.columns:
                logger.warning(f"Treatment column '{self.treatment}' not found in data - skipping causal analysis")
                return {"causal_effect": 0.0, "status": "missing_treatment_column", "treatment": self.treatment}
            
            if self.outcome not in df.columns:
                logger.warning(f"Outcome column '{self.outcome}' not found in data - skipping causal analysis")
                return {"causal_effect": 0.0, "status": "missing_outcome_column", "outcome": self.outcome}
            
            # Filter common causes to only existing columns
            available_causes = [c for c in self.common_causes if c in df.columns]
            
            if not available_causes:
                logger.warning("No common causes available, skipping causal analysis")
                return {"causal_effect": 0.0, "status": "no_confounders"}
            
            # Validate treatment and outcome have sufficient variance
            if df[self.treatment].nunique() < 2:
                logger.warning(f"Treatment '{self.treatment}' has no variance - cannot estimate causal effect")
                return {"causal_effect": 0.0, "status": "no_treatment_variance"}
            
            if df[self.outcome].nunique() < 2:
                logger.warning(f"Outcome '{self.outcome}' has no variance - cannot estimate causal effect")
                return {"causal_effect": 0.0, "status": "no_outcome_variance"}
            
            # Use CausalModel from DoWhy instead of undefined CausalEngine
            try:
                from dowhy import CausalModel
                
                # Create graphical model with common causes
                gml_graph = f"digraph {{{';'.join(available_causes)}->{self.treatment}->{self.outcome}}}"
                
                model = CausalModel(
                    data=df,
                    treatment=self.treatment,
                    outcome=self.outcome,
                    common_causes=available_causes,
                    graphs=gml_graph
                )
                
                # Identify causal effect
                identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
                
                # Estimate using OLS
                estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                effect = estimate.value if hasattr(estimate, 'value') else 0.0
                
                return {
                    "causal_effect": float(effect) if not np.isnan(effect) else 0.0,
                    "treatment": self.treatment,
                    "outcome": self.outcome,
                    "common_causes": available_causes,
                    "status": "success"
                }
            except ImportError:
                logger.error("DoWhy library not available for causal analysis")
                return {"causal_effect": 0.0, "status": "dowhy_not_available"}
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}", exc_info=True)
            return {"causal_effect": 0.0, "status": "error", "error": str(e)}


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
            raise ValueError("Data must contain treatment, outcome, and all common cause columns.")

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
            raise

    def identify_effect(self):
        """
        Identifies the causal estimand (the query to be answered).
        """
        if not self._model:
            raise RuntimeError("Model has not been created.")
        
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
            return effect_value
        except Exception as e:
            logger.error(f"Error during causal effect estimation: {e}", exc_info=True)
            return np.nan

    def run_refutation_tests(self, **kwargs) -> dict:
        """
        Runs refutation tests to check the robustness of the causal estimate.

        Returns:
            dict: A summary of the refutation test results.
        """
        if not hasattr(self, 'identified_estimand'):
            raise RuntimeError("Cannot run refutation without an identified estimand.")

        refutation_results = {}
        
        # Example: Random Common Cause
        try:
            res_random = self._model.refute_estimate(self.identified_estimand, self._model.latest_estimate, method_name="random_common_cause")
            refutation_results['random_common_cause'] = str(res_random)
            logger.info(f"Refutation (Random Common Cause): {res_random.new_effect} (p-value: {res_random.p_value})")
        except Exception as e:
            logger.warning(f"Could not run random_common_cause refutation: {e}")

        # Example: Data Subset Refuter
        try:
            res_subset = self._model.refute_estimate(self.identified_estimand, self._model.latest_estimate, method_name="data_subset_refuter", subset_fraction=0.8)
            refutation_results['data_subset_refuter'] = str(res_subset)
            logger.info(f"Refutation (Data Subset): {res_subset.new_effect} (p-value: {res_subset.p_value})")
        except Exception as e:
            logger.warning(f"Could not run data_subset_refuter refutation: {e}")

        return refutation_results
