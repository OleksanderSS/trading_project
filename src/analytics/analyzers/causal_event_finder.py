import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List
import networkx as nx
from dowhy import CausalModel
from src.core.logging.logger import ProjectLogger

from ..interfaces import IAnalyzer

logger = ProjectLogger.get_logger(__name__)

# Compatibility shim for networkx/dowhy API differences
if not hasattr(nx.algorithms, 'd_separated') and hasattr(nx.algorithms, 'd_separation'):
    nx.algorithms.d_separated = nx.algorithms.d_separation
    logging.getLogger(__name__).info("Patched networkx.algorithms.d_separated alias for dowhy compatibility")

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
            df = self._prepare_analysis_data(data)
            if not isinstance(df, pd.DataFrame):
                return df  # Return error status from preparation
            
            validation_result = self._validate_data_columns(df)
            if validation_result.get("status") != "valid":
                return validation_result
            
            available_causes = self._filter_available_causes(df)
            if not available_causes:
                return {"causal_effect": 0.0, "status": "no_confounders"}
            
            variance_check = self._validate_treatment_outcome_variance(df)
            if variance_check.get("status") != "valid":
                return variance_check
            
            return self._run_causal_analysis(df, available_causes)
            
        except Exception as e:
            logger.error(f"Causal analysis failed: {e}", exc_info=True)
            return {"causal_effect": 0.0, "status": "error", "error": str(e)}
    
    def _filter_available_causes(self, df: pd.DataFrame) -> List[str]:
        """Filter common causes to only existing columns."""
        available_causes = [c for c in self.common_causes if c in df.columns]
        
        if not available_causes:
            logger.warning("No common causes available, skipping causal analysis")
        
        return available_causes
    
    def _validate_treatment_outcome_variance(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate treatment and outcome have sufficient variance."""
        if df[self.treatment].nunique() < 2:
            logger.warning(f"Treatment '{self.treatment}' has no variance - cannot estimate causal effect")
            return {"causal_effect": 0.0, "status": "no_treatment_variance"}
        
        if df[self.outcome].nunique() < 2:
            logger.warning(f"Outcome '{self.outcome}' has no variance - cannot estimate causal effect")
            return {"causal_effect": 0.0, "status": "no_outcome_variance"}
        
        return {"status": "valid"}
    
    def _run_causal_analysis(self, df: pd.DataFrame, available_causes: List[str]) -> Dict[str, Any]:
        """Run causal analysis using DoWhy library."""
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

    def _prepare_analysis_data(self, data: Any) -> Any:
        """Prepare and validate input data for analysis."""
        if isinstance(data, dict):
            return self._prepare_dict_data(data)
        else:
            return self._validate_dataframe(data)
    
    def _prepare_dict_data(self, data: dict) -> Any:
        """Prepare data from dictionary input."""
        price_data = data.get('price_data')
        macro_data = data.get('macro_data')
        
        if self._no_data_available(price_data, macro_data):
            return {"causal_effect": 0.0, "status": "no_data"}
        
        df = self._merge_data_sources(price_data, macro_data)
        return self._validate_dataframe(df)
    
    def _no_data_available(self, price_data: Any, macro_data: Any) -> bool:
        """Check if no data is available."""
        if price_data is None and macro_data is None:
            logger.warning("Both price_data and macro_data are None - skipping causal analysis")
            return True
        return False
    
    def _merge_data_sources(self, price_data: Any, macro_data: Any) -> pd.DataFrame:
        """Merge price and macro data sources."""
        if price_data is not None and macro_data is not None:
            return pd.merge(price_data, macro_data, 
                          left_index=True, right_index=True, how='left', on=None,
                          validate='one_to_one')
        elif price_data is not None:
            return price_data
        else:
            return macro_data
    
    def _validate_dataframe(self, df: pd.DataFrame) -> Any:
        """Validate dataframe has sufficient data."""
        if df.empty or len(df) < 10:
            logger.warning("Insufficient data for causal analysis")
            return {"causal_effect": 0.0, "status": "insufficient_data"}
        
        return df

    def _validate_data_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate that required columns exist in the data."""
        if self.treatment not in df.columns:
            logger.warning(f"Treatment column '{self.treatment}' not found in data - skipping causal analysis")
            return {"causal_effect": 0.0, "status": "missing_treatment_column", "treatment": self.treatment}
        
        if self.outcome not in df.columns:
            logger.warning(f"Outcome column '{self.outcome}' not found in data - skipping causal analysis")
            return {"causal_effect": 0.0, "status": "missing_outcome_column", "outcome": self.outcome}
        
        return {"status": "valid"}


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
