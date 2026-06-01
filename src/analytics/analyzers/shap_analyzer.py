from typing import Any, Dict

import shap

from src.analytics.interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger
from src.core.exceptions import DataProcessingError

logger = ProjectLogger.get_logger(__name__)

class ShapAnalyzer(IAnalyzer):
    """
    Analyzer for model explainability using SHAP.
    """

    def __init__(self, model: Any = None):
        self.model = model
        self.explainer = None
        self.logger = logger

    def analyze(self, data: Any, **kwargs) -> Dict[str, Any]:
        """
        Calculates SHAP values for the provided data.

        Expects 'model' and 'features_data' in the data dictionary.
        """
        model = kwargs.get("model") or data.get("model")
        features = data.get("features_data")
        
        if model is None or features is None:
            raise DataProcessingError("Missing model or features_data for SHAP analysis.")
            
        try:
            self.explainer = shap.TreeExplainer(model)
            shap_values = self.explainer.shap_values(features)
            return {"shap_values": shap_values, "status": "success"}
        except Exception as e:
            self.logger.error(f"SHAP analysis failed: {e}", exc_info=True)
            raise DataProcessingError(f"SHAP analysis failed: {e}") from e
