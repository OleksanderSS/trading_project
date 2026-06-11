
import logging
from src.analytics.analyzers.causal_event_finder import CausalEventFinder
from src.analytics.analyzers.drift_analyzer import DriftAnalyzer
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
from src.analytics.analyzers.shap_analyzer import ShapAnalyzer
from src.analytics.analyzers.wrappers import DrawdownAnalyzer, FamaFrenchAnalyzer, VolatilityAnalyzer
from src.analytics.context.ensemble_selector import EnsembleSelector
from src.analytics.interfaces import IAnalyzer

logger = logging.getLogger(__name__)

# Registry
ANALYZER_REGISTRY: dict[str, type[IAnalyzer]] = {
    "drift": DriftAnalyzer,
    "hedge_fund": HedgeFundAnalyzer,
    "causal_event": CausalEventFinder,
    "shap": ShapAnalyzer,
    "drawdown": DrawdownAnalyzer,
    "volatility": VolatilityAnalyzer,
    "fama_french": FamaFrenchAnalyzer,
    "ensemble_selector": EnsembleSelector,  # type: ignore
}


def get_analyzer(name: str, config: dict = None) -> IAnalyzer:
    """Factory method to instantiate an analyzer by name."""
    if name not in ANALYZER_REGISTRY:
        raise ValueError(f"Analyzer '{name}' not found in registry. Available: {list(ANALYZER_REGISTRY.keys())}")

    analyzer_class = ANALYZER_REGISTRY[name]
    
    # Handle different analyzer constructor signatures
    try:
        # Try with config parameter first
        return analyzer_class(config=config or {})
    except TypeError as e:
        if "config" in str(e):
            # Fallback: try without config parameter
            try:
                return analyzer_class()
            except (TypeError, ValueError) as e2:
                logger.exception(f"Failed to instantiate analyzer '{name}' without config")
                # Fallback: try with **kwargs
                try:
                    return analyzer_class(**(config or {}))
                except (TypeError, ValueError) as e3:
                    logger.exception(f"Failed to instantiate analyzer '{name}' with **kwargs")
                    raise ValueError(f"Failed to instantiate analyzer '{name}': {e3}") from e3
        else:
            raise ValueError(f"Failed to instantiate analyzer '{name}': {e}") from e
