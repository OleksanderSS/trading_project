
from src.analytics.analyzers.causal_event_finder import CausalEventFinder
from src.analytics.analyzers.drift_analyzer import DriftAnalyzer
from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
from src.analytics.analyzers.shap_analyzer import ShapAnalyzer
from src.analytics.analyzers.wrappers import DrawdownAnalyzer, FamaFrenchAnalyzer, VolatilityAnalyzer
from src.analytics.context.ensemble_selector import EnsembleSelector
from src.analytics.interfaces import IAnalyzer

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

    return ANALYZER_REGISTRY[name](config=config or {})
