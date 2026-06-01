import logging
from dataclasses import dataclass
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Type-safe container for ensemble results."""

    model_names: List[str]
    weights: Dict[str, float]
    metrics: Dict[str, Any]
    meta_info: Dict[str, Any]


class EnsembleComposer:
    """
    Handles robust combination of models into ensembles.
    Decoupled from Stage 4 Modeling orchestrator logic.
    """

    def __init__(self, composer_type: str = "weighted_avg"):
        self.composer_type = composer_type

    def compose(self, models: Dict[str, Any], results: Dict[str, Any], ticker: str, target: str) -> EnsembleResult:
        """
        Creates an ensemble from top models and returns a type-safe result.
        """
        logger.info(f"Composing ensemble for {ticker}_{target} using {self.composer_type}")

        # Example logic: extract model names and simple uniform weights
        model_names = list(models.keys())
        weights = {name: 1.0 / len(model_names) for name in model_names}

        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(results)

        return EnsembleResult(
            model_names=model_names,
            weights=weights,
            metrics=aggregated_metrics,
            meta_info={"ticker": ticker, "target": target, "type": self.composer_type},
        )

    def _aggregate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Simple aggregation logic for ensemble metrics."""
        # This can be expanded based on specific requirements
        return {"status": "composed", "n_models": len(results)}
