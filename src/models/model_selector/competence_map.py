# models/model_selector/competence_map.py

import json
from pathlib import Path
from typing import Dict, Tuple, Any, List, Optional


def _load_history(results_path: str) -> List[Dict]:
    """Load history from JSON file."""
    path = Path(results_path)
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        try:
            return [json.loads(line) for line in f]
        except Exception:
            return []


def _extract_record_key(record: Dict) -> Optional[Tuple[str, str]]:
    """Extract and validate record key."""
    try:
        return (record["target"], json.dumps(record["context"], sort_keys=True))
    except Exception:
        return None


def _validate_record(record: Dict) -> bool:
    """Validate record has required fields."""
    required = ["target", "context", "metrics", "model"]
    return all(k in record for k in required) and record.get("metrics")


def _extract_classification_metrics(metrics: Dict, tolerance: float) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Extract main metric and warnings for classification task."""
    main_metric = None
    main_score = None
    warnings = []

    for m in ["F1", "accuracy", "roc_auc"]:
        if m in metrics:
            main_metric = m
            main_score = metrics[m]
            break

    if main_metric:
        aux = {k: v for k, v in metrics.items()
               if k in ["F1", "accuracy", "roc_auc"] and k != main_metric}
        for k, v in aux.items():
            if abs(main_score - v) > tolerance:
                warnings.append(
                    f"[WARN] Metric inconsistency: {main_metric}={main_score:.3f} and {k}={v:.3f}"
                )

    return main_metric, main_score, warnings


def _extract_regression_metrics(metrics: Dict) -> Tuple[Optional[str], Optional[float], List[str]]:
    """Extract main metric and warnings for regression task."""
    main_metric = None
    main_score = None
    warnings = []

    if "mae" in metrics:
        main_metric = "mae"
        main_score = metrics["mae"]
    elif "r2" in metrics:
        main_metric = "r2"
        main_score = metrics["r2"]

    if "mae" in metrics and "r2" in metrics:
        if metrics["r2"] < 0.2 and metrics["mae"] < 1.0:
            warnings.append("[WARN] Low r2 despite good mae - data issues")

    return main_metric, main_score, warnings


def _is_better_model(
    task: str,
    main_metric: str,
    main_score: float,
    prev_record: Optional[Dict]
) -> bool:
    """Determine if current model is better than previous."""
    if not prev_record:
        return True

    prev_score = prev_record.get("main_score")
    if prev_score is None:
        return True

    if task == "classification":
        return main_score > prev_score
    if task == "regression":
        if main_metric == "mae":
            return main_score < prev_score
        if main_metric == "r2":
            return main_score > prev_score

    return False


def build_competence_map(
    results_path: str = "results.json",
    tolerance: float = 0.1
) -> dict:
    """
    Build competence map from results and logs with model performance.
    Maps (target, context) to best performing model and metrics.

    Args:
        results_path: Path to results JSON file
        tolerance: Tolerance for metric consistency checks

    Returns:
        Dictionary mapping (target, context) to best model info
    """
    history = _load_history(results_path)
    competence_map = {}

    for record in history:
        if not _validate_record(record):
            continue

        key = _extract_record_key(record)
        if not key:
            continue

        task = record["context"].get("task", "")
        metrics = record.get("metrics", {})

        if task == "classification":
            main_metric, main_score, warnings = _extract_classification_metrics(metrics, tolerance)
        elif task == "regression":
            main_metric, main_score, warnings = _extract_regression_metrics(metrics)
        else:
            continue

        if main_score is None:
            continue

        prev_record = competence_map.get(key)
        if _is_better_model(task, main_metric, main_score, prev_record):
            competence_map[key] = {
                "best_model": record["model"],
                "metrics": metrics,
                "main_metric": main_metric,
                "main_score": main_score,
                "warnings": warnings
            }

    return competence_map
