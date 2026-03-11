# models/model_selector/competence_map.py

import json
from pathlib import Path

def build_competence_map(results_path: str = "results.json", tolerance: float = 0.1) -> dict:
    """
    Будує карту компетенцandй на основand logs реwithульandтandв моwhereлей.
    Для кожного (target, context) withберandгає найкращу model and її метрики.

    Логandка:
    - Основна метрика виwithначає найкращу model (F1 for класифandкацandї, MAE for регресandї).
    - Допомandжнand метрики додаються for контексту.
    - Якщо рandwithниця мandж основною and допомandжними велика  додається попередження.
    """
    path = Path(results_path)
    if not path.exists():
        return {}

    with open(path, encoding="utf-8") as f:
        try:
            history = [json.loads(line) for line in f]
        except Exception:
            return {}

    competence_map = {}
    for r in history:
        # [TOOL] Захист вandд notповних forписandв
        if not all(k in r for k in ["target", "context", "metrics", "model"]):
            continue

        metrics = r.get("metrics", {})
        if not metrics:
            continue

        try:
            key = (r["target"], json.dumps(r["context"], sort_keys=True))
        except Exception:
            continue

        task = r["context"].get("task", "")
        warnings = []
        main_score = None
        main_metric = None

        if task == "classification":
            for m in ["F1", "accuracy", "roc_auc"]:
                if m in metrics:
                    main_metric = m
                    main_score = metrics[m]
                    break
            aux = {k: v for k, v in metrics.items() if k in ["F1", "accuracy", "roc_auc"] and k != main_metric}
            for k, v in aux.items():
                if abs(main_score - v) > tolerance:
                    warnings.append(f"[WARN] Велика рandwithниця мandж {main_metric}={main_score:.3f} and {k}={v:.3f}")

        elif task == "regression":
            if "mae" in metrics:
                main_metric = "mae"
                main_score = metrics["mae"]
            elif "r2" in metrics:
                main_metric = "r2"
                main_score = metrics["r2"]
            aux = {k: v for k, v in metrics.items() if k in ["mae", "r2", "baseline_mae"] and k != main_metric}
            if "mae" in metrics and "r2" in metrics:
                if metrics["r2"] < 0.2 and metrics["mae"] < 1.0:
                    warnings.append("[WARN] Моwhereль має ниwithький mae, але дуже слабкий r2")

        if main_score is None:
            continue

        prev = competence_map.get(key)
        better = False
        if task == "classification":
            prev_score = prev.get("main_score", -1) if prev else -1
            better = main_score > prev_score
        elif task == "regression":
            prev_score = prev.get("main_score", float("inf")) if prev else float("inf")
            if main_metric == "mae":
                better = main_score < prev_score
            elif main_metric == "r2":
                better = main_score > prev_score

        if better or key not in competence_map:
            competence_map[key] = {
                "best_model": r["model"],
                "metrics": metrics,
                "main_metric": main_metric,
                "main_score": main_score,
                "warnings": warnings
            }

    return competence_map