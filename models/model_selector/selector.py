# models/model_selector/selector.py

import json

def select_model(target: str, context: dict, competence_map: dict, fallback: str = "lgbm") -> str:
    """
    Вибирає найкращу model for forданого andргету and контексту.
    Якщо контекст новий  поверandє fallback (for forмовчуванням lgbm).
    """
    try:
        key = (target, json.dumps(context, sort_keys=True))
    except Exception:
        return fallback

    if key in competence_map:
        return competence_map[key]["best_model"]
    return fallback