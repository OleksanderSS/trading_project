import datetime
from typing import Any


def calculate_model_metrics(y_test, predictions, task_type):
    import numpy as np
    from sklearn.metrics import accuracy_score, mean_squared_error

    if task_type == "regression":
        mse = mean_squared_error(y_test, predictions)
        score = -mse
        return {"mse": float(mse), "rmse": float(np.sqrt(mse)), "score": float(score)}
    else:
        accuracy = accuracy_score(y_test, predictions)
        return {"accuracy": float(accuracy), "score": float(accuracy)}


def create_light_model_champion_info(config) -> dict[str, Any]:
    context_map = {
        "context_fingerprint": config.context_fingerprint,
        "market_regime": config.market_regime,
        "volatility_regime": config.volatility_regime,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return {
        "ticker": config.ticker,
        "target": config.target_name,
        "winner": config.model_type,
        "model_type": config.model_type,
        "champion_reason": f"Light model trained locally with {len(config.selected_features)} features",
        "context": config.context_fingerprint,
        "context_map": context_map,
        "market_regime": config.market_regime,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": config.metrics,
        "model_path": str(config.model_path),
        "model_key": config.model_key,
        "selected_features": config.selected_features,
        "feature_count": len(config.selected_features),
        "model_category": "light",
    }
