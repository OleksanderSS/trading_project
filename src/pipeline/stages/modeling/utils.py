"""
Utility helpers for stage 4 modeling: context fingerprints, diary logging,
and small helper extractors used by `stage_4_modeling.ModelingStage`.
"""

import datetime
from typing import Any, Dict, List

import pandas as pd
import psutil


def get_context_fingerprint(ticker_result: Dict[str, Any]) -> str:
    return ticker_result.get("context_fingerprint", "unknown")


def log_training_debug_info(logger, debug_info) -> None:
    logger.info(f"📊 Stage 4 - {debug_info.context_key}:")
    logger.info(f"   winner: {debug_info.winner_name}")
    logger.info(f"   winner_metrics: {debug_info.winner_metrics}")
    logger.info(f"   all_metrics: {debug_info.all_metrics}")
    logger.info(f"   selected_features: {len(debug_info.selected_features)} features")
    logger.info(f"   First 5 features: {debug_info.selected_features[:5]}")


def create_champion_info(brain: Dict[str, Any], config) -> Dict[str, Any]:
    context_map = {
        "context_fingerprint": config.context_fingerprint,
        "market_regime": config.market_regime,
        "volatility_regime": brain.get("volatility_regime", "normal"),
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return {
        "ticker": config.ticker,
        "target": config.target_name,
        "winner": config.comparison_report.get("champion_model", config.winner_name),
        "champion_reason": config.comparison_report.get("selection_reason", "Top accuracy"),
        "context": config.context_fingerprint,
        "context_map": context_map,
        "market_regime": config.market_regime,
        "timestamp": datetime.datetime.now().isoformat(),
        "metrics": config.winner_metrics,
        "all_models_metrics": config.all_metrics,
        "model_path": config.ticker_result.get("model_path"),
        "selected_features": config.selected_features,
        "feature_count": len(config.selected_features),
    }


def log_to_diary(diary_path, info: Dict[str, Any], tf: str) -> None:
    entry = {
        "timestamp": info["timestamp"],
        "ticker": info["ticker"],
        "tf": tf,
        "target": info["target"],
        "model_name": info["winner"],
        "context_fingerprint": info["context"],
        "is_champion": True,
        "cpu_usage": psutil.cpu_percent(),
        "ram_usage": psutil.virtual_memory().percent,
    }
    pd.DataFrame([entry]).to_csv(diary_path, mode="a", header=False, index=False)


def get_light_model_training_data(prepared_data: Dict[str, Any]):
    X_train = prepared_data.get("light_models", {}).get("X_train")
    y_train = prepared_data.get("light_models", {}).get("y_train")
    X_test = prepared_data.get("light_models", {}).get("X_test")
    y_test = prepared_data.get("light_models", {}).get("y_test")

    if X_train is None or y_train is None:
        return None

    return X_train, y_train, X_test, y_test


def determine_task_type(target_name: str) -> str:
    return "regression" if "return" in target_name or "price" in target_name else "classification"


def get_light_model_types() -> List[str]:
    return ["catboost", "lightgbm", "xgboost", "random_forest", "linear", "svm", "knn"]
