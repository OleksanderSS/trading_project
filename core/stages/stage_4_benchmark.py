# core/stages/stage_4_benchmark.py

import json
import time
import traceback
import pandas as pd
from typing import List, Dict, Any, Optional

from utils.logger import ProjectLogger
from utils.benchmark_logger import log_result

from config.config import TICKERS, TIME_FRAMES
from config.feature_layers import FEATURE_LAYERS, get_features_by_layer
from config.feature_config import TICKER_TARGET_MAP, ALL_FEATURES, SAFE_FILL_FEATURES

from core.stages.stage_4_modeling import run_stage_4_modeling


ALL_MODELS = [
    "lgbm", "rf", "xgb", "catboost",
    "lstm", "cnn", "transformer", "mlp",
    "linear", "svm", "knn", "autoencoder"
]


def _safe_fill(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    Акуратnot forповnotння пропускandв:
    - технandчнand/макро: ffill/bfill
    - новиннand/SAFE_FILL_FEATURES: 0
    - категорandальнand: forповнюємо 'unknown' and даємо OHE у Stage4
    """
    df = merged_df.copy()

    # 1) числовand колонки  спробувати ffill/bfill
    num_cols = df.select_dtypes(include=["number", "float", "int"]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].ffill().bfill()

    # 2) спецandальнand колонки, якand беwithпечно сandвити в 0
    for col in SAFE_FILL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 3) ймовandрнand новиннand лandчильники  0
    for col in ["news_score", "match_count", "daily_sentiment", "news_count", "impact_score", "reverse_impact"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 4) категорandальнand/текстовand  'unknown'
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    for col in cat_cols:
        if col in {"description", "summary", "headline", "clean_text", "keywords", "source"}:
            df[col] = df[col].fillna("unknown")

    return df


def benchmark_all_models(
    merged_df: pd.DataFrame,
    models: Optional[List[str]] = None,
    tickers: Optional[List[str]] = None,
    time_frames: Optional[List[str]] = None,
    background_layers: Optional[List[str]] = None,
    impulse_layers: Optional[List[str]] = None,
    results_path: str = "results.json",
    run_id: Optional[str] = None
) -> pd.DataFrame:
    logger = ProjectLogger.get_logger("Stage4Benchmark")
    t_start = time.time()

    models = models or ALL_MODELS
    tickers = tickers or list(TICKERS.keys())
    time_frames = time_frames or TIME_FRAMES

    all_layers = list(FEATURE_LAYERS.keys())
    background_layers = background_layers or all_layers
    impulse_layers = impulse_layers or all_layers

    df = _safe_fill(merged_df)
    rows: List[Dict[str, Any]] = []

    logger.info(f"[Stage4.0] Сandрт бенчмарку: models={models}, tickers={tickers}, time_frames={time_frames}")

    for ticker in tickers:
        for interval in time_frames:
            for model_name in models:
                try:
                    model, df_results, meta = run_stage_4_modeling(
                        df,
                        model_name=model_name,
                        ticker=ticker,
                        interval=interval,
                        background_layers=background_layers,
                        impulse_layers=impulse_layers,
                        get_features_by_layer=get_features_by_layer
                    )

                    if model is None or meta is None:
                        entry = {
                            "run_id": run_id,
                            "ticker": ticker,
                            "interval": interval,
                            "model": model_name,
                            "status": "skipped",
                            "reason": "no_features_or_constant",
                            "target": None,
                            "metrics": {},
                            "context": {
                                "ticker": ticker,
                                "interval": interval,
                                "features": [],
                                "context_features": [],
                                "task": "unknown"
                            }
                        }
                        log_result(entry, path=results_path)
                        rows.append(entry)
                        logger.warning(f"[Stage4.0] Пропущено: {entry}")
                        continue

                    target_cols = [c for c in df_results.columns if c.startswith("target_")]
                    pred_cols = [c for c in df_results.columns if c.startswith("predicted_")]

                    target_col = target_cols[0] if target_cols else None
                    predicted_col = pred_cols[0] if pred_cols else None

                    metrics = meta.get("metrics", {})
                    layer_contrib = meta.get("layer_contributions")
                    layer_contrib_dict = (
                        layer_contrib.to_dict() if hasattr(layer_contrib, "to_dict") else layer_contrib
                    )

                    entry = {
                        "run_id": run_id,
                        "ticker": ticker,
                        "interval": interval,
                        "model": model_name,
                        "status": "ok",
                        "target": target_col,   # унandфandковано
                        "predicted_col": predicted_col,
                        "metrics": metrics,
                        "layer_contributions": layer_contrib_dict,
                        "context": {            # унandфandковано
                            "ticker": ticker,
                            "interval": interval,
                            "features": meta.get("final_features", []),
                            "context_features": list(layer_contrib_dict.keys()) if layer_contrib_dict else [],
                            "task": meta.get("task_type", "unknown")
                        }
                    }

                    log_result(entry, path=results_path)
                    rows.append(entry)
                    logger.info(f"[Stage4.0] OK: {ticker}-{interval}-{model_name}  {list(metrics.keys())}")

                except Exception as e:
                    tb = traceback.format_exc(limit=2)
                    entry = {
                        "run_id": run_id,
                        "ticker": ticker,
                        "interval": interval,
                        "model": model_name,
                        "status": "error",
                        "error": str(e),
                        "target": None,
                        "metrics": {},
                        "context": {
                            "ticker": ticker,
                            "interval": interval,
                            "features": [],
                            "context_features": [],
                            "task": "unknown"
                        }
                    }
                    log_result(entry, path=results_path)
                    rows.append(entry)
                    logger.exception(f"[Stage4.0] ERROR {ticker}-{interval}-{model_name}: {e}\n{tb}")

    df_results = pd.DataFrame(rows)
    elapsed = round(time.time() - t_start, 2)
    logger.info(f"[Stage4.0] Завершено for {elapsed}s, рядкandв: {len(df_results)}")
    return df_results


def summarize_benchmark(df_results: pd.DataFrame, tolerance: float = 0.1) -> Dict[str, pd.DataFrame]:
    """
    Робить компактнand пandводи for огляду:
    - класифandкацandя: F1 як основна, andншand як допомandжнand
    - регресandя: MAE як основна, andншand як допомandжнand
    - ворнandнги при великandй рandwithницand мandж метриками
    """
    if df_results.empty:
        return {
            "classification": pd.DataFrame(),
            "regression": pd.DataFrame(),
            "errors": pd.DataFrame(),
        }

    ok_results = df_results[df_results["status"] == "ok"].copy()
    err_df = df_results[df_results["status"] == "error"].copy()

    if ok_results.empty:
        return {
            "classification": pd.DataFrame(),
            "regression": pd.DataFrame(),
            "errors": err_df,
        }

    # Роwithгорandння метрик
    metrics_list = ok_results["metrics"].apply(lambda m: m if isinstance(m, dict) else {}).tolist()
    metrics_expanded = pd.json_normalize(metrics_list)
    metrics_expanded.index = ok_results.index
    metrics_df = pd.concat([ok_results, metrics_expanded], axis=1)

    # === Класифandкацandя ===
    cls_cols = ["F1", "accuracy", "roc_auc"]
    existing_cls = [c for c in cls_cols if c in metrics_df.columns]

    if existing_cls:
        cls_df = metrics_df[metrics_df["context"].apply(lambda c: c.get("task") == "classification")].copy()
        cls_df = cls_df.dropna(subset=existing_cls, how="all")

        if not cls_df.empty:
            def classify_score(row):
                main = row.get("F1") or row.get("accuracy") or row.get("roc_auc")
                warnings = []
                for m in existing_cls:
                    if m != "F1" and pd.notna(row.get(m)) and abs(main - row[m]) > tolerance:
                        warnings.append(f"[WARN] {m}={row[m]:.3f} сильно вandдрandwithняється вandд F1={main:.3f}")
                return pd.Series({"score": main, "warnings": "; ".join(warnings)})

            cls_df[["score", "warnings"]] = cls_df.apply(classify_score, axis=1)
            cls_best = cls_df.sort_values(["ticker", "interval", "score"], ascending=[True, True, False])
        else:
            cls_best = pd.DataFrame()
    else:
        cls_best = pd.DataFrame()

    # === Регресandя ===
    reg_cols = ["mae", "r2"]
    existing_reg = [c for c in reg_cols if c in metrics_df.columns]

    if existing_reg:
        reg_df = metrics_df[metrics_df["context"].apply(lambda c: c.get("task") == "regression")].copy()
        reg_df = reg_df.dropna(subset=existing_reg, how="all")

        if not reg_df.empty:
            def regression_score(row):
                if pd.notna(row.get("mae")):
                    score = 1.0 / row["mae"] if row["mae"] != 0 else 0
                    main = row["mae"]
                    main_metric = "mae"
                elif pd.notna(row.get("r2")):
                    score = row["r2"]
                    main = row["r2"]
                    main_metric = "r2"
                else:
                    return pd.Series({"score": None, "warnings": ""})

                warnings = []
                if "r2" in row and "mae" in row:
                    if row["r2"] < 0.2 and row["mae"] < 1.0:
                        warnings.append("[WARN] r2 дуже ниwithький при хорошому mae")

                return pd.Series({"score": score, "warnings": "; ".join(warnings)})

            reg_df[["score", "warnings"]] = reg_df.apply(regression_score, axis=1)
            reg_best = reg_df.sort_values(["ticker", "interval", "score"], ascending=[True, True, False])
        else:
            reg_best = pd.DataFrame()
    else:
        reg_best = pd.DataFrame()

    return {
        "classification": cls_best,
        "regression": reg_best,
        "errors": err_df,
    }