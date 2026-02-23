# models/model_selector/competence_analyzer.py

import json
import pandas as pd
from typing import Dict, Any, List, Optional
from utils.logger import ProjectLogger


def load_results_json(path: str = "results.json") -> pd.DataFrame:
    """
    Заванandжує построчний JSON (results.json) у DataFrame.
    Кожен рядок  окремий forпуск.
    """
    logger = ProjectLogger.get_logger("CompetenceAnalyzer")
    rows: List[Dict[str, Any]] = []

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        df = pd.DataFrame(rows)
        logger.info(f"[CompetenceAnalyzer] Заванandжено {len(df)} рядкandв andwith {path}")
        return df
    except FileNotFoundError:
        logger.warning(f"[CompetenceAnalyzer] Файл {path} not withнайwhereно")
        return pd.DataFrame()


def build_competence_matrix(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Будує матрицю компетенцandй for порandвняння:
    - for класифandкацandєю: average accuracy/F1/roc_auc per (ticker, interval, model)
    - for регресandєю: average mae/r2 per (ticker, interval, model)
    """
    if df.empty:
        return {
            "classification": pd.DataFrame(),
            "regression": pd.DataFrame()
        }

    # Роwithгорнути метрики в колонки
    df = df.copy()
    df["metrics"] = df["metrics"].apply(lambda m: m if isinstance(m, dict) else {})
    metrics_df = df.assign(**df["metrics"].apply(pd.Series))

    # Класифandкацandя
    cls_cols = ["accuracy", "F1", "roc_auc"]
    cls_matrix = (
        metrics_df[["ticker", "interval", "model"] + cls_cols]
        .groupby(["ticker", "interval", "model"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["ticker", "interval"], ascending=[True, True])
    )

    # Регресandя
    reg_cols = ["mae", "r2", "baseline_mae"]
    reg_matrix = (
        metrics_df[["ticker", "interval", "model"] + reg_cols]
        .groupby(["ticker", "interval", "model"], dropna=False)
        .mean(numeric_only=True)
        .reset_index()
        .sort_values(["ticker", "interval"], ascending=[True, True])
    )

    return {
        "classification": cls_matrix,
        "regression": reg_matrix
    }


def pick_best_models(
    matrices: Dict[str, pd.DataFrame],
    top_n: int = 3
) -> Dict[str, pd.DataFrame]:
    """
    Вибирає топ-моwhereлand:
    - for класифandкацandї: сортуємо for середнandм скором (mean accuracy/F1/roc_auc)
    - for регресandї: компоwithитний скор (1/mae + r2), сортуємо спадно
    """
    cls = matrices.get("classification", pd.DataFrame()).copy()
    reg = matrices.get("regression", pd.DataFrame()).copy()

    if not cls.empty:
        cls["score"] = cls[["accuracy", "F1", "roc_auc"]].mean(axis=1, skipna=True)
        best_cls = (
            cls.sort_values(["ticker", "interval", "score"], ascending=[True, True, False])
            .groupby(["ticker", "interval"], as_index=False)
            .head(top_n)
        )
    else:
        best_cls = pd.DataFrame()

    if not reg.empty:
        reg["mae_inv"] = 1.0 / reg["mae"].replace(0, pd.NA)
        reg["score"] = reg[["mae_inv", "r2"]].mean(axis=1, skipna=True)
        best_reg = (
            reg.sort_values(["ticker", "interval", "score"], ascending=[True, True, False])
            .groupby(["ticker", "interval"], as_index=False)
            .head(top_n)
        )
    else:
        best_reg = pd.DataFrame()

    return {
        "classification_best": best_cls,
        "regression_best": best_reg
    }