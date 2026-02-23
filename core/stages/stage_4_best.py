# core/stages/stage_4_best.py

import pandas as pd
from core.stages.stage_4_modeling import run_stage_4_modeling
from utils.logger import ProjectLogger
from config.config import TICKERS, TIME_FRAMES


def safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Акуратnot forповnotння NaN перед Stage4."""
    df = df.copy()
    # числовand колонки  ffill/bfill
    num_cols = df.select_dtypes(include=["number", "float", "int"]).columns
    df[num_cols] = df[num_cols].ffill().bfill()
    # новиннand/лandчильники  0
    for col in ["news_score", "match_count", "daily_sentiment", "news_count", "impact_score", "reverse_impact"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    # текстовand  "unknown"
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        df[col] = df[col].fillna("unknown")
    return df


def run_best_model_for_context(merged_df: pd.DataFrame, benchmark_df: pd.DataFrame, ticker: str, interval: str):
    """
    Вибирає найкращу model for класифandкацandї and регресandї with бенчмарку
    and forпускає them на нових data.
    """
    logger = ProjectLogger.get_logger("BestModelRunner")

    df = safe_fill(merged_df)

    # вибandр найкращої класифandкацandйної моwhereлand
    cls_df = benchmark_df[(benchmark_df["ticker"] == ticker) &
                          (benchmark_df["interval"] == interval) &
                          (benchmark_df["status"] == "ok")].dropna(subset=["metrics"])
    cls_best = None
    if not cls_df.empty:
        cls_df = cls_df.assign(
            score=cls_df["metrics"].apply(lambda m: (m.get("accuracy", 0) + m.get("F1", 0) + m.get("roc_auc", 0)) / 3))
        cls_best = cls_df.sort_values("score", ascending=False).iloc[0]["model"]

    # вибandр найкращої регресandйної моwhereлand
    reg_df = benchmark_df[(benchmark_df["ticker"] == ticker) &
                          (benchmark_df["interval"] == interval) &
                          (benchmark_df["status"] == "ok")].dropna(subset=["metrics"])
    reg_best = None
    if not reg_df.empty:
        reg_df = reg_df.assign(score=reg_df["metrics"].apply(lambda m: (m.get("r2", 0) - m.get("mae", 0))))
        reg_best = reg_df.sort_values("score", ascending=False).iloc[0]["model"]

    results = {}

    if cls_best:
        logger.info(f"[BestModelRunner] Запускаю класифandкацandю: {cls_best}")
        model_cls, df_cls, meta_cls = run_stage_4_modeling(df, model_name=cls_best, ticker=ticker, interval=interval)
        results["classification"] = {"model": cls_best, "metrics": meta_cls.get("metrics")}

    if reg_best:
        logger.info(f"[BestModelRunner] Запускаю регресandю: {reg_best}")
        model_reg, df_reg, meta_reg = run_stage_4_modeling(df, model_name=reg_best, ticker=ticker, interval=interval)
        results["regression"] = {"model": reg_best, "metrics": meta_reg.get("metrics")}

    return results


def run_best_models_overall(merged_df: pd.DataFrame, benchmark_df: pd.DataFrame):
    """
    Запускає найкращand моwhereлand for allх тикерandв and andнтервалandв.
    Поверandє словник with реwithульandandми.
    """
    results = {}
    for ticker in TICKERS.keys():
        for interval in TIME_FRAMES:
            res = run_best_model_for_context(merged_df, benchmark_df, ticker=ticker, interval=interval)
            if res:
                results[(ticker, interval)] = res
    return results