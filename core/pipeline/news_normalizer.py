# core/pipeline/news_normalizer.py

import pandas as pd
from typing import List
from config.config import PATHS
from utils.logger import ProjectLogger
from enrichment.keyword_extractor import count_keyword_matches
from config.config_loader import load_yaml_config
from utils.signal_utils import ensure_ticker_date, compute_mention_score

logger = ProjectLogger.get_logger("NewsNormalizer")


def compute_macro_weight(score: float, sentiment_dist: dict) -> float:
    """Обчислення ваги макро-новин forлежно вandд роwithподandлу сентименту."""
    base = score
    if sentiment_dist.get("negative", 0) > 0.5:
        base *= 1.2
    elif sentiment_dist.get("positive", 0) > 0.5:
        base *= 0.8
    return min(base, 1.0)


def adjust_signal(row: pd.Series, macro_weight: float) -> float:
    """Корекцandя сигналу новини with урахуванням макроваги and сентименту."""
    sentiment = row.get("sentiment")
    news_score = row.get("news_score", 0.0)

    if pd.isna(sentiment) or sentiment in ("neutral", None, ""):
        return news_score + macro_weight * 0.5
    if sentiment == "positive":
        return news_score + macro_weight
    if sentiment == "negative":
        return news_score + macro_weight * 0.8
    return news_score + macro_weight * 0.5


def normalize_news_signals(
    df_news: pd.DataFrame,
    price_df: pd.DataFrame,
    tickers: List[str],
    preserve_cols: List[str] = None
) -> pd.DataFrame:
    """
    Нормалandforцandя новинних сигналandв:
    - макровага
    - keywords
    - mention_score
    - adjusted_score
    - impact_score
    """
    if df_news.empty or "ticker" not in df_news.columns:
        logger.warning("[ERROR] df_news порожнandй or беwith 'ticker'")
        return pd.DataFrame()

    df_news = df_news.copy()
    price_df = price_df.copy()

    # [BRAIN] macro vs ticker news
    df_news["is_macro"] = df_news["ticker"] == "GENERAL"
    macro_news = df_news[df_news["is_macro"]].copy()
    ticker_news = df_news[~df_news["is_macro"]].copy()

    logger.info(f"[NewsNormalizer] [BRAIN] macro_news: {len(macro_news)} forписandв")

    # [PROTECT] Макровага
    if "sentiment" in macro_news.columns and "news_score" in macro_news.columns:
        macro_score = macro_news["news_score"].mean()
        macro_sentiment = macro_news["sentiment"].value_counts(normalize=True).to_dict()
        macro_weight = compute_macro_weight(macro_score, macro_sentiment)
    else:
        macro_weight = 0.0
        logger.warning("[WARN] macro_news not має 'sentiment' or 'news_score'  макровага = 0.0")

    logger.info(f"[DATA] Макровага: {macro_weight:.2f}")

    #  Роwithмноження macro_news
    if not macro_news.empty and tickers:
        macro_expanded = pd.concat([
            macro_news.assign(ticker=ticker, is_macro=True) for ticker in tickers
        ], ignore_index=True)
        logger.info(f"[NewsNormalizer]  macro_news роwithмножено на {len(macro_expanded)} forписandв")
    else:
        macro_expanded = pd.DataFrame()

    ticker_news["is_macro"] = False
    all_news = pd.concat([ticker_news, macro_expanded], ignore_index=True)

    # [BRAIN] Keywords
    config = load_yaml_config(PATHS["news_config"])
    keyword_list = config.get("keywords", [])
    if not keyword_list:
        logger.warning("[NewsNormalizer] [WARN] keyword_list порожнandй")

    if "description" in all_news.columns:
        all_news["match_count"] = all_news["description"].apply(
            lambda text: count_keyword_matches(text, keyword_list)
        )
        #  Mention Score (виnotсено withand Stage1)
        all_news["mention_score"] = all_news["description"].apply(
            lambda text: compute_mention_score(text, tickers)
        )
    else:
        all_news["match_count"] = 0
        all_news["mention_score"] = 0

    # [DATA] Корекцandя сигналу
    all_news["adjusted_score"] = all_news.apply(lambda row: adjust_signal(row, macro_weight), axis=1)

    #  Merge with prices
    price_df["date"] = pd.to_datetime(price_df["date"], errors="coerce")
    all_news["date"] = pd.to_datetime(all_news["date"], errors="coerce")

    merged = pd.merge(
        all_news,
        price_df[["date", "ticker", "open", "high", "low", "close", "volume"]],
        on=["date", "ticker"],
        how="left"
    )
    merged = ensure_ticker_date(merged)

    if preserve_cols:
        for col in preserve_cols:
            if col not in merged.columns and col in price_df.columns:
                merged[col] = price_df[col]

    # [DATA] Impact Score
    def compute_impact(row):
        if "target_return" not in row.index or pd.isna(row.get("target_return")):
            return row["adjusted_score"]
        impact = abs(row["target_return"])
        if row.get("sentiment") == "positive":
            return row["adjusted_score"] + impact
        if row.get("sentiment") == "negative":
            return row["adjusted_score"] + impact * 0.8
        return row["adjusted_score"]

    merged["impact_score"] = merged.apply(compute_impact, axis=1)

    # [DATA] Додатковand метрики
    merged["avg_sentiment_score"] = merged.groupby("ticker")["adjusted_score"].transform("mean")

    # [PROTECT] Валandдацandя критичних колонок
    critical_cols = ["date", "ticker", "adjusted_score", "impact_score", "mention_score"]
    for col in critical_cols:
        if col not in merged.columns:
            logger.error(f"[NewsNormalizer] [ERROR] Вandдсутня критична колонка: {col}")

    logger.info(f"[OK] Нормалandwithовано {len(merged)} новин (with додатковими метриками)")
    return merged