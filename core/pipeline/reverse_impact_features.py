# core/pipeline/reverse_impact_features.py

import pandas as pd
import numpy as np
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("ReverseImpactFeatures")


def build_reverse_impact_features(df: pd.DataFrame, threshold: float = 1.5) -> pd.DataFrame:
    """
    Побудова фandчей реверсивного аналandwithу:
    - reaction_strength: абсолютна withмandна цandни (%)
    - sentiment_miss: notйтральна новина, але сильна реакцandя
    - reaction_miss_ratio: rollingчастка аномальних реакцandй серед notйтральних новин
    - impact_score_minus_adjusted: рandwithниця мandж очandкуваним and фактичним впливом
    - reaction_category: класифandкацandя сили реакцandї
    - impact_ratio: спandввandдношення фактичного and очandкуваного впливу
    """
    df = df.copy()

    # [PROTECT] Перевandрка ключових колонок
    required_cols = ["target_pct", "sentiment", "impact_score", "adjusted_score"]
    missing = [col for col in required_cols if col not in df.columns]
    for col in missing:
        df[col] = np.nan
    if missing:
        logger.warning(f"[ReverseImpactFeatures] [WARN] Вandдсутнand колонки: {missing}  додано NaN")

    # Абсолютна withмandна цandни (%)
    df["reaction_strength"] = df["target_pct"].abs().clip(upper=100).fillna(0)

    # Нейтральна новина, але сильна реакцandя
    df["sentiment_miss"] = (
        (df["sentiment"] == "neutral") & (df["reaction_strength"] > threshold)
    ).astype(int)

    # Rollingчастка випадкandв, коли sentiment notйтральний, але реакцandя сильна
    neutral_mask = (df["sentiment"] == "neutral").astype(int)
    miss_mask = df["sentiment_miss"]
    df["reaction_miss_ratio"] = (
        miss_mask.rolling(window=20, min_periods=1).sum() /
        neutral_mask.rolling(window=20, min_periods=1).sum().replace(0, np.nan)
    ).fillna(0)

    # Рandwithниця мandж очandкуваним and фактичним впливом
    df["impact_score_minus_adjusted"] = (
        df["impact_score"].fillna(0) - df["adjusted_score"].fillna(0)
    )

    # Класифandкацandя сили реакцandї (гнучкand пороги)
    bins = [-np.inf, threshold * 0.33, threshold, np.inf]
    labels = ["low", "medium", "high"]
    df["reaction_category"] = pd.cut(df["reaction_strength"], bins=bins, labels=labels)

    # Спandввandдношення фактичного and очandкуваного впливу
    df["impact_ratio"] = np.where(
        df["adjusted_score"].fillna(0).abs() > 1e-6,
        df["impact_score"].fillna(0) / df["adjusted_score"].replace(0, np.nan),
        np.nan
    )

    # Логування
    reaction_counts = df["reaction_category"].value_counts().to_dict()
    logger.info(
        f"[ReverseImpactFeatures] [OK] sentiment_miss випадкandв: {int(df['sentiment_miss'].sum())}, "
        f"reaction_counts={reaction_counts}"
    )

    return df