# core/pipeline/ensemble.py

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


def adaptive_weighted_signal(
    ticker_signals: Dict[str, Dict[str, str]],
    daily_trend: str = 'HOLD',
    df_daily: Optional[pd.DataFrame] = None,
    df_short_tf: Optional[Dict[str, pd.DataFrame]] = None,
    sentiment_score: float = 0.0
) -> str:
    """
    Формує адаптивний withважений сигнал BUY/HOLD/SELL with урахуванням:
    - волатильностand (daily + short TF),
    - тренду,
    - сентименту,
    - новинних тригерandв.
    """
    mapping = {'BUY': 1, 'HOLD': 0, 'SELL': -1}
    weights_base = {'1d': 0.5, '60m': 0.2, '15m': 0.15}
    vol_factor = 1.0

    def compute_vol_factor(df: pd.DataFrame) -> float:
        if df is None or df.empty or 'close' not in df.columns:
            return 1.0
        closes = df['close'].dropna()
        if len(closes) < 2:
            return 1.0
        vol = np.log(closes).diff().dropna().std()
        if vol <= 0:
            return 1.0
        return max(0.5, min(1.5, 0.05 / vol))

    # Волатильнandсть daily
    try:
        vol_factor *= compute_vol_factor(df_daily)
    except Exception as e:
        logger.warning(f"[Ensemble] Daily volatility skipped: {e}")

    # Волатильнandсть short TF
    try:
        if df_short_tf:
            for tf, df_tf in df_short_tf.items():
                vol_factor *= compute_vol_factor(df_tf)
    except Exception as e:
        logger.warning(f"[Ensemble] Short timeframe volatility skipped: {e}")

    total, total_weight = 0.0, 0.0
    try:
        for tf, sigs in ticker_signals.items():
            final_signal = sigs.get('final_signal', 'HOLD') if sigs else 'HOLD'
            val = mapping.get(final_signal, 0)
            weight = weights_base.get(tf, 0.1)

            # Корекцandя на тренд
            if daily_trend == 'BUY' and val < 0:
                val *= 0.5
            if daily_trend == 'SELL' and val > 0:
                val *= 0.5

            # Додавання сентименту
            val += sentiment_score * weight

            #  Корекцandя на новиннand тригери
            if sigs:
                if sigs.get("mention_spikes", 0) > 0:
                    val += 0.2 * weight
                if sigs.get("sentiment_extremes", 0) > 0:
                    val += 0.3 * weight
                if sigs.get("volatility_anomalies", 0) > 0:
                    val -= 0.2 * weight

            total += val * weight
            total_weight += weight
    except Exception as e:
        logger.warning(f"[Ensemble] Signal aggregation skipped: {e}")

    if total_weight == 0:
        logger.info("[Ensemble] Total weight is zero, returning default 'HOLD'")
        return 'HOLD'

    avg = (total / total_weight) * vol_factor
    logger.info(f"[Ensemble] Final avg={avg:.3f}, vol_factor={vol_factor:.2f}")

    if avg > 0.3 * vol_factor:
        return 'BUY'
    elif avg < -0.3 * vol_factor:
        return 'SELL'
    else:
        return 'HOLD'