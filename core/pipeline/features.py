# core/pipeline/features.py

import os
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
from utils.features_utils import FeatureUtils
from utils.trading_signals import (
    generate_signals,
    DEFAULT_FORECAST_THRESHOLDS,
    DEFAULT_RSI_THRESHOLDS,
    DEFAULT_SENTIMENT_THRESHOLDS,
)
from utils.backtesting import Backtester
from utils.visualization import plot_model_performance
import logging
from config.thresholds import get_forecast_threshold
from utils.trigger_features import compute_trigger_features
from utils.trigger_signals import generate_trigger_signals
from utils.trading_calendar import TradingCalendar
from enrichment.sentiment_analyzer import SentimentEnricher
from utils.cache_utils import CacheManager
from config.config import TICKERS
from core.pipeline.ensemble import adaptive_weighted_signal
from core.pipeline.features_final import compute_final_features


logger = logging.getLogger(__name__)
feature_utils = FeatureUtils(short_window=3, long_window=200)

SIGNAL_MAPPING = {"BUY": 1, "HOLD": 0, "SELL": -1}
PREFERRED_BASE_TF = "1d"


def sanitize_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].apply(lambda x: x.normalize() if pd.notna(x) else x)
            df = df[df[col].notna()]
    return df


def sanitize_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(columns=[col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])], errors="ignore")
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


def smooth_signals(signals_dict: dict, window: int = 3) -> dict:
    if not signals_dict:
        return {}
    df_sig = pd.DataFrame([signals_dict])
    df_num = df_sig.apply(lambda col: col.map(lambda x: SIGNAL_MAPPING.get(x, 0))).astype(float)
    df_smooth = df_num.rolling(window=window, min_periods=1).mean()

    def num_to_signal(val):
        if val > 0.33:
            return "BUY"
        elif val < -0.33:
            return "SELL"
        else:
            return "HOLD"

    df_final = df_smooth.apply(lambda col: col.map(num_to_signal))
    return df_final.iloc[-1].to_dict()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df


def has_rsi(df: pd.DataFrame) -> bool:
    for c in df.columns:
        if isinstance(c, str) and (c.lower() == "rsi" or c.lower().endswith("rsi")):
            if not df[c].isna().all():
                return True
    return False


def ensure_base_tf(df_dict: Dict[str, pd.DataFrame], base_tf: str = PREFERRED_BASE_TF, key: Optional[str] = None) -> \
Dict[str, pd.DataFrame]:
    if base_tf in df_dict and not df_dict[base_tf].empty:
        return df_dict

    for candidate in ["1h", "15m"]:
        if candidate in df_dict and not df_dict[candidate].empty:
            logger.warning(f"[Features] base_tf '{base_tf}' вandдсутнandй  ресемплюємо with {candidate}")
            df_resampled = df_dict[candidate].copy()
            try:
                # [IDEA] ВИПРАВЛЕНО: Використовуємо andнwhereкс for ресемплandнгу
                df_resampled = df_resampled.set_index(pd.to_datetime(df_resampled.index)).resample("1D").ffill()
                df_resampled = df_resampled.reset_index().rename(columns={"index": "date"})

                #  Вandдновлення 'ticker' пandсля ресемплandнгу
                if "ticker" not in df_resampled.columns:
                    if "ticker" in df_dict[candidate].columns:
                        df_resampled["ticker"] = df_dict[candidate]["ticker"].iloc[0]
                    elif key is not None:
                        df_resampled["ticker"] = key
                    else:
                        logger.warning(f"[Features] [WARN] 'ticker' not withнайwhereно and key not передано  колонка will пропущена")
            except Exception as e:
                logger.error(f"[Features] Resample error: {e}")
                df_resampled = pd.DataFrame(columns=["close"])

            df_dict[base_tf] = df_resampled
            return df_dict

    logger.error(f"[Features] base_tf '{base_tf}' вandдсутнandй and notма data for ресемплandнгу")
    df_dict[base_tf] = pd.DataFrame(columns=["close"])
    return df_dict


# ----------------------------------------------------------------------
# [START] ПЕРЕПИСАНА ОСНОВНА ФУНКЦІЯ (Multi-Ticker/Single-Pass FinBERT)
# ----------------------------------------------------------------------

def prepare_features_and_signals(
        financial_data_all: Dict[str, Dict[str, pd.DataFrame]],  # {Ticker: {TF: DF}}
        df_news: pd.DataFrame,
        df_macro: pd.DataFrame,
        calendar: Optional[TradingCalendar] = None,
        MODELS_DICT: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Головний метод for обчислення фandчей and сигналandв for ВСІХ тandкерandв.

    Виnotсено andнandцandалandforцandю SentimentEnricher with циклу for пandдвищення продуктивностand.
    """
    logger.info("[Еandп 4]  Сandрт prepare_features_and_signals...")
    all_processed_data = {}

    if not financial_data_all or all(all(df.empty for df in data.values()) for data in financial_data_all.values()):
        raise ValueError("[Pipeline] Empty financial_data_all. Завершення.")

    # 1. [IDEA] Інandцandалandforцandя forгальних об'єктandв (FinBERT)
    feature_engineer_instance = feature_utils
    cache = CacheManager()

    # Використовуємо all тandкери як ключовand слова for SentimentEnricher
    keyword_list = list(TICKERS.keys())
    keyword_dict = {"all": keyword_list}
    sentiment_enricher_instance = SentimentEnricher(cache_manager=cache, keyword_dict=keyword_dict)
    logger.info("[OK] SentimentEnricher (FinBERT) andнandцandалandwithовано 1 раwith for allх тandкерandв.")

    #  Попередня обробка новин and макроandндикаторandв (Зсув дат новин)
    df_macro = sanitize_dates(df_macro)
    if calendar and not df_news.empty and "published_at" in df_news.columns:
        df_news["date"] = pd.to_datetime(df_news["published_at"], errors="coerce").dt.normalize()

        if df_news["date"].dt.tz is not None:
            df_news["date"] = df_news["date"].dt.tz_convert(None)

        try:
            # [IDEA] ВИПРАВЛЕННЯ: Використовуємо .apply() for ефективного withсуву
            df_news["date"] = df_news["date"].apply(
                lambda d: calendar.shift_to_next_trading_day(d) if pd.notna(d) else d)
        except Exception as e:
            logger.warning(f"[Features] Error при withсувand новин до торгових дат: {e}")
    df_news = sanitize_dates(df_news)

    # 2. Перебandр Тandкерandв
    for ticker_key, data_for_ticker in financial_data_all.items():
        all_processed_data[ticker_key] = {}
        logger.info(f"[{ticker_key}] -------------------- Обробка тandкера --------------------")

        # 2.1. Фandльтрацandя новин for поточного тandкера
        ticker_news = df_news[df_news[
                                  'ticker'] == ticker_key].copy() if not df_news.empty and 'ticker' in df_news.columns else df_news.copy()

        # 2.2. Забеwithпечення баwithового TF and нормалandforцandя колонок
        data_for_ticker = {tf: normalize_columns(df) for tf, df in data_for_ticker.items()}
        data_for_ticker = ensure_base_tf(data_for_ticker, base_tf=PREFERRED_BASE_TF, key=ticker_key)

        # Отримання 1d цandни for тригерandв
        df_prices_1d = data_for_ticker.get(PREFERRED_BASE_TF)

        # 3. Перебandр Таймфреймandв
        for tf, df_price in data_for_ticker.items():
            if df_price.empty:
                logger.warning(f"[{ticker_key}:{tf}] Цandновand данand порожнand. Пропуск.")
                continue

            # 4. Виклик compute_final_features (обчислює фandчand, контекст and TARGET)
            df_features = compute_final_features(
                data_all_tf=data_for_ticker,
                df_news=ticker_news,
                df_macro=df_macro,
                feature_engineer=feature_engineer_instance,
                sentiment_enricher=sentiment_enricher_instance,
                calendar=calendar,
                ticker=ticker_key,
                tf=tf,
            )

            if df_features.empty or 'target_pct' not in df_features.columns:
                logger.warning(f"[{ticker_key}:{tf}] df_features порожнandй or беwith 'target_pct'. Пропуск.")
                continue

            # 5. Геnotрацandя сигналandв
            df_features = sanitize_features(df_features)

            # 5.1 Тригернand сигнали (ОБЧИСЛЕННЯ ТРИГЕРІВ)
            trigger_data = {}
            if not ticker_news.empty and df_prices_1d is not None and not df_prices_1d.empty:
                try:
                    # [IDEA] Обчислюємо тригери, використовуючи 1D цandну як контекст for волатильностand
                    df_prices_1d_indexed = df_prices_1d.set_index(
                        "date") if "date" in df_prices_1d.columns else df_prices_1d

                    trigger_data = compute_trigger_features(
                        df_news=ticker_news,
                        df_prices=df_prices_1d_indexed,  # Використовуємо 1D
                        mention_threshold=10,
                        sentiment_threshold=0.8,
                        volatility_threshold=0.02
                    )
                    trigger_signals = generate_trigger_signals(trigger_data)

                    #  Злиття тригерних фandчей у df_features
                    # Створюємо DF with тригер-фandчами and forбеwithпечуємо, що andнwhereкс вandдповandдає df_features
                    df_trigger_features = pd.DataFrame(trigger_data).set_index(df_features.index.name)
                    df_features = df_features.merge(
                        df_trigger_features, left_index=True, right_index=True, how="left"
                    ).fillna(0)

                    # [IDEA] Тригернand сигнали can add як бandнарнand фandчand for моwhereлand
                    for sig_name, sig_value in trigger_signals.items():
                        df_features[f"trigger_{sig_name}"] = SIGNAL_MAPPING.get(sig_value, 0)

                except Exception as e:
                    logger.warning(f"[{ticker_key}:{tf}] [WARN] Error обчислення/геnotрацandї тригерandв: {e}")
                    trigger_signals = {}
            else:
                trigger_signals = {}

            # 5.2 Основнand сигнали (BUY/SELL)
            signals = {}
            if df_features["target_pct"].notna().sum() > 0:
                try:
                    # [IDEA] ВИПРАВЛЕННЯ: Обчислюємо усередnotний сентимент на основand обчислених фandчей
                    if 'sentiment_score' in df_features.columns and df_features['sentiment_score'].notna().any():
                        # Беремо середнє values сентименту for весь period
                        avg_s = df_features['sentiment_score'].mean()
                        # Просand евристика for avg_sentiment
                        current_avg_sentiment = {
                            "positive": max(0, avg_s),
                            "negative": max(0, -avg_s),
                            "neutral": 1.0 - abs(avg_s)  # Просand нормалandforцandя
                        }
                    else:
                        # Fallback (як було ранandше, але тепер як forпасний варandант)
                        current_avg_sentiment = {"positive": 0.5, "negative": 0.5, "neutral": 0.0}

                    signals = generate_signals(
                        df=df_features,
                        avg_sentiment=current_avg_sentiment,
                        ticker=ticker_key,
                        thresholds=thresholds or {
                            "forecast": DEFAULT_FORECAST_THRESHOLDS,
                            "rsi": DEFAULT_RSI_THRESHOLDS,
                            "sentiment": DEFAULT_SENTIMENT_THRESHOLDS,
                        },
                        MODELS_DICT=MODELS_DICT,
                    )
                    df_features["trading_signal"] = signals.get("signal", "HOLD")

                    #  Додаємо ensemble_signal
                    try:
                        ensemble_signal = adaptive_weighted_signal(
                            ticker_signals={tf: {"final_signal": df_features["trading_signal"].iloc[-1]}},
                            daily_trend=df_features.get("trend_label", pd.Series(["HOLD"])).iloc[-1],
                            df_daily=df_prices_1d,
                            df_short_tf={tf: df_price},
                            sentiment_score=current_avg_sentiment["positive"] - current_avg_sentiment["negative"]
                        )
                        df_features["ensemble_signal"] = ensemble_signal
                        logger.info(f"[{ticker_key}:{tf}] [OK] Ensemble сигнал withгеnotровано: {ensemble_signal}")
                    except Exception as e:
                        logger.warning(f"[{ticker_key}:{tf}] [WARN] Error геnotрацandї ensemble_signal: {e}")

                    logger.info(f"[{ticker_key}:{tf}] [OK] Сигнали withгеnotровано. Фandнальний DF shape: {df_features.shape}")
                except Exception as e:
                    logger.exception(f"[{ticker_key}:{tf}] [ERROR] Error геnotрацandї сигналandв: {e}")

            # 6. Збереження фandнального DF
            all_processed_data[ticker_key][tf] = df_features

    logger.info("[Еandп 4] [OK] prepare_features_and_signals forвершено. Поверandємо оброблений словник.")
    return all_processed_data