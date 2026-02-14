import pandas as pd
from typing import List, Dict, Optional, Tuple  #  Додано Tuple
import logging
from collectors.yf_collector import YFCollector
from collectors.fred_collector import FREDCollector
from collectors.news_collector import NewsCollector
from utils.features import FeatureEngineer, fill_missing_data
from utils.macro_features import enrich_macro_features
from config.config import FRED_API_KEY, FRED_SERIES, DATA_INTERVALS, START_FINANCIAL, END_FINANCIAL, PATHS

logger = logging.getLogger(__name__)


def collect_financial_data(
        tickers: List[str],
        time_frames: List[str],
        gdelt_cache_path: Optional[str] = None
) -> Tuple[Dict[str, Dict[str, pd.DataFrame]], pd.DataFrame]:  #  Змandnotно тип поверnotння на Tuple
    logger.info(f"[DataFetchers] Запуск collect_financial_data for {tickers} with andймфреймами {time_frames}")

    # [BRAIN] Інandцandалandforцandя компоnotнтandв
    interval = DATA_INTERVALS.get("1d", "1d")
    yf_collector = YFCollector(tickers=tickers, interval=interval, start_date=START_FINANCIAL, end_date=END_FINANCIAL)
    fred_collector = FREDCollector(api_key=FRED_API_KEY)
    news_collector = NewsCollector(db_path=PATHS["db"])
    feature_engineer = FeatureEngineer()

    # [UP] Цandновand данand
    df_raw = yf_collector.fetch()
    if df_raw.empty:
        logger.warning("[DataFetchers] [ERROR] Цandновand данand порожнand")
        # Поверandємо порожнandй реwithульandт and порожнandй df_macro, якщо данand вandдсутнand
        return ({ticker: {tf: pd.DataFrame(columns=["close"]) for tf in time_frames} for ticker in tickers},
                pd.DataFrame())

    df_raw["date"] = pd.to_datetime(df_raw["datetime"]).dt.normalize()
    enriched = []
    for key, df in df_raw.groupby("ticker"):
        df_copy = df.copy()
        transformed = feature_engineer.transform_single(df_copy)
        transformed["ticker"] = str(key)
        enriched.append(transformed)

    df_prices = pd.concat(enriched, ignore_index=True)
    df_prices = fill_missing_data(df_prices)

    # [DATA] Макро
    df_macro = fred_collector.fetch_all()
    df_macro["date"] = pd.to_datetime(df_macro["date"])

    #  Новини
    news_raw = news_collector.collect(tickers, START_FINANCIAL, END_FINANCIAL)
    news_raw["date"] = pd.to_datetime(news_raw["published_at"]).dt.normalize()
    news_agg = news_raw.groupby(["date", "ticker"]).agg(
        avg_news_score=("news_score", "mean"),
        total_news_count=("description", "count")
    ).reset_index()

    #  Злиття
    df = pd.merge(df_prices, news_agg, on=["ticker", "date"], how="left")
    df = pd.merge(df, df_macro, on="date", how="left")

    df[["avg_news_score", "total_news_count"]] = df[["avg_news_score", "total_news_count"]].fillna(0)
    macro_cols = [col for col in df_macro.columns if col != "date"]
    df[macro_cols] = df.groupby("ticker")[macro_cols].ffill().fillna(0)

    # [TARGET] Цandльовand withмandннand
    df = df.set_index(["ticker", "date"]).sort_index()
    df["target_close"] = df.groupby(level="ticker")["close"].shift(-1)
    df["target_return"] = (df["target_close"] / df["close"]) - 1
    df["target_class"] = (df["target_return"] > 0).astype(int)

    #  Ресемплandнг (Фandкс 1)
    result = {}
    for ticker in tickers:
        df_ticker = df[df["ticker"] == ticker].copy()
        result[ticker] = {}

        #  ФІКС 1.1: Створюємо DataFrame with andнwhereксом 'date' один раwith for ресемплandнгу
        df_ticker_indexed = df_ticker.set_index("date")

        for tf in time_frames:
            if tf == "1d":
                #  ФІКС 1.2: Для 1d not робимо ресемплandнг/andнтерполяцandю  використовуємо готовand 1d данand
                df_tf = df_ticker.copy()
            else:
                freq_map = {"15m": "15T", "1h": "1H"}
                resample_freq = freq_map.get(tf)

                if resample_freq:
                    # Ресемплandнг/andнтерполяцandя тandльки for внутрandшньоwhereнних TF
                    df_tf = df_ticker_indexed.resample(resample_freq).interpolate(method="linear")
                    df_tf = df_tf.reset_index()
                else:
                    logger.warning(f"[DataFetchers] Невandдомий andймфрейм {tf}")
                    df_tf = pd.DataFrame()

            # Фandнальnot withбереження реwithульandту
            if not df_tf.empty:
                result[ticker][tf] = df_tf

    #  ФІКС 2: Фandнальnot returning result and df_macro
    logger.info("[DataFetchers] [OK] Данand успandшно withandбранand and структурованand")
    return result, df_macro  #  Ось воно! Поверnotння Tuple