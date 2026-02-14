# utils/price_utils.py

import pandas as pd
from utils.logger import ProjectLogger
from config.price_config import PRICE_METRICS

logger = ProjectLogger.get_logger("TradingProjectLogger")

def normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        logger.warning("[price_utils] [WARN] Порожнandй DataFrame, поверandємо каркас колонок")
        return pd.DataFrame(columns=["datetime","date","ticker"] + PRICE_METRICS)

    df = df.copy()

    # flatten MultiIndex
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns]

    # melt у tidy
    value_cols = [c for c in df.columns if any(m in c for m in PRICE_METRICS)]
    id_cols = [c for c in df.columns if c not in value_cols]

    tidy = df.melt(id_vars=id_cols, value_vars=value_cols,
                   var_name="metric", value_name="value")
    tidy["ticker"] = tidy["metric"].str.split("_").str[-1]
    tidy["metric"] = tidy["metric"].str.split("_").str[0]

    # pivot наforд
    price_df = tidy.pivot_table(index=["datetime","ticker"],
                                columns="metric", values="value").reset_index()

    # [OK] додаємо колонку date
    price_df["datetime"] = pd.to_datetime(price_df["datetime"], errors="coerce")
    price_df["date"] = price_df["datetime"].dt.normalize()

    # logging
    tickers = price_df["ticker"].nunique()
    logger.info(f"[price_utils] [OK] Нормалandwithовано данand: {len(price_df)} рядкandв, {tickers} тикерandв")

    return price_df