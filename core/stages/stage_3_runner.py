# core/stages/stage_3_runner.py

from core.stages.stage_3_features import prepare_stage3_datasets
from utils.data_storage import load_from_storage
from utils.data_cleaning import harmonize_dataframe, safe_fill
from core.stages.stage_3_utils import print_stage3_stats
from utils.trading_calendar import TradingCalendar
from config.config import TICKERS, DATA_INTERVALS
import pandas as pd
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def run_stage3(merged_full, calendar_year=2025, base_path="/content/drive/MyDrive/trading_project/data", 
                tickers=None, timeframes=None):
    """
    🆕 ПОКРАЩЕНА ВЕРСІЯ з гнучкими параметрами
    
    Args:
        merged_full: DataFrame з етапу 2
        calendar_year: Рік календаря
        base_path: Базовий шлях
        tickers: Гнучкий список тікерів
        timeframes: Гнучкий список таймфреймів
    """
    # 🆕 ГНУЧКІ ПАРАМЕТРИ
    if tickers is None:
        tickers = list(TICKERS.keys())
    if timeframes is None:
        timeframes = list(DATA_INTERVALS.keys())
    
    logger.info(f"[Stage3] [START] ENHANCED Runner - Processing {len(tickers)} tickers, {len(timeframes)} timeframes")
    
    calendar = TradingCalendar.from_year(calendar_year, tickers=tickers)

    # 🆕 ВИКЛИКАЄМО ПОКРАЩЕНУ ФУНКЦІЮ
    merged_stage3, context_df, features_df, trigger_data = prepare_stage3_datasets(
        merged_full, calendar, tickers, timeframes
    )

    # [TOOL] FIX: НЕ всandновлюємо trade_date+event_time як andнwhereкс!
    if isinstance(merged_stage3.index, pd.MultiIndex):
        if set(merged_stage3.index.names) & {'trade_date', 'event_time'}:
            print("[TOOL] Resetting MultiIndex (contains trade_date/event_time)")
            merged_stage3 = merged_stage3.reset_index()
    elif merged_stage3.index.name in ['trade_date', 'event_time']:
        print("[TOOL] Resetting index")
        merged_stage3 = merged_stage3.reset_index()

    # [PROTECT] Reset index for features_df
    if isinstance(features_df.index, pd.MultiIndex) or features_df.index.name in ["date","trade_date","event_time"]:
        features_df = features_df.reset_index()

    # [PROTECT] Reset index for кожного context_df
    for layer_name, df in context_df.items():
        if isinstance(df.index, pd.MultiIndex) or df.index.name in ["date","trade_date","event_time"]:
            context_df[layer_name] = df.reset_index()

    #  Дandагностика merged_stage3
    print("\n=== Stage3: merged_stage3 дandагностика ===")
    print("Shape:", merged_stage3.shape)
    print("Інwhereкс:",
          merged_stage3.index.names if isinstance(merged_stage3.index, pd.MultiIndex) else merged_stage3.index.name)
    print("Першand 20 колонок:", list(merged_stage3.columns)[:20])

    # Перевandрка наявностand колонок for кожного тandкера
    for ticker in TICKERS.keys():
        t = ticker.lower()
        print(f"\n[UP] {ticker}:")
        for interval in DATA_INTERVALS.keys():
            has_open = f"{interval}_open_{t}" in merged_stage3.columns
            has_close = f"{interval}_close_{t}" in merged_stage3.columns
            print(f"  {interval}: open={has_open}, close={has_close}")

    if "ticker" in merged_stage3.columns:
        print("Унandкальнand tickers:", merged_stage3["ticker"].unique())
    print(merged_stage3.head(3))

    print_stage3_stats(merged_stage3, TICKERS.keys())

    # --- Features ---
    features_df = harmonize_dataframe(features_df)
    if "trade_date" in features_df.columns:
        features_df["trade_date"] = pd.to_datetime(features_df["trade_date"], errors="coerce").dt.tz_convert(None)
    # [PROTECT] Alias for date
    if "date" not in features_df.columns and "trade_date" in features_df.columns:
        features_df["date"] = features_df["trade_date"]

    features_path = f"{base_path}/stage3_features.parquet"
    if os.path.exists(features_path):
        old_features = pd.read_parquet(features_path)
        if "date" in old_features.columns:
            old_features["date"] = pd.to_datetime(old_features["date"], errors="coerce").dt.tz_convert(None)
        features_df = pd.concat([old_features, features_df], ignore_index=True, sort=False).drop_duplicates()

    max_date = features_df['date'].dropna().max() if 'date' in features_df.columns else "N/A"
    features_df.to_parquet(features_path, index=False)
    print(f" stage3_features: {len(features_df)} rows, {features_df.shape[1]} columns, last date = {max_date}")

    # --- Context layers ---
    for layer_name, df in context_df.items():
        safe_df = harmonize_dataframe(df)
        if "trade_date" in safe_df.columns:
            safe_df["trade_date"] = pd.to_datetime(safe_df["trade_date"], errors="coerce").dt.tz_convert(None)
        # [PROTECT] Alias for date
        if "date" not in safe_df.columns and "trade_date" in safe_df.columns:
            safe_df["date"] = safe_df["trade_date"]

        path = f"{base_path}/stage3_context_{layer_name}.parquet"
        if os.path.exists(path):
            old_df = pd.read_parquet(path)
            if "date" in old_df.columns:
                old_df["date"] = pd.to_datetime(old_df["date"], errors="coerce").dt.tz_localize(None)
            safe_df = pd.concat([old_df, safe_df], ignore_index=True, sort=False).drop_duplicates()
        safe_df.to_parquet(path, index=False)
        max_date = safe_df['date'].max() if 'date' in safe_df.columns else "N/A"
        print(f" stage3_context_{layer_name}: {len(safe_df)} rows, {safe_df.shape[1]} columns, last date = {max_date}")

    # --- Trigger data ---
    print("\n=== Stage3: trigger_data обробка ===")
    trigger_df_data = {}
    for k, v in trigger_data.items():
        try:
            if isinstance(v, (list, pd.Series, np.ndarray)):
                trigger_df_data[k] = v
            else:
                trigger_df_data[k] = [v]
        except Exception as e:
            print(f"[WARN] Skipping trigger key '{k}': {e}")

    if not trigger_df_data:
        print("[WARN] trigger_data порожнandй - створюю мandнandмальний")
        trigger_df = pd.DataFrame({'dummy': [0]})
    else:
        trigger_df = pd.DataFrame(trigger_df_data).reset_index(drop=True)

    # [PROTECT] Alias for date
    if "date" not in trigger_df.columns and "trade_date" in trigger_df.columns:
        trigger_df["date"] = trigger_df["trade_date"]

    trigger_df = harmonize_dataframe(trigger_df)
    if "trade_date" in trigger_df.columns and trigger_df["trade_date"].dtype != 'datetime64[ns]':
        trigger_df["trade_date"] = pd.to_datetime(trigger_df["trade_date"], errors="coerce")

    trigger_path = f"{base_path}/stage3_context_trigger.parquet"
    if os.path.exists(trigger_path):
        old_trigger = pd.read_parquet(trigger_path)
        if "date" in old_trigger.columns:
            old_trigger["date"] = pd.to_datetime(old_trigger["date"], errors="coerce").dt.tz_convert(None)
        trigger_df = pd.concat([old_trigger, trigger_df], ignore_index=True, sort=False).drop_duplicates()

    trigger_df.to_parquet(trigger_path, index=False)
    print(f" stage3_context_trigger: {len(trigger_df)} rows, {trigger_df.shape[1]} columns")

    merged_stage3 = safe_fill(merged_stage3)
    print("Залишилось NaN пandсля safe_fill:", merged_stage3.isna().sum().sum())

    print("[OK] Stage 3 реwithульandти withбережено")
    return merged_stage3, context_df, features_df, trigger_data