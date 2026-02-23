# pipeline/data.py
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional, Union
import pandas as pd
from utils.logger import ProjectLogger
from pipeline.data_fetchers import polygon_fetch_ohlcv
from utils.cache_utils import CacheManager
from typing import Dict, Optional, Union
from typing import Optional, Union

logger = ProjectLogger.get_logger("DataPipeline")


def safe_load_financial_data(ticker: str, tf: str, start_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Обгортка for fetcher'а Polygon with беwithпечним поверnotнням DataFrame."""
    try:
        data = polygon_fetch_ohlcv(ticker, tf, start_date)
        if not data or "results" not in data:
            logger.warning(f"[safe_load_financial_data] [ERROR] No data for {ticker} {tf}")
            return pd.DataFrame()

        df = pd.DataFrame(data["results"])
        df['datetime'] = pd.to_datetime(df['t'], unit='ms')
        df = df.rename(columns={'c': 'close', 'o': 'open', 'h': 'high',
                                'l': 'low', 'v': 'volume'})
        df = df.drop(columns=['t'], errors='ignore')
        df = df.drop_duplicates(subset=['datetime']).sort_values('datetime')
        return df

    except Exception as e:
        logger.error(f"[safe_load_financial_data] Failed {ticker} {tf}: {e}")
        return pd.DataFrame()


def collect_financial_data(
        tickers: List[str],
        time_frames: List[str],
        data_path: Union[str, Path],
        cache_manager: CacheManager,
        use_cache: bool = True,
        force_refresh: bool = False,
        max_workers: int = 8
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Паралельnot forванandження фandнансових data andwith кешем CacheManager.
    - Використовує TTL кешу.
    - Уникає дублювання рядкandв.
    - force_refresh andгнорує кеш.
    """
    data_path = Path(data_path)
    data: Dict[str, Dict[str, pd.DataFrame]] = {t: {} for t in tickers}

    def fetch_and_cache(ticker: str, tf: str) -> (str, str, pd.DataFrame):
        cache_file = f"{ticker}_{tf}.parquet"

        # Заванandження with кешу
        cached_df = pd.DataFrame()
        if use_cache and not force_refresh:
            try:
                cached_df = cache_manager.get_df(cache_file, tf=tf, fallback=pd.DataFrame())
            except Exception:
                logger.info(f"[collect_financial_data] Cache miss or expired: {cache_file}")

        last_date = cached_df['datetime'].max() if not cached_df.empty else None

        # Заванandження нових data
        df_new = safe_load_financial_data(ticker, tf, start_date=last_date)

        # Комбandнування and removing duplicates
        combined = pd.concat([cached_df, df_new], ignore_index=True) if not cached_df.empty else df_new
        combined = combined.drop_duplicates(subset=['datetime']).sort_values('datetime')

        # Збереження в кеш
        if not combined.empty:
            cache_manager.set_df(cache_file, combined)

        return ticker, tf, combined

    tasks = [(t, tf) for t in tickers for tf in time_frames]
    with ThreadPoolExecutor(max_workers=min(max_workers, len(tasks))) as executor:
        futures = {executor.submit(fetch_and_cache, t, tf): (t, tf) for t, tf in tasks}
        for future in as_completed(futures):
            t, tf = futures[future]
            try:
                ticker, tf, df = future.result()
                data[ticker][tf] = df
            except Exception as e:
                logger.error(f"[collect_financial_data] Error for {t} {tf}: {e}")
                data[t][tf] = pd.DataFrame()

    return data
