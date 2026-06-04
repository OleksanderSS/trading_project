from pathlib import Path
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


class PipelineDataLoader:
    """Helper to handle data loading for pipeline execution."""

    @staticmethod
    def load_parquet_from_path(path: Path, label: str) -> pd.DataFrame | None:
        """Load a parquet file if it exists."""
        if path.exists():
            try:
                df = pd.read_parquet(path)
                logger.info(f"Loaded {label}: {df.shape}")
                return df
            except Exception as e:
                logger.error(f"Error loading {label} from {path}: {e}")
        else:
            logger.error(f"{label.capitalize()} file not found: {path}")
        return None

    @staticmethod
    def load_news_economic_data(batch_dir: Path) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
        """Load news and economic data with fallbacks."""
        news_data = PipelineDataLoader.load_parquet_from_path(batch_dir / "news_data.parquet", "news data")
        if news_data is None:
            persistent_news = Path("data/processed/features/news_data.parquet")
            news_data = PipelineDataLoader.load_parquet_from_path(persistent_news, "news data (persistent fallback)")
        econ_data = PipelineDataLoader.load_parquet_from_path(batch_dir / "economic_data.parquet", "economic data")
        if econ_data is None:
            for path in [
                Path("data/processed/features/macro_data.parquet"),
                Path("data/processed/features/economic_data.parquet"),
            ]:
                econ_data = PipelineDataLoader.load_parquet_from_path(path, "economic data (persistent fallback)")
                if econ_data is not None:
                    break
        return news_data, econ_data

    @staticmethod
    def load_from_db_fallback(
        orchestrator: Any, news_data: pd.DataFrame | None, economic_data: pd.DataFrame | None
    ):
        """DuckDB Database Fallback for news and macro data."""
        try:
            from src.data.management.data_manager import DataManager
            from src.processing.deduplication_utils import deduplicate_dataframe

            db_manager = DataManager(orchestrator.config_manager)
            table_names = db_manager.get_all_table_names()
            collector_configs = orchestrator.config_manager.get_config("collectors", {})
            all_news_dfs = []
            macro_dfs = []
            for table_name in table_names:
                if table_name in [
                    "cache_metadata",
                    "huggingface_data",
                    "enriched_features",
                    "experience_diary",
                    "market_data",
                ]:
                    continue
                df = db_manager.fetch_data_from_table(table_name)
                if df is None or df.empty:
                    continue
                collector_info = {}
                for config in collector_configs.values():
                    if config.get("table_name") == table_name:
                        collector_info = config
                        break
                if not collector_info:
                    collector_info = collector_configs.get(table_name, {})
                data_type = collector_info.get("data_type")
                if data_type == "news":
                    all_news_dfs.append(df)
                elif "fred" in table_name.lower() or "macro" in table_name.lower() or data_type == "macro_data":
                    macro_dfs.append(df)
            if news_data is None and all_news_dfs:
                news_data = PipelineDataLoader.reconstruct_from_db(all_news_dfs, "news", deduplicate_dataframe)
            if economic_data is None and macro_dfs:
                economic_data = PipelineDataLoader.reconstruct_from_db(macro_dfs, "economic", deduplicate_dataframe)
        except Exception as ex:
            logger.error(f"Виникла помилка: {ex}", exc_info=True)
            logger.warning(f"⚠️ Failed to load news/macro fallback from database: {ex}")
            raise
        return news_data, economic_data

    @staticmethod
    def reconstruct_from_db(dfs: list[pd.DataFrame], label: str, deduplicate_func: Any) -> pd.DataFrame:
        """Reconstruct a dataframe from multiple DB tables."""
        df = pd.concat(dfs, ignore_index=True)
        hashable_cols = [
            col
            for col in df.columns
            if df[col].apply(lambda x: isinstance(x, (str, int, float, bool, type(None)))).all()
        ]
        df, _ = deduplicate_func(df, hashable_cols)
        logger.info(f"✅ Reconstructed {label} data from database fallback: {df.shape}")
        return df
