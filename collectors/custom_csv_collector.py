# collectors/custom_csv_collector.py

import pandas as pd
from collectors.base_collector import BaseCollector
from typing import List, Dict
import logging

logger = logging.getLogger("trading_project.custom_csv_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class CustomCSVCollector(BaseCollector):
    def __init__(self, file_path: str, source_name: str = "CSV",
                 table_name: str = "csv_data", db_path: str = ":memory:", **kwargs):
        super().__init__(db_path=db_path, table_name=table_name, **kwargs)
        self.file_path = file_path
        self.source_name = source_name

    def fetch(self) -> pd.DataFrame:
        try:
            df_raw = pd.read_csv(self.file_path)
            entries = []
            for _, row in df_raw.iterrows():
                row_dict = row.to_dict()  # перетворюємо Series у dict
                entries.append({
                    "title": row_dict.get("title"),
                    "description": row_dict.get("description"),
                    "summary": row_dict.get("summary"),
                    "published_at": row_dict.get("published"),  # унandфandковаnot поле
                    "url": row_dict.get("url"),
                    "type": row_dict.get("type", "custom"),
                    "source": self.source_name,
                    "value": row_dict.get("value"),
                    "sentiment": row_dict.get("sentiment"),
                    "result": row_dict.get("result"),
                    "ticker": row_dict.get("ticker"),
                    "raw_data_fields": row_dict  # withберandгаємо все andнше
                })
            # якщо harmonize_batch є у BaseCollector  використовуємо
            if hasattr(self, "harmonize_batch"):
                return pd.DataFrame(self.harmonize_batch(entries, source=self.source_name))
            else:
                return pd.DataFrame(entries)
        except Exception as e:
            logger.error(f"[CustomCSVCollector] Error reading {self.file_path}: {e}")
            return pd.DataFrame()

    def collect(self) -> List[Dict]:
        df = self.fetch()
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        self.save(records)
        return records