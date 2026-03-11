
# src/data/collectors/custom_csv_collector.py

import asyncio
import csv
import logging
from typing import List, Dict, Any

from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class CustomCSVCollector(BaseCollector):
    """
    Збирає дані з локального CSV-файлу, вказаного в конфігурації.
    """
    collector_type = "custom_csv"
    data_type = "generic"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Конструктор максимально спрощено

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Асинхронно читає вказаний CSV-файл, виконуючи операції вводу/виводу в окремому потоці.
        """
        file_path = self.config.get('file_path')
        if not file_path:
            self.logger.error(f"В конфігурації для '{self.collector_name}' не вказано 'file_path'.")
            return []

        self.logger.info(f"Читаємо дані з {file_path}...")
        try:
            # Використовуємо asyncio.to_thread, щоб не блокувати цикл подій
            records = await asyncio.to_thread(self._read_csv_sync, file_path)
            self.logger.info(f"Успішно прочитано {len(records)} записів з '{file_path}'.")
            return records
        except FileNotFoundError:
            err = FileNotFoundError(f"CSV-файл не знайдено за вказаним шляхом: {file_path}")
            self.handle_error(err, {"file_path": file_path})
            return []
        except Exception as e:
            self.handle_error(e, {"file_path": file_path})
            return []

    def _read_csv_sync(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Синхронна функція для виконання читання файлу. Виконується в окремому потоці.
        """
        encoding = self.config.get('encoding', 'utf-8')
        records = []
        with open(file_path, mode='r', encoding=encoding) as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                records.append(dict(row))
        return records
