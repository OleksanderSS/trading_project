from typing import List, Dict, Any
import asyncio
import functools

from src.data.collectors.base_collector import BaseCollector

class HuggingFaceCollector(BaseCollector):
    """Колектор для завантаження датасетів з Hugging Face Hub."""
    collector_type = "hugging_face"
    data_type = "market_sentiment" # Або інший відповідний тип

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Специфічна конфігурація для Hugging Face
        self.dataset_name = self.config.get("dataset_name")
        self.subset_name = self.config.get("subset_name")
        self.split = self.config.get("split", "train")

        if not self.dataset_name:
            self.logger.error("Не вказано 'dataset_name' для HuggingFaceCollector.")

    async def fetch_raw_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Завантажує датасет з Hugging Face Hub."""
        if not self.dataset_name:
            return []
        
        # Lazy import datasets, оскільки це важка залежність
        try:
            from datasets import load_dataset
            from dotenv import load_dotenv
            import os
        except ImportError:
            self.logger.error("Для роботи з HuggingFaceCollector необхідно встановити бібліотеку 'datasets'.")
            return []

        try:
            self.logger.info(f"Завантаження датасету '{self.dataset_name}' (підмножина: {self.subset_name}, частина: {self.split})...")
            
            # Force loading of the .env file
            load_dotenv()
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                 self.logger.error("Could not find HuggingFace token in the .env file.")

            # Виконуємо синхронну функцію в окремому потоці, щоб не блокувати asyncio loop
            loop = asyncio.get_running_loop()
            load_func = functools.partial(load_dataset, self.dataset_name, self.subset_name, split=self.split, token=hf_token)
            dataset = await loop.run_in_executor(None, load_func)
            
            self.logger.info("Датасет успішно завантажено.")
            return dataset.to_list()

        except Exception as e:
            self.handle_error(e, {"message": f"Помилка під час завантаження датасету '{self.dataset_name}'"})
            return []
