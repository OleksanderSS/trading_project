import json
import logging
from pathlib import Path

import aiofiles

logger = logging.getLogger(__name__)


class FeatureLoader:
    """Helper to load selected features files, supporting both sync and async."""

    def __init__(self, batch_dir: Path):
        self.batch_dir = batch_dir

    def _get_file_candidates(self, model_type: str, ticker: str, target_name: str) -> list[Path]:
        return [
            self.batch_dir / f"selected_features_{model_type}_{ticker}_{target_name}.json",
            self.batch_dir / f"selected_features_{model_type}_{ticker}.json",
            self.batch_dir / f"selected_features_{model_type}_{target_name}.json",
            self.batch_dir / f"selected_features_{model_type}.json",
        ]

    async def load_async(self, model_type: str, ticker: str, target_name: str) -> list[str]:
        """Load features asynchronously."""
        candidates = self._get_file_candidates(model_type, ticker, target_name)
        for candidate in candidates:
            if candidate.exists():
                try:
                    async with aiofiles.open(candidate, encoding="utf-8") as f:
                        content = await f.read()
                        data = json.loads(content)
                        return data.get("selected_features", [])
                except Exception as e:
                    self.logger.error(f"Виникла помилка: {e}", exc_info=True)
                    self.logger.warning(f"Failed to load {candidate}: {e}", exc_info=True)
                    raise
        return []

    def load_sync(self, model_type: str, ticker: str, target_name: str) -> list[str]:
        """Load features synchronously."""
        candidates = self._get_file_candidates(model_type, ticker, target_name)
        for candidate in candidates:
            if candidate.exists():
                try:
                    with open(candidate, encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("selected_features", [])
                except Exception as e:
                    self.logger.error(f"Виникла помилка: {e}", exc_info=True)
                    self.logger.warning(f"Failed to load {candidate}: {e}", exc_info=True)
                    raise
        return []
