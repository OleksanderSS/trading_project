import json
from pathlib import Path
from typing import Any

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


async def load_stage6_results(batch_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    stage_6_file = batch_dir / "stage_6_results.json"
    if not stage_6_file.exists():
        return [], [], {}

    import aiofiles

    try:
        async with aiofiles.open(stage_6_file, encoding="utf-8") as f:
            content = await f.read()
            data = json.loads(content)
            signals = data.get("predictions", [])
            trading_activity = data.get("trade_history", [])
            portfolio_summary = data.get("portfolio_summary", {})
            return signals, trading_activity, portfolio_summary
    except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
        logger.exception(f"Error loading stage 6 results from {stage_6_file}: {e}")
        raise


def save_evaluation_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)


async def save_evaluation_summary_async(path: Path, summary: dict[str, Any]) -> None:
    import aiofiles

    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(summary, indent=2, default=str))
