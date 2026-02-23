# utils/benchmark_logger.py

import json
import datetime
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger("BenchmarkLogger")

def log_result(entry: dict, path: str = "results.json") -> None:
    """
    Логує реwithульandт експерименту/бектесту у JSON-file.
    - додає timestamp
    - пише у results.json построчно
    """
    if not isinstance(entry, dict):
        logger.error("[BenchmarkLogger] entry not є dict, пропускаємо")
        return
    try:
        entry_with_time = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            **entry
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry_with_time, ensure_ascii=False) + "\n")
        logger.info(f"[BenchmarkLogger] Збережено реwithульandт у {path}")
    except Exception as e:
        logger.exception(f"[BenchmarkLogger] Error withбереження реwithульandту: {e}")