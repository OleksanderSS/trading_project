# utils/zip_forecast_logic.py

import os
import zipfile
from datetime import datetime
from typing import Optional, Dict
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def _is_valid_file(filepath: str, max_size_mb: int = 50) -> bool:
    """Перевandряє чи file валandдний for архandвування"""
    if not os.path.exists(filepath) or not os.access(filepath, os.R_OK):
        return False
    
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return size_mb <= max_size_mb
    except OSError:
        return False


def archive_forecasts(forecast_dir: str, output_dir: str = "archives") -> Optional[Dict[str, int]]:

    """Архandвує прогноwithи в ZIP file"""
    if not os.path.exists(forecast_dir):
        logger.warning(f" Директорandя not andснує: {forecast_dir}")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"forecasts_{os.path.basename(forecast_dir)}_{timestamp}.zip"
    archive_path = os.path.join(output_dir, archive_name)
    
    files_added = 0
    files_skipped = 0
    
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(forecast_dir):
                for file in files:
                    if file.endswith(('.tmp', '.log', '.pyc')):
                        files_skipped += 1
                        continue
                        
                    filepath = os.path.join(root, file)
                    if not _is_valid_file(filepath):
                        files_skipped += 1
                        continue
                        
                    arcname = os.path.relpath(filepath, forecast_dir)
                    zf.write(filepath, arcname)
                    files_added += 1
        
        logger.info(f" Архandвовано: {files_added} fileandв, пропущено: {files_skipped}")
        return {"files_added": files_added, "files_skipped": files_skipped}
        
    except Exception as e:
        logger.error(f"[ERROR] Error архandвування: {e}")
        if os.path.exists(archive_path):
            os.remove(archive_path)
        return None
