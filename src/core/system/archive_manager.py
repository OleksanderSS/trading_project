# src/core/system/archive_manager.py

import os
import zipfile
from datetime import datetime
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


def _is_valid_file(filepath: str, max_size_mb: int = 50) -> bool:
    """Перевіряє чи file валідний для архівування"""
    if not os.path.exists(filepath) or not os.access(filepath, os.R_OK):
        return False
    
    try:
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        return size_mb <= max_size_mb
    except OSError:
        return False


def archive_directory(source_dir: str, output_dir: str = "archives", archive_prefix: str = "archive") -> Optional[Dict[str, int]]:
    """Архівує вказану директорію в ZIP file"""
    if not os.path.exists(source_dir):
        logger.warning(f" Директорія не існує: {source_dir}")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_name = f"{archive_prefix}_{os.path.basename(source_dir)}_{timestamp}.zip"
    archive_path = os.path.join(output_dir, archive_name)
    
    files_added = 0
    files_skipped = 0
    
    try:
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.endswith(('.tmp', '.log', '.pyc')):
                        files_skipped += 1
                        continue
                        
                    filepath = os.path.join(root, file)
                    if not _is_valid_file(filepath):
                        files_skipped += 1
                        continue
                        
                    arcname = os.path.relpath(filepath, source_dir)
                    zf.write(filepath, arcname)
                    files_added += 1
        
        logger.info(f" Архівовано: {files_added} файлів, пропущено: {files_skipped}")
        return {"files_added": files_added, "files_skipped": files_skipped}
        
    except Exception as e:
        logger.error(f"[ERROR] Error архівування: {e}")
        if os.path.exists(archive_path):
            os.remove(archive_path)
        return None
