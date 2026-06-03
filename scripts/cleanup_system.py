import os
import sys
from pathlib import Path
import psutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("cleanup_system")


def cleanup():
    logger.info("--- Starting system cleanup ---")
    
    # 1. Kill all python processes EXCEPT this one
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            if proc.info['name'] == 'python.exe' and proc.info['pid'] != current_pid:
                logger.info(f"   Killing python process: {proc.info['pid']}")
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # 2. Cleanup DuckDB lock files
    data_dir = Path('data')
    if data_dir.exists():
        for lock_file in data_dir.glob('*.duckdb.wal'):
            logger.info(f"   Removing stale WAL file: {lock_file}")
            try:
                os.remove(lock_file)
            except Exception as e:
                logger.error(f"   Failed to remove {lock_file}: {e}", exc_info=True)
                # Fail gracefully as this is a cleanup script
        
        for lock_file in data_dir.glob('*.duckdb.tmp'):
             logger.info(f"   Removing stale TMP file: {lock_file}")
             try:
                 os.remove(lock_file)
             except Exception as e:
                 logger.error(f"   Failed to remove {lock_file}: {e}", exc_info=True)
                 # Fail gracefully

    # 3. Clear logs
    log_file = Path('logs/system.log')
    if log_file.exists():
        logger.info("   Clearing system.log...")
        try:
            with open(log_file, 'w') as f:
                f.write(f"--- LOG RESET ---")
        except Exception as e:
            logger.error(f"   Failed to clear log: {e}", exc_info=True)
            # Fail gracefully

    logger.info("--- Cleanup complete ---")

if __name__ == "__main__":
    cleanup()
