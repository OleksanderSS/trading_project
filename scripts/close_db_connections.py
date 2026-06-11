#!/usr/bin/env python3
"""
Script to close all DuckDB connections and release locks.
Run this before starting a new pipeline if you encounter lock errors.
"""

import sys
import os
import psutil
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger(__name__)


def find_processes_using_file(file_path: str):
    """Find all processes that have the file open."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'open_files']):
        try:
            if proc.info['open_files']:
                for file in proc.info['open_files']:
                    if file_path in file.path:
                        processes.append({
                            'pid': proc.info['pid'],
                            'name': proc.info['name'],
                            'path': file.path
                        })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return processes


def close_all_connections():
    """Close all DataManager connections."""
    logger.info("🔒 Closing all DataManager connections...")
    try:
        DataManager.close_all_connections()
        logger.info("✅ All DataManager connections closed")
    except Exception as e:
        logger.error(f"❌ Error closing connections: {e}")


def check_database_locks():
    """Check for processes holding database locks."""
    db_path = project_root / "data" / "trading_data.duckdb"
    
    if not db_path.exists():
        logger.info(f"📁 Database file not found: {db_path}")
        return
    
    logger.info(f"🔍 Checking for processes using: {db_path}")
    processes = find_processes_using_file(str(db_path))
    
    if not processes:
        logger.info("✅ No processes found using the database")
    else:
        logger.warning(f"⚠️ Found {len(processes)} process(es) using the database:")
        for proc in processes:
            logger.warning(f"   PID {proc['pid']}: {proc['name']}")
            logger.warning(f"   Path: {proc['path']}")
        
        logger.info("\n💡 To close these processes, run:")
        for proc in processes:
            logger.info(f"   taskkill /PID {proc['pid']} /F")


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("🔧 DuckDB Connection Cleanup Tool")
    logger.info("=" * 60)
    
    # Step 1: Close all DataManager connections
    close_all_connections()
    
    # Step 2: Check for remaining locks
    check_database_locks()
    
    logger.info("=" * 60)
    logger.info("✅ Cleanup complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
