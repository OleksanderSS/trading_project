#!/usr/bin/env python3
"""
Enhanced Backup System - Покращена система бекапу та відновлення
"""

import os
import shutil
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import logging
import zipfile
import threading

logger = logging.getLogger(__name__)

class EnhancedBackupSystem:
    """Покращена система бекапу з накопичувальною базою"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.backup_root = Path(self.config.get('backup', {}).get('root_path', 'backups/'))
        self.accumulated_db_path = Path(self.config.get('data', {}).get('accumulated_db', 'data/accumulated_data.parquet'))
        self.max_backups = self.config.get('backup', {}).get('max_backups', 10)
        
        # Створюємо структуру директорій
        self.backup_root.mkdir(parents=True, exist_ok=True)
        (self.backup_root / 'full').mkdir(exist_ok=True)
        (self.backup_root / 'accumulated').mkdir(exist_ok=True)
        (self.backup_root / 'metadata').mkdir(exist_ok=True)
        
        # Метадані
        self.metadata_file = self.backup_root / 'metadata' / 'backup_metadata.json'
        self.metadata = self._load_metadata()
        self.lock = threading.Lock()
        
        logger.info("[EnhancedBackupSystem] Initialized")
    
    def _load_metadata(self) -> Dict:
        """Завантаження метаdata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[BackupSystem] Error loading metadata: {e}")
        
        return {
            'last_full_backup': None,
            'last_accumulated_backup': None,
            'backup_history': [],
            'accumulated_snapshots': []
        }
    
    def _save_metadata(self):
        """Збереження метаdata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"[BackupSystem] Error saving metadata: {e}")
    
    def create_full_backup(self, project_root: str = None) -> str:
        """Створення повного бекапу проекту"""
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"full_backup_{timestamp}"
            backup_path = self.backup_root / 'full' / backup_name
            
            if project_root is None:
                project_root = Path.cwd().parent
            
            logger.info(f"[BackupSystem] Creating full backup: {backup_name}")
            
            try:
                backup_path.mkdir(parents=True, exist_ok=True)
                
                # Важливі директорії
                important_dirs = ['config', 'core', 'data', 'models', 'utils', 'pipeline']
                
                for dir_name in important_dirs:
                    src_dir = Path(project_root) / dir_name
                    if src_dir.exists():
                        dst_dir = backup_path / dir_name
                        shutil.copytree(src_dir, dst_dir, 
                                      ignore=shutil.ignore_patterns('*.pyc', '__pycache__', '.git'))
                        logger.info(f"[BackupSystem] Backed up: {dir_name}")
                
                # Маніфест
                manifest = {
                    'backup_type': 'full',
                    'timestamp': timestamp,
                    'created_at': datetime.now().isoformat(),
                    'size_mb': self._get_dir_size(backup_path) / 1024 / 1024
                }
                
                with open(backup_path / 'backup_manifest.json', 'w') as f:
                    json.dump(manifest, f, indent=2)
                
                # Стискаємо
                zip_path = backup_path.with_suffix('.zip')
                self._create_zip(backup_path, zip_path)
                shutil.rmtree(backup_path)
                
                # Оновлюємо метадані
                self.metadata['last_full_backup'] = str(zip_path)
                self.metadata['backup_history'].append({
                    'type': 'full',
                    'path': str(zip_path),
                    'timestamp': timestamp
                })
                
                self._cleanup_old_backups('full')
                self._save_metadata()
                
                logger.info(f"[BackupSystem] Full backup created: {zip_path}")
                return str(zip_path)
                
            except Exception as e:
                logger.error(f"[BackupSystem] Error creating full backup: {e}")
                raise
    
    def backup_accumulated_data(self) -> str:
        """Бекап накопичувальної бази"""
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"accumulated_backup_{timestamp}"
            backup_path = self.backup_root / 'accumulated' / backup_name
            
            logger.info(f"[BackupSystem] Creating accumulated backup: {backup_name}")
            
            try:
                backup_path.mkdir(parents=True, exist_ok=True)
                
                if self.accumulated_db_path.exists():
                    # Копіюємо дані
                    dst_db = backup_path / 'accumulated_data.parquet'
                    shutil.copy2(self.accumulated_db_path, dst_db)
                    
                    # Метадані
                    df = pd.read_parquet(self.accumulated_db_path)
                    metadata = {
                        'backup_type': 'accumulated',
                        'timestamp': timestamp,
                        'rows': len(df),
                        'columns': len(df.columns),
                        'size_mb': self.accumulated_db_path.stat().st_size / 1024 / 1024
                    }
                    
                    with open(backup_path / 'metadata.json', 'w') as f:
                        json.dump(metadata, f, indent=2)
                    
                    # Снепшот
                    snapshot_path = self.backup_root / 'accumulated' / 'latest_snapshot.parquet'
                    shutil.copy2(self.accumulated_db_path, snapshot_path)
                    
                    # Оновлюємо метадані
                    self.metadata['last_accumulated_backup'] = str(backup_path)
                    self.metadata['accumulated_snapshots'].append({
                        'path': str(backup_path),
                        'timestamp': timestamp,
                        'rows': metadata['rows']
                    })
                    
                    self._cleanup_old_backups('accumulated')
                    self._save_metadata()
                    
                    logger.info(f"[BackupSystem] Accumulated backup created: {backup_path}")
                    return str(backup_path)
                    
            except Exception as e:
                logger.error(f"[BackupSystem] Error creating accumulated backup: {e}")
                raise
    
    def restore_accumulated_data(self, backup_path: str = None) -> bool:
        """Відновлення накопичувальної бази"""
        with self.lock:
            try:
                if backup_path is None:
                    backup_path = self.metadata.get('last_accumulated_backup')
                
                if not backup_path or not Path(backup_path).exists():
                    logger.error(f"[BackupSystem] Backup not found: {backup_path}")
                    return False
                
                backup_path = Path(backup_path)
                logger.info(f"[BackupSystem] Restoring from: {backup_path}")
                
                # Бекап поточного стану
                if self.accumulated_db_path.exists():
                    current_backup = self.accumulated_db_path.with_suffix(f'.backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.parquet')
                    shutil.copy2(self.accumulated_db_path, current_backup)
                
                # Відновлення
                src_db = backup_path / 'accumulated_data.parquet'
                if src_db.exists():
                    self.accumulated_db_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_db, self.accumulated_db_path)
                    
                    df = pd.read_parquet(self.accumulated_db_path)
                    logger.info(f"[BackupSystem] Restored: {df.shape}")
                    return True
                    
            except Exception as e:
                logger.error(f"[BackupSystem] Error restoring: {e}")
                return False
    
    def _get_dir_size(self, path: Path) -> int:
        """Розмір директорії"""
        total = 0
        for item in path.rglob('*'):
            if item.is_file():
                total += item.stat().st_size
        return total
    
    def _create_zip(self, src_path: Path, dst_path: Path):
        """Створення ZIP архіву"""
        with zipfile.ZipFile(dst_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in src_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(src_path)
                    zipf.write(file_path, arcname)
    
    def _cleanup_old_backups(self, backup_type: str):
        """Очищення старих бекапів"""
        backup_dir = self.backup_root / backup_type
        
        if backup_type == 'full':
            backups = list(backup_dir.glob('*.zip'))
        else:
            backups = [d for d in backup_dir.iterdir() if d.is_dir()]
        
        backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        for old_backup in backups[self.max_backups:]:
            if old_backup.is_file():
                old_backup.unlink()
            else:
                shutil.rmtree(old_backup)
            logger.info(f"[BackupSystem] Deleted old backup: {old_backup.name}")
    
    def get_backup_status(self) -> Dict:
        """Статус бекапів"""
        return {
            'last_full_backup': self.metadata.get('last_full_backup'),
            'last_accumulated_backup': self.metadata.get('last_accumulated_backup'),
            'total_backups': len(self.metadata.get('backup_history', [])),
            'accumulated_snapshots': len(self.metadata.get('accumulated_snapshots', [])),
            'backup_root': str(self.backup_root)
        }


if __name__ == "__main__":
    print("Enhanced Backup System - готовий до використання")
    print("[SAVE] Покращена система бекапу та відновлення")
