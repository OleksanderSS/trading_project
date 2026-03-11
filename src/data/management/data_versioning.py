# src/core/data/data_versioning.py
"""
DATA VERSIONING UTILITIES
Система версіонування та перевірки свіжості даних
"""

import os
import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class DataVersioning:
    """
    Система версіонування даних для уникнення застарілих файлів
    """
    
    def __init__(self, version_dir: str = "data/versions"):
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Файл метаданих версій
        self.metadata_file = self.version_dir / "data_versions.json"
        self.metadata = self._load_metadata()
        
        # Максимальний вік даних (днів)
        self.max_age_days = {
            'daily_data': 1,
            'intraday_data': 0.25,  # 6 годин
            'news_data': 0.5,  # 12 годин
            'technical_indicators': 1,
            'targets': 1,
            'cache': 0.1  # 2.4 години
        }
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Завантаження метаданих версій"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Збереження метаданих версій"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
    
    def _get_file_hash(self, file_path: Path) -> str:
        """Обчислення хешу файлу"""
        if not file_path.exists():
            return ""
        
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {e}")
            return ""
    
    def _get_file_size(self, file_path: Path) -> int:
        """Отримання розміру файлу"""
        if not file_path.exists():
            return 0
        return file_path.stat().st_size
    
    def _get_file_mtime(self, file_path: Path) -> datetime:
        """Отримання часу модифікації файлу"""
        if not file_path.exists():
            return datetime.min
        return datetime.fromtimestamp(file_path.stat().st_mtime)
    
    def register_file(self, file_path: str, data_type: str, 
                     description: str = "", metadata: Dict = None) -> Dict[str, Any]:
        """
        Реєстрація файлу в системі версіонування
        
        Args:
            file_path: Шлях до файлу
            data_type: Тип даних
            description: Опис файлу
            metadata: Додаткові метадані
            
        Returns:
            Dict[str, Any]: Інформація про версію
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"File not found: {file_path}")
            return {}
        
        version_info = {
            'file_path': str(file_path),
            'data_type': data_type,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'file_size': self._get_file_size(file_path),
            'file_hash': self._get_file_hash(file_path),
            'file_mtime': self._get_file_mtime(file_path).isoformat(),
            'metadata': metadata or {}
        }
        
        # Додаємо до метаданих
        file_key = str(file_path.relative_to(Path.cwd()))
        self.metadata[file_key] = version_info
        
        # Зберігаємо метадані
        self._save_metadata()
        
        self.logger.info(f"[OK] Registered file: {file_key} ({data_type})")
        return version_info
    
    def is_file_fresh(self, file_path: str, data_type: str = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Перевірка чи файл є свіжим
        
        Args:
            file_path: Шлях до файлу
            data_type: Тип даних
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (is_fresh, info)
        """
        file_path = Path(file_path)
        file_key = str(file_path.relative_to(Path.cwd()))
        
        if not file_path.exists():
            return False, {'reason': 'file_not_exists'}
        
        # Отримуємо інформацію про файл
        current_mtime = self._get_file_mtime(file_path)
        current_size = self._get_file_size(file_path)
        current_hash = self._get_file_hash(file_path)
        
        # Перевіряємо вік файлу
        if data_type and data_type in self.max_age_days:
            max_age = timedelta(days=self.max_age_days[data_type])
            file_age = datetime.now() - current_mtime
            
            if file_age > max_age:
                return False, {
                    'reason': 'file_too_old',
                    'file_age_days': file_age.days,
                    'max_age_days': self.max_age_days[data_type]
                }
        
        # Перевіряємо метадані
        if file_key in self.metadata:
            stored_info = self.metadata[file_key]
            
            # Перевіряємо хеш
            if stored_info.get('file_hash') != current_hash:
                return False, {'reason': 'file_modified'}
            
            # Перевіряємо розмір
            if stored_info.get('file_size') != current_size:
                return False, {'reason': 'file_size_changed'}
            
            # Перевіряємо вік з метаданих
            stored_mtime = datetime.fromisoformat(stored_info['file_mtime'])
            metadata_age = datetime.now() - stored_mtime
            
            if data_type and data_type in self.max_age_days:
                max_age = timedelta(days=self.max_age_days[data_type])
                if metadata_age > max_age:
                    return False, {
                        'reason': 'metadata_too_old',
                        'metadata_age_days': metadata_age.days,
                        'max_age_days': self.max_age_days[data_type]
                    }
        
        return True, {'reason': 'file_fresh'}
    
    def get_fresh_files(self, data_type: str = None) -> List[Dict[str, Any]]:
        """
        Отримання списку свіжих файлів
        
        Args:
            data_type: Тип даних
            
        Returns:
            List[Dict[str, Any]]: Список свіжих файлів
        """
        fresh_files = []
        
        for file_key, version_info in self.metadata.items():
            if data_type and version_info.get('data_type') != data_type:
                continue
            
            is_fresh, info = self.is_file_fresh(file_key, version_info.get('data_type'))
            
            if is_fresh:
                fresh_files.append({
                    'file_key': file_key,
                    'version_info': version_info,
                    'fresh_info': info
                })
        
        return fresh_files
    
    def get_stale_files(self, data_type: str = None) -> List[Dict[str, Any]]:
        """
        Отримання списку застарілих файлів
        
        Args:
            data_type: Тип даних
            
        Returns:
            List[Dict[str, Any]]: Список застарілих файлів
        """
        stale_files = []
        
        for file_key, version_info in self.metadata.items():
            if data_type and version_info.get('data_type') != data_type:
                continue
            
            is_fresh, info = self.is_file_fresh(file_key, version_info.get('data_type'))
            
            if not is_fresh:
                stale_files.append({
                    'file_key': file_key,
                    'version_info': version_info,
                    'stale_info': info
                })
        
        return stale_files
    
    def cleanup_stale_files(self, data_type: str = None, dry_run: bool = True) -> Dict[str, Any]:
        """
        Очищення застарілих файлів
        
        Args:
            data_type: Тип даних
            dry_run: Чи виконувати в режимі симуляції
            
        Returns:
            Dict[str, Any]: Результати очищення
        """
        stale_files = self.get_stale_files(data_type)
        
        results = {
            'total_stale': len(stale_files),
            'deleted_files': [],
            'failed_deletions': [],
            'dry_run': dry_run
        }
        
        for file_info in stale_files:
            file_path = Path(file_info['file_key'])
            
            if dry_run:
                results['deleted_files'].append({
                    'file_path': str(file_path),
                    'reason': file_info['stale_info'].get('reason', 'unknown')
                })
            else:
                try:
                    if file_path.exists():
                        file_path.unlink()
                        results['deleted_files'].append({
                            'file_path': str(file_path),
                            'reason': file_info['stale_info'].get('reason', 'unknown')
                        })
                        
                        # Видаляємо з метаданих
                        if file_info['file_key'] in self.metadata:
                            del self.metadata[file_info['file_key']]
                        
                except Exception as e:
                    results['failed_deletions'].append({
                        'file_path': str(file_path),
                        'error': str(e)
                    })
        
        if not dry_run:
            self._save_metadata()
        
        self.logger.info(f"{'[SEARCH] Dry run' if dry_run else ' Cleanup'}: {len(results['deleted_files'])} files processed")
        
        return results
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Отримання інформації про файл
        
        Args:
            file_path: Шлях до файлу
            
        Returns:
            Dict[str, Any]: Інформація про файл
        """
        file_path = Path(file_path)
        file_key = str(file_path.relative_to(Path.cwd()))
        
        info = {
            'file_path': str(file_path),
            'file_exists': file_path.exists(),
            'registered': file_key in self.metadata
        }
        
        if file_path.exists():
            info.update({
                'file_size': self._get_file_size(file_path),
                'file_mtime': self._get_file_mtime(file_path).isoformat(),
                'file_hash': self._get_file_hash(file_path)
            })
        
        if file_key in self.metadata:
            info['metadata'] = self.metadata[file_key]
        
        return info
    
    def generate_report(self, data_type: str = None) -> Dict[str, Any]:
        """
        Генерація звіту про стан даних
        
        Args:
            data_type: Тип даних
            
        Returns:
            Dict[str, Any]: Звіт про стан даних
        """
        all_files = list(self.metadata.keys())
        fresh_files = self.get_fresh_files(data_type)
        stale_files = self.get_stale_files(data_type)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'data_type': data_type,
            'total_files': len(all_files),
            'fresh_files': len(fresh_files),
            'stale_files': len(stale_files),
            'freshness_rate': len(fresh_files) / len(all_files) * 100 if all_files else 0,
            'files_by_type': {},
            'age_distribution': {},
            'recommendations': []
        }
        
        # Розподіл по типах
        for file_key, version_info in self.metadata.items():
            file_type = version_info.get('data_type', 'unknown')
            if file_type not in report['files_by_type']:
                report['files_by_type'][file_type] = 0
            report['files_by_type'][file_type] += 1
        
        # Рекомендації
        if report['freshness_rate'] < 80:
            report['recommendations'].append("Low freshness rate - consider updating data")
        
        if len(stale_files) > 0:
            report['recommendations'].append(f"Found {len(stale_files)} stale files - consider cleanup")
        
        return report

def create_data_versioning(version_dir: str = "data/versions") -> DataVersioning:
    """
    Factory function для створення системи версіонування
    
    Args:
        version_dir: Директорія для версій
        
    Returns:
        DataVersioning: Екземпляр системи версіонування
    """
    return DataVersioning(version_dir)
