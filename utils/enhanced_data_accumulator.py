# enhanced_data_accumulator.py - Покращена система накопичення data

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging
import tarfile
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class EnhancedDataAccumulator:
    """
    Покращена система накопичення data для локального тренування та Colab
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Створює структуру папок"""
        self.raw_dir = self.base_dir / "raw"
        self.enriched_dir = self.base_dir / "enriched"
        self.colab_packages_dir = self.base_dir / "colab_packages"
        
        # Створюємо папки
        for dir_path in [self.raw_dir, self.enriched_dir, self.colab_packages_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Папки для специфічних data
        (self.raw_dir / "stage1_latest").mkdir(exist_ok=True)
        (self.raw_dir / "daily").mkdir(exist_ok=True)
        (self.enriched_dir / "daily_batches").mkdir(exist_ok=True)
        
        logger.info(f"[EnhancedAccumulator] Директорії created в {self.base_dir}")
    
    def save_stage1_data(self, stage1_data: Dict[str, pd.DataFrame], 
                         create_daily_backup: bool = True) -> Dict[str, str]:
        """
        Зберігає дані етапу 1
        
        Returns:
            Dict: шляхи до збережених файлів
        """
        logger.info("[EnhancedAccumulator] Збереження Stage 1 data...")
        
        saved_files = {}
        stage1_dir = self.raw_dir / "stage1_latest"
        
        # Зберігаємо кожен тип data
        for data_type, df in stage1_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_path = stage1_dir / f"{data_type}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files[data_type] = str(file_path)
                logger.info(f"[EnhancedAccumulator] Збережено {data_type}: {df.shape}")
        
        # Створюємо метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_types': list(saved_files.keys()),
            'shapes': {dt: pd.read_parquet(path).shape for dt, path in saved_files.items()}
        }
        
        metadata_path = stage1_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Щоденний бекап
        if create_daily_backup:
            daily_dir = self.raw_dir / "daily" / datetime.now().strftime("%Y-%m-%d")
            daily_dir.mkdir(parents=True, exist_ok=True)
            
            for file_path in saved_files.values():
                src = Path(file_path)
                dst = daily_dir / src.name
                src.copy(dst)
        
        logger.info(f"[EnhancedAccumulator] Stage 1 saved: {len(saved_files)} файлів")
        return saved_files
    
    def accumulate_stage2_data(self, merged_df: pd.DataFrame, 
                             batch_size: Optional[int] = None) -> Dict[str, str]:
        """
        Накопичує дані етапу 2
        
        Args:
            merged_df: DataFrame з об'єднаними даними
            batch_size: Розмір батчу для автоматичного розділення
            
        Returns:
            Dict: шляхи до збережених файлів
        """
        logger.info(f"[EnhancedAccumulator] Накопичення Stage 2 data: {merged_df.shape}")
        
        if merged_df.empty:
            logger.warning("[EnhancedAccumulator] DataFrame порожній, пропускаємо")
            return {}
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Зберігаємо останні дані
        latest_file = self.enriched_dir / "stage2_latest.parquet"
        merged_df.to_parquet(latest_file, index=False)
        saved_files['latest'] = str(latest_file)
        
        # 2. Накопичуємо до основного файлу
        accumulated_file = self.enriched_dir / "stage2_accumulated.parquet"
        
        if accumulated_file.exists():
            # Завантажуємо існуючі дані
            existing_df = pd.read_parquet(accumulated_file)
            logger.info(f"[EnhancedAccumulator] Існуючі дані: {existing_df.shape}")
            
            # Видаляємо дублікати
            combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
            
            # Визначаємо колонки для унікальності
            unique_cols = []
            for col in ['published_at', 'title', 'url', 'timestamp']:
                if col in combined_df.columns:
                    unique_cols.append(col)
            
            if unique_cols:
                before_dedup = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=unique_cols, keep='last')
                after_dedup = len(combined_df)
                logger.info(f"[EnhancedAccumulator] Видалено дублікатів: {before_dedup - after_dedup}")
        else:
            combined_df = merged_df
        
        # Зберігаємо накопичені дані
        combined_df.to_parquet(accumulated_file, index=False)
        saved_files['accumulated'] = str(accumulated_file)
        logger.info(f"[EnhancedAccumulator] Накопичено: {combined_df.shape}")
        
        # 3. Створюємо щоденний батч
        batch_file = self.enriched_dir / "daily_batches" / f"batch_{timestamp}.parquet"
        merged_df.to_parquet(batch_file, index=False)
        saved_files['batch'] = str(batch_file)
        
        # 4. Розділяємо на батчі якщо needed
        if batch_size and len(merged_df) > batch_size:
            self._create_size_batches(merged_df, timestamp, batch_size)
        
        # 5. Створюємо метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'latest_shape': merged_df.shape,
            'accumulated_shape': combined_df.shape,
            'batch_count': len(list((self.enriched_dir / "daily_batches").glob("*.parquet"))),
            'total_accumulated_rows': len(combined_df)
        }
        
        metadata_path = self.enriched_dir / "accumulation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[EnhancedAccumulator] Stage 2 накопичено: {len(saved_files)} файлів")
        return saved_files
    
    def _create_size_batches(self, df: pd.DataFrame, timestamp: str, 
                           batch_size: int):
        """Створює батчі за розміром"""
        batches_dir = self.enriched_dir / "size_batches"
        batches_dir.mkdir(exist_ok=True)
        
        for i, start_idx in enumerate(range(0, len(df), batch_size)):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_file = batches_dir / f"size_batch_{timestamp}_{i+1:03d}.parquet"
            batch_df.to_parquet(batch_file, index=False)
            
        logger.info(f"[EnhancedAccumulator] Створено {i+1} батчів розміром {batch_size}")
    
    def create_colab_packages(self, package_sizes: Dict[str, int] = None) -> Dict[str, str]:
        """
        Створює пакети для Colab
        
        Args:
            package_sizes: {'light': 50, 'medium': 200, 'full': 1000} в MB
            
        Returns:
            Dict: шляхи до створених пакетів
        """
        if package_sizes is None:
            package_sizes = {'light': 50, 'medium': 200, 'full': 1000}
        
        logger.info("[EnhancedAccumulator] Створення пакетів для Colab...")
        
        packages = {}
        accumulated_file = self.enriched_dir / "stage2_accumulated.parquet"
        
        if not accumulated_file.exists():
            logger.warning("[EnhancedAccumulator] Немає накопичених data для пакування")
            return {}
        
        # Завантажуємо дані
        df = pd.read_parquet(accumulated_file)
        logger.info(f"[EnhancedAccumulator] Дані для пакування: {df.shape}")
        
        for package_name, max_size_mb in package_sizes.items():
            package_file = self._create_package(df, package_name, max_size_mb)
            if package_file:
                packages[package_name] = package_file
        
        logger.info(f"[EnhancedAccumulator] Створено {len(packages)} пакетів для Colab")
        return packages
    
    def _create_package(self, df: pd.DataFrame, package_name: str, 
                       max_size_mb: int) -> Optional[str]:
        """Створює один пакет"""
        try:
            # Розраховуємо кількість рядків для пакета
            row_size_mb = df.memory_usage(deep=True).sum() / (1024 * 1024) / len(df)
            max_rows = int(max_size_mb / row_size_mb)
            
            if len(df) <= max_rows:
                package_df = df
            else:
                # Беремо останні рядки
                package_df = df.tail(max_rows)
            
            # Створюємо тимчасову папку
            temp_dir = self.colab_packages_dir / f"temp_{package_name}"
            temp_dir.mkdir(exist_ok=True)
            
            # Зберігаємо дані
            data_file = temp_dir / "data.parquet"
            package_df.to_parquet(data_file, index=False)
            
            # Створюємо метадані
            metadata = {
                'package_name': package_name,
                'created_at': datetime.now().isoformat(),
                'shape': package_df.shape,
                'size_mb': max_size_mb,
                'columns': list(package_df.columns),
                'total_rows_original': len(df)
            }
            
            metadata_file = temp_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Створюємо README
            readme_content = f"""# Trading Data Package - {package_name.title()}

## Інформація
- Розмір: ~{max_size_mb}MB
- Рядків: {package_df.shape[0]}
- Колонок: {package_df.shape[1]}
- Створено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Використання в Colab
```python
import pandas as pd
import tarfile

# Розпаковка
with tarfile.open('package.tar.gz', 'r:gz') as tar:
    tar.extractall()

# Завантаження data
df = pd.read_parquet('data.parquet')
print(f"Дані: {df.shape}")
```

## Колонки
{', '.join(package_df.columns[:10])}{'...' if len(package_df.columns) > 10 else ''}
"""
            
            readme_file = temp_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Створюємо tar.gz архів
            archive_name = f"trading_data_{package_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
            archive_path = self.colab_packages_dir / archive_name
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_dir, arcname=archive_name.replace('.tar.gz', ''))
            
            # Видаляємо тимчасову папку
            import shutil
            shutil.rmtree(temp_dir)
            
            # Перевіряємо розмір
            actual_size_mb = archive_path.stat().st_size / (1024 * 1024)
            logger.info(f"[EnhancedAccumulator] Пакет {package_name}: {actual_size_mb:.1f}MB")
            
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"[EnhancedAccumulator] Помилка створення пакету {package_name}: {e}")
            return None
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Повертає статистику накопичення"""
        stats = {
            'stage1': {},
            'stage2': {},
            'packages': {}
        }
        
        # Stage 1 статистика
        stage1_dir = self.raw_dir / "stage1_latest"
        if stage1_dir.exists():
            files = list(stage1_dir.glob("*.parquet"))
            stats['stage1'] = {
                'files_count': len(files),
                'files': [f.name for f in files],
                'total_size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024)
            }
        
        # Stage 2 статистика
        accumulated_file = self.enriched_dir / "stage2_accumulated.parquet"
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            stats['stage2'] = {
                'accumulated_rows': len(df),
                'accumulated_shape': df.shape,
                'latest_file': (self.enriched_dir / "stage2_latest.parquet").exists(),
                'batch_count': len(list((self.enriched_dir / "daily_batches").glob("*.parquet")))
            }
        
        # Packages статистика
        packages = list(self.colab_packages_dir.glob("*.tar.gz"))
        stats['packages'] = {
            'count': len(packages),
            'files': [p.name for p in packages],
            'total_size_mb': sum(p.stat().st_size for p in packages) / (1024 * 1024)
        }
        
        return stats

# Глобальний екземпляр
enhanced_accumulator = EnhancedDataAccumulator()
