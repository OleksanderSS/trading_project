# simple_accumulator.py - Проста система накопичення після кожного етапу

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SimpleAccumulator:
    """
    Проста система накопичення - зберігаємо після кожного етапу
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Створює базові папки"""
        self.raw_dir = self.base_dir / "raw"
        self.enriched_dir = self.base_dir / "enriched" 
        self.features_dir = self.base_dir / "features"
        
        for dir_path in [self.raw_dir, self.enriched_dir, self.features_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"[SimpleAccumulator] Папки created в {self.base_dir}")
    
    def save_stage1(self, stage1_data: dict) -> dict:
        """Зберігає дані після етапу 1"""
        logger.info("[SimpleAccumulator] Збереження Stage 1...")
        
        saved_files = {}
        stage1_dir = self.raw_dir / "stage1_latest"
        stage1_dir.mkdir(exist_ok=True)
        
        for data_type, df in stage1_data.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                file_path = stage1_dir / f"{data_type}.parquet"
                df.to_parquet(file_path, index=False)
                saved_files[data_type] = str(file_path)
                logger.info(f"[SimpleAccumulator] Stage 1 {data_type}: {df.shape}")
        
        # Метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'data_types': list(saved_files.keys()),
            'stage': 1
        }
        
        with open(stage1_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[SimpleAccumulator] Stage 1 saved: {len(saved_files)} файлів")
        return saved_files
    
    def save_stage2(self, merged_df: pd.DataFrame) -> dict:
        """Зберігає та накопичує дані після етапу 2"""
        logger.info(f"[SimpleAccumulator] Збереження Stage 2: {merged_df.shape}")
        
        if merged_df.empty:
            logger.warning("[SimpleAccumulator] Stage 2 DataFrame порожній")
            return {}
        
        # 1. Зберігаємо останні дані
        latest_file = self.enriched_dir / "stage2_latest.parquet"
        merged_df.to_parquet(latest_file, index=False)
        
        # 2. Накопичуємо
        accumulated_file = self.enriched_dir / "stage2_accumulated.parquet"
        
        if accumulated_file.exists():
            existing_df = pd.read_parquet(accumulated_file)
            combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
            
            # Видаляємо дублікати
            if 'date' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        else:
            combined_df = merged_df
        
        combined_df.to_parquet(accumulated_file, index=False)
        
        # Метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'latest_shape': merged_df.shape,
            'accumulated_shape': combined_df.shape,
            'stage': 2
        }
        
        with open(self.enriched_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[SimpleAccumulator] Stage 2 накопичено: {combined_df.shape}")
        return {
            'latest': str(latest_file),
            'accumulated': str(accumulated_file)
        }
    
    def save_stage3(self, features_df: pd.DataFrame) -> dict:
        """Зберігає та накопичує фічі після етапу 3"""
        logger.info(f"[SimpleAccumulator] Збереження Stage 3: {features_df.shape}")
        
        if features_df.empty:
            logger.warning("[SimpleAccumulator] Stage 3 DataFrame порожній")
            return {}
        
        # 1. Зберігаємо останні фічі
        latest_file = self.features_dir / "stage3_latest.parquet"
        features_df.to_parquet(latest_file, index=False)
        
        # 2. Накопичуємо
        accumulated_file = self.features_dir / "stage3_accumulated.parquet"
        
        if accumulated_file.exists():
            existing_df = pd.read_parquet(accumulated_file)
            combined_df = pd.concat([existing_df, features_df], ignore_index=True)
            
            # Видаляємо дублікати
            if 'date' in combined_df.columns:
                combined_df = combined_df.drop_duplicates(subset=['date'], keep='last')
        else:
            combined_df = features_df
        
        combined_df.to_parquet(accumulated_file, index=False)
        
        # 3. Створюємо датасет для моделей
        model_ready_file = self.features_dir / "model_ready.parquet"
        combined_df.to_parquet(model_ready_file, index=False)
        
        # Метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'latest_shape': features_df.shape,
            'accumulated_shape': combined_df.shape,
            'feature_count': len(features_df.columns),
            'stage': 3
        }
        
        with open(self.features_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[SimpleAccumulator] Stage 3 накопичено: {combined_df.shape}")
        return {
            'latest': str(latest_file),
            'accumulated': str(accumulated_file),
            'model_ready': str(model_ready_file)
        }
    
    def get_status(self) -> dict:
        """Повертає статус збереження"""
        status = {
            'stage1': {},
            'stage2': {},
            'stage3': {}
        }
        
        # Stage 1
        stage1_dir = self.raw_dir / "stage1_latest"
        if stage1_dir.exists():
            files = list(stage1_dir.glob("*.parquet"))
            status['stage1'] = {
                'files_count': len(files),
                'files': [f.name for f in files]
            }
        
        # Stage 2
        accumulated_file = self.enriched_dir / "stage2_accumulated.parquet"
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            status['stage2'] = {
                'accumulated_rows': len(df),
                'shape': df.shape
            }
        
        # Stage 3
        accumulated_file = self.features_dir / "stage3_accumulated.parquet"
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            status['stage3'] = {
                'accumulated_rows': len(df),
                'shape': df.shape,
                'features': len(df.columns)
            }
        
        return status

# Глобальний екземпляр
simple_accumulator = SimpleAccumulator()
