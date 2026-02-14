# stage3_accumulator.py - Накопичення data після етапу 3 (Features)

import os
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging
import tarfile
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class Stage3Accumulator:
    """
    Накопичення data після етапу 3 - Feature Engineering
    Це НАЙВАЖЛИВІШЕ для тренування моделей
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.setup_directories()
        
    def setup_directories(self):
        """Створює структуру папок для Stage 3"""
        self.features_dir = self.base_dir / "features"
        self.model_ready_dir = self.features_dir / "model_ready"
        self.colab_packages_dir = self.features_dir / "colab_packages"
        
        # Створюємо папки
        for dir_path in [self.features_dir, self.model_ready_dir, self.colab_packages_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"[Stage3Accumulator] Директорії Stage 3 created в {self.base_dir}")
    
    def accumulate_stage3_data(self, features_df: pd.DataFrame, 
                             context_df: Optional[pd.DataFrame] = None,
                             trigger_data: Optional[Any] = None,
                             create_model_ready: bool = True) -> Dict[str, str]:
        """
        Накопичує дані етапу 3 - фічі для тренування моделей
        
        Args:
            features_df: Основний DataFrame з фічами
            context_df: Контекстні дані
            trigger_data: Дані тригерів
            create_model_ready: Чи створювати дані готові для моделей
            
        Returns:
            Dict: шляхи до збережених файлів
        """
        logger.info(f"[Stage3Accumulator] Накопичення Stage 3 фіч: {features_df.shape}")
        
        if features_df.empty:
            logger.warning("[Stage3Accumulator] Features DataFrame порожній, пропускаємо")
            return {}
        
        saved_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. Зберігаємо останні фічі
        latest_file = self.features_dir / "stage3_latest.parquet"
        features_df.to_parquet(latest_file, index=False)
        saved_files['latest'] = str(latest_file)
        
        # 2. Накопичуємо до основного файлу
        accumulated_file = self.features_dir / "stage3_accumulated.parquet"
        
        if accumulated_file.exists():
            # Завантажуємо існуючі дані
            existing_df = pd.read_parquet(accumulated_file)
            logger.info(f"[Stage3Accumulator] Існуючі фічі: {existing_df.shape}")
            
            # Об'єднуємо дані
            combined_df = pd.concat([existing_df, features_df], ignore_index=True)
            
            # Видаляємо дублікати (важливо для фіч!)
            unique_cols = []
            for col in ['date', 'timestamp', 'trade_date']:
                if col in combined_df.columns:
                    unique_cols.append(col)
            
            if len(unique_cols) >= 1:
                before_dedup = len(combined_df)
                combined_df = combined_df.drop_duplicates(subset=unique_cols, keep='last')
                after_dedup = len(combined_df)
                logger.info(f"[Stage3Accumulator] Видалено дублікатів фіч: {before_dedup - after_dedup}")
        else:
            combined_df = features_df
        
        # Зберігаємо накопичені фічі
        combined_df.to_parquet(accumulated_file, index=False)
        saved_files['accumulated'] = str(accumulated_file)
        logger.info(f"[Stage3Accumulator] Накопичено фічі: {combined_df.shape}")
        
        # 3. Створюємо щоденний батч
        batch_file = self.features_dir / f"stage3_batch_{timestamp}.parquet"
        features_df.to_parquet(batch_file, index=False)
        saved_files['batch'] = str(batch_file)
        
        # 4. Зберігаємо додаткові дані
        if context_df is not None and not context_df.empty:
            context_file = self.features_dir / f"context_{timestamp}.parquet"
            context_df.to_parquet(context_file, index=False)
            saved_files['context'] = str(context_file)
        
        # 5. Створюємо дані готові для моделей
        if create_model_ready:
            model_files = self._create_model_ready_datasets(combined_df, timestamp)
            saved_files.update(model_files)
        
        # 6. Створюємо метадані
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'latest_shape': features_df.shape,
            'accumulated_shape': combined_df.shape,
            'feature_count': len(features_df.columns),
            'batch_count': len(list(self.features_dir.glob("stage3_batch_*.parquet"))),
            'total_accumulated_rows': len(combined_df),
            'has_context': context_df is not None,
            'model_ready_created': create_model_ready
        }
        
        metadata_path = self.features_dir / "accumulation_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"[Stage3Accumulator] Stage 3 накопичено: {len(saved_files)} файлів")
        return saved_files
    
    def _create_model_ready_datasets(self, df: pd.DataFrame, timestamp: str) -> Dict[str, str]:
        """
        Створює датасети готові для тренування моделей
        """
        logger.info("[Stage3Accumulator] Створення датасетів для моделей...")
        
        model_files = {}
        
        # 1. Визначаємо колонки фіч та таргетів
        feature_cols = []
        target_cols = []
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['target', 'label', 'y_']):
                target_cols.append(col)
            elif col not in ['date', 'timestamp', 'trade_date']:
                feature_cols.append(col)
        
        logger.info(f"[Stage3Accumulator] Знайдено фіч: {len(feature_cols)}, таргетів: {len(target_cols)}")
        
        if len(target_cols) == 0:
            logger.warning("[Stage3Accumulator] Не found таргетів, створюємо синтетичні...")
            # Створюємо синтетичні таргети на основі ціни
            for col in feature_cols:
                if 'close' in col.lower() or 'price' in col.lower():
                    target_col = f"target_{col}_1d"
                    if target_col not in df.columns:
                        df[target_col] = df[col].pct_change().fillna(0)
                        target_cols.append(target_col)
        
        # 2. Легкий датасет для простих моделей (Random Forest, XGBoost)
        if len(feature_cols) > 0:
            # Вибираємо найважливіші фічі
            light_features = feature_cols[:50]  # Перші 50 фіч
            light_targets = target_cols[:5]     # Перші 5 таргетів
            
            light_cols = ['date'] if 'date' in df.columns else []
            light_cols.extend(light_features)
            light_cols.extend(light_targets)
            
            light_df = df[light_cols].copy()
            light_file = self.model_ready_dir / f"light_features_{timestamp}.parquet"
            light_df.to_parquet(light_file, index=False)
            model_files['light_features'] = str(light_file)
            
            logger.info(f"[Stage3Accumulator] Легкий датасет: {light_df.shape}")
        
        # 3. Важкий датасет для складних моделей (LSTM, Transformer)
        heavy_cols = ['date'] if 'date' in df.columns else []
        heavy_cols.extend(feature_cols)  # Всі фічі
        heavy_cols.extend(target_cols)   # Всі таргети
        
        heavy_df = df[heavy_cols].copy()
        heavy_file = self.model_ready_dir / f"heavy_features_{timestamp}.parquet"
        heavy_df.to_parquet(heavy_file, index=False)
        model_files['heavy_features'] = str(heavy_file)
        
        logger.info(f"[Stage3Accumulator] Важкий датасет: {heavy_df.shape}")
        
        # 4. Окремо зберігаємо таргети
        if len(target_cols) > 0:
            target_df = df[['date'] + target_cols].copy() if 'date' in df.columns else df[target_cols].copy()
            target_file = self.model_ready_dir / f"targets_{timestamp}.parquet"
            target_df.to_parquet(target_file, index=False)
            model_files['targets'] = str(target_file)
            
            logger.info(f"[Stage3Accumulator] Таргети: {target_df.shape}")
        
        return model_files
    
    def create_colab_packages(self, package_configs: Dict[str, Dict] = None) -> Dict[str, str]:
        """
        Створює оптимізовані пакети для Colab
        
        Args:
            package_configs: Конфігурації пакетів
            
        Returns:
            Dict: шляхи до створених пакетів
        """
        if package_configs is None:
            package_configs = {
                'light': {
                    'max_size_mb': 50,
                    'features_count': 50,
                    'targets_count': 3,
                    'description': 'For quick testing and simple models'
                },
                'medium': {
                    'max_size_mb': 200,
                    'features_count': 200,
                    'targets_count': 10,
                    'description': 'For medium complexity models'
                },
                'full': {
                    'max_size_mb': 1000,
                    'features_count': None,  # Всі фічі
                    'targets_count': None,   # Всі таргети
                    'description': 'For complex models (LSTM, Transformer)'
                }
            }
        
        logger.info("[Stage3Accumulator] Створення пакетів для Colab...")
        
        packages = {}
        accumulated_file = self.features_dir / "stage3_accumulated.parquet"
        
        if not accumulated_file.exists():
            logger.warning("[Stage3Accumulator] Немає накопичених фіч для пакування")
            return {}
        
        # Завантажуємо дані
        df = pd.read_parquet(accumulated_file)
        logger.info(f"[Stage3Accumulator] Фічі для пакування: {df.shape}")
        
        for package_name, config in package_configs.items():
            package_file = self._create_colab_package(df, package_name, config)
            if package_file:
                packages[package_name] = package_file
        
        logger.info(f"[Stage3Accumulator] Створено {len(packages)} пакетів для Colab")
        return packages
    
    def _create_colab_package(self, df: pd.DataFrame, package_name: str, 
                           config: Dict) -> Optional[str]:
        """Створює один пакет для Colab"""
        try:
            # Вибираємо колонки
            feature_cols = []
            target_cols = []
            
            for col in df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['target', 'label', 'y_']):
                    target_cols.append(col)
                elif col not in ['date', 'timestamp', 'trade_date']:
                    feature_cols.append(col)
            
            # Обмежуємо кількість фіч та таргетів
            if config.get('features_count'):
                feature_cols = feature_cols[:config['features_count']]
            if config.get('targets_count'):
                target_cols = target_cols[:config['targets_count']]
            
            # Формуємо датасет
            cols = ['date'] if 'date' in df.columns else []
            cols.extend(feature_cols)
            cols.extend(target_cols)
            
            package_df = df[cols].copy()
            
            # Обмежуємо розмір
            max_size_mb = config['max_size_mb']
            row_size_mb = package_df.memory_usage(deep=True).sum() / (1024 * 1024) / len(package_df)
            max_rows = int(max_size_mb / row_size_mb)
            
            if len(package_df) > max_rows:
                package_df = package_df.tail(max_rows)
            
            # Створюємо тимчасову папку
            temp_dir = self.colab_packages_dir / f"temp_{package_name}"
            temp_dir.mkdir(exist_ok=True)
            
            # Зберігаємо дані
            data_file = temp_dir / "features.parquet"
            package_df.to_parquet(data_file, index=False)
            
            # Створюємо метадані
            metadata = {
                'package_name': package_name,
                'description': config['description'],
                'created_at': datetime.now().isoformat(),
                'shape': package_df.shape,
                'size_mb': max_size_mb,
                'features': feature_cols,
                'targets': target_cols,
                'total_rows_original': len(df)
            }
            
            metadata_file = temp_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Створюємо README
            readme_content = f"""# Trading Features Package - {package_name.title()}

## Опис
{config['description']}

## Інформація
- Розмір: ~{max_size_mb}MB
- Рядків: {package_df.shape[0]}
- Фіч: {len(feature_cols)}
- Таргетів: {len(target_cols)}
- Створено: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Використання в Colab
```python
import pandas as pd
import tarfile

# Розпаковка
with tarfile.open('package.tar.gz', 'r:gz') as tar:
    tar.extractall()

# Завантаження фіч
features_df = pd.read_parquet('features.parquet')
print(f"Фічі: {features_df.shape}")

# Розділення фіч та таргетів
feature_cols = {feature_cols[:10]}{'...' if len(feature_cols) > 10 else ''}
target_cols = {target_cols[:5]}{'...' if len(target_cols) > 5 else ''}

X = features_df[feature_cols]
y = features_df[target_cols]
```

## Моделі
- **Light**: Random Forest, XGBoost, LightGBM
- **Medium**: Gradient Boosting, Neural Networks
- **Full**: LSTM, Transformer, Deep Learning
"""
            
            readme_file = temp_dir / "README.md"
            with open(readme_file, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            
            # Створюємо tar.gz архів
            archive_name = f"stage3_features_{package_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
            archive_path = self.colab_packages_dir / archive_name
            
            with tarfile.open(archive_path, 'w:gz') as tar:
                tar.add(temp_dir, arcname=archive_name.replace('.tar.gz', ''))
            
            # Видаляємо тимчасову папку
            import shutil
            shutil.rmtree(temp_dir)
            
            # Перевіряємо розмір
            actual_size_mb = archive_path.stat().st_size / (1024 * 1024)
            logger.info(f"[Stage3Accumulator] Пакет {package_name}: {actual_size_mb:.1f}MB")
            
            return str(archive_path)
            
        except Exception as e:
            logger.error(f"[Stage3Accumulator] Помилка створення пакету {package_name}: {e}")
            return None
    
    def get_accumulation_stats(self) -> Dict[str, Any]:
        """Повертає статистику накопичення Stage 3"""
        stats = {
            'features': {},
            'model_ready': {},
            'packages': {}
        }
        
        # Features статистика
        accumulated_file = self.features_dir / "stage3_accumulated.parquet"
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            stats['features'] = {
                'accumulated_rows': len(df),
                'accumulated_shape': df.shape,
                'feature_columns': len([c for c in df.columns if 'target' not in c.lower()]),
                'target_columns': len([c for c in df.columns if 'target' in c.lower()]),
                'latest_file': (self.features_dir / "stage3_latest.parquet").exists(),
                'batch_count': len(list(self.features_dir.glob("stage3_batch_*.parquet")))
            }
        
        # Model ready статистика
        model_files = list(self.model_ready_dir.glob("*.parquet"))
        stats['model_ready'] = {
            'files_count': len(model_files),
            'files': [f.name for f in model_files],
            'total_size_mb': sum(f.stat().st_size for f in model_files) / (1024 * 1024)
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
stage3_accumulator = Stage3Accumulator()
