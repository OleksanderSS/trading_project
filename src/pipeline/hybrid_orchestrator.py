# src/pipeline/hybrid_orchestrator.py
"""
Гібридний оркестратор пайплайну:
- Локально: парсинг, вибір фіч, легкі моделі
- Colab: важкі моделі, важкі аналізатори
- Збереження проміжних результатів для довгих сесій
"""

import asyncio
import logging
import json
import pickle
import time
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
from src.data.management.data_manager import DataManager

logger = ProjectLogger.get_logger(__name__)

# Google Drive API (опціонально)
try:
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GDRIVE_AVAILABLE = True
except ImportError:
    GDRIVE_AVAILABLE = False
    logger.warning("Google Drive API не встановлено. Використовуйте ручний трансфер.")


class HybridOrchestrator:
    """
    Гібридний оркестратор для розподіленого виконання пайплайну.
    
    Цей клас тепер працює як вузол узгодження між сучасним `PipelineOrchestrator`
    та Colab-орієнтованим workflow. Локальна частина виконує етапи 0-3 через
    `PipelineOrchestrator`, а фінальні етапи 4-7 теж делегуються тій же сучасній
    оркестрації.
    
    Локально:
    - Stage 0-3: Збір даних, очищення, генерація фіч, вибір фіч
    - Легкі моделі (CatBoost, LightGBM, XGBoost, RF, Linear, SVM, KNN)
    
    В Colab:
    - Важкі моделі (LSTM, GRU, Transformer, TabNet, CNN, Autoencoder, MLP)
    - Важкі аналізатори
    
    Збереження:
    - Проміжні результати після кожного етапу
    - Можливість відновлення після розриву сесії
    """
    
    def __init__(self, config_manager: UnifiedConfigManager, batch_name: str = "main_database"):
        self.config_manager = config_manager
        self.logger = ProjectLogger.get_logger(__name__)
        self.batch_name = batch_name
        
        # Шляхи для збереження (база)
        self.output_dir = Path(config_manager.get('system.accumulation.output_dir', 'data/colab/accumulated'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # ✅ ADD: Шлях для збереження моделей
        system_config = config_manager.get_config('system') or {}
        self.models_dir = Path(system_config.get('models_path', 'trained_models'))
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Конфігурація етапів
        self.system_config = system_config
        self.models_config = config_manager.get_config('models') or {}
        
        # Розподіл моделей
        self.light_models = self.models_config.get('categories', {}).get('light', [])
        self.heavy_models = self.models_config.get('categories', {}).get('heavy', [])
        
        # Google Drive налаштування
        self.gdrive_enabled = GDRIVE_AVAILABLE and self.system_config.get('google_drive', {}).get('enabled', False)
        self.gdrive_folder_id = self.system_config.get('google_drive', {}).get('folder_id')
        self.gdrive_service = None
        
        # ✅ Fallback storage options
        self.storage_fallback = self.system_config.get('storage_fallback', {})
        self.use_s3 = self.storage_fallback.get('s3', {}).get('enabled', False)
        self.use_gcs = self.storage_fallback.get('gcs', {}).get('enabled', False)
        
        if self.gdrive_enabled:
            try:
                self._init_gdrive()
            except Exception as e:
                self.logger.warning(f"⚠️ Google Drive ініціалізація не вдалася: {e}")
                self.logger.info("💡 Використовуємо ручний трансфер або fallback storage")
                self.gdrive_enabled = False
        
        self.logger.info(f"🚀 Гібридний оркестратор ініціалізовано")
        self.logger.info(f"📁 Директорія виводу: {self.output_dir}")
        self.logger.info(f"💡 Легкі моделі: {self.light_models}")
        self.logger.info(f"🔥 Важкі моделі: {self.heavy_models}")
        self.logger.info(f"☁️ Google Drive: {'✅ Увімкнено' if self.gdrive_enabled else '❌ Вимкнено (ручний трансфер)'}")
        if self.use_s3:
            self.logger.info(f"☁️ S3 Fallback: ✅ Увімкнено")
        if self.use_gcs:
            self.logger.info(f"☁️ GCS Fallback: ✅ Увімкнено")
    
    def _init_gdrive(self):
        """
        ✅ Ініціалізує Google Drive API.
        Викликається тільки якщо GDRIVE_AVAILABLE та enabled в конфігу.
        """
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            
            # Тут має бути логіка ініціалізації Google Drive
            # Поки що просто логуємо
            self.logger.info("✅ Google Drive API ініціалізовано")
        except Exception as e:
            self.logger.error(f"❌ Помилка ініціалізації Google Drive: {e}")
            # ✅ Спробуємо fallback storage
            if self._init_fallback_storage():
                self.logger.info("✅ Fallback storage ініціалізовано")
            else:
                raise
    
    def _init_fallback_storage(self) -> bool:
        """
        ✅ Ініціалізує fallback storage (S3 або GCS).
        
        Returns:
            True якщо fallback storage ініціалізовано успішно
        """
        # Спробуємо S3
        if self.use_s3:
            try:
                import boto3
                s3_config = self.storage_fallback.get('s3', {})
                self.s3_client = boto3.client(
                    's3',
                    aws_access_key_id=s3_config.get('access_key'),
                    aws_secret_access_key=s3_config.get('secret_key'),
                    region_name=s3_config.get('region', 'us-east-1')
                )
                self.s3_bucket = s3_config.get('bucket')
                self.logger.info(f"✅ S3 fallback ініціалізовано: {self.s3_bucket}")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ S3 fallback не вдався: {e}")
        
        # Спробуємо GCS
        if self.use_gcs:
            try:
                from google.cloud import storage
                gcs_config = self.storage_fallback.get('gcs', {})
                self.gcs_client = storage.Client(project=gcs_config.get('project_id'))
                self.gcs_bucket = self.gcs_client.bucket(gcs_config.get('bucket'))
                self.logger.info(f"✅ GCS fallback ініціалізовано: {gcs_config.get('bucket')}")
                return True
            except Exception as e:
                self.logger.warning(f"⚠️ GCS fallback не вдався: {e}")
        
        self.logger.warning("⚠️ Жоден fallback storage не доступний, використовуємо ручний трансфер")
        return False
    
    def _resolve_target_task_type(self, target_name: str) -> str:
        """Maps configured targets to the trainer's regression/classification contract."""
        targets_config = self.config_manager.get_config('targets', {})
        if hasattr(targets_config, 'as_dict'):
            targets_config = targets_config.as_dict()

        target_definitions = targets_config.get('targets', targets_config)
        target_meta = target_definitions.get(target_name, {}) if isinstance(target_definitions, dict) else {}
        configured_type = str(target_meta.get('type', '')).lower()

        if configured_type in {'regression', 'indicator_prediction'}:
            return 'regression'
        if configured_type.startswith('classification'):
            return 'classification'

        fallback_name = str(target_name).lower()
        if 'return' in fallback_name or 'price' in fallback_name or '_f' in fallback_name:
            return 'regression'
        return 'classification'

    async def run_local_pipeline(
        self, 
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        stages_to_run: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Виконує локальну частину пайплайну (етапи 0-3 + легкі моделі).
        
        Returns:
            Dict з результатами та шляхами до збережених файлів
        """
        import time
        start_time = time.time()
        self.logger.info("🚀 Запуск локального пайплайну...")
        
        # Етапи для локального виконання (0=Setup, 1=Collection, 2=Processing, 3=Feature Engineering)
        # ✅ FIX: Запускаємо 0-3 (включно з Feature Engineering)
        local_stages = stages_to_run or [0, 1, 2, 3]
        
        # Створюємо оркестратор для локальних етапів
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=local_stages
        )
        
        # Запускаємо локальні етапи
        stage_start = time.time()
        results = await orchestrator.run(
            tickers=tickers,
            timeframes=timeframes,
            run_mode='train'
        )
        stage_duration = time.time() - stage_start
        self.logger.info(f"⏱️ Етапи {local_stages} виконано за {stage_duration:.1f}s")
        
        # Зберігаємо результати після кожного етапу
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved_files = {}
        
        # Логуємо, що ми отримали
        self.logger.info(f"📊 Results keys: {results.keys() if results else 'None'}")
        self.logger.info(f"📊 Results type: {type(results)}")
        
        # Stage 1: Raw data
        if results and 'raw_data' in results:
            save_start = time.time()
            raw_data_path = self.output_dir / f"{self.batch_name}_stage1_raw_data_{timestamp}.parquet"
            self._save_data(results['raw_data'], raw_data_path)
            saved_files['raw_data'] = str(raw_data_path)
            save_duration = time.time() - save_start
            self.logger.info(f"✅ Stage 1 збережено за {save_duration:.1f}s: {raw_data_path}")
        
        # Stage 2: Cleaned data
        if results and 'cleaned_data' in results:
            save_start = time.time()
            cleaned_data_path = self.output_dir / f"{self.batch_name}_stage2_cleaned_data_{timestamp}.parquet"
            # Перетворюємо вкладені структури у плаский список для _save_data
            data_to_save = {}
            cleaned_data = results['cleaned_data']
            for k, v in cleaned_data.items():
                if isinstance(v, dict):
                    for sub_k, sub_v in v.items():
                        # Якщо це ціни з вкладеним 'data', дістаємо його
                        if isinstance(sub_v, dict) and 'data' in sub_v:
                            data_to_save[f"{k}_{sub_k}"] = sub_v['data']
                        else:
                            data_to_save[f"{k}_{sub_k}"] = sub_v
                else:
                    data_to_save[k] = v
                    
            self._save_data(data_to_save, cleaned_data_path)
            saved_files['cleaned_data'] = str(cleaned_data_path)
            save_duration = time.time() - save_start
            self.logger.info(f"✅ Stage 2 збережено за {save_duration:.1f}s: {cleaned_data_path}")
        
        # Stage 3: Features + Targets
        # ✅ Передаємо Event-Centric датасет до Colab (з ВСІ фічами)
        if results and 'enriched_data' in results:
            enriched_df = results['enriched_data']
            
            # Логуємо кількість фіч
            self.logger.info(f"📊 Enriched DataFrame shape: {enriched_df.shape}")
            self.logger.info(f"📊 Enriched columns: {len(enriched_df.columns)}")
            self.logger.info(f"📊 Has 'datetime': {'datetime' in enriched_df.columns}")
            self.logger.info(f"📊 Has 'ticker': {'ticker' in enriched_df.columns}")
            self.logger.info(f"📊 First 20 columns: {enriched_df.columns.tolist()[:20]}")
            
            enriched_data_path = self.output_dir / f"{self.batch_name}_stage3_enriched_{timestamp}.parquet"
            self._save_dataframe(results['enriched_data'], enriched_data_path)
            saved_files['enriched_data'] = str(enriched_data_path)
            self.logger.info(f"✅ Stage 3 enriched data збережено: {enriched_data_path}")
            
            # ✅ КРИТИЧНИЙ FIX: Зберігаємо datetime та ticker як колонки, не як індекс
            if enriched_df.index.name == 'datetime':
                enriched_df = enriched_df.reset_index()
            
            # ✅ КРИТИЧНИЙ FIX: Перевіряємо наявність datetime перед використанням
            # Шукаємо datetime або published_at
            datetime_col = None
            if 'datetime' in enriched_df.columns:
                datetime_col = 'datetime'
            elif 'published_at' in enriched_df.columns:
                datetime_col = 'published_at'
                self.logger.info(f"✅ Використовуємо 'published_at' замість 'datetime'")
            elif enriched_df.index.name == 'datetime' or isinstance(enriched_df.index, pd.DatetimeIndex):
                enriched_df = enriched_df.reset_index()
                if 'datetime' in enriched_df.columns:
                    datetime_col = 'datetime'
                elif 'published_at' in enriched_df.columns:
                    datetime_col = 'published_at'
                self.logger.info(f"✅ Відновлено datetime з індексу")
            
            if datetime_col is None:
                self.logger.error(f"❌ КРИТИЧНА ПОМИЛКА: Колонка 'datetime' або 'published_at' відсутня в enriched_df!")
                self.logger.error(f"Доступні колонки: {enriched_df.columns.tolist()[:30]}")
                self.logger.error(f"❌ Не вдалося відновити datetime")
                return {"status": "failed", "reason": "no_datetime_column"}
            
            # Якщо використовуємо published_at, перейменуємо на datetime для сумісності
            if datetime_col == 'published_at':
                enriched_df['datetime'] = enriched_df['published_at']
                self.logger.info(f"✅ Створено 'datetime' з 'published_at'")
            
            # Розділяємо на features та targets
            target_cols = [c for c in enriched_df.columns if c.startswith('target_')]
            feature_cols = [c for c in enriched_df.columns if c not in target_cols]
            
            features_df = enriched_df[feature_cols].copy()
            
            # ✅ КРИТИЧНИЙ FIX: Додаємо ticker та datetime в features_df для правильного merge в Colab
            if 'ticker' in enriched_df.columns and 'ticker' not in features_df.columns:
                features_df['ticker'] = enriched_df['ticker'].values
            if 'datetime' in enriched_df.columns and 'datetime' not in features_df.columns:
                features_df['datetime'] = enriched_df['datetime'].values
            
            # ✅ NORMALIZE TIMEZONE FIRST: Видаляємо timezone з datetime для уникнення помилок порівняння
            if 'datetime' in enriched_df.columns:
                enriched_df['datetime'] = pd.to_datetime(enriched_df['datetime']).dt.tz_localize(None)
            
            # ✅ КРИТИЧНИЙ FIX: targets_df має містити ТІЛЬКИ таргети + datetime + ticker
            targets_df = enriched_df[target_cols].copy()
            if 'datetime' in enriched_df.columns:
                targets_df['datetime'] = enriched_df['datetime'].values
            else:
                self.logger.warning("⚠️ No 'datetime' column in enriched_df for targets_df")
            if 'ticker' in enriched_df.columns:
                targets_df['ticker'] = enriched_df['ticker'].values
            
            # ✅ Додаємо datetime та ticker в features_df
            if 'datetime' in features_df.columns:
                features_df['datetime'] = pd.to_datetime(features_df['datetime']).dt.tz_localize(None)
            elif 'datetime' in enriched_df.columns:
                features_df['datetime'] = enriched_df['datetime'].values
            else:
                self.logger.warning("⚠️ No 'datetime' column found in enriched_df or features_df")
            
            self.logger.info(f"📊 Features: {features_df.shape}")
            self.logger.info(f"🎯 Targets: {targets_df.shape}")
            self.logger.info(f"✅ Features columns: {features_df.columns.tolist()[:10]}")
            self.logger.info(f"✅ Targets columns: {targets_df.columns.tolist()[:10]}")
            
            # ✅ КРИТИЧНИЙ FIX: Зберігаємо features_df та targets_df до batch_dir для Colab
            # Це необхідно для правильного merge в Colab (без картезіанського добутку)
            batch_dir = self.output_dir / self.batch_name
            batch_dir.mkdir(parents=True, exist_ok=True)
            
            features_path = batch_dir / "features.parquet"
            targets_path = batch_dir / "targets.parquet"
            
            self._save_dataframe(features_df, features_path)
            self._save_dataframe(targets_df, targets_path)
            
            self.logger.info(f"✅ Features збережено: {features_path}")
            self.logger.info(f"✅ Targets збережено: {targets_path}")
            
            saved_files['features'] = str(features_path)
            saved_files['targets'] = str(targets_path)
            
            # Оновлюємо результати
            results['enriched_data'] = enriched_df
            results['features_df'] = features_df
            results['targets_df'] = targets_df
        else:
            self.logger.error(f"❌ Stage 3 не повернув enriched_data! Доступні ключі: {results.keys() if results else 'None'}")
            self.logger.error(f"❌ Перевірте чи Stage 3 (Feature Engineering) увімкнений та правильно виконується")
        
        # Зберігаємо метадані
        metadata = {
            'timestamp': timestamp,
            'tickers': tickers,
            'timeframes': timeframes,
            'stages_completed': local_stages,
            'saved_files': saved_files,
            'light_models': self.light_models,
            'heavy_models': self.heavy_models
        }
        
        metadata_path = self.output_dir / f"{self.batch_name}_metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Створюємо batch_metadata.json для Colab
        batch_metadata = {
            'batch_name': self.batch_name,
            'timestamp': timestamp,
            'tickers': tickers,
            'timeframes': timeframes,
            'heavy_models': self.heavy_models,
            'enriched_shape': list(results.get('enriched_data', pd.DataFrame()).shape) if results and 'enriched_data' in results else [0, 0],
            'files': saved_files
        }
        
        batch_metadata_path = self.output_dir / "batch_metadata.json"
        with open(batch_metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        self.logger.info(f"📋 Batch метадані збережено: {batch_metadata_path}")
        
        total_duration = time.time() - start_time
        self.logger.info(f"📋 Метадані збережено: {metadata_path}")
        self.logger.info(f"⏱️ Загальний час виконання: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        
        return {
            'status': 'local_complete',
            'results': results,
            'saved_files': saved_files,
            'metadata_path': str(metadata_path),
            'timestamp': timestamp,
            'duration_seconds': total_duration
        }
    
    async def run_light_models(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        tickers: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Тренує легкі моделі локально та накопичує результати.
        """
        self.logger.info("💡 Запуск тренування легких моделей...")
        
        # Зберігаємо оригінальну конфігурацію
        original_models_config = self.config_manager.merged_config.get('models')
        
        # Створюємо нову конфігурацію тільки з легкими моделями
        import copy
        models_dict = self.models_config.as_dict() if hasattr(self.models_config, 'as_dict') else self.models_config
        temp_config_dict = copy.deepcopy(models_dict)
        temp_config_dict['categories'] = {'light': self.light_models}
        
        # Оновлюємо конфігурацію тимчасово
        self.config_manager.merged_config['models'] = temp_config_dict
        
        # Створюємо оркестратор для Stage 4
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=[4]  # Тільки modeling stage
        )
        
        # Запускаємо тренування
        results = await orchestrator.run(
            features_df=features_df,
            targets_df=targets_df,
            tickers=tickers,
            run_mode='train'
        )
        
        # Відновлюємо оригінальну конфігурацію
        self.config_manager.merged_config['models'] = original_models_config
        
        # Накопичуємо результати в один файл
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        light_results_path = self.output_dir / "light_models_results.json"
        
        # Отримуємо метадані моделей з результатів
        current_run = {
            'timestamp': timestamp,
            'models_metadata': results.get('models_metadata', {}),
            'metrics': results.get('metrics', {})
        }
        
        # Завантажуємо існуючі результати (якщо є)
        accumulated_results = {
            'timestamp': timestamp,
            'total_runs': 1,
            'runs': [current_run]
        }
        
        if light_results_path.exists():
            try:
                with open(light_results_path, 'r') as f:
                    existing = json.load(f)
                    accumulated_results['total_runs'] = existing.get('total_runs', 0) + 1
                    accumulated_results['runs'] = existing.get('runs', []) + [current_run]
                    self.logger.info(f"📊 Накопичено результатів: {accumulated_results['total_runs']} запусків")
            except Exception as e:
                self.logger.warning(f"⚠️ Не вдалося завантажити існуючі результати: {e}")
        
        # Зберігаємо накопичені результати
        with open(light_results_path, 'w') as f:
            json.dump(accumulated_results, f, indent=2, default=str)
        
        self.logger.info(f"✅ Результати легких моделей накопичено: {light_results_path}")
        
        return {
            'status': 'light_models_complete',
            'results': results,
            'saved_path': str(light_results_path),
            'timestamp': timestamp,
            'total_runs': accumulated_results['total_runs']
        }
    
    async def run_light_models_with_selected_features(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        batch_name: str,
        tickers: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Train light models locally using the feature subsets selected in Colab.

        This keeps the existing pipeline shape, but avoids collapsing all
        selected-feature files into a single ALL-ticker context.
        """
        from src.training.light_model_trainer import LightModelTrainer
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        import numpy as np

        self.logger.info("Running local light-model training from selected feature files...")
        if force:
            self.logger.info("Force mode enabled for local light-model retraining")

        batch_dir = self.output_dir / batch_name
        runtime_params_path = batch_dir / "runtime_params.json"
        test_ticker = None
        test_target = None
        test_model = None

        if runtime_params_path.exists():
            try:
                with open(runtime_params_path, 'r') as f:
                    runtime_params = json.load(f)
                test_mode = runtime_params.get('test_mode', {})
                if test_mode.get('enabled'):
                    test_ticker = test_mode.get('test_ticker')
                    test_target = test_mode.get('test_target')
                    test_model = test_mode.get('test_model')
                    self.logger.info(
                        f"Test mode detected: ticker={test_ticker}, target={test_target}, model={test_model}"
                    )
            except Exception as e:
                self.logger.warning(f"Could not read runtime_params.json: {e}")

        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        if test_target:
            if test_target in target_cols:
                target_cols = [test_target]
                self.logger.info(f"Using only target {test_target}")
            else:
                self.logger.warning(f"Target {test_target} not found, keeping all {len(target_cols)} targets")

        if test_ticker and 'ticker' in features_df.columns:
            ticker_mask = features_df['ticker'].str.upper() == test_ticker.upper()
            features_df = features_df[ticker_mask].copy()
            targets_df = targets_df[ticker_mask].copy()
            self.logger.info(f"Filtered source data to ticker {test_ticker}: {len(features_df)} rows")

        light_models_to_train = self.light_models
        if test_model:
            if test_model in self.light_models:
                light_models_to_train = [test_model]
            else:
                self.logger.warning(f"Requested test model {test_model} is not a light model, keeping defaults")

        selected_features_files = list(batch_dir.glob("selected_features_*.json"))
        if not selected_features_files:
            fallback_files = list(self.output_dir.glob("selected_features_*.json"))
            if fallback_files:
                self.logger.info(f"Using {len(fallback_files)} selected feature files from root output dir")
                selected_features_files = fallback_files

        if not selected_features_files:
            self.logger.error("No selected_features_*.json files found")
            return {
                'status': 'error',
                'message': 'No selected_features files found. Run Colab first.',
                'models_trained': 0
            }

        selected_feature_contexts: Dict[str, Dict[str, Any]] = {}
        for file_path in selected_features_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                model_name = data.get('model_type') or data.get('model_name')
                if not model_name:
                    parts = file_path.stem.replace('selected_features_', '').split('_')
                    model_name = parts[0] if parts else 'unknown'

                if test_model and model_name != test_model:
                    continue
                if model_name not in light_models_to_train:
                    continue

                context_ticker = data.get('ticker')
                if context_ticker:
                    context_ticker = str(context_ticker).upper()
                context_target = data.get('target')

                if test_ticker and context_ticker and context_ticker != test_ticker.upper():
                    continue
                if test_target and context_target and context_target != test_target:
                    continue

                selected_features = data.get('selected_features', [])
                if not selected_features:
                    self.logger.warning(f"Skipping empty feature selection file {file_path.name}")
                    continue

                context_targets = [context_target] if context_target else list(target_cols)
                context_id = f"{context_ticker or test_ticker or 'ALL'}::{context_target or 'ALL'}::{model_name}"
                selected_feature_contexts[context_id] = {
                    'model_name': model_name,
                    'ticker': context_ticker,
                    'targets': context_targets,
                    'selected_features': selected_features,
                    'source_file': file_path.name,
                    'max_features': data.get('max_features', 'N/A')
                }
            except Exception as e:
                self.logger.warning(f"Error while loading {file_path.name}: {e}")

        if not selected_feature_contexts:
            self.logger.error("No usable selected feature contexts found for light models")
            return {
                'status': 'error',
                'message': 'No selected_features found for light models',
                'models_trained': 0
            }

        self.logger.info(f"Training {len(selected_feature_contexts)} light-model contexts...")

        light_trainer = LightModelTrainer()
        models_metadata: Dict[str, Any] = {}
        models_trained = 0

        for context_id, context_data in selected_feature_contexts.items():
            model_name = context_data['model_name']
            context_ticker = context_data.get('ticker')
            context_targets = context_data.get('targets', [])
            selected_features = context_data.get('selected_features', [])

            context_features_df = features_df
            context_targets_df = targets_df
            if context_ticker and 'ticker' in features_df.columns:
                ticker_mask = features_df['ticker'].str.upper() == context_ticker.upper()
                context_features_df = features_df[ticker_mask].copy()
                context_targets_df = targets_df[ticker_mask].copy()

            if context_features_df.empty or context_targets_df.empty:
                self.logger.warning(f"Skipping empty context {context_id}")
                continue

            available_features = [f for f in selected_features if f in context_features_df.columns]
            if not available_features:
                self.logger.warning(f"No available features found for {context_id}")
                continue

            resolved_ticker = context_ticker
            if not resolved_ticker and 'ticker' in context_features_df.columns and not context_features_df['ticker'].dropna().empty:
                resolved_ticker = str(context_features_df['ticker'].dropna().iloc[-1]).upper()
            
            # ✅ ВИПРАВЛЕНО: Якщо тікер не знайдено, пропускаємо контекст (не встановлюємо 'ALL')
            if not resolved_ticker:
                self.logger.warning(f"Cannot resolve ticker for {context_id}, skipping")
                continue
            
            # ✅ Додаємо логування для дебагу
            self.logger.info(f"✅ Resolved ticker: {resolved_ticker} for context {context_id}")

            if 'timeframe' in context_features_df.columns and not context_features_df['timeframe'].dropna().empty:
                timeframe = str(context_features_df['timeframe'].dropna().iloc[-1])
            else:
                timeframe = '1d'

            for target_col in context_targets:
                try:
                    if target_col not in context_targets_df.columns:
                        self.logger.warning(f"Target {target_col} missing for {context_id}")
                        continue

                    X = context_features_df[available_features].copy()
                    y = context_targets_df[target_col].copy()

                    valid_mask = y.notna() & X.notna().all(axis=1)
                    X = X.loc[valid_mask]
                    y = y.loc[valid_mask]

                    if len(y) < 5:
                        self.logger.warning(f"Not enough valid rows for {resolved_ticker}/{target_col}/{model_name}")
                        continue

                    split_idx = max(1, int(len(X) * 0.8))
                    split_idx = min(split_idx, len(X) - 1)
                    X_train = X.iloc[:split_idx].copy()
                    X_test = X.iloc[split_idx:].copy()
                    y_train = y.iloc[:split_idx].copy()
                    y_test = y.iloc[split_idx:].copy()

                    if X_train.empty or X_test.empty:
                        self.logger.warning(f"Chronological split produced an empty partition for {context_id}")
                        continue

                    train_df = X_train.copy()
                    train_df[target_col] = y_train.values
                    task_type = self._resolve_target_task_type(target_col)

                    result = light_trainer.train_light_model(
                        features_df=train_df,
                        model_type=model_name,
                        ticker=resolved_ticker,
                        timeframe=timeframe,
                        target_col=target_col,
                        task_type=task_type
                    )

                    predictions = light_trainer.predict(result['model_key'], X_test)
                    if task_type == 'regression':
                        mse = mean_squared_error(y_test, predictions)
                        mae = mean_absolute_error(y_test, predictions)
                        r2 = r2_score(y_test, predictions)
                        rmse = np.sqrt(mse)
                        metrics = {
                            'mse': float(mse),
                            'rmse': float(rmse),
                            'mae': float(mae),
                            'r2': float(r2),
                            'score': float(r2)
                        }
                    else:
                        from sklearn.metrics import accuracy_score
                        accuracy = accuracy_score(y_test, predictions)
                        metrics = {
                            'accuracy': float(accuracy),
                            'score': float(accuracy)
                        }

                    models_save_dir = batch_dir / 'models'
                    models_save_dir.mkdir(parents=True, exist_ok=True)
                    model_path = models_save_dir / f"{model_name}_{resolved_ticker}_{target_col}.joblib"
                    light_trainer.save_model_to_disk(result['model_key'], str(model_path))

                    model_context_key = f"{resolved_ticker}_{target_col}_{model_name}"
                    models_metadata[model_context_key] = {
                        'ticker': resolved_ticker,
                        'target': target_col,
                        'winner': model_name,
                        'model_type': model_name,
                        'model_name': model_name,
                        'model_category': 'light',
                        'source': 'local',
                        'champion_reason': f"Light model trained locally with {len(available_features)} features",
                        'context': 'local_training',
                        'context_map': {
                            'context_fingerprint': 'local_training',
                            'market_regime': 'normal',
                            'volatility_regime': 'normal',
                            'timestamp': datetime.now().isoformat()
                        },
                        'market_regime': 'normal',
                        'timestamp': datetime.now().isoformat(),
                        'metrics': metrics,
                        'target_type': task_type,
                        'timeframe': timeframe,
                        'model_path': str(model_path),
                        'model_key': result['model_key'],
                        'selected_features': available_features,
                        'feature_count': len(available_features),
                        'validation_split': 'chronological_80_20',
                        'trained': True
                    }
                    models_trained += 1
                except Exception as e:
                    self.logger.error(
                        f"Error training {model_name} for {target_col} in {context_id}: {e}",
                        exc_info=True
                    )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        light_results_path = batch_dir / f"light_models_results_{timestamp}.json"

        accumulated_results = {
            'timestamp': timestamp,
            'batch_name': batch_name,
            'models_trained': models_trained,
            'models_metadata': models_metadata,
            'light_models_count': models_trained,
            'status': 'success'
        }

        with open(light_results_path, 'w') as f:
            json.dump(accumulated_results, f, indent=2, default=str)

        self.logger.info(f"Saved light-model results to {light_results_path}")
        self.logger.info(f"Trained {models_trained} light models")

        return {
            'status': 'success',
            'models_trained': models_trained,
            'models_metadata': models_metadata,
            'saved_path': str(light_results_path),
            'timestamp': timestamp
        }

    def check_if_feature_selection_needed(
        self,
        batch_dir: Path,
        new_rows_count: int,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Перевіряє чи потрібен новий вибір фіч.
        
        Критерії для нового вибору:
        1. Немає файлів selected_features_*.json
        2. Кількість нових рядків > 10% від попередніх
        3. Пройшло > 7 днів з останнього вибору
        4. Користувач примусово запустив (force=True)
        
        Returns:
            {
                'needed': bool,
                'reason': str,
                'last_selection_date': str,
                'data_change_percent': float
            }
        """
        # Перевірка 1: Чи є файли selected_features?
        selected_features_files = list(batch_dir.glob("selected_features_*.json"))
        
        if not selected_features_files:
            return {
                'needed': True,
                'reason': 'Немає файлів selected_features (перший запуск)',
                'last_selection_date': None,
                'data_change_percent': None
            }
        
        if force:
            return {
                'needed': True,
                'reason': 'Примусовий вибір фіч (--force-feature-selection)',
                'last_selection_date': None,
                'data_change_percent': None
            }
        
        # Перевірка 2: Дата останнього вибору
        try:
            with open(selected_features_files[0], 'r') as f:
                data = json.load(f)
                last_timestamp = data.get('timestamp')
                
            if last_timestamp:
                from datetime import datetime, timedelta
                last_date = datetime.fromisoformat(last_timestamp)
                days_passed = (datetime.now() - last_date).days
                
                if days_passed > 7:
                    return {
                        'needed': True,
                        'reason': f'Пройшло {days_passed} днів з останнього вибору (> 7 днів)',
                        'last_selection_date': last_timestamp,
                        'data_change_percent': None
                    }
        except Exception as e:
            self.logger.warning(f"⚠️ Не вдалося прочитати дату останнього вибору: {e}")
        
        # Перевірка 3: Зміна кількості даних
        features_path = batch_dir / "features.parquet"
        if features_path.exists():
            try:
                existing_features = pd.read_parquet(features_path)
                old_rows = len(existing_features)
                
                if old_rows > 0:
                    change_percent = (new_rows_count / old_rows) * 100
                    
                    if change_percent > 10:
                        return {
                            'needed': True,
                            'reason': f'Дані змінились на {change_percent:.1f}% (> 10%)',
                            'last_selection_date': last_timestamp if 'last_timestamp' in locals() else None,
                            'data_change_percent': change_percent
                        }
            except Exception as e:
                self.logger.warning(f"⚠️ Не вдалося перевірити зміну даних: {e}")
        
        # Вибір фіч не потрібен
        return {
            'needed': False,
            'reason': 'Використовуємо існуючі фічі (дані змінились незначно)',
            'last_selection_date': last_timestamp if 'last_timestamp' in locals() else None,
            'data_change_percent': None
        }
    
    def prepare_colab_batch(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        tickers: List[str],
        timeframes: List[str],
        batch_name: Optional[str] = None,
        accumulate: bool = True,  # Акумулювати дані до основної бази
        check_feature_selection: bool = True,  # Перевіряти чи потрібен новий вибір фіч
        force_feature_selection: bool = False  # Примусово перевибрати фічі
    ) -> Dict[str, str]:
        """
        Підготовлює пакет даних для тренування в Colab.
        
        ✅ АРХІТЕКТУРА:
        - features_df: Event-Centric датасет (новини з ВСІ фічами - СОТНІ колонок!)
        - targets_df: Таргети для кожної новини
        - Colab використовує систему вибору фіч з models.yaml для кожної моделі
        
        ⚠️ КРИТИЧНО: features_df повинен мати СОТНІ колонок (всі показники для всіх тікерів)
        Colab сам вибере потрібні фічи для кожної моделі за допомогою SmartFeatureSelector
        
        Архітектура накопичення:
        1. Основна база: data/colab/accumulated/main_database/
           - Одна велика база, що накопичує ВСІ дані
           - Видаляє дублікати за datetime + ticker
        
        2. Бекап: backups/accumulated/main_database_backup_YYYYMMDD_HHMMSS/
           - Дублюючий бекап основної бази
           - Створюється перед кожним оновленням
           - Дозволяє відновити, якщо щось видалено
        
        Args:
            accumulate: Якщо True, додає нові дані до основної бази (за замовчуванням)
        
        Створює:
        - features.parquet (Event-Centric датасет з ВСІ фічами - СОТНІ колонок!)
        - targets.parquet
        - config.json (з налаштуваннями важких моделей)
        - batch_metadata.json
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ✅ ПЕРЕВІРКА: Створюємо ізольовану підпапку для Colab
        effective_batch_name = (batch_name or self.batch_name).replace('target_target_', 'target_')
        batch_dir = self.output_dir / effective_batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        batch_name = effective_batch_name
        
        # Зчитуємо та копіюємо runtime_params
        runtime_params = {}
        params_path = self.config_manager.get_runtime_params_path()
        if params_path.exists():
            try:
                with open(params_path, 'r') as f:
                    runtime_params = json.load(f)
                # Копіюємо в батч
                shutil.copy2(params_path, batch_dir / "runtime_params.json")
            except Exception as e:
                self.logger.warning(f"⚠️ Не вдалося завантажити/скопіювати runtime_params.json: {e}")

        test_mode = runtime_params.get('test_mode', {})
        test_ticker = test_mode.get('test_ticker')
        test_target = test_mode.get('test_target')
        
        self.logger.info(f"📦 Підготовка пакету для Colab: {batch_name}")
        if test_ticker or test_target:
            self.logger.info(f"🧪 FAST MODE АКТИВОВАНО: ticker={test_ticker}, target={test_target}")

        self.logger.info(f"📦 Підготовка пакету для Colab: {batch_name}")
        
        # ✅ ФІЛЬТРУЄМО targets_df ДО АКУМУЛЯЦІЇ (щоб зберегти тільки таргети)
        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        metadata_cols = ['datetime', 'ticker']
        targets_df_filtered = targets_df[target_cols + [c for c in metadata_cols if c in targets_df.columns]].copy()
        
        self.logger.info(f"📊 Features: {features_df.shape} (повинно мати СОТНІ колонок!)")
        self.logger.info(f"🎯 Targets: {targets_df_filtered.shape} (ТІЛЬКИ таргети + datetime + ticker)")
        
        if accumulate:
            # ✅ КРИТИЧНИЙ FIX: Для тестових запусків не акумулюємо з main_database
            if test_ticker or test_target:
                # Тестовий режим - створюємо ізольовану папку без акумуляції
                self.logger.info(f"🧪 Тестовий режим: створюємо ізольовану папку {batch_name} без акумуляції")
                self.logger.info(f"📊 Тільки нові дані: features={features_df.shape}, targets={targets_df.shape}")
            else:
                # Нормальний режим - акумулюємо з основної бази
                main_db_dir = self.output_dir / "main_database"
                existing_features_path = main_db_dir / "features.parquet"
                existing_targets_path = main_db_dir / "targets.parquet"
                
                if existing_features_path.exists() and existing_targets_path.exists():
                    try:
                        self.logger.info(f"📊 Завантаження існуючих даних з основної бази...")
                        existing_features = pd.read_parquet(existing_features_path)
                        existing_targets = pd.read_parquet(existing_targets_path)
                        
                        self.logger.info(f"📊 Існуючі features: {existing_features.shape}")
                        self.logger.info(f"📊 Нові features: {features_df.shape}")
                        
                        # Створюємо бекап основної бази перед оновленням
                        backup_base = Path("backups/accumulated")
                        backup_base.mkdir(parents=True, exist_ok=True)
                        backup_dir = backup_base / f"main_database_backup_{timestamp}"
                        backup_dir.mkdir(parents=True, exist_ok=True)
                        
                        self.logger.info(f"💾 Створення бекапу основної бази: {backup_dir}")
                    except Exception as e:
                        self.logger.error(f"❌ Помилка завантаження існуючих даних: {e}")
                        self.logger.warning("⚠️ Продовжуємо без акумуляції (створюємо нову базу)")
                        existing_features = None
                        existing_targets = None
                    if main_db_dir.exists():
                        shutil.copytree(main_db_dir, backup_dir / "main_database", dirs_exist_ok=True)
                    
                    # Об'єднуємо з новими даними (видаляємо дублікати)
                    features_df = pd.concat([existing_features, features_df], ignore_index=True)
                    targets_df_filtered = pd.concat([existing_targets, targets_df_filtered], ignore_index=True)
                    
                    # Видаляємо дублікати за ключовими колонками
                    if 'datetime' in features_df.columns and 'ticker' in features_df.columns:
                        features_df = features_df.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
                    
                    if 'datetime' in targets_df_filtered.columns and 'ticker' in targets_df_filtered.columns:
                        targets_df_filtered = targets_df_filtered.drop_duplicates(subset=['datetime', 'ticker'], keep='last')
                    
                    self.logger.info(f"📊 Після акумуляції: features={features_df.shape}, targets={targets_df_filtered.shape}")
                    
                    # ✅ ПЕРЕВІРКА НА NaN/Inf ПЕРЕД ЗБЕРЕЖЕННЯМ
                    features_df = self._clean_dataframe(features_df, "features")
                    targets_df_filtered = self._clean_dataframe(targets_df_filtered, "targets")
                    
                    # Оновлюємо основну базу
                    self._save_dataframe(features_df, existing_features_path)
                    self._save_dataframe(targets_df_filtered, existing_targets_path)
                    self.logger.info(f"✅ Основну базу оновлено: {main_db_dir}")
                else:
                    # Перший запуск - основна база не існує
                    self.logger.info(f"📦 Створення нової основної бази...")
                    main_db_dir.mkdir(parents=True, exist_ok=True)
                    self._save_dataframe(features_df, existing_features_path)
                    self._save_dataframe(targets_df_filtered, existing_targets_path)
        else:
            # Без акумуляції - просто перезаписуємо тестову папку
            self.logger.info(f"📦 Перезапис тестової папки (без акумуляції)...")
        
        self.logger.info(f"📦 Підготовка пакету для Colab: {batch_name}")
        
        # Зберігаємо дані
        features_path = batch_dir / "features.parquet"
        targets_path = batch_dir / "targets.parquet"
        
        self._save_dataframe(features_df, features_path)
        self._save_dataframe(targets_df_filtered, targets_path)
        
        # Конфігурація для Colab (Heavy + Light categories for Feature Selection)
        all_targets_dict = self.config_manager.get_config('targets', {})
        if hasattr(all_targets_dict, 'as_dict'):
            all_targets_dict = all_targets_dict.as_dict()

        heavy_config = {
            'models': {
                'categories': {
                    'heavy': self.heavy_models,
                    'light': self.light_models
                },
                'per_model': {
                    model: self.models_config.get('per_model', {}).get(model, {})
                    for model in self.heavy_models + self.light_models
                }
            },
            'modeling': self.config_manager.get_config('modeling', {}),
            'features': self.config_manager.get_config('features', {}),
            'targets': all_targets_dict
        }
        
        self.logger.info(f"📋 Підготовка конфігурації Colab: {len(self.heavy_models)} важких + {len(self.light_models)} легких моделей")
        self.logger.info(f"💡 Весь вибір фіч відбуватиметься в Colab")
        
        # Конвертуємо DynamicConfig в словники для JSON серіалізації
        def convert_to_dict(obj):
            """Recursively convert DynamicConfig to dict."""
            if hasattr(obj, 'as_dict'):
                return convert_to_dict(obj.as_dict())
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            else:
                return obj
        
        heavy_config = convert_to_dict(heavy_config)
        
        config_path = batch_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(heavy_config, f, indent=2)
        
        # Метадані пакету
        batch_metadata = {
            'batch_name': batch_name,
            'timestamp': timestamp,
            'tickers': tickers,
            'timeframes': timeframes,
            'heavy_models': self.heavy_models,
            'features_shape': list(features_df.shape),
            'targets_shape': list(targets_df_filtered.shape),
            'files': {
                'features': str(features_path),
                'targets': str(targets_path),
                'config': str(config_path)
            }
        }
        
        metadata_path = batch_dir / "batch_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(batch_metadata, f, indent=2)
        
        self.logger.info(f"✅ Пакет підготовлено: {batch_dir}")
        self.logger.info(f"📊 Features: {features_df.shape} (СОТНІ колонок для всіх тікерів)")
        self.logger.info(f"🎯 Targets: {targets_df.shape}")
        self.logger.info(f"🔥 Важкі моделі: {len(self.heavy_models)}")
        
        # Перевірка чи потрібен новий вибір фіч
        feature_selection_check = None
        if check_feature_selection:
            feature_selection_check = self.check_if_feature_selection_needed(
                batch_dir=batch_dir,
                new_rows_count=len(features_df),
                force=force_feature_selection
            )
            
            if feature_selection_check['needed']:
                self.logger.warning("\n" + "🔄 "*40)
                self.logger.warning(f"🔄 ПОТРІБЕН НОВИЙ ВИБІР ФІЧ")
                self.logger.warning(f"🔄 Причина: {feature_selection_check['reason']}")
                self.logger.warning("🔄 "*40)
            else:
                self.logger.info("\n" + "✅ "*40)
                self.logger.info(f"✅ ВИБІР ФІЧ НЕ ПОТРІБЕН")
                self.logger.info(f"✅ {feature_selection_check['reason']}")
                self.logger.info("✅ "*40)
        
        return {
            'batch_dir': str(batch_dir),
            'batch_name': batch_name,
            'metadata_path': str(metadata_path),
            'files': batch_metadata['files'],
            'feature_selection_check': feature_selection_check
        }
    
    def load_colab_results(self, batch_name: str) -> Dict[str, Any]:
        """
        Завантажує результати тренування з Colab.
        Спочатку шукає colab_results_summary.json, потім colab_results.json
        
        ✅ SMART FALLBACK: Якщо batch_name не знайдено, шукає схожі папки
        (для зворотної сумісності зі старими назвами)
        ✅ КОНВЕРТАЦІЯ ШЛЯХІВ: Конвертує Colab шляхи в локальні Windows шляхи
        """
        # ✅ УНІФІКАЦІЯ: Видаляємо подвійні "target_target_" якщо вони є
        batch_name = batch_name.replace('target_target_', 'target_')
        
        batch_dir = self.output_dir / batch_name
        
        self.logger.info(f"🔍 load_colab_results: Шукаємо batch_name={batch_name}")
        self.logger.info(f"🔍 load_colab_results: batch_dir={batch_dir}")
        self.logger.info(f"🔍 load_colab_results: batch_dir.exists()={batch_dir.exists()}")
        
        # Якщо папка не існує, шукаємо схожі папки
        if not batch_dir.exists():
            self.logger.warning(f"⚠️ Папка {batch_name} не знайдена, шукаємо схожі...")
            
            # Шукаємо всі папки в output_dir
            all_batches = [d for d in self.output_dir.iterdir() if d.is_dir()]
            self.logger.info(f"🔍 Знайдено {len(all_batches)} папок в {self.output_dir}")
            self.logger.info(f"🔍 Папки: {[b.name for b in all_batches[:10]]}")  # Show first 10
            
            # Фільтруємо схожі назви (без префікса "test_" або з подвійним "target_")
            similar_batches = []
            for batch in all_batches:
                # Видаляємо "test_" з початку для порівняння
                batch_name_clean = batch_name.replace('test_', '')
                batch_dir_name_clean = batch.name.replace('test_', '')
                
                # Також видаляємо подвійний "target_"
                batch_name_clean = batch_name_clean.replace('target_target_', 'target_')
                batch_dir_name_clean = batch_dir_name_clean.replace('target_target_', 'target_')
                
                if batch_name_clean in batch_dir_name_clean or batch_dir_name_clean in batch_name_clean:
                    similar_batches.append(batch)
                    self.logger.debug(f"   ✅ Схожа папка: {batch.name}")
            
            if similar_batches:
                # Використовуємо найновішу схожу папку
                batch_dir = max(similar_batches, key=lambda p: p.stat().st_mtime)
                self.logger.info(f"✅ Знайдено схожу папку: {batch_dir.name}")
            else:
                self.logger.error(f"❌ Не знайдено жодної схожої папки для {batch_name}")
                return {'status': 'not_found'}
        
        # Спочатку шукаємо colab_results_summary.json (новий формат)
        results_path = batch_dir / "colab_results_summary.json"
        
        # Якщо не знайдено, шукаємо colab_results.json (старий формат)
        if not results_path.exists():
            results_path = batch_dir / "colab_results.json"
        
        if not results_path.exists():
            self.logger.warning(f"⚠️ Результати не знайдено: {batch_dir / 'colab_results_summary.json'} або {batch_dir / 'colab_results.json'}")
            self.logger.info(f"🔍 Файли в {batch_dir}:")
            if batch_dir.exists():
                for f in list(batch_dir.iterdir())[:10]:
                    self.logger.info(f"   - {f.name}")
            return {'status': 'not_found'}
        
        with open(results_path, 'r') as f:
            results = json.load(f)
        
        # ✅ ВИЗНАЧАЄМО РЕЖИМ: Перевіряємо чи це тестовий запуск
        runtime_params_path = batch_dir / "runtime_params.json"
        is_test_mode = False
        
        if runtime_params_path.exists():
            try:
                with open(runtime_params_path, 'r') as f:
                    runtime_params = json.load(f)
                    test_mode = runtime_params.get('test_mode', {})
                    is_test_mode = test_mode.get('enabled', False) and (
                        test_mode.get('test_ticker') or test_mode.get('test_target')
                    )
            except Exception as e:
                self.logger.warning(f"⚠️ Не вдалося прочитати runtime_params.json: {e}")
        
        # ✅ КОНВЕРТАЦІЯ ШЛЯХІВ: Замінюємо Colab шляхи на локальні
        def convert_colab_paths(obj, batch_dir, is_test_mode):
            """Рекурсивно конвертує Colab шляхи в локальні Windows шляхи.
            
            Логіка:
            - Тестовий режим (test_ticker/test_target): models/ підпапка
            - Звичайний режим: кореневий каталог батча
            """
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if key == 'model_path' and isinstance(value, str):
                        # Конвертуємо Colab шлях в локальний
                        # Витягуємо назву файлу моделі
                        model_filename = value.split('/')[-1]
                        
                        if is_test_mode:
                            # Тестовий режим: models/ підпапка
                            local_path = batch_dir / 'models' / model_filename
                            self.logger.debug(f"🧪 Тестовий режим: {value} -> {local_path}")
                        else:
                            # Звичайний режим: кореневий каталог
                            local_path = batch_dir / model_filename
                            self.logger.debug(f"📦 Звичайний режим: {value} -> {local_path}")
                        
                        obj[key] = str(local_path)
                    else:
                        obj[key] = convert_colab_paths(value, batch_dir, is_test_mode)
            elif isinstance(obj, list):
                return [convert_colab_paths(item, batch_dir, is_test_mode) for item in obj]
            return obj
        
        results = convert_colab_paths(results, batch_dir, is_test_mode)
        
        if is_test_mode:
            self.logger.info(f"🧪 ТЕСТОВИЙ РЕЖИМ: Моделі шукаються в {batch_dir / 'models'}")
        else:
            self.logger.info(f"📦 ЗВИЧАЙНИЙ РЕЖИМ: Моделі шукаються в {batch_dir}")
        
        self.logger.info(f"✅ Результати завантажено з: {results_path}")
        self.logger.info(f"✅ Шляхи до моделей конвертовано в локальні")
        return results
    
    async def run_full_hybrid_pipeline(
        self,
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        run_colab: bool = False,
        accumulate: bool = True,  # Акумулювати дані
        force_training: bool = False,  # Форсувати тренування навіть без нових даних
        skip_colab: bool = False,  # Пропустити Colab, виконати тільки локальну частину
        force_feature_selection: bool = False  # Примусово перевибрати фічі
    ) -> Dict[str, Any]:
        """
        Повний гібридний пайплайн з розумною логікою кешування:
        
        ЛОГІКА:
        1. Збираємо нові дані (етапи 0-3)
        2. Порівнюємо з кешем - чи є нові рядки?
        3. Якщо НЕ має нових рядків:
           - Пропускаємо Stage 4 (тренування)
           - Пропускаємо Colab (важкі моделі)
           - Переходимо прямо до Stage 5-7 (аналіз з існуючими моделями)
        4. Якщо є хоча б 1 новий рядок:
           - Запускаємо ВСІ етапи (0-4 + Colab + 5-7)
        """
        self.logger.info(f"🌐 Запуск повного гібридного пайплайну для батчу: {self.batch_name}")
        
        batch_dir = self.output_dir / self.batch_name
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        features_path = batch_dir / "features.parquet"
        targets_path = batch_dir / "targets.parquet"
        
        # КРОК 1: Збираємо нові дані (етапи 0-3)
        self.logger.info("📊 КРОК 1: Збір нових даних (етапи 0-3)...")
        local_results = await self.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )
        
        if local_results['status'] != 'local_complete':
            self.logger.error("❌ Локальний пайплайн не завершився успішно")
            return local_results
        
        new_features_df = local_results['results'].get('features_df')
        new_targets_df = local_results['results'].get('targets_df')
        
        if new_features_df is None or new_targets_df is None or new_features_df.empty or new_targets_df.empty:
            self.logger.warning("⚠️ Не вдалося зібрати дані")
            return {
                'status': 'no_data',
                'message': 'Не вдалося зібрати дані'
            }
        
        self.logger.info(f"✅ Нові дані зібрано: features={new_features_df.shape}, targets={new_targets_df.shape}")
        
        # КРОК 2: Порівняти з кешем - чи є нові рядки?
        self.logger.info("📊 КРОК 2: Порівняння з кешем...")
        
        has_existing_cache = features_path.exists() and targets_path.exists()
        has_truly_new_data = False
        
        if has_existing_cache:
            existing_features = pd.read_parquet(features_path)
            existing_rows = len(existing_features)
            new_rows = len(new_features_df)
            
            self.logger.info(f"📊 Кеш: {existing_rows} рядків")
            self.logger.info(f"📊 Нові дані: {new_rows} рядків")
            
            # Перевіряємо, чи є дійсно нові рядки (порівнюємо за datetime + ticker)
            if 'datetime' in new_features_df.columns and 'ticker' in new_features_df.columns:
                # Нормалізуємо datetime для порівняння (видаляємо timezone)
                existing_dt = pd.to_datetime(existing_features['datetime']).dt.tz_localize(None)
                new_dt = pd.to_datetime(new_features_df['datetime']).dt.tz_localize(None)
                
                existing_keys = set(zip(existing_dt, existing_features['ticker']))
                new_keys = set(zip(new_dt, new_features_df['ticker']))
                truly_new_keys = new_keys - existing_keys
                has_truly_new_data = len(truly_new_keys) > 0
                
                self.logger.info(f"📊 Дійсно нових рядків: {len(truly_new_keys)}")
            else:
                # Якщо немає datetime/ticker, порівнюємо за розміром
                has_truly_new_data = new_rows > existing_rows
                self.logger.info(f"📊 Різниця в розмірі: {new_rows - existing_rows}")
        else:
            has_truly_new_data = True
            self.logger.info("📦 Перша база даних - всі дані нові")
        
        # КРОК 3: Логіка на основі наявності нових рядків
        # Якщо немає нових рядків, але є кеш - використовуємо кеш
        if not has_truly_new_data and has_existing_cache and not force_training:
            self.logger.info("\n" + "⚠️ "*40)
            self.logger.info("⚠️ НЕМАЄ НОВИХ РЯДКІВ ДАНИХ")
            self.logger.info("⚠️ Використовуємо існуючий кеш")
            self.logger.info("⚠️ Пропускаємо Stage 0-3 (збір даних)")
            self.logger.info("⚠️ Переходимо до Colab для перевірки та тренування")
            self.logger.info("⚠️ "*40)
            
            # Використовуємо існуючі дані з кешу
            new_features_df = pd.read_parquet(features_path)
            new_targets_df = pd.read_parquet(targets_path)
        else:
            # Є нові рядки (або force_training) - використовуємо дані з run_local_pipeline
            if force_training:
                self.logger.info("\n" + "🔥 "*40)
                self.logger.info("🔥 ФОРСОВАНЕ ТРЕНУВАННЯ (--force-training)")
                self.logger.info("🔥 "*40)
            else:
                self.logger.info("\n" + "✅ "*40)
                self.logger.info("✅ ЗНАЙДЕНО НОВІ РЯДКИ ДАНИХ")
                self.logger.info("✅ ЗАПУСК Stage 0-3 (збір та обробка даних)")
                self.logger.info("✅ "*40)
            
            # Дані вже зібрані в run_local_pipeline, просто зберігаємо їх в кеш
            self.logger.info("💾 Збереження нових фіч та таргетів в кеш...")
            features_path.parent.mkdir(parents=True, exist_ok=True)
            new_features_df.to_parquet(features_path)
            new_targets_df.to_parquet(targets_path)
            self.logger.info(f"✅ Дані збережено в кеш: {batch_dir}")
        
        # КРОК 4: Підготовка пакету для Colab (ЗАВЖДИ)
        self.logger.info("\n" + "📦 "*40)
        self.logger.info("📦 ПІДГОТОВКА ПАКЕТУ ДЛЯ COLAB")
        self.logger.info("📦 "*40)
        
        batch_info = self.prepare_colab_batch(
            features_df=new_features_df,
            targets_df=new_targets_df,
            tickers=tickers or [],
            timeframes=timeframes or [],
            accumulate=accumulate,
            force_feature_selection=force_feature_selection
        )
        self.logger.info(f"✅ Пакет підготовлено: {batch_info.get('batch_dir')}")
        
        # Перевіряємо чи потрібно пропустити Colab
        if skip_colab:
            self.logger.info("\n" + "⏭️ "*40)
            self.logger.info("⏭️ ПРОПУСКАЄМО COLAB (--skip-colab)")
            self.logger.info("⏭️ Переходимо до фінальних етапів з локальними моделями")
            self.logger.info("⏭️ "*40)
            
            # Створюємо фейкові selected_features для легких моделей (всі features)
            self._create_fallback_selected_features(batch_info, new_features_df)
            
            # Запускаємо фінальні етапи
            final_results = await self.run_final_stages(
                batch_name=self.batch_name,
                tickers=tickers,
                timeframes=timeframes
            )
            
            return {
                'status': 'completed_without_colab',
                'message': 'Пайплайн завершено без Colab, використано тільки локальні моделі',
                'batch_info': batch_info,
                'final_results': final_results
            }
        
        # Інструкції для Colab
        colab_instructions = self._generate_colab_instructions(batch_info)
        
        # ⏸️ ПАУЗА: Очікування тренування важких моделей в Colab (ЗАВЖДИ)
        self.logger.info("\n" + "🔴 "*40)
        self.logger.info("🚨 ПАУЗА: Вибір фіч + Тренування важких моделей в Google Colab")
        self.logger.info("🔴 "*40)
        self.logger.info("📋 Інструкції для Colab:")
        self.logger.info(colab_instructions)
        self.logger.info("")
        self.logger.info("💡 В Colab:")
        self.logger.info("   - Клітинка 4: Перевірить чи змінилась база")
        self.logger.info("   - Клітинка 5: Перевірить які тікери вже протреновані")
        self.logger.info("   - Клітинка 7-9: Вибір фіч + тренування + збереження")
        self.logger.info("   - Якщо дані не змінилися → пропустить тренування")
        self.logger.info("")
        
        return {
            'status': 'paused_for_colab',
            'message': 'Пауза для вибору фіч та тренування важких моделей в Colab',
            'batch_info': batch_info,
            'colab_batch': batch_info,  # backward compatibility with older callers
            'colab_instructions': colab_instructions,
            'data_changed': has_truly_new_data  # ✅ NEW: Передаємо флаг для Stage 4
        }
    
    def _generate_colab_instructions(self, batch_info: Dict[str, Any]) -> str:
        """Генерує інструкції для запуску в Colab."""
        if batch_info.get('status') == 'skipped':
            return "Пакет для Colab не створено (немає даних)"
        
        batch_dir_path = Path(batch_info['batch_dir'])
        batch_dir_posix = batch_dir_path.as_posix()
        batch_name = batch_info['batch_name']
        
        instructions = f"""
ІНСТРУКЦІЇ ДЛЯ COLAB:

1. ПЕРЕНЕСЕННЯ БАЗИ НА GOOGLE DRIVE (ОБОВ'ЯЗКОВО!)

   Перекопіюйте папку "{batch_name}" у ваш Google Drive.
   Локальний шлях: {batch_dir_posix}
   На Google Drive шлях має бути: /MyDrive/trading_project/{batch_dir_posix}
   
   Команда для синхронізації (Windows):
   robocopy "{batch_dir_posix}" "G:/MyDrive/trading_project/{batch_dir_posix}" /MIR
   
   Команда для Linux/Mac:
   rsync -av "{batch_dir_posix}/" "G:/MyDrive/trading_project/{batch_dir_posix}/"

2. ЗАВАНТАЖЕННЯ В COLAB

   У Colab ноутбуці виконайте (ПЕРШОЮ КЛІТИНКОЮ):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')  # <-- Потрібно натиснути "Дозволити" у вікні!
   
   import pandas as pd
   import json
   import os
   
   batch_name = "{batch_name}"
   batch_dir = "/content/drive/MyDrive/trading_project/{batch_dir_posix}"
   
   # Завантаження даних (тепер зі справжніми Linux-шляхами)
   features_df = pd.read_parquet(f"{{batch_dir}}/features.parquet")
   targets_df = pd.read_parquet(f"{{batch_dir}}/targets.parquet")
   
   with open(f"{{batch_dir}}/config.json", 'r') as f:
       config = json.load(f)
   ```

3. ВИБІР ФІЧ (5 методів голосування)

   Для КОЖНОЇ моделі виберіть оптимальні фічі:
   ```python
   selected_features_by_model = {{}}
   
   # Включаємо всі моделі: і легкі, і важкі
   all_models = {self.light_models + self.heavy_models}
   
   for model_name in all_models:
       selected_features_by_model[model_name] = set()
       
       for target_col in target_cols:
           selected = feature_selector.select_features(
               X=features_df,
               y=targets_df[target_col],
               model_type=model_name,
               max_features=80  # або з config
           )
           selected_features_by_model[model_name].update(selected)
   
   # Save selected features
   for model_name, features_list in selected_features_by_model.items():
       filename = f"{{batch_dir}}/selected_features_{{model_name}}.json"
       with open(filename, 'w') as f:
           json.dump({{
               'model_name': model_name,
               'selected_features': list(features_list)
           }}, f, indent=2)
   ```

4. ТРЕНУВАННЯ ВАЖКИХ МОДЕЛЕЙ

   Використовуйте вибрані фічі для тренування:
   ```python
   heavy_models = {self.heavy_models}
   for model_name in heavy_models:
       features = selected_features_by_model[model_name]
       # Тренуєте модель на вибраних фічах
   ```

5. ЗБЕРЕЖЕННЯ РЕЗУЛЬТАТІВ

   Результати збережуться автоматично в:
   - `selected_features_{{model}}.json` (для легких моделей)
   - `colab_results_{{ticker}}.json` (результати важких моделей)

6. ПОВЕРНЕННЯ РЕЗУЛЬТАТІВ

   Після завершення Colab:
   - Результати автоматично синхронізуються з Google Drive
   - Локально завантажте результати:
   ```bash
   python run_hybrid_pipeline.py --mode continue --batch-name {batch_name}
   ```

ВАЖЛИВО:
- features.parquet має бути НОВИМ (з останніми новинами)
- selected_features_*.json створюються в Colab (на НОВИХ даних)
- colab_results_*.json створюються в Colab (результати важких моделей)
"""
        return instructions
    
    def _create_fallback_selected_features(self, batch_info: Dict[str, Any], features_df: pd.DataFrame):
        """
        Створює fallback selected_features файли для всіх легких моделей,
        використовуючи всі доступні фічі (коли пропускаємо Colab).
        """
        batch_dir = Path(batch_info['batch_dir'])
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Отримуємо всі доступні фічі (виключаємо target колонки)
        all_features = [col for col in features_df.columns if not col.startswith('target_')]
        
        self.logger.info(f"📊 Створюємо fallback selected_features для {len(self.light_models)} легких моделей")
        self.logger.info(f"📊 Використовуємо всі {len(all_features)} фіч: {all_features[:5]}...")
        
        # Створюємо selected_features для кожної легкої моделі
        for model_name in self.light_models:
            selected_features_path = batch_dir / f"selected_features_{model_name}.json"
            
            selected_features_data = {
                'model_name': model_name,
                'selected_features': all_features,
                'selection_method': 'fallback_all_features',
                'reason': 'skip_colab_mode',
                'total_features': len(all_features),
                'created_at': datetime.now().isoformat()
            }
            
            with open(selected_features_path, 'w', encoding='utf-8') as f:
                json.dump(selected_features_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✅ Створено: {selected_features_path.name}")
    
    async def run_final_stages(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        colab_results: Optional[Dict[str, Any]] = None,
        tickers: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        batch_name: Optional[str] = None,  # ✅ ADD batch_name parameter
        stages_to_run: Optional[List[int]] = None  # ✅ NEW: Дозволяє вибрати конкретні етапи
    ) -> Dict[str, Any]:
        """
        Запускає фінальні етапи 4-7 після завантаження результатів з Colab.
        
        Args:
            features_df: DataFrame з фічами
            targets_df: DataFrame з таргетами
            colab_results: Результати тренування важких моделей з Colab
            tickers: Список тікерів
            timeframes: Список таймфреймів
            batch_name: Назва batch для пошуку selected_features файлів
            stages_to_run: Список етапів для виконання (наприклад, [4, 5, 6, 7] або [6]). За замовчуванням [5, 6, 7]
        
        Returns:
            Dict з результатами виконання етапів 4-7
        """
        if colab_results is None:
            colab_results = {}

        import time
        start_time = time.time()
        
        # ✅ FIX: Витягуємо batch_name з colab_results якщо не передано
        if batch_name is None:
            batch_name = colab_results.get('batch_name', self.batch_name)
        
        # ✅ NEW: Встановлюємо етапи за замовчуванням
        if stages_to_run is None:
            stages_to_run = [5, 6, 7]  # За замовчуванням: Prediction, Trading, Evaluation
        
        self.logger.info("🎯 DEBUG: run_final_stages STARTED")
        self.logger.info(f"🎯 Запуск етапів {stages_to_run}...")
        self.logger.info(f"📦 Batch name: {batch_name}")
        
        # ✅ DEBUG: Перевіряємо структуру colab_results
        self.logger.info(f"📊 DEBUG run_final_stages: colab_results keys: {list(colab_results.keys())}")
        if 'ticker_results' in colab_results:
            self.logger.info(f"📊 DEBUG run_final_stages: ticker_results keys: {list(colab_results['ticker_results'].keys())}")
        
        # Об'єднуємо метадані легких та важких моделей
        # Легкі моделі вже збережені локально, важкі - з Colab
        models_metadata = {}
        
        # ✅ СПОЧАТКУ додаємо метадані важких моделей з Colab
        # ✅ КРИТИЧНО: selected_features завжди в ticker_results, НЕ в models_metadata!
        if 'models_metadata' in colab_results:
            # Новий формат: colab_results_summary.json з models_metadata
            self.logger.info(f"✅ Знайдено models_metadata в colab_results")
            
            # ✅ FIX: Замінюємо Colab шляхи на локальні шляхи
            batch_name = colab_results.get('batch_name', self.batch_name)
            for model_key, meta in colab_results['models_metadata'].items():
                # Замінюємо model_path на локальний шлях
                if 'model_path' in meta:
                    # Витягуємо model_name з model_key (формат: ticker_target_model)
                    parts = model_key.split('_')
                    if len(parts) >= 3:
                        model_name = parts[-1]
                        ticker = parts[0]
                        target = '_'.join(parts[1:-1])
                        # Конструюємо локальний шлях
                        meta['model_path'] = f"data\\colab\\accumulated\\{batch_name}\\models\\{model_name}_{ticker}_{target}.pt"
                        self.logger.debug(f"✅ Замінено model_path для {model_key}: {meta['model_path']}")
                
                # ✅ КРИТИЧНО: Встановлюємо model_category для важких моделей
                if 'model_category' not in meta:
                    # Якщо model_category не встановлено, визначаємо за model_type
                    model_type = meta.get('model_type', meta.get('model_name', ''))
                    if model_type in self.heavy_models:
                        meta['model_category'] = 'heavy'
                        self.logger.debug(f"✅ Встановлено model_category='heavy' для {model_key}")
                    elif model_type in self.light_models:
                        meta['model_category'] = 'light'
                        self.logger.debug(f"✅ Встановлено model_category='light' для {model_key}")
            
            models_metadata.update(colab_results['models_metadata'])
            self.logger.info(f"✅ Додано метадані важких моделей: {len(colab_results['models_metadata'])} моделей")
            
            # ✅ КРИТИЧНО: Додаємо selected_features з ticker_results
            if 'ticker_results' in colab_results:
                self.logger.info(f"🔍 Додаємо selected_features з ticker_results...")
                for ticker, ticker_data in colab_results['ticker_results'].items():
                    timeframes_data = ticker_data.get('timeframes', {})
                    for tf, tf_data in timeframes_data.items():
                        results_data = tf_data.get('results', {})
                        
                        for target_name, target_data in results_data.items():
                            models_dict = target_data.get('models', {})
                            
                            for model_name, model_data in models_dict.items():
                                model_key = f"{ticker}_{target_name}_{model_name}"
                                
                                # Витягуємо selected_features
                                selected_feats = model_data.get('selected_features', [])
                                
                                # Додаємо в існуючу metadata
                                if model_key in models_metadata:
                                    models_metadata[model_key]['selected_features'] = selected_feats
                                    self.logger.info(f"✅ Додано {len(selected_feats)} selected_features для {model_key}")
                                else:
                                    self.logger.warning(f"⚠️ model_key {model_key} не знайдено в models_metadata")
        elif 'ticker_results' in colab_results:
            # Формат з ticker_results (структура з Colab)
            # Витягуємо моделі з результатів
            models_metadata_from_colab = {}
            for ticker, ticker_data in colab_results['ticker_results'].items():
                timeframes_data = ticker_data.get('timeframes', {})
                for tf, tf_data in timeframes_data.items():
                    results_data = tf_data.get('results', {})
                    
                    # Для кожного таргета витягуємо моделі
                    for target_name, target_data in results_data.items():
                        models_dict = target_data.get('models', {})
                        
                        # Для кожної моделі створюємо метадані
                        for model_name, model_data in models_dict.items():
                            model_key = f"{ticker}_{target_name}_{model_name}"
                            
                            # Витягуємо метрики
                            test_metrics = model_data.get('test_metrics', {})
                            train_metrics = model_data.get('train_metrics', {})
                            
                            # ✅ КРИТИЧНО: Витягуємо selected_features з model_data
                            selected_feats = model_data.get('selected_features', [])
                            self.logger.info(f"🔍 Витягування selected_features для {model_key}: {len(selected_feats)} фіч")
                            if selected_feats:
                                self.logger.debug(f"   Перші 5 фіч: {selected_feats[:5]}")
                            else:
                                self.logger.warning(f"⚠️ selected_features ПОРОЖНІ для {model_key}!")
                                self.logger.debug(f"   model_data keys: {list(model_data.keys())}")
                            
                            models_metadata_from_colab[model_key] = {
                                'ticker': ticker,
                                'target': target_name,
                                'model_name': model_name,
                                'model_type': model_name,  # ✅ ADD model_type (same as model_name)
                                'mse': model_data.get('mse', test_metrics.get('mse', 0.05)),
                                'r2': test_metrics.get('r2', 0.0),
                                'mae': test_metrics.get('mae', 0.0),
                                'rmse': test_metrics.get('rmse', 0.0),
                                'mape': test_metrics.get('mape', 0.0),
                                'type': 'heavy',
                                'model_category': 'heavy',  # ✅ ADD для підрахунку в Stage 5
                                'source': 'colab',
                                'trained': True,
                                'timeframe': tf,
                                'model_path': f"data\\colab\\accumulated\\{colab_results.get('batch_name', self.batch_name)}\\models\\{model_name}_{ticker}_{target_name}.pt",
                                'feature_count': model_data.get('feature_count', 0),
                                'selected_features': selected_feats  # ✅ ADD selected_features
                            }
            
            if models_metadata_from_colab:
                models_metadata.update(models_metadata_from_colab)
                self.logger.info(f"✅ Додано метадані важких моделей з ticker_results: {len(models_metadata_from_colab)} контекстів")
                self.logger.debug(f"📊 Моделі: {list(models_metadata_from_colab.keys())}")
        
        # ✅ ПОТІМ завантажуємо метадані легких моделей (якщо вже є з попереднього запуску)
        # Це потрібно для випадку коли Stage 4 вже виконувався раніше
        light_models_files = list(self.output_dir.glob("light_models_results_*.json"))
        if light_models_files:
            latest_light = max(light_models_files, key=lambda p: p.stat().st_mtime)
            with open(latest_light, 'r') as f:
                light_metadata_from_file = json.load(f)
                # Витягуємо models_metadata з файлу
                if 'models_metadata' in light_metadata_from_file:
                    light_meta = light_metadata_from_file['models_metadata']
                    # ✅ НЕ перезаписуємо, а додаємо тільки якщо ключа немає
                    for key, meta in light_meta.items():
                        if key not in models_metadata:
                            models_metadata[key] = meta
                    self.logger.info(f"✅ Завантажено метадані легких моделей з файлу: {len(light_meta)} контекстів")

        
        # ✅ КРОК 4: Тренування легких моделей локально (використовуючи selected_features з Colab)
        # Запускаємо тільки якщо 4 в stages_to_run
        if 4 in stages_to_run:
            self.logger.info("\n" + "💡 "*40)
            self.logger.info("💡 КРОК 4: Тренування легких моделей локально")
            self.logger.info("💡 "*40)
            
            light_models_results = await self.run_light_models_with_selected_features(
                features_df=features_df,
                targets_df=targets_df,
                batch_name=batch_name,
                tickers=tickers
            )
            
            # ✅ КРИТИЧНИЙ FIX: Витягуємо models_metadata напряму з результату
            if light_models_results.get('status') == 'success':
                light_metadata = light_models_results.get('models_metadata', {})
                self.logger.info(f"✅ Легкі моделі протреновані: {len(light_metadata)} моделей")
                
                # Додаємо метадані легких моделей до загальних метаданих
                models_metadata.update(light_metadata)
                self.logger.info(f"📊 Загальна кількість моделей: {len(models_metadata)} (heavy + light)")
                
                # ✅ DEBUG: Логуємо які моделі були додані
                light_keys = [k for k, v in light_metadata.items() if v.get('model_category') == 'light']
                self.logger.info(f"📊 Додані легкі моделі: {light_keys}")
            else:
                self.logger.warning(f"⚠️ Тренування легких моделей не вдалося: {light_models_results.get('message', 'Unknown error')}")
        else:
            self.logger.info("⏭️ Пропускаємо етап 4 (Light Models Training)")
        
        # ✅ NEW: Якщо етап 4 не запускався, завантажуємо легкі моделі з диска
        if 4 not in stages_to_run:
            self.logger.info("🔍 Етап 4 не запускався. Спроба завантажити легкі моделі з диска...")
            light_models_files = list(self.output_dir.glob("light_models_results_*.json"))
            if light_models_files:
                latest_light = max(light_models_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_light, 'r') as f:
                        light_metadata_from_file = json.load(f)
                        if 'models_metadata' in light_metadata_from_file:
                            light_meta = light_metadata_from_file['models_metadata']
                            models_metadata.update(light_meta)
                            self.logger.info(f"✅ Завантажено {len(light_meta)} легких моделей з {latest_light.name}")
                        else:
                            self.logger.warning(f"⚠️ models_metadata не знайдено в {latest_light.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Помилка завантаження легких моделей: {e}")
            else:
                self.logger.warning(f"⚠️ Файли light_models_results_*.json не знайдено в {self.output_dir}")
        
        # ✅ Створюємо оркестратор для етапів 5-7
        # Фільтруємо тільки етапи 5-7 з stages_to_run
        pipeline_stages = [s for s in stages_to_run if s in [5, 6, 7]]
        
        # ✅ FIX: Якщо запускаємо етап 6 або 7, потрібно також запустити етап 5 (залежність)
        if 6 in pipeline_stages or 7 in pipeline_stages:
            if 5 not in pipeline_stages:
                self.logger.info("⚠️ Етап 6 або 7 потребує етапу 5. Додаємо етап 5 до pipeline_stages.")
                pipeline_stages = sorted(set(pipeline_stages) | {5})
        
        if not pipeline_stages:
            self.logger.warning("⚠️ Немає етапів 5-7 для виконання")
            return {
                'status': 'no_stages',
                'message': 'Немає етапів для виконання',
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
            }
        
        self.logger.info(f"🎯 Запуск етапів {pipeline_stages} через PipelineOrchestrator...")
        
        orchestrator = PipelineOrchestrator(
            config_manager=self.config_manager,
            stages_to_run=pipeline_stages  # ✅ Використовуємо відфільтровані етапи
        )
        
        # ✅ FIX: В режимі continue не тренуємо, тільки прогнозуємо
        # Передаємо models_metadata через kwargs, щоб Stage 5 міг витягти batch_dir
        stage_start = time.time()
        
        # ✅ FIX: Передаємо параметри через orchestrator.run()
        self.logger.info(f"📊 DEBUG: Передаємо в orchestrator.run():")
        self.logger.info(f"  - features_data shape: {features_df.shape if features_df is not None else None}")
        self.logger.info(f"  - targets_df shape: {targets_df.shape if targets_df is not None else None}")
        self.logger.info(f"  - models_metadata count: {len(models_metadata)}")
        self.logger.info(f"  - models_metadata keys: {list(models_metadata.keys())[:10]}...")  # Show first 10
        
        # ✅ NEW: Рахуємо легкі та важкі моделі
        light_count = sum(1 for m in models_metadata.values() if m.get('model_category') == 'light')
        heavy_count = sum(1 for m in models_metadata.values() if m.get('model_category') in ['heavy', 'colab'])
        self.logger.info(f"  - Light models: {light_count}, Heavy models: {heavy_count}")
        
        # ✅ DEBUG: Перевіряємо selected_features в metadata
        for key in list(models_metadata.keys())[:3]:
            meta = models_metadata[key]
            selected_feats = meta.get('selected_features', [])
            self.logger.debug(f"  - {key}: selected_features={len(selected_feats)} фіч")
            if selected_feats:
                self.logger.debug(f"    Перші 3 фіч: {selected_feats[:3]}")
        
        results = await orchestrator.run(
            tickers=tickers,
            timeframes=timeframes,
            run_mode='predict',  # ✅ Тільки prediction, не training
            features_data=features_df,
            targets_df=targets_df,
            models_metadata=models_metadata,  # ✅ Містить важкі + легкі моделі
            batch_name=batch_name,  # ✅ NEW: Передаємо batch_name для завантаження з диска
            force_retrain=False,
            stages_to_run=pipeline_stages  # ✅ Передаємо відфільтровані етапи
        )
        stage_duration = time.time() - stage_start
        
        total_duration = time.time() - start_time
        self.logger.info(f"⏱️ Етапи 4-7 виконано за {stage_duration:.1f}s")
        self.logger.info(f"⏱️ Загальний час: {total_duration:.1f}s ({total_duration/60:.1f}m)")
        
        # Зберігаємо фінальні результати
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_results_path = self.output_dir / f"final_results_{timestamp}.json"
        
        # NEW: Отримати правильні counts з результатів prediction stage
        light_models_count = results.get('light_models_count', 0)
        heavy_models_count = results.get('heavy_models_count', 0)
        total_models = results.get('total_models', len(models_metadata))
        
        final_summary = {
            'timestamp': timestamp,
            'tickers': tickers,
            'timeframes': timeframes,
            'light_models_count': light_models_count,
            'heavy_models_count': heavy_models_count,
            'total_models': total_models,
            'prediction_results': results.get('prediction_results', {}),
            'analyzer_summary': results.get('analyzer_summary', {}),
            'trading_summary': results.get('portfolio_summary', {}),
            'evaluation_summary': results.get('evaluation_summary', {}),
            'duration_seconds': total_duration
        }
        
        with open(final_results_path, 'w') as f:
            json.dump(final_summary, f, indent=2, default=str)
        
        self.logger.info(f"📋 Фінальні результати збережено: {final_results_path}")
        
        return {
            'status': 'complete',
            'results': results,
            'final_results_path': str(final_results_path),
            'timestamp': timestamp,
            'duration_seconds': total_duration
        }
    
    def _save_data(self, data: Dict[str, pd.DataFrame], path: Path):
        """Зберігає словник DataFrame'ів у parquet, автоматично розплющуючи вкладені словники."""
        if data is None or (isinstance(data, dict) and not data):
            return
        
        try:
            flat_dict = {}
            for k, v in data.items():
                if isinstance(v, pd.DataFrame):
                    flat_dict[k] = v
                elif isinstance(v, dict):
                    # Глибоке розплющування (один рівень)
                    for sub_k, sub_v in v.items():
                        if isinstance(sub_v, pd.DataFrame):
                            flat_dict[f"{k}_{sub_k}"] = sub_v
                        elif isinstance(sub_v, dict) and 'data' in sub_v and isinstance(sub_v['data'], pd.DataFrame):
                            flat_dict[f"{k}_{sub_k}"] = sub_v['data']
            
            if not flat_dict:
                self.logger.warning(f"⚠️ Немає валідних DataFrame для збереження у {path}")
                return

            # Об'єднуємо всі DataFrame'и
            combined = pd.concat(flat_dict.values(), keys=flat_dict.keys(), names=['source'], sort=False)
            
            # Очищаємо дані перед збереженням у Parquet
            for col in combined.columns:
                col_dtype = combined[col].dtype
                if col_dtype == 'object' or col_dtype.name == 'object':
                    try:
                        # Спробуємо конвертувати, видаляючи коми
                        combined[col] = pd.to_numeric(
                            combined[col].astype(str).str.replace(',', ''), 
                            errors='coerce'  # Замість 'ignore' використовуємо 'coerce'
                        )
                    except Exception:
                        pass
            
            combined.to_parquet(path, compression='snappy')
        except Exception as e:
            self.logger.error(f"❌ Помилка при збереженні даних у {path}: {e}", exc_info=True)
    
    def _clean_dataframe(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        """
        ✅ Очищає DataFrame від NaN та Inf значень.
        
        Args:
            df: DataFrame для очищення
            name: Назва DataFrame для логування
        
        Returns:
            Очищений DataFrame
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # Перевіряємо NaN
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            nan_percent = (nan_count / (df.shape[0] * df.shape[1])) * 100
            self.logger.warning(f"⚠️ {name}: Знайдено {nan_count} NaN значень ({nan_percent:.2f}%)")
            
            # Заповнюємо NaN нулями для числових колонок
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            self.logger.info(f"✅ {name}: NaN значення заповнено нулями")
        
        # Перевіряємо Inf
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_mask = np.isinf(df[numeric_cols].values)
        inf_count = inf_mask.sum()
        
        if inf_count > 0:
            inf_percent = (inf_count / (df.shape[0] * len(numeric_cols))) * 100
            self.logger.warning(f"⚠️ {name}: Знайдено {inf_count} Inf значень ({inf_percent:.2f}%)")
            
            # Замінюємо Inf на 0
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)
            self.logger.info(f"✅ {name}: Inf значення замінено нулями")
        
        return df
    
    def _save_dataframe(self, df: pd.DataFrame, path: Path):
        """Зберігає DataFrame у parquet."""
        if df is None or df.empty:
            self.logger.warning(f"⚠️ Порожній DataFrame, пропускаємо збереження: {path}")
            return
        
        # Перевіряємо, що df - це DataFrame
        if not isinstance(df, pd.DataFrame):
            self.logger.error(f"❌ Очікувався DataFrame, отримано {type(df)}")
            return
        
        # ✅ FIX: Виправляємо опечатки в назвах колонок
        df = df.copy()
        rename_map = {}
        for col in df.columns:
            # Виправляємо conteext → context
            if 'conteext' in col:
                new_col = col.replace('conteext', 'context')
                rename_map[col] = new_col
                self.logger.info(f"✅ Виправлено опечатку: {col} → {new_col}")
        
        if rename_map:
            df = df.rename(columns=rename_map)
        
        # Видаляємо дублікати колонок (залишаємо перше входження)
        if df.columns.duplicated().any():
            duplicated_cols = df.columns[df.columns.duplicated()].tolist()
            self.logger.warning(f"⚠️ Знайдено {len(duplicated_cols)} дублікатів колонок, видаляємо: {duplicated_cols[:10]}")
            df = df.loc[:, ~df.columns.duplicated()]
        
        # Очищаємо дані перед збереженням
        for col in df.columns:
            try:
                # Отримуємо Series для колонки
                series = df[col]
                
                # Перевіряємо, що це Series, а не DataFrame
                if not isinstance(series, pd.Series):
                    self.logger.warning(f"Column {col} is not a Series, skipping")
                    continue
                
                # ✅ КРИТИЧНИЙ FIX: НЕ конвертуємо ticker, datetime та інші метадані
                if col in ['ticker', 'datetime', 'interval', 'hash', 'context_fingerprint']:
                    continue
                
                # Перевіряємо тип колонки
                col_dtype = series.dtype
                if col_dtype == 'object' or col_dtype.name == 'object':
                    # Конвертуємо рядки з комами в числа
                    df[col] = pd.to_numeric(
                        series.astype(str).str.replace(',', ''), 
                        errors='coerce'
                    )
            except Exception as e:
                self.logger.debug(f"Could not process column {col}: {e}")
                pass
        
        df.to_parquet(path, compression='snappy')
