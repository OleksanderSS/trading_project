"""
Batch Training Manager
Пакетний меnotджер тренування for Colab with роwithбиттям на частини
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import time
from datetime import datetime, timedelta

# Додаємо шлях до проекту
import sys
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_tickers_dict
from core.targets.universal_target_manager import UniversalTargetManager, ModelType
from utils.your_working_colab_cell import create_multi_targets, train_heavy_models, train_light_models

logger = logging.getLogger("BatchTrainingManager")

@dataclass
class BatchConfig:
    """Конфandгурацandя пакету тренування"""
    batch_id: str
    tickers: List[str]
    timeframes: List[str]
    model_types: List[ModelType]
    data_points: int
    max_models_per_batch: int = 5
    save_intermediate: bool = True
    resume_from_checkpoint: bool = True

@dataclass
class TrainingProgress:
    """Прогрес тренування"""
    total_batches: int
    completed_batches: int
    total_models: int
    completed_models: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    current_batch: str = ""
    current_model: str = ""
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class BatchTrainingManager:
    """Пакетний меnotджер тренування"""
    
    def __init__(self, output_dir: str = "data/batch_training"):
        self.logger = logging.getLogger("BatchTrainingManager")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Інandцandалandwithуємо system
        self.target_manager = UniversalTargetManager()
        self.tickers_dict = get_tickers_dict()
        self.all_tickers = get_tickers("all")
        
        # Створюємо директорandї
        self.batches_dir = self.output_dir / "batches"
        self.models_dir = self.output_dir / "models"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"
        
        for dir_path in [self.batches_dir, self.models_dir, self.checkpoints_dir, self.logs_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger.info("Batch Training Manager initialized")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Available tickers: {len(self.all_tickers)}")
    
    def create_training_batches(self, 
                              tickers: List[str] = None,
                              timeframes: List[str] = None,
                              model_types: List[ModelType] = None,
                              max_models_per_batch: int = 5,
                              data_points_per_ticker: int = 1000) -> List[BatchConfig]:
        """
        Create пакети тренування
        
        Args:
            tickers: Список тandкерandв (якщо None, то all)
            timeframes: Список andймфреймandв (якщо None, то ['15m', '60m', '1d'])
            model_types: Список типandв моwhereлей (якщо None, то all)
            max_models_per_batch: Максимальна кandлькandсть моwhereлей на пакет
            data_points_per_ticker: Кandлькandсть data на тandкер
            
        Returns:
            List[BatchConfig]: Список пакетandв
        """
        if tickers is None:
            tickers = self.all_tickers[:20]  # Обмежуємо for тестування
        
        if timeframes is None:
            timeframes = ['15m', '60m', '1d']
        
        if model_types is None:
            model_types = [
                ModelType.LIGHTGBM,
                ModelType.RANDOM_FOREST,
                ModelType.LINEAR,
                ModelType.MLP,
                ModelType.GRU,
                ModelType.LSTM,
                ModelType.TRANSFORMER,
                ModelType.CNN,
                ModelType.TABNET,
                ModelType.AUTOENCODER
            ]
        
        # Створюємо all комбandнацandї
        all_combinations = []
        for ticker in tickers:
            for timeframe in timeframes:
                for model_type in model_types:
                    all_combinations.append({
                        'ticker': ticker,
                        'timeframe': timeframe,
                        'model_type': model_type,
                        'data_points': data_points_per_ticker
                    })
        
        # Роwithбиваємо на пакети
        batches = []
        current_batch = []
        current_batch_tickers = set()
        current_batch_timeframes = set()
        current_batch_models = []
        
        for i, combo in enumerate(all_combinations):
            # Перевandряємо чи can add до поточного пакету
            can_add = (
                len(current_batch) < max_models_per_batch and
                (combo['ticker'] not in current_batch_tickers or 
                 len(current_batch_tickers) < 5) and
                (combo['timeframe'] not in current_batch_timeframes or 
                 len(current_batch_timeframes) < 2)
            )
            
            if can_add:
                current_batch.append(combo)
                current_batch_tickers.add(combo['ticker'])
                current_batch_timeframes.add(combo['timeframe'])
                current_batch_models.append(combo['model_type'])
            else:
                # Створюємо пакет with поточними даними
                if current_batch:
                    batch_id = f"batch_{len(batches) + 1:03d}"
                    batch_config = BatchConfig(
                        batch_id=batch_id,
                        tickers=list(current_batch_tickers),
                        timeframes=list(current_batch_timeframes),
                        model_types=list(set(current_batch_models)),
                        data_points=data_points_per_ticker,
                        max_models_per_batch=max_models_per_batch
                    )
                    batches.append(batch_config)
                
                # Починаємо новий пакет
                current_batch = [combo]
                current_batch_tickers = {combo['ticker']}
                current_batch_timeframes = {combo['timeframe']}
                current_batch_models = [combo['model_type']]
        
        # Додаємо осandннandй пакет
        if current_batch:
            batch_id = f"batch_{len(batches) + 1:03d}"
            batch_config = BatchConfig(
                batch_id=batch_id,
                tickers=list(current_batch_tickers),
                timeframes=list(current_batch_timeframes),
                model_types=list(set(current_batch_models)),
                data_points=data_points_per_ticker,
                max_models_per_batch=max_models_per_batch
            )
            batches.append(batch_config)
        
        self.logger.info(f"Created {len(batches)} batches for {len(all_combinations)} combinations")
        self.logger.info(f"Average models per batch: {len(all_combinations) / len(batches):.1f}")
        
        return batches
    
    def prepare_batch_data(self, batch_config: BatchConfig) -> pd.DataFrame:
        """
        Пandдготувати данand for пакету
        
        Args:
            batch_config: Конфandгурацandя пакету
            
        Returns:
            pd.DataFrame: Пandдготовленand данand
        """
        self.logger.info(f"Preparing data for batch {batch_config.batch_id}")
        
        # Створюємо синтетичнand данand for тестування
        # В реальностand тут will forванandження with баwithи data
        np.random.seed(42)
        
        # Створюємо данand for кожного тandкера/andймфрейму
        all_data = []
        
        for ticker in batch_config.tickers:
            for timeframe in batch_config.timeframes:
                # Виwithначаємо кandлькandсть точок data
                if timeframe == '15m':
                    points_per_day = 78  # 6.5 годин * 12 (15-хвилиннand andнтервали)
                    days = batch_config.data_points // points_per_day
                elif timeframe == '60m':
                    points_per_day = 6   # 6.5 годин * 1 (годиннand andнтервали)
                    days = batch_config.data_points // points_per_day
                else:  # 1d
                    points_per_day = 1
                    days = batch_config.data_points
                
                # Створюємо дати
                dates = pd.date_range(
                    start=datetime.now() - timedelta(days=days),
                    periods=batch_config.data_points,
                    freq='15T' if timeframe == '15m' else '1H' if timeframe == '60m' else '1D'
                )
                
                # Створюємо цandни
                base_price = 100 + np.random.uniform(-50, 200)
                price_changes = np.random.normal(0, 0.02, batch_config.data_points)
                prices = base_price * np.cumprod(1 + price_changes)
                
                # Створюємо OHLCV
                data = {
                    'date': dates,
                    'ticker': ticker,
                    'timeframe': timeframe,
                    'close': prices,
                    'high': prices * (1 + np.abs(np.random.normal(0, 0.01, batch_config.data_points))),
                    'low': prices * (1 - np.abs(np.random.normal(0, 0.01, batch_config.data_points))),
                    'open': np.roll(prices, 1),
                    'volume': np.random.randint(1000000, 10000000, batch_config.data_points)
                }
                
                df = pd.DataFrame(data)
                df.set_index('date', inplace=True)
                all_data.append(df)
        
        # Об'єднуємо all данand
        combined_df = pd.concat(all_data, ignore_index=False)
        
        # Створюємо andргети
        self.logger.info("Creating targets with Universal Target Manager...")
        combined_df = create_multi_targets(combined_df, batch_config.tickers, batch_config.timeframes)
        
        # Зберandгаємо данand пакету
        batch_data_path = self.batches_dir / f"{batch_config.batch_id}_data.parquet"
        combined_df.to_parquet(batch_data_path)
        
        self.logger.info(f"Batch data saved to: {batch_data_path}")
        self.logger.info(f"Data shape: {combined_df.shape}")
        
        return combined_df
    
    def train_batch(self, batch_config: BatchConfig, data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Тренувати пакет
        
        Args:
            batch_config: Конфandгурацandя пакету
            data: Данand for тренування (якщо None, forванandжує with fileу)
            
        Returns:
            Dict[str, Any]: Реwithульandти тренування
        """
        self.logger.info(f"Training batch {batch_config.batch_id}")
        
        # Заванandжуємо данand якщо not надано
        if data is None:
            batch_data_path = self.batches_dir / f"{batch_config.batch_id}_data.parquet"
            if batch_data_path.exists():
                data = pd.read_parquet(batch_data_path)
            else:
                data = self.prepare_batch_data(batch_config)
        
        results = {
            'batch_id': batch_config.batch_id,
            'tickers': batch_config.tickers,
            'timeframes': batch_config.timeframes,
            'model_types': [mt.value for mt in batch_config.model_types],
            'start_time': datetime.now(),
            'models_trained': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Тренуємо light моwhereлand
            light_models = [mt for mt in batch_config.model_types if mt in [
                ModelType.LIGHTGBM, ModelType.RANDOM_FOREST, ModelType.LINEAR, ModelType.MLP
            ]]
            
            if light_models:
                self.logger.info(f"Training {len(light_models)} light models...")
                light_results = train_light_models(data)
                results['models_trained'].extend(light_results)
                results['metrics']['light'] = self._calculate_metrics(light_results)
            
            # Тренуємо heavy моwhereлand
            heavy_models = [mt for mt in batch_config.model_types if mt in [
                ModelType.GRU, ModelType.LSTM, ModelType.TRANSFORMER, ModelType.CNN, 
                ModelType.TABNET, ModelType.AUTOENCODER
            ]]
            
            if heavy_models:
                self.logger.info(f"Training {len(heavy_models)} heavy models...")
                heavy_results = train_heavy_models(data)
                results['models_trained'].extend(heavy_results)
                results['metrics']['heavy'] = self._calculate_metrics(heavy_results)
            
            results['end_time'] = datetime.now()
            results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
            results['status'] = 'completed'
            
        except Exception as e:
            self.logger.error(f"Error training batch {batch_config.batch_id}: {e}")
            results['errors'].append(str(e))
            results['status'] = 'failed'
            results['end_time'] = datetime.now()
        
        # Зберandгаємо реwithульandти
        results_path = self.batches_dir / f"{batch_config.batch_id}_results.json"
        with open(results_path, 'w') as f:
            # Конвертуємо datetime в string for JSON
            json_results = self._convert_datetime_to_string(results)
            json.dump(json_results, f, indent=2)
        
        self.logger.info(f"Batch results saved to: {results_path}")
        
        return results
    
    def _calculate_metrics(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Роwithрахувати метрики for реwithульandтandв"""
        if not model_results:
            return {}
        
        metrics = {
            'total_models': len(model_results),
            'successful_models': len([r for r in model_results if r.get('status') == 'success']),
            'failed_models': len([r for r in model_results if r.get('status') == 'failed']),
            'average_accuracy': 0,
            'average_f1': 0,
            'model_types': {}
        }
        
        # Calculating середнand метрики
        accuracies = []
        f1_scores = []
        
        for result in model_results:
            if result.get('status') == 'success':
                if 'accuracy' in result:
                    accuracies.append(result['accuracy'])
                if 'f1_score' in result:
                    f1_scores.append(result['f1_score'])
                
                # Групуємо for типами моwhereлей
                model_type = result.get('model_type', 'unknown')
                if model_type not in metrics['model_types']:
                    metrics['model_types'][model_type] = {'count': 0, 'success': 0}
                metrics['model_types'][model_type]['count'] += 1
                metrics['model_types'][model_type]['success'] += 1
        
        if accuracies:
            metrics['average_accuracy'] = np.mean(accuracies)
        if f1_scores:
            metrics['average_f1'] = np.mean(f1_scores)
        
        return metrics
    
    def _convert_datetime_to_string(self, obj):
        """Конвертувати datetime в string for JSON"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._convert_datetime_to_string(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_datetime_to_string(item) for item in obj]
        else:
            return obj
    
    def run_batch_training(self, 
                          tickers: List[str] = None,
                          timeframes: List[str] = None,
                          model_types: List[ModelType] = None,
                          max_models_per_batch: int = 5,
                          resume_from_checkpoint: bool = True) -> TrainingProgress:
        """
        Запустити пакетnot тренування
        
        Args:
            tickers: Список тandкерandв
            timeframes: Список andймфреймandв
            model_types: Список типandв моwhereлей
            max_models_per_batch: Максимальна кandлькandсть моwhereлей на пакет
            resume_from_checkpoint: Продовжити with чекпоandнту
            
        Returns:
            TrainingProgress: Прогрес тренування
        """
        self.logger.info("Starting batch training...")
        
        # Створюємо пакети
        batches = self.create_training_batches(
            tickers=tickers,
            timeframes=timeframes,
            model_types=model_types,
            max_models_per_batch=max_models_per_batch
        )
        
        # Інandцandалandwithуємо прогрес
        progress = TrainingProgress(
            total_batches=len(batches),
            completed_batches=0,
            total_models=sum(len(batch.model_types) * len(batch.tickers) * len(batch.timeframes) for batch in batches),
            completed_models=0,
            start_time=datetime.now()
        )
        
        # Перевandряємо чекпоandнти
        completed_batches = []
        if resume_from_checkpoint:
            completed_batches = self._get_completed_batches()
            progress.completed_batches = len(completed_batches)
            progress.completed_models = sum(
                len(batch.model_types) * len(batch.tickers) * len(batch.timeframes) 
                for batch in batches if batch.batch_id in completed_batches
            )
        
        # Тренуємо пакети
        for i, batch_config in enumerate(batches):
            if batch_config.batch_id in completed_batches:
                self.logger.info(f"Skipping completed batch {batch_config.batch_id}")
                continue
            
            try:
                progress.current_batch = batch_config.batch_id
                
                # Тренуємо пакет
                results = self.train_batch(batch_config)
                
                # Оновлюємо прогрес
                progress.completed_batches += 1
                progress.completed_models += len(batch_config.model_types) * len(batch_config.tickers) * len(batch_config.timeframes)
                
                # Calculating оцandнку forвершення
                elapsed = (datetime.now() - progress.start_time).total_seconds()
                if progress.completed_batches > 0:
                    avg_time_per_batch = elapsed / progress.completed_batches
                    remaining_batches = progress.total_batches - progress.completed_batches
                    estimated_remaining_seconds = avg_time_per_batch * remaining_batches
                    progress.estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_seconds)
                
                # Зберandгаємо прогрес
                self._save_progress(progress)
                
                self.logger.info(f"Batch {batch_config.batch_id} completed. Progress: {progress.completed_batches}/{progress.total_batches}")
                
            except Exception as e:
                self.logger.error(f"Error in batch {batch_config.batch_id}: {e}")
                progress.errors.append(f"Batch {batch_config.batch_id}: {str(e)}")
                continue
        
        # Фandнальnot withбереження
        self._save_progress(progress)
        
        self.logger.info(f"Batch training completed. Total: {progress.completed_batches}/{progress.total_batches} batches")
        
        return progress
    
    def _get_completed_batches(self) -> List[str]:
        """Отримати список forвершених пакетandв"""
        completed = []
        for file_path in self.batches_dir.glob("*_results.json"):
            try:
                with open(file_path, 'r') as f:
                    results = json.load(f)
                    if results.get('status') == 'completed':
                        batch_id = results.get('batch_id')
                        if batch_id:
                            completed.append(batch_id)
            except Exception as e:
                self.logger.warning(f"Error reading {file_path}: {e}")
        return completed
    
    def _save_progress(self, progress: TrainingProgress):
        """Зберегти прогрес"""
        progress_path = self.checkpoints_dir / "training_progress.json"
        with open(progress_path, 'w') as f:
            json_progress = asdict(progress)
            json_progress['start_time'] = progress.start_time.isoformat()
            if progress.estimated_completion:
                json_progress['estimated_completion'] = progress.estimated_completion.isoformat()
            json.dump(json_progress, f, indent=2)
    
    def create_colab_training_script(self, batch_config: BatchConfig) -> str:
        """
        Create script for тренування в Colab
        
        Args:
            batch_config: Конфandгурацandя пакету
            
        Returns:
            str: Шлях до scriptу
        """
        script_content = f'''#!/usr/bin/env python3
"""
Colab Training Script for {batch_config.batch_id}
Generated on: {datetime.now().isoformat()}
"""

import sys
import os
from pathlib import Path

# Додаємо шлях до проекту
project_root = Path("/content/drive/MyDrive/trading_project")
sys.path.append(str(project_root))

# Імпорти
import pandas as pd
import numpy as np
import logging
from datetime import datetime

# Налаштування logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Імпорти with проекту
from utils.your_working_colab_cell import auto_load_latest_files, create_multi_targets, train_heavy_models, train_light_models
from core.targets.universal_target_manager import UniversalTargetManager, ModelType

def main():
    """Головна функцandя тренування"""
    logger.info("Starting Colab training for {batch_config.batch_id}")
    
    # Конфandгурацandя
    tickers = {batch_config.tickers}
    timeframes = {batch_config.timeframes}
    model_types = {batch_config.model_types}
    
    logger.info(f"Configuration:")
    logger.info(f"  Tickers: {{len(tickers)}}")
    logger.info(f"  Timeframes: {{len(timeframes)}}")
    logger.info(f"  Model types: {{len(model_types)}}")
    
    try:
        # Крок 1: Заванandження data
        logger.info("Step 1: Loading data...")
        features_df = auto_load_latest_files()
        
        if features_df is None:
            logger.error("Failed to load data")
            return False
        
        logger.info(f"Data loaded: {{features_df.shape}}")
        
        # Крок 2: Створення andргетandв
        logger.info("Step 2: Creating targets...")
        features_df = create_multi_targets(features_df, tickers, timeframes)
        
        if features_df is None:
            logger.error("Failed to create targets")
            return False
        
        logger.info(f"Targets created: {{features_df.shape}}")
        
        # Крок 3: Тренування light моwhereлей
        logger.info("Step 3: Training light models...")
        light_model_types = [mt for mt in model_types if mt in [
            ModelType.LIGHTGBM, ModelType.RANDOM_FOREST, ModelType.LINEAR, ModelType.MLP
        ]]
        
        light_results = []
        if light_model_types:
            light_results = train_light_models(features_df)
            logger.info(f"Light models trained: {{len(light_results)}}")
        
        # Крок 4: Тренування heavy моwhereлей
        logger.info("Step 4: Training heavy models...")
        heavy_model_types = [mt for mt in model_types if mt in [
            ModelType.GRU, ModelType.LSTM, ModelType.TRANSFORMER, ModelType.CNN, 
            ModelType.TABNET, ModelType.AUTOENCODER
        ]]
        
        heavy_results = []
        if heavy_model_types:
            heavy_results = train_heavy_models(features_df)
            logger.info(f"Heavy models trained: {{len(heavy_results)}}")
        
        # Крок 5: Збереження реwithульandтandв
        logger.info("Step 5: Saving results...")
        results = {{
            'batch_id': '{batch_config.batch_id}',
            'tickers': tickers,
            'timeframes': timeframes,
            'model_types': [mt.value for mt in model_types],
            'light_results': light_results,
            'heavy_results': heavy_results,
            'completion_time': datetime.now().isoformat()
        }}
        
        # Зберandгаємо в Google Drive
        output_path = f"/content/drive/MyDrive/trading_project/colab_results/{{batch_config.batch_id}}_results.json"
        import json
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {{output_path}}")
        logger.info("Training completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"Training failed: {{e}}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("[OK] Training completed successfully!")
    else:
        print("[ERROR] Training failed!")
'''
        
        # Зберandгаємо script
        script_path = self.output_dir / f"{batch_config.batch_id}_colab_script.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        self.logger.info(f"Colab script created: {script_path}")
        
        return str(script_path)

def main():
    """Тестування пакетного тренування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch Training Manager')
    parser.add_argument('--tickers', nargs='+', default=['SPY', 'QQQ', 'NVDA', 'TSLA'],
                       help='Tickers to train')
    parser.add_argument('--timeframes', nargs='+', default=['15m', '60m', '1d'],
                       help='Timeframes to train')
    parser.add_argument('--models', nargs='+', 
                       choices=['lightgbm', 'random_forest', 'linear', 'mlp', 'gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder'],
                       default=['lightgbm', 'random_forest', 'gru', 'lstm'],
                       help='Model types to train')
    parser.add_argument('--max-models-per-batch', type=int, default=3,
                       help='Maximum models per batch')
    parser.add_argument('--create-batches-only', action='store_true',
                       help='Only create batches, don\'t train')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Створюємо меnotджер
    manager = BatchTrainingManager()
    
    # Конвертуємо моwhereлand
    model_types = [ModelType(model.upper()) for model in args.models]
    
    # Створюємо пакети
    batches = manager.create_training_batches(
        tickers=args.tickers,
        timeframes=args.timeframes,
        model_types=model_types,
        max_models_per_batch=args.max_models_per_batch
    )
    
    print(f"Created {len(batches)} batches:")
    for batch in batches:
        print(f"  {batch.batch_id}: {len(batch.tickers)} tickers, {len(batch.timeframes)} timeframes, {len(batch.model_types)} model types")
    
    if not args.create_batches_only:
        # Запускаємо тренування
        progress = manager.run_batch_training(
            tickers=args.tickers,
            timeframes=args.timeframes,
            model_types=model_types,
            max_models_per_batch=args.max_models_per_batch,
            resume_from_checkpoint=args.resume
        )
        
        print(f"\\nTraining completed:")
        print(f"  Batches: {progress.completed_batches}/{progress.total_batches}")
        print(f"  Models: {progress.completed_models}/{progress.total_models}")
        print(f"  Duration: {(datetime.now() - progress.start_time).total_seconds():.1f}s")
        if progress.estimated_completion:
            print(f"  Estimated completion: {progress.estimated_completion}")
        
        if progress.errors:
            print(f"  Errors: {len(progress.errors)}")
            for error in progress.errors[:5]:
                print(f"    - {error}")

if __name__ == "__main__":
    main()
