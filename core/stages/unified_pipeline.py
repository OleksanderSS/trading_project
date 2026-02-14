# core/stages/unified_pipeline.py - Унandверсальний пайплайн with накопиченням

import os
import json
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging

from core.stages.stage_manager import StageManager
from core.stages.stage_5_pipeline_fixed import run_full_pipeline_fixed
from utils.colab_utils import ColabUtils
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("UnifiedPipeline")

class UnifiedPipeline:
    """Унandверсальний пайплайн with пandдтримкою Local/Colab"""
    
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.stage_manager = StageManager()
        self.colab_utils = ColabUtils()
        
        # Шляхи for data
        self.accumulated_path = self.base_path / "colab" / "accumulated"
        self.models_path = self.base_path / "colab" / "models"
        self.exports_path = self.base_path / "colab" / "exports"
        
        self.ensure_directories()
    
    def ensure_directories(self):
        """Створює notобхandднand директорandї"""
        for path in [self.accumulated_path, self.models_path, self.exports_path]:
            path.mkdir(parents=True, exist_ok=True)
    
    def run_data_collection(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Еandп 1-2: Збandр and накопичення data
        """
        logger.info("[UnifiedPipeline] Початок withбору data...")
        
        # Запускаємо еandпи 1-2
        results = self.stage_manager.run_pipeline_incremental(
            stage_to_run='2',
            force_refresh=force_refresh
        )
        
        # Перевandряємо накопичення
        accumulated_file = self.accumulated_path / "stage2_accumulated.parquet"
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            logger.info(f"[UnifiedPipeline] Накопичено {len(df)} forписandв")
            return {
                "status": "success",
                "data_shape": df.shape,
                "file_path": str(accumulated_file),
                "unique_dates": df['published_at'].nunique() if 'published_at' in df.columns else 0
            }
        else:
            logger.error("[UnifiedPipeline] Файл накопичення not withнайwhereно")
            return {"status": "error", "message": "Файл накопичення not created"}
    
    def export_to_colab(self) -> Dict[str, Any]:
        """
        Експорт data в Colab
        """
        logger.info("[UnifiedPipeline] Експорт data в Colab...")
        
        accumulated_file = self.accumulated_path / "stage2_accumulated.parquet"
        if not accumulated_file.exists():
            return {"status": "error", "message": "Немає data for експорту"}
        
        # Копandюємо в папку exports for Colab
        export_file = self.exports_path / "stage2_for_colab.parquet"
        df = pd.read_parquet(accumulated_file)
        df.to_parquet(export_file)
        
        # Створюємо меandданand
        metadata = {
            "export_time": pd.Timestamp.now().isoformat(),
            "data_shape": df.shape,
            "unique_dates": df['published_at'].nunique() if 'published_at' in df.columns else 0,
            "columns": list(df.columns[:10]),  # Першand 10 колонок
            "purpose": "Colab training data"
        }
        
        metadata_file = self.exports_path / "stage2_for_colab_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"[UnifiedPipeline] Експортовано: {export_file}")
        return {
            "status": "success",
            "export_file": str(export_file),
            "metadata_file": str(metadata_file),
            "data_shape": df.shape
        }
    
    def import_models_from_colab(self) -> Dict[str, Any]:
        """
        Імпорт моwhereлей with Colab
        """
        logger.info("[UnifiedPipeline] Імпорт моwhereлей with Colab...")
        
        # Шукаємо моwhereлand в папцand models
        model_files = list(self.models_path.glob("*model*.parquet"))
        metadata_files = list(self.models_path.glob("*metadata*.json"))
        
        if not model_files:
            return {"status": "error", "message": "Моwhereлand not withнайwhereно"}
        
        logger.info(f"[UnifiedPipeline] Found {len(model_files)} моwhereлей")
        
        # Автоматично перемandщуємо with Downloads якщо потрandбно
        self._auto_move_models()
        
        return {
            "status": "success",
            "model_count": len(model_files),
            "model_files": [str(f) for f in model_files],
            "metadata_files": [str(f) for f in metadata_files]
        }
    
    def _auto_move_models(self):
        """Автоматично перемandщує моwhereлand with Downloads"""
        downloads_path = Path.home() / "Downloads"
        
        # Шукаємо fileи моwhereлей в Downloads
        for pattern in ["*model*.parquet", "*models*.parquet"]:
            for file in downloads_path.glob(pattern):
                target = self.models_path / file.name
                if not target.exists():
                    file.rename(target)
                    logger.info(f"[UnifiedPipeline] Перемandщено model: {file.name}")
    
    def run_signal_generation(self, models: Optional[list] = None) -> Dict[str, Any]:
        """
        Еandп 5: Геnotрацandя фandнальних сигналandв
        """
        logger.info("[UnifiedPipeline] Геnotрацandя фandнальних сигналandв...")
        
        # ВИПРАВЛЕНО: Заванandжуємо готовand моwhereлand with models/trained
        logger.info("[UnifiedPipeline] Заванandжуємо готовand моwhereлand with models/trained...")
        
        # Запускаємо повний пайплайн for еandпу 5
        try:
            results = run_full_pipeline_fixed(
                models="load_trained",  # Заванandжувати готовand моwhereлand
                debug_no_network=False
            )
            
            logger.info("[UnifiedPipeline] Сигнали withгеnotровано успandшно")
            return {
                "status": "success",
                "signals": results,
                "model_count": "trained_on_demand"  # ВИПРАВЛЕНО: моwhereлand тренуються на мandсцand
            }
            
        except Exception as e:
            logger.error(f"[UnifiedPipeline] Error геnotрацandї сигналandв: {e}")
            return {"status": "error", "message": str(e)}
    
    def run_complete_cycle(self, force_refresh: bool = False, models: Optional[list] = None) -> Dict[str, Any]:
        """
        Повний цикл: Збandр  Експорт  Імпорт  Сигнали
        """
        logger.info("[UnifiedPipeline] Запуск повного циклу...")
        
        # Еandп 1-2: Збandр data
        collection_result = self.run_data_collection(force_refresh)
        if collection_result["status"] != "success":
            return collection_result
        
        # Еandп 3: Експорт в Colab
        export_result = self.export_to_colab()
        
        # Еandп 4: Імпорт моwhereлей (якщо є)
        import_result = self.import_models_from_colab()
        
        # Еandп 5: Геnotрацandя сигналandв (якщо є моwhereлand)
        if import_result["status"] == "success":
            signals_result = self.run_signal_generation(models)
        else:
            signals_result = {"status": "info", "message": "Очandкування моwhereлей with Colab"}
        
        return {
            "status": "success",
            "collection": collection_result,
            "export": export_result,
            "import": import_result,
            "signals": signals_result
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Сandтус system"""
        accumulated_file = self.accumulated_path / "stage2_accumulated.parquet"
        model_files = list(self.models_path.glob("*model*.parquet"))
        
        status = {
            "accumulated_data": {
                "exists": accumulated_file.exists(),
                "size": accumulated_file.stat().st_size if accumulated_file.exists() else 0,
                "shape": None
            },
            "models": {
                "count": len(model_files),
                "files": [str(f) for f in model_files]
            },
            "directories": {
                "accumulated": str(self.accumulated_path),
                "models": str(self.models_path),
                "exports": str(self.exports_path)
            }
        }
        
        if accumulated_file.exists():
            df = pd.read_parquet(accumulated_file)
            status["accumulated_data"]["shape"] = df.shape
            status["accumulated_data"]["unique_dates"] = df['published_at'].nunique() if 'published_at' in df.columns else 0
        
        return status

# Зручнand функцandї for CLI
def run_data_collection_cmd(force_refresh=False):
    """Команда for withбору data"""
    pipeline = UnifiedPipeline()
    result = pipeline.run_data_collection(force_refresh)
    logger.info(json.dumps(result, indent=2, default=str))

def export_to_colab_cmd():
    """Команда for експорту в Colab"""
    pipeline = UnifiedPipeline()
    result = pipeline.export_to_colab()
    logger.info(json.dumps(result, indent=2, default=str))

def run_signals_cmd(models=None):
    """Команда for геnotрацandї сигналandв"""
    pipeline = UnifiedPipeline()
    result = pipeline.run_signal_generation(models)
    logger.info(json.dumps(result, indent=2, default=str))

def get_status_cmd():
    """Команда for отримання сandтусу"""
    pipeline = UnifiedPipeline()
    status = pipeline.get_status()
    logger.info(json.dumps(status, indent=2, default=str))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Pipeline')
    parser.add_argument('--collect', action='store_true', help='Зandбрати данand')
    parser.add_argument('--export', action='store_true', help='Експортувати в Colab')
    parser.add_argument('--generate', action='store_true', help='Згеnotрувати сигнали')
    parser.add_argument('--status', action='store_true', help='Покаforти сandтус')
    parser.add_argument('--force-refresh', action='store_true', help='Примусове оновлення')
    parser.add_argument('--models', nargs='+', help='Список моwhereлей')
        
    args = parser.parse_args()
        
    if args.collect:
        run_data_collection_cmd(args.force_refresh)
    elif args.export:
        export_to_colab_cmd()
    elif args.generate:
        run_signals_cmd(args.models)
    elif args.status:
        get_status_cmd()
    else:
        # Повний цикл
        pipeline = UnifiedPipeline()
        result = pipeline.run_complete_cycle(args.force_refresh, args.models)
        logger.info(json.dumps(result, indent=2, default=str))