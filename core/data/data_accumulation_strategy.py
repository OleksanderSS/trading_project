"""
Data Accumulation Strategy
Стратегandя накопичення data for переходу with 4 на 119 тandкерandв
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Додаємо шлях до проекту
import sys
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from config.tickers import get_tickers, get_tickers_dict
from core.data.enhanced_training_data_generator import EnhancedTrainingDataGenerator
from utils.your_working_colab_cell import create_multi_targets

logger = logging.getLogger("DataAccumulationStrategy")

class DataAccumulationStrategy:
    """Стратегandя накопичення data"""
    
    def __init__(self, output_dir: str = "data/accumulation"):
        self.logger = logging.getLogger("DataAccumulationStrategy")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Інandцandалandwithуємо system
        self.data_generator = EnhancedTrainingDataGenerator()
        self.tickers_dict = get_tickers_dict()
        self.all_tickers = get_tickers("all")
        
        # Створюємо директорandї
        self.migration_dir = self.output_dir / "migration"
        self.backup_dir = self.output_dir / "backup"
        self.new_data_dir = self.output_dir / "new_data"
        
        for dir_path in [self.migration_dir, self.backup_dir, self.new_data_dir]:
            dir_path.mkdir(exist_ok=True)
        
        self.logger.info("Data Accumulation Strategy initialized")
        self.logger.info(f"Total tickers available: {len(self.all_tickers)}")
    
    def analyze_current_data(self) -> Dict[str, Any]:
        """
        Аналandwithувати поточнand данand
        
        Returns:
            Dict[str, Any]: Реwithульandти аналandwithу
        """
        self.logger.info("[SEARCH] Analyzing current data...")
        
        analysis = {
            "old_data": {},
            "new_requirements": {},
            "migration_plan": {}
        }
        
        # Аналandwithуємо сandрand данand
        old_data_files = [
            "colab_data/stage3_enhanced_targets.parquet",
            "colab_data/proper_2year_dataset.parquet",
            "data/database/current/stage2_latest.parquet"
        ]
        
        for file_path in old_data_files:
            full_path = Path(file_path)
            if full_path.exists():
                try:
                    df = pd.read_parquet(full_path)
                    
                    file_analysis = {
                        "path": file_path,
                        "shape": df.shape,
                        "size_mb": full_path.stat().st_size / 1024**2,
                        "null_percentage": df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100,
                        "date_range": None,
                        "tickers": [],
                        "target_columns": 0
                    }
                    
                    # Аналandwithуємо тandкери
                    if 'ticker' in df.columns:
                        file_analysis["tickers"] = df['ticker'].unique().tolist()
                    else:
                        # Шукаємо тandкери в наwithвах колонок
                        ticker_cols = []
                        for col in df.columns:
                            for ticker in ['SPY', 'QQQ', 'NVDA', 'TSLA']:
                                if ticker.lower() in col.lower():
                                    ticker_cols.append(ticker)
                                    break
                        file_analysis["tickers"] = list(set(ticker_cols))
                    
                    # Аналandwithуємо andргети
                    target_cols = [col for col in df.columns if 'target_' in col]
                    file_analysis["target_columns"] = len(target_cols)
                    
                    # Аналandwithуємо дати
                    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
                        try:
                            file_analysis["date_range"] = {
                                "start": str(df.index.min()),
                                "end": str(df.index.max())
                            }
                        except:
                            pass
                    
                    analysis["old_data"][file_path] = file_analysis
                    
                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path}: {e}")
                    analysis["old_data"][file_path] = {"error": str(e)}
        
        # Вимоги до нових data
        analysis["new_requirements"] = {
            "total_tickers": len(self.all_tickers),
            "target_tickers": self.all_tickers[:20],  # Першand 20 for прandоритету
            "timeframes": ["15m", "60m", "1d"],
            "target_types": ["volatility", "return", "direction", "trend", "risk"],
            "data_period": "2y",
            "max_null_percentage": 10,
            "min_data_points": 1000,
            "required_features": [
                "technical_indicators",
                "universal_targets",
                "multi_timeframe",
                "quality_metrics"
            ]
        }
        
        # План мandграцandї
        analysis["migration_plan"] = self._create_migration_plan(analysis)
        
        return analysis
    
    def _create_migration_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create план мandграцandї"""
        plan = {
            "strategy": "full_rebuild",  # Повна перебудова
            "reasons": [],
            "steps": [],
            "timeline": {},
            "risks": []
        }
        
        # Аналandwithуємо причини
        old_data = analysis.get("old_data", {})
        
        for file_path, file_info in old_data.items():
            if "error" not in file_info:
                # Перевandряємо якandсть data
                if file_info.get("null_percentage", 0) > 20:
                    plan["reasons"].append(f"High null percentage in {file_path}: {file_info['null_percentage']:.1f}%")
                
                # Перевandряємо кandлькandсть тandкерandв
                if len(file_info.get("tickers", [])) < 10:
                    plan["reasons"].append(f"Insufficient tickers in {file_path}: {len(file_info['tickers'])}")
                
                # Перевandряємо роwithмandр data
                if file_info.get("shape", [0])[0] < 5000:
                    plan["reasons"].append(f"Insufficient data points in {file_path}: {file_info['shape'][0]}")
        
        # Додаємо причини for нових вимог
        plan["reasons"].append("Need to support 119 tickers instead of 4")
        plan["reasons"].append("Need universal target system for each model type")
        plan["reasons"].append("Need multi-timeframe data (15m, 60m, 1d)")
        plan["reasons"].append("Need enhanced technical indicators")
        plan["reasons"].append("Need quality metrics and validation")
        
        # Створюємо кроки
        plan["steps"] = [
            {
                "step": 1,
                "name": "Backup existing data",
                "description": "Create backup of all current data files",
                "estimated_time": "5 minutes",
                "priority": "high"
            },
            {
                "step": 2,
                "name": "Download historical data",
                "description": "Download 2 years of data for all 119 tickers",
                "estimated_time": "30-60 minutes",
                "priority": "high"
            },
            {
                "step": 3,
                "name": "Generate multi-timeframe data",
                "description": "Create 15m, 60m, 1d data from daily data",
                "estimated_time": "15-20 minutes",
                "priority": "high"
            },
            {
                "step": 4,
                "name": "Calculate technical indicators",
                "description": "Add 50+ technical indicators for all timeframes",
                "estimated_time": "10-15 minutes",
                "priority": "high"
            },
            {
                "step": 5,
                "name": "Create universal targets",
                "description": "Generate targets for each model type and ticker",
                "estimated_time": "20-30 minutes",
                "priority": "high"
            },
            {
                "step": 6,
                "name": "Quality validation",
                "description": "Validate data quality and fix issues",
                "estimated_time": "10 minutes",
                "priority": "medium"
            },
            {
                "step": 7,
                "name": "Create Colab package",
                "description": "Create optimized dataset for Colab training",
                "estimated_time": "5-10 minutes",
                "priority": "medium"
            }
        ]
        
        # Timeline
        total_time = sum([
            self._parse_time(step["estimated_time"]) 
            for step in plan["steps"]
        ])
        
        plan["timeline"] = {
            "total_estimated_minutes": total_time,
            "total_estimated_hours": total_time / 60,
            "completion_time": datetime.now() + timedelta(minutes=total_time)
        }
        
        # Риски
        plan["risks"] = [
            "API rate limits when downloading data",
            "Memory issues with large datasets",
            "Data quality issues for some tickers",
            "Long processing time for 119 tickers"
        ]
        
        return plan
    
    def _parse_time(self, time_str: str) -> int:
        """Парсити час у хвилини"""
        if "hour" in time_str.lower():
            return int(time_str.split("-")[0]) * 60
        elif "minute" in time_str.lower():
            return int(time_str.split("-")[0])
        else:
            return 10  # Default
    
    def execute_migration(self, 
                          priority_tickers: List[str] = None,
                          max_tickers: int = 20,
                          create_colab_package: bool = True) -> Dict[str, Any]:
        """
        Виконати мandграцandю data
        
        Args:
            priority_tickers: Прandоритетнand тandкери
            max_tickers: Максимальна кandлькandсть тandкерandв for обробки
            create_colab_package: Чи створювати пакет for Colab
            
        Returns:
            Dict[str, Any]: Реwithульandти мandграцandї
        """
        self.logger.info("[START] Starting data migration...")
        
        results = {
            "status": "in_progress",
            "start_time": datetime.now(),
            "steps_completed": [],
            "errors": [],
            "final_dataset_path": None,
            "colab_package_path": None
        }
        
        try:
            # Крок 1: Бекап andснуючих data
            self.logger.info(" Step 1: Creating backup...")
            backup_result = self._create_backup()
            results["steps_completed"].append({
                "step": "backup",
                "status": "completed",
                "result": backup_result
            })
            
            # Крок 2: Виwithначаємо тandкери for обробки
            if priority_tickers is None:
                # Обираємо прandоритетнand тandкери
                priority_categories = ["core_etfs", "tech_stocks", "major_indices"]
                priority_tickers = []
                
                for category in priority_categories:
                    if category in self.tickers_dict:
                        priority_tickers.extend(self.tickers_dict[category][:5])
                
                # Додаємо унandкальнand тandкери
                priority_tickers = list(set(priority_tickers))[:max_tickers]
            
            self.logger.info(f"[TARGET] Processing {len(priority_tickers)} priority tickers: {priority_tickers}")
            
            # Крок 3: Заванandження andсторичних data
            self.logger.info("[DATA] Step 2: Downloading historical data...")
            historical_data = self.data_generator.download_historical_data(
                priority_tickers, 
                period="2y"
            )
            
            if not historical_data:
                raise ValueError("No historical data downloaded")
            
            results["steps_completed"].append({
                "step": "download",
                "status": "completed",
                "tickers_downloaded": len(historical_data)
            })
            
            # Крок 4: Створення покращеного даandсету
            self.logger.info("[TOOL] Step 3-5: Creating enhanced dataset...")
            enhanced_dataset = self.data_generator.create_enhanced_training_dataset(
                tickers=priority_tickers,
                period="2y",
                include_targets=True
            )
            
            # Зберandгаємо основний даandсет
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            dataset_path = self.new_data_dir / f"enhanced_dataset_{timestamp}.parquet"
            enhanced_dataset.to_parquet(dataset_path)
            
            results["steps_completed"].append({
                "step": "enhanced_dataset",
                "status": "completed",
                "dataset_path": str(dataset_path),
                "shape": enhanced_dataset.shape,
                "tickers": enhanced_dataset['ticker'].unique().tolist() if 'ticker' in enhanced_dataset.columns else priority_tickers
            })
            
            results["final_dataset_path"] = str(dataset_path)
            
            # Крок 5: Валandдацandя якостand
            self.logger.info("[OK] Step 6: Validating data quality...")
            quality_report = self._validate_data_quality(enhanced_dataset)
            results["steps_completed"].append({
                "step": "quality_validation",
                "status": "completed",
                "quality_report": quality_report
            })
            
            # Крок 6: Створення пакету for Colab
            if create_colab_package:
                self.logger.info(" Step 7: Creating Colab package...")
                colab_path = self.data_generator.create_colab_ready_dataset(
                    tickers=priority_tickers,
                    max_size_mb=500
                )
                
                results["steps_completed"].append({
                    "step": "colab_package",
                    "status": "completed",
                    "colab_path": colab_path
                })
                
                results["colab_package_path"] = colab_path
            
            # Фandнальнand реwithульandти
            results["status"] = "completed"
            results["end_time"] = datetime.now()
            results["duration"] = (results["end_time"] - results["start_time"]).total_seconds()
            
            self.logger.info(f"[OK] Migration completed in {results['duration']:.1f} seconds")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Migration failed: {e}")
            results["status"] = "failed"
            results["errors"].append(str(e))
            results["end_time"] = datetime.now()
        
        return results
    
    def _create_backup(self) -> Dict[str, Any]:
        """Create бекап andснуючих data"""
        backup_info = {
            "files_backed_up": [],
            "total_size_mb": 0,
            "backup_path": str(self.backup_dir)
        }
        
        # Список fileandв for бекапу
        files_to_backup = [
            "colab_data/stage3_enhanced_targets.parquet",
            "colab_data/proper_2year_dataset.parquet",
            "data/database/current/stage2_latest.parquet",
            "data/database/current/stage2_accumulated_final.parquet"
        ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for file_path in files_to_backup:
            source_path = Path(file_path)
            if source_path.exists():
                try:
                    # Створюємо andм'я бекапу
                    backup_name = f"{source_path.stem}_backup_{timestamp}{source_path.suffix}"
                    backup_path = self.backup_dir / backup_name
                    
                    # Копandюємо file
                    import shutil
                    shutil.copy2(source_path, backup_path)
                    
                    file_size_mb = source_path.stat().st_size / 1024**2
                    backup_info["files_backed_up"].append({
                        "original": str(source_path),
                        "backup": str(backup_path),
                        "size_mb": file_size_mb
                    })
                    backup_info["total_size_mb"] += file_size_mb
                    
                    self.logger.info(f"[OK] Backed up: {file_path} -> {backup_name}")
                    
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to backup {file_path}: {e}")
        
        return backup_info
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Валandдувати якandсть data"""
        quality_report = {
            "overall_score": 0,
            "metrics": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # Роwithмandр data
            total_cells = df.shape[0] * df.shape[1]
            null_cells = df.isnull().sum().sum()
            null_percentage = (null_cells / total_cells) * 100
            
            quality_report["metrics"]["null_percentage"] = null_percentage
            quality_report["metrics"]["total_rows"] = df.shape[0]
            quality_report["metrics"]["total_columns"] = df.shape[1]
            
            # Tickers
            if 'ticker' in df.columns:
                unique_tickers = df['ticker'].nunique()
                quality_report["metrics"]["unique_tickers"] = unique_tickers
                
                if unique_tickers < 5:
                    quality_report["issues"].append(f"Low ticker count: {unique_tickers}")
            
            # Таргети
            target_cols = [col for col in df.columns if 'target_' in col]
            quality_report["metrics"]["target_columns"] = len(target_cols)
            
            if len(target_cols) < 10:
                quality_report["issues"].append(f"Low target count: {len(target_cols)}")
            
            # Таймфрейми
            if 'timeframe' in df.columns:
                unique_timeframes = df['timeframe'].nunique()
                quality_report["metrics"]["unique_timeframes"] = unique_timeframes
            
            # Calculating forгальний бал
            score = 100
            
            if null_percentage > 10:
                score -= (null_percentage - 10) * 2
            
            if quality_report["metrics"].get("unique_tickers", 0) < 5:
                score -= 20
            
            if quality_report["metrics"].get("target_columns", 0) < 10:
                score -= 15
            
            quality_report["overall_score"] = max(0, score)
            
            # Рекомендацandї
            if null_percentage > 5:
                quality_report["recommendations"].append("Consider filling missing values")
            
            if quality_report["metrics"].get("unique_tickers", 0) < 10:
                quality_report["recommendations"].append("Add more tickers for better model training")
            
            if quality_report["metrics"].get("target_columns", 0) < 20:
                quality_report["recommendations"].append("Generate more target types")
            
        except Exception as e:
            quality_report["error"] = str(e)
            quality_report["overall_score"] = 0
        
        return quality_report

def main():
    """Тестування стратегandї накопичення"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Data Accumulation Strategy')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze current data, don\'t migrate')
    parser.add_argument('--tickers', nargs='+', 
                       help='Priority tickers to process')
    parser.add_argument('--max-tickers', type=int, default=20,
                       help='Maximum tickers to process')
    parser.add_argument('--no-colab', action='store_true',
                       help='Don\'t create Colab package')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Створюємо стратегandю
    strategy = DataAccumulationStrategy()
    
    try:
        if args.analyze_only:
            # Тandльки аналandwith
            analysis = strategy.analyze_current_data()
            
            print("[DATA] DATA ANALYSIS RESULTS:")
            print("=" * 50)
            
            for file_path, file_info in analysis["old_data"].items():
                print(f"\n {file_path}:")
                if "error" in file_info:
                    print(f"  [ERROR] Error: {file_info['error']}")
                else:
                    print(f"  [DATA] Shape: {file_info['shape']}")
                    print(f"   Size: {file_info['size_mb']:.1f}MB")
                    print(f"  [SEARCH] Null %: {file_info['null_percentage']:.1f}%")
                    print(f"  [TARGET] Tickers: {file_info['tickers']}")
                    print(f"  [TARGET] Targets: {file_info['target_columns']}")
            
            print(f"\n NEW REQUIREMENTS:")
            reqs = analysis["new_requirements"]
            print(f"  [TARGET] Total tickers: {reqs['total_tickers']}")
            print(f"  [TARGET] Priority tickers: {len(reqs['target_tickers'])}")
            print(f"  [DATA] Timeframes: {reqs['timeframes']}")
            print(f"  [DATA] Target types: {reqs['target_types']}")
            print(f"  [DATA] Data period: {reqs['data_period']}")
            
            print(f"\n[START] MIGRATION PLAN:")
            plan = analysis["migration_plan"]
            print(f"   Strategy: {plan['strategy']}")
            print(f"    Estimated time: {plan['timeline']['total_estimated_minutes']:.0f} minutes")
            print(f"  [NOTE] Reasons: {len(plan['reasons'])}")
            for reason in plan['reasons'][:3]:
                print(f"    - {reason}")
            
        else:
            # Повна мandграцandя
            results = strategy.execute_migration(
                priority_tickers=args.tickers,
                max_tickers=args.max_tickers,
                create_colab_package=not args.no_colab
            )
            
            print("\n[START] MIGRATION RESULTS:")
            print("=" * 50)
            print(f"[DATA] Status: {results['status']}")
            print(f"  Duration: {results.get('duration', 0):.1f} seconds")
            
            for step in results["steps_completed"]:
                print(f"[OK] {step['step']}: {step['status']}")
                if 'dataset_path' in step:
                    print(f"    Dataset: {step['dataset_path']}")
                if 'colab_path' in step:
                    print(f"    Colab: {step['colab_path']}")
            
            if results["errors"]:
                print(f"\n[ERROR] Errors: {len(results['errors'])}")
                for error in results["errors"]:
                    print(f"   - {error}")
            
            if results["final_dataset_path"]:
                print(f"\n Final dataset: {results['final_dataset_path']}")
            
            if results["colab_package_path"]:
                print(f" Colab package: {results['colab_package_path']}")
    
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
