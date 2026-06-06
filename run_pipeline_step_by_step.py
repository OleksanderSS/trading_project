#!/usr/bin/env python3
"""
Скрипт для поетапного запуску та аналізу пайплайну.

Запускає кожен етап окремо з детальним аналізом результатів.
"""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.pipeline.hybrid_orchestrator import HybridOrchestrator

# Налаштування кодування для Windows
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8')

logger = ProjectLogger.get_logger(__name__)


class StepByStepAnalyzer:
    """Аналізатор для поетапного запуску пайплайну."""
    
    def __init__(self, batch_name: str = 'step_by_step_analysis'):
        self.batch_name = batch_name
        self.config_manager = UnifiedConfigManager()
        self.orchestrator = HybridOrchestrator(self.config_manager, batch_name=batch_name)
        self.results = {}
        self.analysis_report = {
            'batch_name': batch_name,
            'start_time': datetime.now().isoformat(),
            'stages': {}
        }
        
        # Налаштування для тестування
        self.tickers = ['SPY']  # Почнемо з одного тікера для швидкості
        self.timeframes = ['15m']  # Почнемо з одного таймфрейму
        
    def print_section_header(self, title: str):
        """Вивести заголовок секції."""
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}\n")
        
    def analyze_dataframe(self, df: pd.DataFrame, stage_name: str):
        """Проаналізувати DataFrame з результатами етапу."""
        if df is None or df.empty:
            print(f"⚠️  {stage_name}: DataFrame порожній або None")
            return {
                'rows': 0,
                'columns': 0,
                'empty': True,
                'dtypes': {},
                'memory_mb': 0
            }
        
        analysis = {
            'rows': len(df),
            'columns': len(df.columns),
            'empty': df.empty,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'column_names': list(df.columns),
            'sample_data': df.head(3).to_dict(orient='records') if len(df) > 0 else []
        }
        
        print(f"📊 {stage_name} Analysis:")
        print(f"   Rows: {analysis['rows']:,}")
        print(f"   Columns: {analysis['columns']}")
        print(f"   Memory: {analysis['memory_mb']:.2f} MB")
        print(f"   Columns: {', '.join(analysis['column_names'][:10])}")
        if len(analysis['column_names']) > 10:
            print(f"   ... and {len(analysis['column_names']) - 10} more")
        
        if not df.empty:
            print(f"\n   Sample data (first 3 rows):")
            for i, row in enumerate(analysis['sample_data'][:3], 1):
                print(f"   Row {i}: {dict(list(row.items())[:5])}")
        
        return analysis
    
    def analyze_dict_result(self, result: dict, stage_name: str):
        """Проаналізувати словник з результатами етапу."""
        if not result:
            print(f"⚠️  {stage_name}: Результат порожній")
            return {'empty': True}
        
        analysis = {
            'keys': list(result.keys()),
            'empty': not result
        }
        
        print(f"📊 {stage_name} Analysis:")
        print(f"   Keys: {', '.join(analysis['keys'])}")
        
        # Перевірка на DataFrame в результаті
        for key, value in result.items():
            if isinstance(value, pd.DataFrame):
                print(f"\n   Sub-analysis for '{key}':")
                self.analyze_dataframe(value, f"{stage_name}.{key}")
        
        return analysis
    
    async def run_stage_0_setup(self):
        """Запустити Stage 0: Setup."""
        self.print_section_header("STAGE 0: SETUP")
        
        try:
            print("🚀 Запуск Stage 0 (Setup)...")
            result = await self.orchestrator.run_local_pipeline(
                tickers=self.tickers,
                timeframes=self.timeframes,
                stages_to_run=[0]
            )
            
            self.results['stage_0'] = result
            self.analysis_report['stages']['stage_0'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'result_analysis': self.analyze_dict_result(result, 'Stage 0')
            }
            
            print(f"✅ Stage 0 завершено успішно")
            return True
            
        except Exception as e:
            print(f"❌ Stage 0 завершено з помилкою: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_report['stages']['stage_0'] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
    
    async def run_stage_1_collection(self):
        """Запустити Stage 1: Data Collection."""
        self.print_section_header("STAGE 1: DATA COLLECTION")
        
        try:
            print("🚀 Запуск Stage 1 (Data Collection)...")
            result = await self.orchestrator.run_local_pipeline(
                tickers=self.tickers,
                timeframes=self.timeframes,
                stages_to_run=[1]
            )
            
            self.results['stage_1'] = result
            self.analysis_report['stages']['stage_1'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'result_analysis': self.analyze_dict_result(result, 'Stage 1')
            }
            
            print(f"✅ Stage 1 завершено успішно")
            return True
            
        except Exception as e:
            print(f"❌ Stage 1 завершено з помилкою: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_report['stages']['stage_1'] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
    
    async def run_stage_2_processing(self):
        """Запустити Stage 2: Processing."""
        self.print_section_header("STAGE 2: PROCESSING")
        
        try:
            print("🚀 Запуск Stage 2 (Processing)...")
            result = await self.orchestrator.run_local_pipeline(
                tickers=self.tickers,
                timeframes=self.timeframes,
                stages_to_run=[2]
            )
            
            self.results['stage_2'] = result
            self.analysis_report['stages']['stage_2'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'result_analysis': self.analyze_dict_result(result, 'Stage 2')
            }
            
            print(f"✅ Stage 2 завершено успішно")
            return True
            
        except Exception as e:
            print(f"❌ Stage 2 завершено з помилкою: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_report['stages']['stage_2'] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
    
    async def run_stage_3_features(self):
        """Запустити Stage 3: Feature Engineering."""
        self.print_section_header("STAGE 3: FEATURE ENGINEERING")
        
        try:
            print("🚀 Запуск Stage 3 (Feature Engineering)...")
            result = await self.orchestrator.run_local_pipeline(
                tickers=self.tickers,
                timeframes=self.timeframes,
                stages_to_run=[3]
            )
            
            self.results['stage_3'] = result
            self.analysis_report['stages']['stage_3'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'result_analysis': self.analyze_dict_result(result, 'Stage 3')
            }
            
            # Особливий аналіз для features_df та targets_df
            if 'results' in result:
                results = result['results']
                if 'features_df' in results:
                    print(f"\n📊 Features DataFrame Analysis:")
                    self.analyze_dataframe(results['features_df'], 'Features')
                if 'targets_df' in results:
                    print(f"\n📊 Targets DataFrame Analysis:")
                    self.analyze_dataframe(results['targets_df'], 'Targets')
            
            print(f"✅ Stage 3 завершено успішно")
            return True
            
        except Exception as e:
            print(f"❌ Stage 3 завершено з помилкою: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_report['stages']['stage_3'] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
    
    async def run_local_pipeline_full(self):
        """Запустити повний локальний пайплайн (stages 0-3)."""
        self.print_section_header("FULL LOCAL PIPELINE (STAGES 0-3)")
        
        try:
            print("🚀 Запуск повного локального пайплайну...")
            result = await self.orchestrator.run_local_pipeline(
                tickers=self.tickers,
                timeframes=self.timeframes,
                stages_to_run=[0, 1, 2, 3]
            )
            
            self.results['local_full'] = result
            self.analysis_report['stages']['local_full'] = {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'result_analysis': self.analyze_dict_result(result, 'Full Local Pipeline')
            }
            
            # Детальний аналіз результатів
            if 'results' in result:
                results = result['results']
                for key, value in results.items():
                    if isinstance(value, pd.DataFrame):
                        print(f"\n📊 Analysis for '{key}':")
                        self.analyze_dataframe(value, key)
            
            print(f"✅ Повний локальний пайплайн завершено успішно")
            return True
            
        except Exception as e:
            print(f"❌ Повний локальний пайплайн завершено з помилкою: {e}")
            import traceback
            traceback.print_exc()
            self.analysis_report['stages']['local_full'] = {
                'status': 'failed',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
            return False
    
    def save_analysis_report(self):
        """Зберегти звіт аналізу."""
        self.analysis_report['end_time'] = datetime.now().isoformat()
        
        report_path = Path(f"analysis_reports/{self.batch_name}_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.analysis_report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📋 Звіт аналізу збережено: {report_path}")
        return report_path
    
    def print_summary(self):
        """Вивести підсумок аналізу."""
        self.print_section_header("ANALYSIS SUMMARY")
        
        print(f"Batch Name: {self.batch_name}")
        print(f"Tickers: {', '.join(self.tickers)}")
        print(f"Timeframes: {', '.join(self.timeframes)}")
        print(f"\nStage Results:")
        
        for stage_name, stage_data in self.analysis_report['stages'].items():
            status = stage_data.get('status', 'unknown')
            status_icon = '✅' if status == 'success' else '❌'
            print(f"  {status_icon} {stage_name}: {status}")
            
            if status == 'failed' and 'error' in stage_data:
                print(f"      Error: {stage_data['error']}")
        
        # Підрахунок успішних етапів
        successful = sum(1 for s in self.analysis_report['stages'].values() if s.get('status') == 'success')
        total = len(self.analysis_report['stages'])
        
        print(f"\nTotal: {successful}/{total} stages successful")


async def main():
    """Головна функція."""
    print("🔍 Step-by-Step Pipeline Analysis")
    print("=" * 80)
    
    analyzer = StepByStepAnalyzer(batch_name='step_by_step_analysis')
    
    # Запуск етапів по черзі
    stages = [
        ('Stage 0 Setup', analyzer.run_stage_0_setup),
        ('Stage 1 Collection', analyzer.run_stage_1_collection),
        ('Stage 2 Processing', analyzer.run_stage_2_processing),
        ('Stage 3 Features', analyzer.run_stage_3_features),
    ]
    
    for stage_name, stage_func in stages:
        success = await stage_func()
        if not success:
            print(f"\n⚠️  {stage_name} не вдалося. Продовження...")
            # Продовжуємо з наступним етапом для повного аналізу
    
    # Збереження звіту
    analyzer.save_analysis_report()
    
    # Вивід підсумку
    analyzer.print_summary()


if __name__ == "__main__":
    asyncio.run(main())
