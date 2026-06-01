#!/usr/bin/env python3
"""
Скрипт для перевірки інтеграції enrichers на етапах 0-3 pipeline
Перевіряє чи всі enrichers правильно задіяні та працюють
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Додаємо проект в шлях
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import get_current_config
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CheckEnrichersIntegration")

class EnrichersChecker:
    """Перевіряє інтеграцію enrichers в pipeline"""
    
    def __init__(self):
        self.config_manager = get_current_config()
        self.results = {
            'stage_0': [],
            'stage_1': [],
            'stage_2': [],
            'stage_3': [],
            'issues': [],
            'recommendations': []
        }
        
    def create_sample_data(self) -> pd.DataFrame:
        """Створює тестові дані для перевірки"""
        dates = pd.date_range('2024-01-01', periods=100, freq='15min')
        np.random.seed(42)
        
        data = {
            'datetime': dates,
            'ticker': 'SPY',
            'open': 100 + np.cumsum(np.random.randn(100) * 0.01),
            'high': None,
            'low': None,
            'close': None,
            'volume': np.random.randint(1000, 10000, 100)
        }
        
        df = pd.DataFrame(data)
        df['close'] = df['open'] + np.random.randn(100) * 0.5
        df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(100) * 0.2
        df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(100) * 0.2
        df = df.set_index('datetime')
        
        return df
        
    def create_sample_news_data(self) -> pd.DataFrame:
        """Створює тестові новинні дані"""
        dates = pd.date_range('2024-01-01', periods=20, freq='1h')
        np.random.seed(42)
        
        data = {
            'published_at': dates,
            'title': [f'News headline {i}' for i in range(20)],
            'sentiment_score': np.random.uniform(-1, 1, 20),
            'ticker': ['SPY' if i % 3 == 0 else None for i in range(20)],
            'source': 'test'
        }
        
        return pd.DataFrame(data)
        
    def check_stage_0_setup(self):
        """Перевіряє Stage 0 - Setup"""
        logger.info("🔍 Перевірка Stage 0 - Setup...")
        
        try:
            from src.pipeline.stages.stage_0_setup import Stage0Setup
            
            stage = Stage0Setup(config_manager=self.config_manager, error_handler=None)
            sample_data = self.create_sample_data()
            
            # Перевіряємо чи може stage обробляти дані
            import asyncio
            result = asyncio.run(stage.run(data=sample_data))
            
            self.results['stage_0'].append({
                'component': 'Stage0Setup',
                'status': '✅ PASS',
                'details': f'Оброблено {len(result)} рядків',
                'input_cols': len(sample_data.columns),
                'output_cols': len(result.columns) if hasattr(result, 'columns') else 'N/A'
            })
            
        except Exception:
            logger.exception("❌ Stage 0 Setup Failed")
            self.results['stage_0'].append({
                'component': 'Stage0Setup',
                'status': '❌ FAIL',
                'error': 'Critical failure during stage 0 setup'
            })
            self.results['issues'].append('Stage 0: Failed - see logs for details')
            
    def check_stage_1_collectors(self):
        """Перевіряє Stage 1 - Data Collection"""
        logger.info("🔍 Перевірка Stage 1 - Data Collection...")
        
        collectors_to_check = [
            ('yf_collector', 'YFCollector'),
            ('newsapi_collector', 'NewsAPICollector'), 
            ('fred_collector', 'FredCollector'),
            ('economic_calendar_collector', 'EconomicCalendarCollector')
        ]
        
        for module_name, class_name in collectors_to_check:
            try:
                module_path = f"src.data.collectors.{module_name}"
                module = __import__(module_path, fromlist=[class_name])
                collector_class = getattr(module, class_name)
                
                self.results['stage_1'].append({
                    'component': module_name,
                    'status': '✅ PASS',
                    'details': 'Імпорт успішний'
                })
                
            except Exception as e:
                logger.error(f'Stage 1 - {module_name} import failed: {e}', exc_info=True)
                self.results['stage_1'].append({
                    'component': module_name,
                    'status': '❌ FAIL',
                    'error': str(e)
                })
                self.results['issues'].append(f'Stage 1 - {module_name}: {e}')
                raise
                
    def check_stage_2_cleaning(self):
        """Перевіряє Stage 2 - Data Cleaning"""
        logger.info("🔍 Перевірка Stage 2 - Data Cleaning...")
        
        try:
            from src.pipeline.stages.stage_2_processing import ProcessingStage
            
            stage = ProcessingStage(config_manager=self.config_manager, error_handler=None)
            sample_data = self.create_sample_data()
            
            # Додаємо проблемні дані для тестування
            sample_data.loc[sample_data.index[10:15], 'close'] = np.nan
            sample_data.loc[sample_data.index[20:25], 'volume'] = 0
            
            result = asyncio.run(stage.run(data=sample_data))
            
            # Перевіряємо тип результату
            result_cols = len(result.columns) if hasattr(result, 'columns') else len(result) if isinstance(result, dict) else 0
            self.results['stage_2'].append({
                'component': 'ProcessingStage',
                'status': '✅ PASS',
                'details': f'Очищено {len(result)} рядків',
                'input_cols': len(sample_data.columns),
                'output_cols': result_cols
            })
            
        except Exception as e:
            logger.error(f'Stage 2 processing failed: {e}', exc_info=True)
            self.results['stage_2'].append({
                'component': 'ProcessingStage',
                'status': '❌ FAIL',
                'error': str(e)
            })
            self.results['issues'].append(f'Stage 2: {e}')
            raise
            
    def check_stage_3_features(self):
        """Перевіряє Stage 3 - Feature Engineering з усіма enrichers"""
        logger.info("🔍 Перевірка Stage 3 - Feature Engineering...")
        
        try:
            from src.pipeline.stages.stage_3_feature_engineering import FeatureEngineeringStage
            from unittest.mock import MagicMock
            
            mock_db = MagicMock()
            stage = FeatureEngineeringStage(config_manager=self.config_manager, error_handler=None, db_manager=mock_db)
            sample_data = self.create_sample_data()
            sample_news = self.create_sample_news_data()
            
            # Перевіряємо основні enrichers
            enrichers_to_check = [
                ('technical_analysis_enricher', 'TechnicalAnalysisEnricher'),
                ('sentiment_features_enricher', 'SentimentFeaturesEnricher'),
                ('macro_features_enricher', 'MacroFeaturesEnricher'),
                ('derived_features_enricher', 'DerivedFeaturesEnricher'),
                ('nlp_features_enricher', 'NLPFeaturesEnricher')
            ]
            
            for module_name, class_name in enrichers_to_check:
                try:
                    module_path = f"src.features.enrichers.{module_name}"
                    module = __import__(module_path, fromlist=[class_name])
                    enricher_class = getattr(module, class_name)
                    enricher = enricher_class()
                    
                    # Тестуємо enricher
                    if class_name == 'SentimentFeaturesEnricher':
                        result = enricher._enrich_impl(sample_data, news=sample_news)
                    else:
                        result = enricher._enrich_impl(sample_data)
                    
                    new_features = len(result.columns) - len(sample_data.columns)
                    
                    self.results['stage_3'].append({
                        'component': class_name,
                        'status': '✅ PASS',
                        'details': f'Додано {new_features} нових фіч',
                        'input_cols': len(sample_data.columns),
                        'output_cols': len(result.columns)
                    })
                    
                except Exception as e:
                    logger.error(f'Stage 3 enricher {class_name} failed: {e}', exc_info=True)
                    self.results['stage_3'].append({
                        'component': class_name,
                        'status': '❌ FAIL',
                        'error': str(e)
                    })
                    self.results['issues'].append(f'Stage 3 - {class_name}: {e}')
                    raise
            
            # Перевіряємо повний Stage 3
            try:
                stage_result = asyncio.run(stage.run(data=sample_data, news=sample_news))
                # Перевіряємо тип результату
                result_cols = len(stage_result.columns) if hasattr(stage_result, 'columns') else len(stage_result) if isinstance(stage_result, dict) else 0
                self.results['stage_3'].append({
                    'component': 'FeatureEngineeringStage (Full)',
                    'status': '✅ PASS',
                    'details': f'Повна обробка {len(stage_result)} рядків',
                    'input_cols': len(sample_data.columns),
                    'output_cols': result_cols
                })
            except Exception as e:
                logger.error(f'Stage 3 full run failed: {e}', exc_info=True)
                self.results['stage_3'].append({
                    'component': 'FeatureEngineeringStage (Full)',
                    'status': '❌ FAIL',
                    'error': str(e)
                })
                self.results['issues'].append(f'Stage 3 Full: {e}')
                raise
                
        except Exception as e:
            logger.error(f'Stage 3 overall check failed: {e}', exc_info=True)
            self.results['stage_3'].append({
                'component': 'FeatureEngineeringStage',
                'status': '❌ FAIL',
                'error': str(e)
            })
            self.results['issues'].append(f'Stage 3 Import: {e}')
            raise
            
    def check_adaptive_indicators(self):
        """Перевіряє адаптивні індикатори"""
        logger.info("🔍 Перевірка адаптивних індикаторів...")
        
        try:
            from src.features.utils.modular_adaptive_technical_indicators import ModularAdaptiveTechnicalIndicators
            
            adaptive = ModularAdaptiveTechnicalIndicators()
            sample_data = self.create_sample_data()
            
            result = adaptive.calculate_all_adaptive_indicators(sample_data)
            
            indicators_count = len(result)
            self.results['stage_3'].append({
                'component': 'ModularAdaptiveTechnicalIndicators',
                'status': '✅ PASS',
                'details': f'Розраховано {indicators_count} адаптивних індикаторів',
                'indicators': list(result.keys())
            })
            
        except Exception as e:
            logger.error(f'Adaptive indicators check failed: {e}', exc_info=True)
            self.results['stage_3'].append({
                'component': 'ModularAdaptiveTechnicalIndicators',
                'status': '❌ FAIL',
                'error': str(e)
            })
            self.results['issues'].append(f'Adaptive Indicators: {e}')
            raise
            
    def check_configuration(self):
        """Перевіряє конфігурацію enrichers"""
        logger.info("🔍 Перевірка конфігурації...")
        
        try:
            config = self.config_manager.get_config('unified')
            
            # Перевіряємо чи є конфігурація для enrichers
            enrichment_config = config.get('enrichment', {}) if config else {}
            features_config = config.get('features', {}) if config else {}
            
            self.results['recommendations'].append({
                'type': 'Configuration',
                'message': f'Знайдено {len(enrichment_config)} enriched конфігурацій',
                'details': list(enrichment_config.keys())
            })
            
            # Перевіряємо технічний аналіз
            ta_config = features_config.get('technical', {})
            if ta_config and isinstance(ta_config, dict):
                # В `features.yaml` індикатори мають вкладений ключ `enabled`
                # Також відфільтровуємо не-словникові ключі (наприклад, метадані)
                enabled_indicators = [
                    k for k, v in ta_config.items() 
                    if isinstance(v, dict) and v.get('enabled', False)
                ]
                self.results['recommendations'].append({
                    'type': 'Technical Analysis',
                    'message': f'Увімкнено {len(enabled_indicators)} індикаторів з {len([k for k, v in ta_config.items() if isinstance(v, dict)])}',
                    'details': enabled_indicators
                })
            else:
                self.results['recommendations'].append({
                    'type': 'Technical Analysis',
                    'message': 'Технічний аналіз не налаштований або ключ "technical" відсутній у features'
                })
                
        except Exception as e:
            logger.error(f'Configuration check failed: {e}', exc_info=True)
            self.results['issues'].append(f'Configuration check: {e}')
            raise
            
    def generate_report(self):
        """Генерує звіт про перевірку"""
        print("\n" + "="*80)
        print("📊 ЗВІТ ПРО ІНТЕГРАЦІЮ ENRICHERS (ЕТАПИ 0-3)")
        print("="*80)
        
        # Stage 0
        print(f"\n🎯 STAGE 0 - SETUP")
        print("-" * 40)
        for result in self.results['stage_0']:
            status = result['status']
            print(f"{status} {result['component']}")
            if 'error' in result:
                print(f"   ❌ Помилка: {result['error']}")
            else:
                print(f"   ✅ {result['details']}")
        
        # Stage 1
        print(f"\n📥 STAGE 1 - DATA COLLECTION")
        print("-" * 40)
        for result in self.results['stage_1']:
            status = result['status']
            print(f"{status} {result['component']}")
            if 'error' in result:
                print(f"   ❌ Помилка: {result['error']}")
            else:
                print(f"   ✅ {result['details']}")
        
        # Stage 2
        print(f"\n🧹 STAGE 2 - DATA CLEANING")
        print("-" * 40)
        for result in self.results['stage_2']:
            status = result['status']
            print(f"{status} {result['component']}")
            if 'error' in result:
                print(f"   ❌ Помилка: {result['error']}")
            else:
                print(f"   ✅ {result['details']}")
        
        # Stage 3
        print(f"\n⚙️  STAGE 3 - FEATURE ENGINEERING")
        print("-" * 40)
        for result in self.results['stage_3']:
            status = result['status']
            print(f"{status} {result['component']}")
            if 'error' in result:
                print(f"   ❌ Помилка: {result['error']}")
            else:
                print(f"   ✅ {result['details']}")
                if 'indicators' in result:
                    print(f"   📈 Індикатори: {', '.join(result['indicators'])}")
        
        # Проблеми
        if self.results['issues']:
            print(f"\n⚠️  ПРОБЛЕМИ ({len(self.results['issues'])})")
            print("-" * 40)
            for i, issue in enumerate(self.results['issues'], 1):
                print(f"{i}. {issue}")
        
        # Рекомендації
        if self.results['recommendations']:
            print(f"\n💡 РЕКОМЕНДАЦІЇ ({len(self.results['recommendations'])})")
            print("-" * 40)
            for i, rec in enumerate(self.results['recommendations'], 1):
                print(f"{i}. {rec['message']}")
                if 'details' in rec:
                    print(f"   📋 {rec['details']}")
        
        # Підсумок
        total_checks = sum(len(self.results[key]) for key in ['stage_0', 'stage_1', 'stage_2', 'stage_3'])
        failed_checks = len(self.results['issues'])
        success_rate = ((total_checks - failed_checks) / total_checks * 100) if total_checks > 0 else 0
        
        print(f"\n📈 ПІДСУМОК")
        print("-" * 40)
        print(f"Всього перевірок: {total_checks}")
        print(f"Успішно: {total_checks - failed_checks}")
        print(f"Проблем: {failed_checks}")
        print(f"Рейтинг успішності: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 Інтеграція enrichers в хорошому стані!")
        elif success_rate >= 60:
            print("⚠️  Інтеграція потребує уваги")
        else:
            print("❌ Інтеграція потребує значних виправлень")
            
        return success_rate
        
    def run_all_checks(self):
        """Запускає всі перевірки"""
        logger.info("🚀 Початок перевірки інтеграції enrichers...")
        
        self.check_configuration()
        self.check_stage_0_setup()
        self.check_stage_1_collectors()
        self.check_stage_2_cleaning()
        self.check_stage_3_features()
        self.check_adaptive_indicators()
        
        return self.generate_report()

def main():
    """Головна функція"""
    checker = EnrichersChecker()
    success_rate = checker.run_all_checks()
    
    # Повертаємо код виходу на основі успішності
    if success_rate >= 80:
        return 0
    elif success_rate >= 60:
        return 1
    else:
        return 2

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
