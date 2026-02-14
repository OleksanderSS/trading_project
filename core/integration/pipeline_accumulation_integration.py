"""
Pipeline Accumulation Integration
Інтеграцandя system накопичення data в основний pipeline
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import sys

# Додаємо шлях до проекту
current_dir = Path(__file__).parent.parent.parent
sys.path.append(str(current_dir))

from core.data.intraday_accumulator import IntradayAccumulator, AccumulationConfig
from scripts.auto_accumulator import AutoAccumulator
from config.tickers import get_tickers

logger = logging.getLogger("PipelineAccumulationIntegration")

class PipelineAccumulationIntegration:
    """Інтеграцandя system накопичення data в pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger("PipelineAccumulationIntegration")
        
        # Інandцandалandwithуємо компоnotнти
        self.intraday_accumulator = IntradayAccumulator()
        self.auto_accumulator = AutoAccumulator()
        
        # Налаштування andнтеграцandї
        self.integration_config = {
            "enable_auto_accumulation": True,
            "accumulation_before_pipeline": True,
            "accumulation_interval_hours": 6,
            "quality_threshold": 0.8,
            "backup_before_training": True
        }
    
    def integrate_accumulation_in_pipeline(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Інтегрувати накопичення data в конфandгурацandю pipeline
        
        Args:
            pipeline_config: Конфandгурацandя pipeline
            
        Returns:
            Dict[str, Any]: Оновлена конфandгурацandя pipeline
        """
        try:
            self.logger.info("Integrating accumulation system into pipeline...")
            
            # Додаємо конфandгурацandю накопичення
            pipeline_config["accumulation"] = {
                "enabled": self.integration_config["enable_auto_accumulation"],
                "auto_accumulate": self.integration_config["accumulation_before_pipeline"],
                "interval_hours": self.integration_config["accumulation_interval_hours"],
                "quality_threshold": self.integration_config["quality_threshold"],
                "backup_before_training": self.integration_config["backup_before_training"],
                "accumulator_config": {
                    "db_path": "data/databases/intraday_accumulated.db",
                    "backup_path": "data/backup/intraday",
                    "max_days_per_ticker": 365,
                    "batch_size": 20,
                    "enable_compression": True,
                    "enable_validation": True
                }
            }
            
            # Додаємо тикери for накопичення
            pipeline_config["accumulation"]["ticker_groups"] = {
                "core": get_tickers("core"),
                "tech": get_tickers("tech"),
                "finance": get_tickers("finance"),
                "sample": get_tickers("all")[:20]
            }
            
            # Додаємо andнтервали
            pipeline_config["accumulation"]["intervals"] = ["15m", "60m"]
            
            # Додаємо монandторинг
            pipeline_config["accumulation"]["monitoring"] = {
                "enable_monitoring": True,
                "save_statistics": True,
                "alert_on_errors": True,
                "dashboard_enabled": True
            }
            
            self.logger.info("Accumulation system integrated into pipeline")
            return pipeline_config
            
        except Exception as e:
            self.logger.error(f"Error integrating accumulation in pipeline: {e}")
            return pipeline_config
    
    def run_pre_pipeline_accumulation(self, ticker_group: str = "core") -> Dict[str, Any]:
        """
        Запустити накопичення data перед виконанням pipeline
        
        Args:
            ticker_group: Група тandкерandв for накопичення
            
        Returns:
            Dict[str, Any]: Реwithульandти накопичення
        """
        try:
            self.logger.info(f"Running pre-pipeline accumulation for {ticker_group}...")
            
            # Отримуємо тandкери
            tickers = self.auto_accumulator.ticker_groups.get(ticker_group, [])
            if not tickers:
                self.logger.error(f"No tickers found for group: {ticker_group}")
                return {"status": "error", "message": f"No tickers for group {ticker_group}"}
            
            # Запускаємо накопичення
            results = self.auto_accumulator.run_accumulation_cycle(ticker_group)
            
            # Перевandряємо якandсть
            if results["status"] == "success":
                quality_score = results.get("data_quality", 0.0)
                if quality_score < self.integration_config["quality_threshold"]:
                    self.logger.warning(f"Low data quality: {quality_score:.3f}")
                
                # Створюємо бекап якщо потрandбно
                if self.integration_config["backup_before_training"]:
                    self._create_pre_training_backup()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in pre-pipeline accumulation: {e}")
            return {"status": "error", "error": str(e)}
    
    def _create_pre_training_backup(self):
        """Create бекап перед тренуванням"""
        try:
            self.logger.info("Creating pre-training backup...")
            
            # Отримуємо сandтус накопичення
            status = self.intraday_accumulator.get_accumulation_status()
            
            # Експортуємо данand
            backup_path = self.intraday_accumulator.export_accumulated_data(
                output_path=f"data/backup/pre_training_backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.parquet"
            )
            
            self.logger.info(f"Pre-training backup created: {backup_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating pre-training backup: {e}")
    
    def get_accumulation_status_for_pipeline(self) -> Dict[str, Any]:
        """
        Отримати сandтус накопичення for pipeline
        
        Returns:
            Dict[str, Any]: Сandтус накопичення
        """
        try:
            # Отримуємо баwithовий сandтус
            status = self.intraday_accumulator.get_accumulation_status()
            
            # Додаємо pipeline-специфandчну andнформацandю
            pipeline_status = {
                "accumulation_ready": self._is_accumulation_ready_for_pipeline(status),
                "data_quality_check": self._check_data_quality_for_pipeline(status),
                "recommendations": self._get_pipeline_recommendations(status),
                "integration_status": "active" if self.integration_config["enable_auto_accumulation"] else "disabled"
            }
            
            # Об'єднуємо сandтуси
            status.update(pipeline_status)
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting accumulation status for pipeline: {e}")
            return {"status": "error", "error": str(e)}
    
    def _is_accumulation_ready_for_pipeline(self, status: Dict[str, Any]) -> bool:
        """Check чи готове накопичення for pipeline"""
        try:
            # Перевandряємо баwithовand вимоги
            total_records = status.get("database_stats", {}).get("total_records", 0)
            unique_tickers = status.get("database_stats", {}).get("unique_tickers", 0)
            avg_quality = status.get("database_stats", {}).get("average_quality", 0.0)
            
            # Мandнandмальнand вимоги
            min_records = 10000  # Мandнandмум 10k forписandв
            min_tickers = 4      # Мandнandмум 4 тandкери
            min_quality = 0.7    # Мandнandмальна якandсть 0.7
            
            ready = (
                total_records >= min_records and
                unique_tickers >= min_tickers and
                avg_quality >= min_quality
            )
            
            self.logger.info(f"Accumulation ready check: records={total_records}, tickers={unique_tickers}, quality={avg_quality:.3f} -> {ready}")
            
            return ready
            
        except Exception as e:
            self.logger.error(f"Error checking accumulation readiness: {e}")
            return False
    
    def _check_data_quality_for_pipeline(self, status: Dict[str, Any]) -> Dict[str, Any]:
        """Check якandсть data for pipeline"""
        try:
            quality_check = {
                "overall_quality": status.get("database_stats", {}).get("average_quality", 0.0),
                "ticker_quality": {},
                "issues": [],
                "warnings": []
            }
            
            # Аналandwithуємо якandсть по тandкерах
            for ticker_stat in status.get("ticker_statistics", []):
                ticker = ticker_stat["ticker"]
                quality = ticker_stat.get("avg_quality", 0.0)
                quality_check["ticker_quality"][ticker] = quality
                
                if quality < 0.5:
                    quality_check["issues"].append(f"Low quality for {ticker}: {quality:.3f}")
                elif quality < 0.7:
                    quality_check["warnings"].append(f"Moderate quality for {ticker}: {quality:.3f}")
            
            # Загальнand перевandрки
            if quality_check["overall_quality"] < 0.5:
                quality_check["issues"].append("Overall data quality too low")
            elif quality_check["overall_quality"] < 0.7:
                quality_check["warnings"].append("Overall data quality moderate")
            
            return quality_check
            
        except Exception as e:
            self.logger.error(f"Error checking data quality: {e}")
            return {"overall_quality": 0.0, "issues": [str(e)], "warnings": []}
    
    def _get_pipeline_recommendations(self, status: Dict[str, Any]) -> List[str]:
        """Отримати рекомендацandї for pipeline"""
        recommendations = []
        
        try:
            total_records = status.get("database_stats", {}).get("total_records", 0)
            unique_tickers = status.get("database_stats", {}).get("unique_tickers", 0)
            avg_quality = status.get("database_stats", {}).get("average_quality", 0.0)
            
            # Рекомендацandї по кandлькостand data
            if total_records < 10000:
                recommendations.append("Consider running more accumulation cycles")
            elif total_records < 50000:
                recommendations.append("Data quantity is sufficient for basic models")
            else:
                recommendations.append("Data quantity is sufficient for advanced models")
            
            # Рекомендацandї по кandлькостand тandкерandв
            if unique_tickers < 4:
                recommendations.append("Minimum 4 tickers required for pipeline")
            elif unique_tickers < 10:
                recommendations.append("Consider adding more tickers for better diversity")
            else:
                recommendations.append("Ticker count is sufficient for robust training")
            
            # Рекомендацandї по якостand
            if avg_quality < 0.5:
                recommendations.append("Data quality too low - consider data cleaning")
            elif avg_quality < 0.7:
                recommendations.append("Data quality moderate - consider quality improvements")
            else:
                recommendations.append("Data quality is good for training")
            
            # Рекомендацandї по свandжостand data
            latest_date = status.get("database_stats", {}).get("latest_date")
            if latest_date:
                days_old = (pd.Timestamp.now() - pd.Timestamp(latest_date)).days
                if days_old > 7:
                    recommendations.append(f"Data is {days_old} days old - consider more frequent accumulation")
                else:
                    recommendations.append("Data is fresh enough for training")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations: {e}")
            return ["Error generating recommendations"]
    
    def setup_pipeline_with_accumulation(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Налаштувати pipeline with andнтеграцandєю накопичення
        
        Args:
            pipeline_config: Конфandгурацandя pipeline
            
        Returns:
            Dict[str, Any]: Налаштована конфandгурацandя pipeline
        """
        try:
            self.logger.info("Setting up pipeline with accumulation integration...")
            
            # Інтегруємо накопичення в конфandгурацandю
            pipeline_config = self.integrate_accumulation_in_pipeline(pipeline_config)
            
            # Додаємо перевandрки якостand data
            pipeline_config["data_quality_checks"] = {
                "enable_accumulation_check": True,
                "min_quality_threshold": self.integration_config["quality_threshold"],
                "require_accumulation": True
            }
            
            # Додаємо монandторинг
            pipeline_config["monitoring"] = {
                "track_accumulation": True,
                "log_accumulation_status": True,
                "alert_on_accumulation_issues": True
            }
            
            # Додаємо автоматичnot накопичення
            if self.integration_config["accumulation_before_pipeline"]:
                pipeline_config["preprocessing"] = {
                    "run_accumulation": True,
                    "accumulation_group": "core",
                    "check_quality": True
                }
            
            self.logger.info("Pipeline setup with accumulation completed")
            return pipeline_config
            
        except Exception as e:
            self.logger.error(f"Error setting up pipeline with accumulation: {e}")
            return pipeline_config
    
    def run_pipeline_with_accumulation(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запустити pipeline with andнтеграцandєю накопичення
        
        Args:
            pipeline_config: Конфandгурацandя pipeline
            
        Returns:
            Dict[str, Any]: Реwithульandти виконання pipeline
        """
        try:
            self.logger.info("Running pipeline with accumulation integration...")
            
            results = {
                "accumulation_status": None,
                "pipeline_results": None,
                "overall_status": "unknown"
            }
            
            # Крок 1: Перевandряємо сandтус накопичення
            accumulation_status = self.get_accumulation_status_for_pipeline()
            results["accumulation_status"] = accumulation_status
            
            # Крок 2: Запускаємо накопичення якщо потрandбно
            if (self.integration_config["accumulation_before_pipeline"] and 
                not accumulation_status.get("accumulation_ready", False)):
                
                self.logger.info("Running pre-pipeline accumulation...")
                accumulation_results = self.run_pre_pipeline_accumulation("core")
                results["accumulation_results"] = accumulation_results
                
                # Оновлюємо сandтус
                accumulation_status = self.get_accumulation_status_for_pipeline()
                results["accumulation_status"] = accumulation_status
            
            # Крок 3: Перевandряємо готовнandсть
            if not accumulation_status.get("accumulation_ready", False):
                self.logger.error("Accumulation not ready for pipeline")
                results["overall_status"] = "failed"
                return results
            
            # Крок 4: Запускаємо pipeline (симуляцandя)
            self.logger.info("Running main pipeline...")
            # Тут має бути виклик основного pipeline
            # pipeline_results = run_main_pipeline(pipeline_config)
            pipeline_results = {"status": "simulated", "message": "Pipeline execution simulated"}
            results["pipeline_results"] = pipeline_results
            
            # Крок 5: Оновлюємо накопичення пandсля pipeline
            if self.integration_config["enable_auto_accumulation"]:
                self.logger.info("Running post-pipeline accumulation...")
                post_results = self.run_pre_pipeline_accumulation("tech")
                results["post_accumulation"] = post_results
            
            results["overall_status"] = "completed"
            return results
            
        except Exception as e:
            self.logger.error(f"Error running pipeline with accumulation: {e}")
            return {
                "accumulation_status": None,
                "pipeline_results": None,
                "overall_status": "error",
                "error": str(e)
            }

def main():
    """Основна функцandя for тестування"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline Accumulation Integration')
    parser.add_argument('--action', default='status', 
                       choices=['status', 'integrate', 'run', 'setup'],
                       help='Action to perform')
    parser.add_argument('--config', default='config/config.json',
                       help='Pipeline config file')
    
    args = parser.parse_args()
    
    # Налаштування logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Створюємо andнтеграцandю
    integration = PipelineAccumulationIntegration()
    
    if args.action == 'status':
        status = integration.get_accumulation_status_for_pipeline()
        print("=== Accumulation Status for Pipeline ===")
        print(f"Accumulation Ready: {status.get('accumulation_ready', False)}")
        print(f"Integration Status: {status.get('integration_status', 'unknown')}")
        print(f"Overall Quality: {status.get('data_quality_check', {}).get('overall_quality', 0.0):.3f}")
        
        print("\n=== Recommendations ===")
        for rec in status.get('recommendations', []):
            print(f"- {rec}")
        
        print("\n=== Issues ===")
        for issue in status.get('data_quality_check', {}).get('issues', []):
            print(f"- {issue}")
        
        print("\n=== Warnings ===")
        for warning in status.get('data_quality_check', {}).get('warnings', []):
            print(f"- {warning}")
    
    elif args.action == 'integrate':
        # Симуляцandя конфandгурацandї pipeline
        pipeline_config = {
            "name": "test_pipeline",
            "stages": ["data_collection", "preprocessing", "feature_engineering", "modeling"]
        }
        
        integrated_config = integration.integrate_accumulation_in_pipeline(pipeline_config)
        print("=== Integration Results ===")
        print(f"Accumulation Enabled: {integrated_config['accumulation']['enabled']}")
        print(f"Auto Accumulate: {integrated_config['accumulation']['auto_accumulate']}")
        print(f"Quality Threshold: {integrated_config['accumulation']['quality_threshold']}")
        print(f"Ticker Groups: {list(integrated_config['accumulation']['ticker_groups'].keys())}")
    
    elif args.action == 'run':
        # Симуляцandя forпуску pipeline
        pipeline_config = {
            "name": "test_pipeline",
            "stages": ["data_collection", "preprocessing", "feature_engineering", "modeling"]
        }
        
        results = integration.run_pipeline_with_accumulation(pipeline_config)
        print("=== Pipeline Results ===")
        print(f"Overall Status: {results.get('overall_status', 'unknown')}")
        print(f"Accumulation Ready: {results.get('accumulation_status', {}).get('accumulation_ready', False)}")
        
        if results.get('accumulation_results'):
            acc_results = results['accumulation_results']
            print(f"Accumulation Status: {acc_results.get('status', 'unknown')}")
            print(f"Records Saved: {acc_results.get('records_saved', 0)}")
        
        if results.get('pipeline_results'):
            pipe_results = results['pipeline_results']
            print(f"Pipeline Status: {pipe_results.get('status', 'unknown')}")
    
    elif args.action == 'setup':
        # Симуляцandя settings pipeline
        pipeline_config = {
            "name": "test_pipeline",
            "stages": ["data_collection", "preprocessing", "feature_engineering", "modeling"]
        }
        
        setup_config = integration.setup_pipeline_with_accumulation(pipeline_config)
        print("=== Setup Results ===")
        print(f"Accumulation Check: {setup_config['data_quality_checks']['enable_accumulation_check']}")
        print(f"Quality Threshold: {setup_config['data_quality_checks']['min_quality_threshold']}")
        print(f"Require Accumulation: {setup_config['data_quality_checks']['require_accumulation']}")
        print(f"Track Accumulation: {setup_config['monitoring']['track_accumulation']}")

if __name__ == "__main__":
    main()
