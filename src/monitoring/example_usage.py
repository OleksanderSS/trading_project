"""
Monitoring System Example - Приклад використання системи моніторингу.

Демонструє:
- Ініціалізацію системи моніторингу
- Інтеграцію з пайплайном
- Різні типи моніторингу
- Генерацію звітів

Використовує:
- Реальні дані для тестування
- Симуляцію роботи системи
- Демонстрацію всіх функцій
"""

import time
import random
import numpy as np
from datetime import datetime, timedelta

from src.monitoring.monitoring_system import MonitoringSystem
from src.monitoring.dashboard import MonitoringDashboardGenerator
from src.core.logging.logger import ProjectLogger

def create_sample_config():
    """Створення прикладу конфігурації"""
    return {
        'collection_interval': 5,  # 5 секунд для демонстрації
        'system_health': {
            'cpu_threshold': 80.0,
            'memory_threshold': 85.0,
            'disk_threshold': 90.0,
            'network_timeout': 30,
            'history_size': 20
        },
        'model_performance': {
            'accuracy_threshold': 0.7,
            'mae_threshold': 0.1,
            'drift_threshold': 0.05
        },
        'data_quality': {
            'missing_threshold': 0.05,
            'outlier_threshold': 0.1,
            'consistency_threshold': 0.95
        },
        'alerts': {
            'channels': ['log'],
            'auto_resolve_hours': 1  # 1 година для демонстрації
        },
        'dashboard': {
            'refresh_interval': 2000,  # 2 секунди
            'history_days': 1,
            'auto_save': False,
            'web': {
                'port': 8050,
                'host': 'localhost'
            }
        }
    }

def simulate_model_training(monitoring_system, model_names):
    """Симуляція тренування моделей"""
    logger = ProjectLogger.get_logger("ModelTraining")

    for model_name in model_names:
        try:
            logger.info(f"Starting training for model: {model_name}")

            # Симуляція тренування (затримка)
            time.sleep(random.uniform(1, 3))

            # Генерація випадкових метрик
            accuracy = random.uniform(0.6, 0.95)
            mae = random.uniform(0.01, 0.2)
            precision = random.uniform(0.5, 0.9)
            recall = random.uniform(0.5, 0.9)

            metrics = {
                'accuracy': accuracy,
                'mae': mae,
                'precision': precision,
                'recall': recall,
                'f1_score': 2 * precision * recall / (precision + recall),
                'training_time': random.uniform(10, 300),
                'timestamp': datetime.now().isoformat()
            }

            # Оновлення метрик моніторингу
            monitoring_system.update_model_metrics(model_name, metrics)

            logger.info(f"Model {model_name} trained successfully. Accuracy: {accuracy:.3f}")

            # Симуляція помилки для одного з моделей
            if random.random() < 0.2:  # 20% шанс помилки
                raise Exception(f"Training failed for {model_name}")

        except Exception as e:
            logger.error(f"Model training failed: {e}")

            # Створення сповіщення про помилку
            monitoring_system.alert_manager.process_alert({
                'id': f'training_error_{model_name}_{int(time.time())}',
                'monitor': 'model_training',
                'message': f'Model training failed for {model_name}: {e}',
                'severity': 'error',
                'timestamp': datetime.now().isoformat(),
                'details': {
                    'model_name': model_name,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                }
            })

def simulate_data_processing(monitoring_system, data_sources):
    """Симуляція обробки даних"""
    logger = ProjectLogger.get_logger("DataProcessing")

    for source_name in data_sources:
        try:
            logger.info(f"Processing data from source: {source_name}")

            # Симуляція обробки даних
            time.sleep(random.uniform(0.5, 2))

            # Генерація звіту якості даних
            total_rows = random.randint(1000, 10000)
            missing_count = random.randint(0, int(total_rows * 0.1))
            completeness = (total_rows - missing_count) / total_rows

            outlier_count = random.randint(0, int(total_rows * 0.05))
            consistency_score = random.uniform(0.85, 0.99)

            quality_report = {
                'completeness': completeness,
                'total_rows': total_rows,
                'missing_count': missing_count,
                'missing_percentage': missing_count / total_rows,
                'outlier_count': outlier_count,
                'outlier_percentage': outlier_count / total_rows,
                'consistency_score': consistency_score,
                'duplicate_count': random.randint(0, int(total_rows * 0.02)),
                'data_types_consistent': random.random() > 0.1,  # 90% шанс консистентності
                'timestamp': datetime.now().isoformat()
            }

            # Оновлення якості даних
            monitoring_system.update_data_quality(source_name, quality_report)

            logger.info(f"Data processing completed for {source_name}. Completeness: {completeness:.1%}")

        except Exception as e:
            logger.error(f"Data processing failed for {source_name}: {e}")

def simulate_system_load():
    """Симуляція системного навантаження для тестування"""
    # Створюємо тимчасові файли для імітації використання диска
    temp_files = []
    try:
        for i in range(10):
            temp_file = f'/tmp/monitoring_test_{i}.tmp'
            with open(temp_file, 'w') as f:
                # Створюємо файл розміром ~1MB
                f.write('x' * (1024 * 1024))
            temp_files.append(temp_file)

        # Затримка для моніторингу
        time.sleep(2)

    finally:
        # Очищення
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except:
                pass

def run_monitoring_demo():
    """Запуск демонстрації системи моніторингу"""
    logger = ProjectLogger.get_logger("MonitoringDemo")

    logger.info("Starting Monitoring System Demo")

    # Створення конфігурації
    config = create_sample_config()

    # Ініціалізація системи моніторингу
    monitoring = MonitoringSystem(config)
    dashboard_gen = MonitoringDashboardGenerator(monitoring, config.get('dashboard', {}))

    try:
        # Запуск моніторингу
        monitoring.start()
        logger.info("Monitoring system started")

        # Список моделей та джерел даних для симуляції
        model_names = ['price_predictor', 'trend_analyzer', 'risk_model', 'portfolio_optimizer']
        data_sources = ['market_data', 'economic_indicators', 'news_feed', 'social_sentiment']

        # Основний цикл демонстрації
        demo_duration = 60  # 60 секунд
        start_time = time.time()

        iteration = 0
        while time.time() - start_time < demo_duration:
            iteration += 1
            logger.info(f"Demo iteration {iteration}")

            # Симуляція тренування моделей (кожні 15 секунд)
            if iteration % 3 == 0:
                simulate_model_training(monitoring, random.sample(model_names, random.randint(1, 2)))

            # Симуляція обробки даних (кожні 10 секунд)
            if iteration % 2 == 0:
                simulate_data_processing(monitoring, random.sample(data_sources, random.randint(1, 2)))

            # Симуляція системного навантаження (кожні 20 секунд)
            if iteration % 4 == 0:
                logger.info("Simulating system load...")
                simulate_system_load()

            # Виведення статусу кожні 10 секунд
            if iteration % 2 == 0:
                health_report = monitoring.get_health_report()
                logger.info(f"System health: {health_report['system_status']}, "
                          f"Active alerts: {health_report['active_alerts']}")

            # Затримка між ітераціями
            time.sleep(5)

        # Генерація фінального звіту
        logger.info("Generating final monitoring report...")

        # Текстовий звіт
        text_report = dashboard_gen.generate_text_report()
        print("\n" + "="*80)
        print("MONITORING SYSTEM DEMO REPORT")
        print("="*80)
        print(text_report)
        print("="*80)

        # Збереження звіту
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = f'monitoring_demo_report_{timestamp}.txt'
        dashboard_gen.save_current_report(report_file)
        logger.info(f"Report saved to: {report_file}")

        # Виведення зведення дашборду
        summary = dashboard_gen.get_dashboard_summary()
        logger.info(f"Demo completed. Final summary: {summary}")

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        # Зупинка моніторингу
        monitoring.stop()
        logger.info("Monitoring system stopped")

def run_interactive_dashboard_demo():
    """Запуск інтерактивного дашборду (якщо доступний)"""
    logger = ProjectLogger.get_logger("DashboardDemo")

    config = create_sample_config()
    monitoring = MonitoringSystem(config)
    dashboard_gen = MonitoringDashboardGenerator(monitoring, config.get('dashboard', {}))

    try:
        monitoring.start()
        logger.info("Monitoring system started for dashboard demo")

        # Додавання деяких тестових даних
        monitoring.update_model_metrics('demo_model', {
            'accuracy': 0.85,
            'mae': 0.05,
            'timestamp': datetime.now().isoformat()
        })

        monitoring.update_data_quality('demo_source', {
            'completeness': 0.95,
            'total_rows': 5000,
            'missing_count': 250,
            'timestamp': datetime.now().isoformat()
        })

        # Запуск дашборду
        logger.info("Starting dashboard... (Press Ctrl+C to stop)")
        logger.info("If web dashboard is available, open http://localhost:8050")

        dashboard_gen.run_web_dashboard(debug=False)

    except KeyboardInterrupt:
        logger.info("Dashboard demo interrupted")
    except Exception as e:
        logger.error(f"Dashboard demo failed: {e}")
    finally:
        monitoring.stop()

def main():
    """Головна функція демонстрації"""
    import argparse

    parser = argparse.ArgumentParser(description='Monitoring System Demo')
    parser.add_argument('--mode', choices=['full', 'dashboard'],
                       default='full', help='Demo mode')
    parser.add_argument('--duration', type=int, default=60,
                       help='Demo duration in seconds')

    args = parser.parse_args()

    if args.mode == 'dashboard':
        run_interactive_dashboard_demo()
    else:
        # Зміна тривалості демонстрації
        global demo_duration
        demo_duration = args.duration
        run_monitoring_demo()

if __name__ == '__main__':
    # Налаштування логування
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    main()