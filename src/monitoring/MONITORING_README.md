# Monitoring System / Система моніторингу

Комплексна система моніторингу для торгового проекту з real-time відстеженням системних метрик, продуктивності моделей та якості даних.

## Основні компоненти / Main Components

### 1. System Health Monitor / Монітор системного здоров'я
- **CPU usage** - навантаження процесора
- **Memory usage** - використання пам'яті
- **Disk usage** - використання диска
- **Network I/O** - мережевий трафік
- **Process count** - кількість процесів

### 2. Model Performance Monitor / Монітор продуктивності моделей
- **Model metrics tracking** - відстеження метрик моделей
- **Drift detection** - виявлення дрейфу моделей
- **Accuracy monitoring** - моніторинг точності
- **Performance alerts** - сповіщення про продуктивність

### 3. Data Quality Monitor / Монітор якості даних
- **Data completeness** - повнота даних
- **Missing values detection** - виявлення пропущених значень
- **Outlier detection** - виявлення викидів
- **Consistency checks** - перевірки консистентності

### 4. Alert Manager / Менеджер сповіщень
- **Multi-channel notifications** - багатоканальні сповіщення
- **Severity levels** - рівні критичності (Info, Warning, Error, Critical)
- **Auto-resolution** - автоматичне вирішення
- **Alert history** - історія сповіщень

### 5. Monitoring Dashboard / Дашборд моніторингу
- **Real-time visualization** - візуалізація в реальному часі
- **Interactive charts** - інтерактивні графіки
- **System status overview** - огляд статусу системи
- **Historical trends** - історичні тенденції

## Архітектура / Architecture

```
MonitoringSystem
├── SystemHealthMonitor
├── ModelPerformanceMonitor
├── DataQualityMonitor
├── AlertManager
└── MonitoringDashboard
    ├── WebDashboard (Dash/Plotly)
    └── TextDashboard
```

## Встановлення та налаштування / Installation & Setup

### Залежності / Dependencies

```bash
pip install psutil plotly dash
```

### Базове використання / Basic Usage

```python
from src.monitoring.monitoring_system import MonitoringSystem

# Створення системи моніторингу
config = {
    'collection_interval': 30,  # секунди
    'system_health': {
        'cpu_threshold': 80.0,
        'memory_threshold': 85.0,
        'disk_threshold': 90.0
    },
    'alerts': {
        'channels': ['log', 'email'],
        'auto_resolve_hours': 24
    }
}

monitoring = MonitoringSystem(config)

# Запуск моніторингу
monitoring.start()

# Отримання даних дашборду
dashboard_data = monitoring.get_dashboard_data()

# Зупинка моніторингу
monitoring.stop()
```

## Конфігурація / Configuration

### Приклад конфігурації / Example Configuration

```yaml
monitoring:
  collection_interval: 30
  system_health:
    cpu_threshold: 80.0
    memory_threshold: 85.0
    disk_threshold: 90.0
    network_timeout: 30
    history_size: 100
  model_performance:
    accuracy_threshold: 0.7
    mae_threshold: 0.1
    drift_threshold: 0.05
  data_quality:
    missing_threshold: 0.05
    outlier_threshold: 0.1
    consistency_threshold: 0.95
  alerts:
    channels: ['log', 'email', 'slack']
    auto_resolve_hours: 24
  dashboard:
    refresh_interval: 5000  # ms
    history_days: 7
    web:
      port: 8050
      host: localhost
      update_interval: 5000
    auto_save: true
    save_interval: 3600  # секунди
    save_path: monitoring_reports
```

## API Reference / Довідка API

### MonitoringSystem

#### Методи / Methods

- `start()` - Запуск системи моніторингу
- `stop()` - Зупинка системи моніторингу
- `get_dashboard_data()` - Отримання даних для дашборду
- `get_health_report()` - Отримання звіту про здоров'я системи
- `update_model_metrics(model_name, metrics)` - Оновлення метрик моделі
- `update_data_quality(source_name, quality_report)` - Оновлення якості даних

### AlertManager

#### Методи / Methods

- `process_alert(alert)` - Обробка нового сповіщення
- `resolve_alert(alert_id, resolution)` - Вирішення сповіщення
- `get_active_alerts(severity=None)` - Отримання активних сповіщень
- `cleanup_old_alerts()` - Очищення старих сповіщень

### MonitoringDashboardGenerator

#### Методи / Methods

- `run_web_dashboard(debug=False)` - Запуск веб-дашборду
- `generate_text_report()` - Генерація текстового звіту
- `save_current_report(filepath=None)` - Збереження поточного звіту
- `get_dashboard_summary()` - Отримання зведення дашборду

## Сповіщення / Alerts

### Типи сповіщень / Alert Types

1. **System Alerts** - Системні сповіщення
   - High CPU usage
   - High memory usage
   - Low disk space
   - Network issues

2. **Model Alerts** - Сповіщення моделей
   - Model drift detected
   - Low accuracy
   - Performance degradation

3. **Data Alerts** - Сповіщення даних
   - Low data completeness
   - High missing values
   - Data inconsistency

### Канали сповіщень / Alert Channels

- **Log** - Запис у лог-файл
- **Email** - Надсилання на email (заглушка)
- **Slack** - Надсилання в Slack (заглушка)

## Дашборд / Dashboard

### Веб-дашборд / Web Dashboard

Запуск веб-дашборду:

```python
from src.monitoring.dashboard import MonitoringDashboardGenerator

dashboard_gen = MonitoringDashboardGenerator(monitoring_system, config)
dashboard_gen.run_web_dashboard(debug=True)
```

Відкрити в браузері: `http://localhost:8050`

### Текстовий дашборд / Text Dashboard

```python
# Генерація звіту
report = dashboard_gen.generate_text_report()
print(report)

# Збереження у файл
dashboard_gen.save_current_report('monitoring_report.txt')
```

## Тестування / Testing

Запуск тестів:

```bash
cd src/monitoring
python -m pytest tests.py -v
```

або

```python
python tests.py
```

## Інтеграція / Integration

### Інтеграція з основною системою / Main System Integration

```python
# В основному пайплайні
from src.monitoring.monitoring_system import MonitoringSystem

class TradingPipeline:
    def __init__(self):
        self.monitoring = MonitoringSystem()
        self.monitoring.start()

    def run_model_training(self, model_name, data):
        try:
            # Тренування моделі
            model = train_model(data)

            # Оновлення метрик моніторингу
            metrics = evaluate_model(model, data)
            self.monitoring.update_model_metrics(model_name, metrics)

        except Exception as e:
            # Сповіщення про помилку
            self.monitoring.alert_manager.process_alert({
                'id': f'training_error_{model_name}',
                'monitor': 'pipeline',
                'message': f'Model training failed: {e}',
                'severity': 'error'
            })

    def process_data(self, source_name, data):
        # Аналіз якості даних
        quality_report = analyze_data_quality(data)
        self.monitoring.update_data_quality(source_name, quality_report)
```

### Інтеграція з логуванням / Logging Integration

```python
import logging
from src.monitoring.monitoring_system import MonitoringSystem

# Налаштування логування
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Створення системи моніторингу
monitoring = MonitoringSystem()

# Інтеграція з логуванням додатку
class MonitoringHandler(logging.Handler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            monitoring.alert_manager.process_alert({
                'id': f'log_error_{record.created}',
                'monitor': 'application',
                'message': f'Application error: {record.getMessage()}',
                'severity': 'error',
                'details': {
                    'level': record.levelname,
                    'module': record.module,
                    'function': record.funcName
                }
            })

# Додавання хендлера
logger.addHandler(MonitoringHandler())
```

## Продуктивність / Performance

### Оптимізації / Optimizations

- **Asynchronous collection** - Асинхронний збір метрик
- **Efficient storage** - Ефективне зберігання даних
- **Configurable intervals** - Налаштовувані інтервали
- **Memory management** - Управління пам'ятю

### Рекомендації / Recommendations

- **Collection interval**: 30-60 секунд для production
- **History size**: 100-1000 записів залежно від потреб
- **Alert thresholds**: Налаштувати під конкретну систему
- **Dashboard refresh**: 5-30 секунд для real-time

## Troubleshooting / Вирішення проблем

### Поширені проблеми / Common Issues

1. **High CPU usage from monitoring**
   - Зменшити `collection_interval`
   - Оптимізувати метрики

2. **Memory leaks**
   - Перевірити `history_size`
   - Очистити старі дані

3. **Missing alerts**
   - Перевірити конфігурацію каналів
   - Перевірити пороги

4. **Dashboard not loading**
   - Перевірити наявність Plotly/Dash
   - Використати текстовий дашборд

### Логування / Logging

Система використовує ProjectLogger для логування. Рівні логування:

- **INFO**: Загальна інформація
- **WARNING**: Попередження
- **ERROR**: Помилки
- **CRITICAL**: Критичні помилки

## Розширення / Extensions

### Додавання нових моніторів / Adding New Monitors

```python
from src.monitoring.monitoring_system import BaseMonitor

class CustomMonitor(BaseMonitor):
    def __init__(self, config=None):
        super().__init__('custom_monitor', config)

    def collect_metrics(self):
        # Ваша логіка збірки метрик
        return {'custom_metric': 42}

# Додавання до системи
monitoring.monitors.append(CustomMonitor())
```

### Додавання нових каналів сповіщень / Adding Alert Channels

```python
class CustomAlertChannel:
    def send_alert(self, alert):
        # Ваша логіка надсилання сповіщень
        pass

# Додавання до менеджера
alert_manager.custom_channels.append(CustomAlertChannel())
```

## Ліцензія / License

Цей проект є частиною торгової системи та використовує відповідну ліцензію.

## Контриб'ютори / Contributors

- Розроблено як частина комплексної торгової системи
- Використовує best practices для моніторингу та алертів