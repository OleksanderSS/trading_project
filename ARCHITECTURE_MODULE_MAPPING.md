# 🏗️ Архітектура проекту: Мапинг модулів по етапах Pipeline

## 📋 Огляд
Цей документ описує, які модулі залучаються на кожному етапі 7-етапного pipeline trading системи.

---

## ✅ **Validation** модулі

### **Постійне залучення (валідація та перевірка):**
- **Етап 0-7:** `src/validation/`
  - **Core Validation:**
    - `validators.py` - основні валідатори
    - `validation_protocols.py` - протоколи валідації
  - **Specialized Validation:**
    - `time_series_validator.py` - валідація часових рядів
    - `data_leakage_detector.py` - детектор витоку даних

### **Специфічне залучення:**
- **Етап 0:** `validators.py` - перевірка готовності системи
- **Етап 1:** `data_leakage_detector.py` - перевірка даних на витоки
- **Етап 2:** `time_series_validator.py` - валідація часових рядів
- **Етап 3:** `validation_protocols.py` - перевірка фіч
- **Етап 4:** `validators.py` - валідація моделей
- **Етап 5:** `validation_protocols.py` - валідація прогнозів
- **Етап 7:** Всі модулі для фінальної оцінки

### **Функціональність:**
- **Data Validation:** Перевірка якості та цілісності даних
- **Leakage Detection:** Виявлення витоку даних між train/test
- **Time Series Validation:** Специфічна валідація часових рядів
- **Protocol Validation:** Стандартизовані протоколи перевірки
- **Model Validation:** Перевірка якості та надійності моделей
- **Cross-Validation:** Перехресна валідація результатів

### **Інтеграція:**
- **Data:** Інтегрується з усіма етапами обробки даних
- **Models:** Працює з усіма ML моделями
- **Pipeline:** Забезпечує якість на кожному етапі

---

## � **Dashboard** модулі

### **Опосередковане залучення (веб інтерфейс):**
- **Етап 0-7:** `src/dashboard/`
  - **Core Dashboard:**
    - `main_app.py` - головний веб додаток

### **Специфічне залучення:**
- **Всі етапи:** `main_app.py` - візуалізація pipeline
- **Етап 7:** `main_app.py` - відображення результатів

### **Функціональність:**
- **Web Interface:** Веб інтерфейс для керування системою
- **Visualization:** Візуалізація результатів та метрик
- **Real-time Monitoring:** Моніторинг в реальному часі

---

## 🎮 **Simulation** модулі

### **Опосередковане залучення (симуляція):**
- **Етап 4-7:** `src/simulation/`
  - **Core Simulation:**
    - `simulation_engine.py` - двигун симуляції

### **Специфічне залучення:**
- **Етап 4:** `simulation_engine.py` - симуляція тренування
- **Етап 6:** `simulation_engine.py` - симуляція торгівлі
- **Етап 7:** `simulation_engine.py` - оцінка симуляцій

### **Функціональність:**
- **Trading Simulation:** Симуляція торгових стратегій
- **Backtesting:** Тестування на історичних даних
- **Risk Analysis:** Аналіз ризиків

---

## 🧪 **Experiments** модулі

### **Опосередковане залучення (експерименти):**
- **Етап 4-7:** `src/experiments/`
  - **Core Experiments:**
    - `compare_layers.py` - порівняння шарів моделей

### **Специфічне залучення:**
- **Етап 4:** `compare_layers.py` - експерименти з моделями
- **Етап 7:** `compare_layers.py` - аналіз результатів

### **Функціональність:**
- **Model Comparison:** Порівняння архітектур моделей
- **Layer Analysis:** Аналіз ефективності шарів
- **Research:** Дослідницькі експерименти

---

## 🏆 **Trained Models** модулі

### **Опосередковане залучення (навчені моделі):**
- **Етап 5-7:** `src/trained_models/`
  - **Progressive Models:**
    - `progressive/` - прогресивні моделі

### **Специфічне залучення:**
- **Етап 5:** `progressive/` - завантаження навчених моделей
- **Етап 7:** `progressive/` - оцінка навчених моделей

### **Функціональність:**
- **Model Storage:** Зберігання навчених моделей
- **Model Loading:** Завантаження збережених моделей
- **Version Control:** Управління версіями моделей

---

## �️ **Utils** модулі

### **Постійне залучення (утиліти та допоміжні функції):**
- **Етап 0-7:** `src/utils/`
  - **Core Utilities:**
    - `dynamic_module_loader.py` - динамічне завантаження модулів
    - `json_utils.py` - робота з JSON даними
    - `rate_limiter.py` - обмеження швидкості запитів
    - `trading_calendar.py` - торговий календар

### **Специфічне залучення:**
- **Етап 0:** `dynamic_module_loader.py` - завантаження модулів pipeline
- **Етап 1:** `rate_limiter.py` - обмеження запитів до API
- **Етап 1:** `trading_calendar.py` - перевірка торгових днів
- **Етап 2-7:** `json_utils.py` - серіалізація даних
- **Всі етапи:** `dynamic_module_loader.py` - динамічна ініціалізація

### **Функціональність:**
- **Module Loading:** Динамічне завантаження та ініціалізація модулів
- **JSON Handling:** Серіалізація та десеріалізація даних
- **Rate Limiting:** Обмеження швидкості для API запитів
- **Trading Calendar:** Робота з торговими днями та годинами
- **Error Handling:** Централізована обробка помилок

### **Інтеграція:**
- **Pipeline:** `dynamic_module_loader.py` для всіх етапів
- **Data:** `rate_limiter.py` для колекторів даних
- **Trading:** `trading_calendar.py` для торгових операцій

---

## 🎓 **Training** модулі

### **Пряме залучення:**
- **Етап 4 - Modeling:** `src/training/`
  - **Core Training:**
    - `unified_training_manager.py` - уніфікований менеджер тренування
    - `adaptive_training_manager.py` - адаптивний менеджер тренування
    - `progressive_trainer.py` - прогресивний тренер
  - **Specialized Training:**
    - `batch_trainer.py` - пакетне тренування
    - `light_model_trainer.py` - тренування легких моделей
    - `pattern_aware_training.py` - тренування з урахуванням патернів
  - **Execution:**
    - `run_training.py` - виконання тренування

### **Опосередковане залучення:**
- **Етап 5 - Prediction:** Використання тренованих моделей
- **Етап 7 - Evaluation:** Оцінка якості тренування

### **Специфічне залучення:**
- **Етап 4:** `unified_training_manager.py` - основне тренування
- **Етап 4:** `adaptive_training_manager.py` - адаптивне тренування
- **Етап 4:** `progressive_trainer.py` - прогресивне тренування
- **Етап 4:** `batch_trainer.py` - пакетне тренування
- **Етап 4:** `light_model_trainer.py` - швидке тренування
- **Етап 4:** `pattern_aware_training.py` - тренування з патернами

### **Функціональність:**
- **Unified Training:** Єдиний інтерфейс для всіх типів моделей
- **Adaptive Training:** Адаптація під час тренування
- **Progressive Training:** Пошагове тренування моделей
- **Batch Processing:** Ефективна обробка великих даних
- **Pattern Awareness:** Урахування ринкових патернів
- **Light Models:** Швидке тренування простих моделей

### **Інтеграція:**
- **Models:** Працює з усіма моделями з `src/models/`
- **Features:** Використовує фічі з `src/features/`
- **Targets:** Використовує таргети з `src/targets/`

---

## 💼 **Trading** модулі

### **Пряме залучення:**
- **Етап 6 - Trading Execution:** `src/trading/`
  - **Core Trading:**
    - `trading_orchestrator.py` - головний оркестратор торгівлі
    - `trader.py` - основний трейдер
  - **Portfolio Management:**
    - `portfolio_manager.py` - управління портфоліо
    - `virtual_portfolio.py` - віртуальне портфоліо для тестування
  - **Consensus & Filtering:**
    - `consensus_engine.py` - консенсус двигун
    - `post_inference_filter.py` - фільтрація після інференсу

### **Опосередковане залучення:**
- **Етап 5 - Prediction:** `consensus_engine.py` - вибір найкращих прогнозів
- **Етап 7 - Evaluation:** Всі модулі для оцінки торгових результатів

### **Специфічне залучення:**
- **Етап 6:** `trading_orchestrator.py` - виконання всіх торгових операцій
- **Етап 6:** `trader.py` - реалізація торгових стратегій
- **Етап 6:** `portfolio_manager.py` - управління капіталом
- **Етап 6:** `virtual_portfolio.py` - тестування стратегій
- **Етап 6:** `consensus_engine.py` - узгодження прогнозів
- **Етап 6:** `post_inference_filter.py` - фінальна фільтрація сигналів

### **Функціональність:**
- **Trade Execution:** Виконання покупок/продажів
- **Portfolio Management:** Управління капіталом та ризиками
- **Consensus Building:** Побудова консенсусу з різних моделей
- **Signal Filtering:** Фільтрація та валідація торгових сигналів
- **Risk Management:** Управління ризиками та позиціями
- **Virtual Trading:** Тестування стратегій без реальних грошей

### **Інтеграція:**
- **Predictions:** Використовує прогнози з `src/predictions/`
- **Models:** Інтегрується з усіма моделями для сигналів
- **Analytics:** Працює з `performance_analyzer.py`

---

## 🎯 **Targets** модулі

### **Пряме залучення:**
- **Етап 3 - Feature Engineering:** `src/targets/`
  - **Core Orchestrator:**
    - `target_orchestrator.py` - центральний оркестратор таргетів
  - **Calculators:**
    - `calculators/regression_calculator.py` - регресійні таргети
    - `calculators/classification_calculator.py` - класифікаційні таргети
    - `calculators/indicator_prediction_calculator.py` - прогнозування індикаторів

### **Опосередковане залучення:**
- **Етап 4 - Modeling:** Всі таргети для тренування моделей
- **Етап 5 - Prediction:** Таргети для валідації прогнозів
- **Етап 7 - Evaluation:** Оцінка точності прогнозів таргетів

### **Специфічне залучення:**
- **Етап 3:** `target_orchestrator.py` - створення всіх типів таргетів
- **Етап 3:** `regression_calculator.py` - цінові таргети
- **Етап 3:** `classification_calculator.py` - напрямки руху
- **Етап 3:** `indicator_prediction_calculator.py` - технічні індикатори

### **Функціональність:**
- **Target Generation:** Створення цільових змінних для ML
- **Regression Targets:** Прогнозування цін та повернень
- **Classification Targets:** Прогнозування напрямків (up/down/sideways)
- **Indicator Targets:** Прогнозування технічних індикаторів
- **Multi-timeframe:** Підтримка різних таймфреймів

### **Інтеграція:**
- **Features:** Інтегрується з `feature_orchestrator.py`
- **Models:** Працює з усіма ML моделями
- **Pipeline:** Передача таргетів в наступні етапи

---

## 💭 **Sentiment** модулі

### **Опосередковане залучення (аналіз настроїв):**
- **Етап 2-7:** `src/sentiment/`
  - **Core Sentiment:**
    - `sentiment_models.py` - моделі аналізу сентименту
    - `optimized_sentiment.py` - оптимізований аналіз сентименту

### **Специфічне залучення:**
- **Етап 2:** `sentiment_models.py` - аналіз сентименту в новинах
- **Етап 3:** `optimized_sentiment.py` - створення сентимент фіч
- **Етап 4:** `sentiment_models.py` - використання в моделях
- **Етап 6:** `optimized_sentiment.py` - торгові сигнали на основі сентименту
- **Етап 7:** Всі модулі для фінальної оцінки

### **Функціональність:**
- **Sentiment Analysis:** Аналіз тексту новин та повідомлень
- **Optimized Processing:** Ефективна обробка великих обсягів тексту
- **Feature Generation:** Створення сентимент фіч для ML моделей
- **Trading Signals:** Генерація торгових сигналів на основі настроїв

### **Інтеграція:**
- **Data:** Інтегрується з новинними колекторами
- **Features:** Працює з `sentiment_features_enricher.py`
- **Models:** Використовується в ML моделях для прогнозування

---

## � **Scripts** модулі

### **Опосередковане залучення (утиліти та інструменти):**
- **Етап 0-7:** `src/scripts/`
  - **Analysis Scripts:**
    - `analysis/generate_context_rules.py` - генерація контекстних правил
  - **Colab Scripts:**
    - `colab/auto_colab_sync.py` - синхронізація з Google Colab
  - **Config Scripts:**
    - `config/ticker_config_updater.py` - оновлення конфігурації тікерів
  - **Data Scripts:**
    - `data/auto_accumulator.py` - автоматичне накопичення даних
  - **Debug Scripts:**
    - `debug/data_merge_debugger.py` - відлагодження об'єднання даних
  - **Fix Scripts:**
    - `fix/data_fixer.py` - виправлення даних
  - **Modeling Scripts:**
    - `modeling/train_consensus_model.py` - тренування консенсус моделей
  - **Monitoring Scripts:**
    - `monitoring/run_health_check.py` - перевірка здоров'я системи

### **Специфічне залучення:**
- **Розробка:** Всі скрипти для розробки та тестування
- **Етап 0:** `run_health_check.py` - перевірка готовності системи
- **Етап 4:** `train_consensus_model.py` - спеціалізоване тренування
- **Етап 7:** `generate_context_rules.py` - аналіз результатів

### **Примітка:**
Scripts використовуються для розробки, тестування, налаштування та обслуговування системи, а не в production pipeline.

---

## 🧹 **Processing** модулі

### **Пряме залучення:**
- **Етап 2 - Processing:** `src/processing/`
  - **Data Cleaning:**
    - `cleaners.py` - очищення та виправлення даних
    - `price_preprocessor.py` - попередня обробка цін
  - **Data Filtering:**
    - `data_filter.py` - фільтрація та валідація даних
  - **Normalization:**
    - `normalization_manager.py` - нормалізація та стандартизація
  - **Optimization:**
    - `parallel_processor.py` - паралельна обробка даних
    - `sampling.py` - семплювання та ресемплювання

### **Опосередковане залучення:**
- **Етап 1:** `price_preprocessor.py` - попередня обробка сирових даних
- **Етап 3:** `normalization_manager.py` - підготовка фіч для моделей
- **Етап 4:** `parallel_processor.py` - прискорення тренування

### **Специфічне залучення:**
- **Етап 2:** Всі модулі для комплексної обробки даних
- **Етап 3:** `normalization_manager.py` - підготовка даних для фіч
- **Етап 4:** `parallel_processor.py` - оптимізація обчислень

### **Функціональність:**
- **Data Cleaning:** Видалення аномалій, дублікатів, пропущених значень
- **Data Filtering:** Валідація якості даних, фільтрація шуму
- **Normalization:** Scaling, стандартизація, нормалізація
- **Parallel Processing:** Багатопотокова обробка великих обсягів даних
- **Price Processing:** Специфічна обробка фінансових часових рядів

---

## 🔮 **Predictions** модулі

### **Пряме залучення:**
- **Етап 5 - Prediction:** `src/predictions/`
  - **Core Prediction:**
    - `models_predict.py` - універсальний предиктор для всіх моделей
    - `deep_predict.py` - спеціалізований предиктор для deep learning моделей
  - **Utilities:**
    - `prediction_utils.py` - утиліти для прогнозування

### **Опосередковане залучення:**
- **Етап 6 - Trading:** Використання прогнозів для торгових сигналів
- **Етап 7 - Evaluation:** Оцінка якості прогнозів

### **Специфічне залучення:**
- **Етап 5:** `models_predict.py` - прогнозування всіх типів моделей
- **Етап 5:** `deep_predict.py` - LSTM, CNN, Transformer прогнози
- **Етап 6:** `models_predict.py` - генерація торгових сигналів
- **Етап 7:** Оцінка точності прогнозів

### **Функціональність:**
- **Universal Prediction:** Підтримка всіх типів моделей (tree, linear, neural)
- **Deep Learning:** Спеціалізовані функції для LSTM, CNN, Transformer
- **Batch Processing:** Ефективна обробка великих обсягів даних
- **GPU Support:** Підтримка CUDA для прискорення
- **Error Handling:** Робота з missing data та outliers

### **Інтеграція:**
- **Models:** Працює з усіма тренованими моделями
- **Trading:** Інтегрується з торговими виконавцями
- **Metrics:** Працює з `ml_evaluator.py` для оцінки

---

## 🔄 **Pipeline** модулі

### **Пряме залучення (центральний оркестратор):**
- **Етап 0-7:** `src/pipeline/`
  - **Core Orchestrator:**
    - `pipeline_orchestrator.py` - головний оркестратор pipeline
  - **Stages (7 етапів):**
    - `stages/stage_0_setup.py` - налаштування системи
    - `stages/stage_1_collection.py` - збір даних
    - `stages/stage_2_processing.py` - обробка та очищення
    - `stages/stage_3_feature_engineering.py` - створення фіч
    - `stages/stage_4_modeling.py` - моделювання
    - `stages/stage_5_prediction.py` - прогнозування
    - `stages/stage_6_trading_execution.py` - виконання торгівлі
    - `stages/stage_7_evaluation.py` - оцінка результатів
  - **Support:**
    - `stages/base_stage.py` - базовий клас для етапів
    - `stages/stage_manager.py` - менеджер етапів
    - `stages/stage_config.py` - конфігурація етапів
    - `stages/incremental_pipeline.py` - інкрементний pipeline

### **Специфічне залучення:**
- **Етап 0:** `stage_0_setup.py` - ініціалізація системи
- **Етап 1:** `stage_1_collection.py` - збір даних
- **Етап 2:** `stage_2_processing.py` - обробка та очищення
- **Етап 3:** `stage_3_feature_engineering.py` - створення фіч
- **Етап 4:** `stage_4_modeling.py` - моделювання
- **Етап 5:** `stage_5_prediction.py` - прогнозування
- **Етап 6:** `stage_6_trading_execution.py` - торгівля
- **Етап 7:** `stage_7_evaluation.py` - оцінка

### **Функціональність:**
- **Pipeline Orchestration:** Централізоване управління всіма етапами
- **Stage Management:** Гнучке перемикання між етапами
- **Error Handling:** Обробка помилок на рівні pipeline
- **Data Flow:** Передача даних між етапами
- **Configuration:** Управління конфігурацією етапів

---

## �🔍 **Patterns** модулі

### **Опосередковане залучення (аналіз патернів):**
- **Етап 2-7:** `src/patterns/`
  - **Core Analysis:**
    - `pattern_analyzer.py` - аналіз патернів (новини + ціни)
    - `pattern_recognition_adjustment.py` - розпізнавання та коригування патернів
    - `pattern_tuning.py` - тюнинг ваг патернів

### **Специфічне залучення:**
- **Етап 2:** `pattern_analyzer.py` - аналіз патернів в даних
- **Етап 3:** `pattern_recognition_adjustment.py` - коригування фіч на основі патернів
- **Етап 4:** `pattern_tuning.py` - оптимізація ваг патернів для моделей
- **Етап 6:** `pattern_analyzer.py` - використання патернів для торгівлі
- **Етап 7:** Всі модулі для фінальної оцінки

### **Функціональність:**
- **Pattern Detection:** Виявлення ринкових патернів (banking crisis, tech breakthrough, etc.)
- **Pattern Recognition:** Розпізнавання та класифікація патернів
- **Pattern Tuning:** Оптимізація ваг патернів для ML моделей
- **Integration:** Інтеграція з `IAnalyzer` інтерфейсом

### **Інтеграція:**
- **Analytics:** Працює з `model_comparison_analyzer.py`
- **Features:** Інтегрується з `significance_features_enricher.py`
- **Trading:** Використовує патерни для торгових сигналів

---

## ⚡ **Optimization** модулі

### **Опосередковане залучення (оптимізація системи):**
- **Етап 4 - Modeling:** `src/optimization/`
  - **Core:**
    - `base.py` - базовий клас оптимізації
    - `factory.py` - фабрика оптимізаторів
  - **Hyperparameter Optimization:**
    - `hyperparameters/bayesian.py` - Bayesian оптимізація гіперпараметрів
  - **Portfolio Optimization:**
    - `portfolio/optimizer.py` - оптимізація портфоліо

### **Специфічне залучення:**
- **Етап 4:** `bayesian.py` - оптимізація гіперпараметрів моделей
- **Етап 6:** `portfolio/optimizer.py` - оптимізація торгового портфоліо
- **Етап 7:** `factory.py` - вибір оптимальних стратегій

### **Інтеграція:**
- **Models:** Працює з усіма моделями для гіперпараметрів
- **Trading:** Інтегрується з портфоліо менеджером
- **Meta-Learning:** Використовує Bayesian оптимізацію

### **Примітка:**
Optimization модулі забезпечують автоматичне покращення системи через Bayesian оптимізацію та управління портфоліо.

---

## 📈 **Monitoring** модулі

### **Постійне залучення (всі етапи):**
- **Етап 0-7:** `src/monitoring/`
  - **Core:**
    - `base.py` - базовий клас моніторингу
    - `health_hub.py` - центральний хаб здоров'я системи
    - `ml_analytics.py` - ML аналітика та метрики
  - **Infrastructure:**
    - `infrastructure/resource_monitor.py` - моніторинг ресурсів (CPU, RAM, GPU)
  - **Reporting:**
    - `reporting/performance_reports.py` - звіти про продуктивність

### **Специфічне залучення:**
- **Етап 0:** `health_hub.py` - перевірка готовності системи
- **Етап 1-7:** `resource_monitor.py` - моніторинг ресурсів під час виконання
- **Етап 4:** `ml_analytics.py` - аналітика тренування моделей
- **Етап 6:** `performance_reports.py` - звіти про торгівлю
- **Етап 7:** `health_hub.py` - фінальна оцінка здоров'я

### **Функціональність:**
- **Resource Monitoring:** CPU, RAM, GPU використання
- **Health Checks:** Перевірка стану компонентів системи
- **ML Analytics:** Метрики тренування та продуктивності моделей
- **Performance Reports:** Детальні звіти про систему
- **Alerts:** Сповіщення про проблеми та аномалії

---

## 🤖 **Models** модулі

### **Пряме залучення:**
- **Етап 4 - Modeling:** `src/models/`
  - **Core:**
    - `interfaces.py` - базові інтерфейси моделей
    - `factory.py` - фабрика створення моделей
  - **Neural Models (9 модулів):**
    - `neural/lstm_model.py` - LSTM для часових рядів
    - `neural/gru_model.py` - GRU для часових рядів
    - `neural/cnn_model.py` - CNN для патернів
    - `neural/transformer_model.py` - Transformer для attention
    - `neural/tabnet_model.py` - TabNet для табличних даних
    - `neural/mlp_model.py` - Multi-Layer Perceptron
    - `neural/autoencoder_model.py` - Autoencoder
    - `neural/base_neural.py` - база для нейромереж
    - `neural/neural_network_model.py` - загальна нейромережа
  - **Tree Models (4 модулі):**
    - `tree/xgboost_model.py` - XGBoost
    - `tree/lightgbm_model.py` - LightGBM
    - `tree/catboost_model.py` - CatBoost
    - `tree/random_forest_model.py` - Random Forest
  - **Linear Models (3 модулі):**
    - `linear/linear_model.py` - Linear Regression
    - `linear/svm_model.py` - Support Vector Machine
    - `linear/knn_model.py` - K-Nearest Neighbors
  - **Ensemble Models:**
    - `ensemble/ensemble_model.py` - ансамблеві моделі
  - **Dean Models:**
    - `dean/dean_bootstrap_system.py` - Dean bootstrap система
  - **Model Selection (9 модулів):**
    - `model_selector/smart_selector.py` - розумний вибір моделей
    - `model_selector/competence_analyzer.py` - аналіз компетенцій
    - `model_selector/context_prediction_mapper.py` - контекстний мапінг
    - та інші...
  - **Adapters (3 модулі):**
    - `adapters/data_preparation.py` - підготовка даних
    - `adapters/sentiment_integration.py` - інтеграція сентименту
    - `adapters/adapters.py` - загальні адаптери

### **Опосередковане залучення:**
- **Етап 5 - Prediction:** Всі треновані моделі для прогнозування
- **Етап 6 - Trading:** Моделі для генерації торгових сигналів
- **Етап 7 - Evaluation:** Оцінка ефективності моделей

### **Специфічне залучення:**
- **Етап 4:** Тренування всіх типів моделей
- **Етап 5:** Прогнозування через `factory.py`
- **Етап 6:** Сигнали від `model_selector/`

---

## 📊 **Metrics** модулі

### **Пряме залучення:**
- **Етап 7 - Evaluation:** `src/metrics/`
  - **Core:**
    - `calculator.py` - центральний калькулятор метрик
    - `base.py` - базові класи для метрик
  - **Financial Metrics:**
    - `financial/portfolio_metrics.py` - фінансові метрики портфоліо
  - **Model Metrics:**
    - `model/ml_evaluator.py` - ML метрики моделей
  - **Utilities:**
    - `utils/calculation_tools.py` - інструменти розрахунків

### **Опосередковане залучення:**
- **Етап 4 - Modeling:** `model/ml_evaluator.py` - оцінка моделей під час тренування
- **Етап 6 - Trading:** `financial/portfolio_metrics.py` - метрики торгівлі
- **Етап 7 - Evaluation:** Всі метрики для фінальної оцінки

### **Специфічне залучення:**
- **Етап 4:** `ml_evaluator.py` - accuracy, precision, recall, F1
- **Етап 6:** `portfolio_metrics.py` - Sharpe ratio, max drawdown, returns
- **Етап 7:** `calculator.py` - комплексна оцінка системи

### **Інтеграція:**
- **Models:** Працює з усіма тренованими моделями
- **Trading:** Інтегрується з торговими результатами
- **Analytics:** Працює з `performance_analyzer.py`

---

## 🧠 **Meta-Learning** модулі

### **Опосередковане залучення (інтелектуальні компоненти):**
- **Етап 0-7:** `src/meta_learning/`
  - **Core:**
    - `base.py` - базовий клас meta-learning
    - `dean_integration.py` - інтеграція Dean Trading Models
    - `dean_trading_models.py` - Dean Actor-Critic моделі
  - **Awareness:**
    - `awareness/context_engine.py` - контекстна обізнаність системи
  - **Evolution:**
    - `evolution/dual_loops.py` - подвійні цикли навчання
    - `evolution/optimization/bayesian_optimizer.py` - Bayesian оптимізація
  - **Memory:**
    - `memory/diary_engine.py` - щоденник досвіду системи

### **Специфічне залучення:**
- **Етап 0:** `awareness/context_engine.py` - аналіз контексту
- **Етап 4:** `evolution/dual_loops.py` - оптимізація моделей
- **Етап 4:** `evolution/optimization/bayesian_optimizer.py` - гіперпараметри
- **Етап 6:** `memory/diary_engine.py` - збереження торгового досвіду
- **Етап 7:** `dean_integration.py` - аналіз ефективності Dean моделей

### **Інтеграція:**
- **Models:** Інтегрується з усіма ML моделями
- **Analytics:** Працює з `adaptive_confidence_analyzer.py`
- **Ensembling:** Використовує `diary_engine.py` для адаптивних ваг
- **Trading:** `context_engine.py` для інтелектуальних рішень

### **Примітка:**
Meta-Learning забезпечує адаптивність та само-покращення системи на основі накопиченого досвіду.

---

## 🎮 **Main** модулі

### **Пряме залучення:**
- **Етап 0-7:** `src/main/`
  - **Core Orchestrator:**
    - `system_orchestrator.py` - головний оркестратор системи
  - **Operating Modes:**
    - `modes/base.py` - базовий режим роботи
    - `modes/train.py` - режим тренування моделей
    - `modes/predict.py` - режим прогнозування
    - `modes/backtest.py` - режим бектестингу
    - `modes/intelligent.py` - інтелектуальний режим
    - `modes/training_data_pipeline.py` - pipeline тренувальних даних
    - `modes/monster_test.py` - комплексне тестування
    - `modes/web_ui.py` - веб інтерфейс

### **Специфічне залучення:**
- **Етап 0:** `system_orchestrator.py` - ініціалізація системи
- **Етап 1-7:** `modes/` - вибір режиму виконання
- **Етап 4:** `modes/train.py` - тренування моделей
- **Етап 5:** `modes/predict.py` - прогнозування
- **Етап 6:** `modes/backtest.py` - тестування стратегій
- **Розробка:** `modes/monster_test.py` - комплексне тестування
- **UI:** `modes/web_ui.py` - веб інтерфейс

### **Функціональність:**
- **System Orchestrator:** Центральний контролер всієї системи
- **Parallel Execution:** Підтримка паралельної обробки тікерів
- **Mode Selection:** Гнучке перемикання між режимами роботи
- **Error Handling:** Централізована обробка помилок

---

## 🔗 **Integrations** модулі

### **Опосередковане залучення (зовнішні сервіси):**
- **Етап 1-7:** `src/integrations/`
  - **Base:**
    - `base.py` - базовий клас для інтеграцій
  - **Data Integrations:**
    - `data/bigquery_client.py` - Google BigQuery клієнт
  - **Infrastructure Integrations:**
    - `infra/github_actions.py` - GitHub Actions інтеграція

### **Специфічне залучення:**
- **Етап 1:** `bigquery_client.py` - збір даних з BigQuery
- **Етап 0-7:** `github_actions.py` - CI/CD інтеграція
- **Розробка:** `base.py` - основа для нових інтеграцій

### **Примітка:**
Integrations використовуються для підключення зовнішніх сервісів та API, розширюючи можливості системи.

---

## 🎯 **Features** модулі

### **Пряме залучення:**
- **Етап 3 - Feature Engineering:** `src/features/`
  - **Core:**
    - `feature_orchestrator.py` - оркестратор всіх enricher'ів
  - **Enrichers (11 модулів):**
    - `enrichers/context_map_enricher.py` - контекстне збагачення
    - `enrichers/technical_analysis_enricher.py` - технічні індикатори
    - `enrichers/sentiment_features_enricher.py` - сентимент фічі
    - `enrichers/macro_features_enricher.py` - макро економічні фічі
    - `enrichers/time_features_enricher.py` - часові фічі
    - `enrichers/nlp_features_enricher.py` - NLP фічі
    - `enrichers/derived_features_enricher.py` - похідні фічі
    - `enrichers/decay_features_enricher.py` - фічі затухання
    - `enrichers/hype_enricher.py` - hype фічі
    - `enrichers/macro_context_enricher.py` - макро контекст
    - `enrichers/significance_features_enricher.py` - значущі події
  - **Selection (3 модулі):**
    - `selection/smart_selector.py` - 5-методний вибір фіч
    - `selection/enhanced_selector.py` - розширений селектор
    - `selection/volatility_driver_selector.py` - селектор волатильності
  - **NLP Stack:**
    - `nlp/models/finbert_pipeline.py` - FinBERT sentiment analysis
    - `nlp/processors/news_processing.py` - обробка новин
    - `nlp/extractors/entity_extractor.py` - витягнення сутностей
    - `nlp/scoring/news_scorer.py` - скорінг новин

### **Опосередковане залучення:**
- **Етап 2 - Processing:** NLP enrichers для sentiment analysis
- **Етап 4 - Modeling:** Відібрані фічі для тренування моделей
- **Етап 5 - Prediction:** Фічі для прогнозування
- **Етап 6 - Trading:** Фічі для торгових сигналів

### **Інтеграція:**
- **Data:** Використовує дані з етапу 2
- **Targets:** Інтегрується з `target_orchestrator.py`
- **Analytics:** Працює з feature selection аналітикою

---

## 🏭 **Factories** модулі

### **Пряме залучення:**
- **Етап 4 - Modeling:** `src/factories/`
  - **Model Factory:**
    - `model_factory.py` - фабрика для створення всіх типів моделей

### **Функціональність:**
- **Створення моделей:** Автоматична інстанціація 12+ типів моделей
- **Tree моделі:** XGBoost, LightGBM, CatBoost, RandomForest
- **Linear моделі:** Linear, SVM
- **Neural моделі:** LSTM, GRU, CNN, Transformer, TabNet
- **Ensemble моделі:** EnsembleModel
- **Конфігурація:** Інтеграція з UnifiedConfigManager

### **Опосередковане залучення:**
- **Етап 0:** Ініціалізація доступних моделей
- **Етап 4:** Масове створення моделей для тренування
- **Етап 5:** Створення моделей для прогнозування
- **Етап 7:** Створення моделей для фінальної оцінки

### **Інтеграція:**
- **Models:** Імпортує всі моделі з `src/models/`
- **Config:** Використовує `UnifiedConfigManager` для налаштувань
- **Interfaces:** Працює з `BaseModel` інтерфейсом

---

## � **Ensembling** модулі

### **Пряме залучення:**
- **Етап 4 - Modeling:** `src/ensembling/`
  - **Core Ensemble:**
    - `ensemble.py` - основний клас StackedEnsemble з meta-learning
    - `ensemble/ensemble_model.py` - модель ансамблю
  - **Функціональність:**
    - StackedEnsemble - мета-модель для комбінації прогнозів
    - Live Efficiency Weighting через Meta-Learning
    - Ridge regression як meta-learner для запобігання overfitting

### **Опосередковане залучення:**
- **Етап 5 - Prediction:** використання ансамблю для фінальних прогнозів
- **Етап 6 - Trading:** ансамблеві сигнали для торгових рішень
- **Етап 7 - Evaluation:** оцінка ефективності ансамблю проти індивідуальних моделей

### **Інтеграція:**
- **Meta-Learning:** `src/meta_learning/memory/diary_engine.py` - Experience Diary Engine
- **Models:** Інтегрується з усіма тренованими моделями (neural, tree, linear)
- **Analytics:** Працює з `model_comparison_analyzer.py` для оцінки

---

## 🛠️ **DevTools** модулі

### **Опосередковане залучення (розробка та тестування):**
- **Етап 0-7:** `src/devtools/`
  - **System Tools:**
    - `system_validator.py` - валідація системи та компонентів
    - `task_manager.py` - управління завданнями розробки
    - `rule_generator.py` - генерація правил та логіки
  - **Experimentation:**
    - `experimentation/run_hyperparameter_tuning.py` - тюнинг гіперпараметрів
  - **Prototypes:**
    - `prototypes/live_trading_ticker_manager.py` - прототип менеджера тікерів

### **Специфічне залучення:**
- **Етап 0:** `system_validator.py` - перевірка готовності системи
- **Етап 4:** `run_hyperparameter_tuning.py` - оптимізація моделей
- **Етап 6:** `live_trading_ticker_manager.py` - тестування live торгівлі
- **Розробка:** `task_manager.py` - управління завданнями розробки
- **Аналіз:** `rule_generator.py` - генерація торгових правил

### **Примітка:**
DevTools використовуються переважно під час розробки, тестування та експериментів, а не в production pipeline.

---

## � **Data** модулі

### **Пряме залучення:**
- **Етап 1 - Collection:** `src/data/collectors/` (16+ колекторів)
  - **Ринкові дані:**
    - `yf_collector.py` - Yahoo Finance (акції, ETF, крипто)
    - `market_data_collector.py` - загальні ринкові дані
  - **Новини та інформація:**
    - `newsapi_collector.py` - NewsAPI новини
    - `rss_collector.py` - RSS стрічки
    - `google_news_collector.py` - Google новини
    - `huggingface_collector.py` - HuggingFace дані
  - **Економічні дані:**
    - `fred_collector.py` - FRED економічні індикатори
    - `economic_calendar_collector.py` - економічний календар
  - **Корпоративні дані:**
    - `sec_filings_collector.py` - SEC звіти
    - `insider_collector.py` - інсайдерські транзакції
  - **Додаткові джерела:**
    - `bigquery_collector.py` - BigQuery дані
    - `free_google_trends_collector.py` - Google Trends
    - `custom_csv_collector.py` - кастомні CSV файли
    - `local_file_collector.py` - локальні файли
    - `synthetic_generator.py` - синтетичні дані

### **Опосередковане залучення:**
- **Етап 2-7:** `src/data/management/`
  - `data_manager.py` - управління потоками даних
  - `asset_manager.py` - управління активами
  - `data_versioning.py` - версіонування даних

### **Специфічне залучення:**
- **Етап 0:** `collector_factory.py` - фабрика колекторів
- **Етап 1:** `base_collector.py` - базовий клас для всіх колекторів
- **Етап 2-3:** Кешування результатів колекторів через `cache/cache_manager.py`

---

## 🏗️ **Core** модулі

### **Постійне залучення (всі етапи):**
- **Infrastructure:**
  - `logging/logger.py` - логування всіх операцій
  - `error_handling/error_handler.py` - обробка помилок
  - `file_management/file_manager.py` - управління файлами
  - `validation/validators.py` - валідація даних

### **Системні утиліти:**
- **Cache:** `cache/cache_manager.py` - кешування даних (опосередковано)
- **System:** `system/batch_processor.py` - пакетна обробка (опосередковано)
- **Security:** `security/secure_secrets_manager.py` - управління секретами (опосередковано)
- **Cloud:** `cloud/gcs_manager.py` - робота з хмарою (опосередковано)

### **Специфічне залучення:**
- **Етап 0-1:** `system/version_manager.py` - перевірка версій
- **Етап 2-4:** `cache/object_cache.py` - кешування об'єктів
- **Етап 6-7:** `clients/http_client_factory.py` - HTTP запити

---

## 🔍 **Analytics** модулі

### **Пряме залучення:**
- **Етап 4 - Modeling:** `src/analytics/`
  - **Analyzers:**
    - `model_comparison_analyzer.py` - порівняння моделей
    - `adaptive_confidence_analyzer.py` - адаптивна довіра до прогнозів
    - `performance_analyzer.py` - аналіз продуктивності
  - **Arena:**
    - `arena/arena_battle.py` - система битв між моделями
    - `arena/model_tournament.py` - турнірні змагання моделей
    - `arena/battle_scoring.py` - система підрахунку очок в битвах

### **Опосередковане залучення:**
- **Етап 5 - Prediction:** для аналізу якості прогнозів та вибору arena переможців
- **Етап 6 - Trading:** для аналізу торгових рішень та використання arena чемпіонів
- **Етап 7 - Evaluation:** для фінальної оцінки результатів та arena турнірів

---

## 🔍 **Analysis** модулі

### **Пряме залучення:**
- **Етап 2 - Processing:** `src/core/analysis/`
  - `news_impact.py` - аналіз впливу новин на ціни
  - `context_advisor_switch.py` - контекстна оптимізація рішень
  - `adaptive_noise_filter.py` - фільтрація шуму в даних

### **Опосередковане залучення:**
- **Етап 3 - Feature Engineering:** через FeatureOrchestrator
- **Етап 4 - Modeling:** через ModelSelector для вибору моделей на основі аналізу
- **Етап 5 - Prediction:** для пост-аналізу прогнозів

---

## 📊 **Pipeline Етапи та їх модулі**

### **🔧 Stage 0: Setup**
**Прямі модулі:**
- `src/pipeline/stages/stage_0_setup.py`
- `src/config/unified_config_manager.py`
- `src/core/logging/logger.py`
- `src/core/error_handling/error_handler.py`

**Опосередковані:**
- Всі конфігураційні модулі з `src/config/`

---

### **📥 Stage 1: Collection**
**Прямі модулі:**
- `src/pipeline/stages/stage_1_collection.py`
- `src/data/collectors/` (всі 16+ колекторів):
  - `yf_collector.py` - Yahoo Finance дані
  - `fred_collector.py` - економічні дані
  - `newsapi_collector.py` - новини
  - `rss_collector.py` - RSS стрічки
  - `sec_filings_collector.py` - SEC звіти
  - та інші...

**Опосередковані:**
- `src/core/significance_detector.py` - детекція значущих подій
- `src/config/triggers_config.py` - конфігурація тригерів

---

### **🧹 Stage 2: Processing**
**Прямі модулі:**
- `src/pipeline/stages/stage_2_processing.py`
- `src/processing/cleaners.py` - очищення даних
- `src/processing/data_filter.py` - інтелектуальна фільтрація
- `src/processing/normalization_manager.py` - нормалізація
- `src/processing/price_preprocessor.py` - препроцесинг цін

**Analysis модулі (прямі):**
- `src/core/analysis/news_impact.py` - вплив новин
- `src/core/analysis/context_advisor_switch.py` - контекстний радник
- `src/core/analysis/adaptive_noise_filter.py` - фільтрація шуму

**NLP модулі (прямі):**
- `src/features/nlp/models/finbert_pipeline.py` - sentiment analysis
- `src/features/nlp/processors/news_processing.py` - обробка новин

**Опосередковані:**
- `src/validation/validators.py` - валідація даних
- `src/monitoring/infrastructure/resource_monitor.py` - моніторинг

---

### **⚙️ Stage 3: Feature Engineering**
**Прямі модулі:**
- `src/pipeline/stages/stage_3_feature_engineering.py`
- `src/features/feature_orchestrator.py`
- `src/features/selection/smart_selector.py` - 5-методний вибір фіч
- `src/features/enrichers/` (всі 11 enricher'ів):
  - `context_map_enricher.py` - контекстне збагачення
  - `technical_analysis_enricher.py` - технічні індикатори
  - `sentiment_features_enricher.py` - сентимент фічі
  - `macro_features_enricher.py` - макро економічні фічі
  - та інші...

**NLP модулі (опосередковані):**
- `src/features/nlp/extractors/` - витягнення сутностей
- `src/features/nlp/scoring/` - скорінг новин

**Targets модулі (прямі):**
- `src/targets/target_orchestrator.py`
- `src/targets/calculators/` - калькулятори цілей

**Опосередковані:**
- Analysis модулі через FeatureOrchestrator

---

### **🤖 Stage 4: Modeling**
**Прямі модулі:**
- `src/pipeline/stages/stage_4_modeling.py`
- `src/training/unified_training_manager.py` - уніфікований тренер
- `src/models/adapters/data_preparation.py` - підготовка даних
- `src/models/factory.py` - фабрика моделей

**ML модулі (прямі):**
- `src/models/neural/` (9 моделей):
  - `lstm_model.py`, `cnn_model.py`, `transformer_model.py`, etc.
- `src/models/tree/` (4 моделі):
  - `xgboost_model.py`, `lightgbm_model.py`, `catboost_model.py`, etc.
- `src/models/linear/` (3 моделі):
  - `linear_model.py`, `svm_model.py`, `knn_model.py`
- `src/models/ensemble/ensemble_model.py`

**Model Selection (прямі):**
- `src/models/model_selector/smart_selector.py`
- `src/models/model_selector/competence_analyzer.py`

**Arena модулі (прямі):**
- `src/analytics/arena/arena_battle.py` - система битв між моделями
- `src/analytics/arena/model_tournament.py` - турнірні змагання
- `src/analytics/arena/battle_scoring.py` - підрахунок очок

**Analytics модулі (прямі):**
- `src/analytics/analyzers/model_comparison_analyzer.py` - порівняння моделей
- `src/analytics/analyzers/adaptive_confidence_analyzer.py` - адаптивна довіра
- `src/analytics/analyzers/performance_analyzer.py` - аналіз продуктивности

**Optimization модулі (прямі):**
- `src/optimization/hyperparameters/bayesian.py` - оптимізація гіперпараметрів

**Опосередковані:**
- Analysis модулі для вибору оптимальних моделей

---

### **📈 Stage 5: Prediction**
**Прямі модулі:**
- `src/pipeline/stages/stage_5_prediction.py`
- `src/predictions/deep_predict.py` - глибоке прогнозування
- `src/predictions/models_predict.py` - прогнозування моделями
- `src/predictions/prediction_utils.py` - утиліти прогнозування

**Опосередковані:**
- Всі треновані моделі зі Stage 4
- Analytics модулі для пост-аналізу прогнозів
- `src/analytics/analyzers/adaptive_confidence_analyzer.py` - адаптивна довіра
- Arena переможці для вибору найкращих прогнозів

---

### **💰 Stage 6: Trading Execution**
**Прямі модулі:**
- `src/pipeline/stages/stage_6_trading_execution.py`
- `src/trading/trading_orchestrator.py` - оркестратор торгівлі
- `src/trading/consensus_engine.py` - консенсус двигун
- `src/trading/portfolio_manager.py` - менеджер портфоліо
- `src/trading/virtual_portfolio.py` - віртуальний портфоліо

**Risk Management (прямі):**
- `src/trading/post_inference_filter.py` - фільтрація ризиків

**Опосередковані:**
- Prediction модулі для отримання сигналів
- Analytics модулі для аналізу ринкового контексту
- `src/analytics/analyzers/performance_analyzer.py` - аналіз продуктивності торгів
- Arena переможці для реальних торгових рішень

---

### **📊 Stage 7: Evaluation**
**Прямі модулі:**
- `src/pipeline/stages/stage_7_evaluation.py`
- `src/metrics/calculator.py` - калькулятор метрик
- `src/metrics/financial/portfolio_metrics.py` - фінансові метрики
- `src/metrics/model/ml_evaluator.py` - ML метрики

**Arena модулі (прямі):**
- `src/analytics/arena/arena_battle.py` - фінальна битва чемпіонів
- `src/analytics/arena/model_tournament.py` - турнірна таблиця результатів
- `src/analytics/arena/battle_scoring.py` - підсумковий скоринг

**Analytics модулі (прямі):**
- `src/analytics/analyzers/model_comparison_analyzer.py` - фінальне порівняння
- `src/analytics/analyzers/performance_analyzer.py` - аналіз продуктивності
- `src/analytics/analyzers/adaptive_confidence_analyzer.py` - аналіз довіри до моделей

**Analysis модулі (прямі):**
- Повний аналіз результатів торгівлі
- Оцінка ефективності моделей

**Опосередковані:**
- Всі попередні етапи для збору даних для оцінки

---

## 🧠 **Meta-Learning та Intelligence**

### **Пряме залучення (крос-етапне):**
- **Етап 3:** `src/meta_learning/awareness/context_engine.py` - контекстна обізнаність
- **Етап 4:** `src/meta_learning/evolution/dual_loops.py` - подвійні цикли навчання
- **Етап 4:** `src/meta_learning/evolution/optimization/bayesian_optimizer.py`
- **Етап 6:** `src/meta_learning/memory/diary_engine.py` - щоденник досвіду

### **Опосередковане залучення:**
- Всі етапи через meta-learning компоненти

---

## 📈 **Monitoring та Infrastructure**

### **Постійне залучення (всі етапи):**
- `src/monitoring/infrastructure/resource_monitor.py` - моніторинг ресурсів
- `src/monitoring/ml_analytics.py` - ML аналітика
- `src/monitoring/health_hub.py` - здоров'я системи
- `src/core/logging/logger.py` - логування

---

## 🎯 **Детальний мапинг модулів по етапах Pipeline**

### **📋 Етап 0 - Setup**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - ініціалізація pipeline
- `src/pipeline/stages/stage_0_setup.py` - налаштування системи
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/main/modes/monster_test.py` - комплексне тестування
- `src/main/modes/web_ui.py` - веб інтерфейс
- `src/utils/dynamic_module_loader.py` - завантаження модулів
- `src/validation/validators.py` - перевірка готовності системи
- `src/monitoring/infrastructure/resource_monitor.py` - моніторинг ресурсів
- `src/monitoring/health_hub.py` - здоров'я системи
- `src/monitoring/ml_analytics.py` - ML аналітика
- `src/core/logging/logger.py` - логування
- `src/scripts/monitoring/run_health_check.py` - перевірка здоров'я

### **📊 Етап 1 - Collection**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_1_collection.py` - збір даних
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/data/collectors/yf_collector.py` - Yahoo Finance
- `src/data/collectors/newsapi_collector.py` - NewsAPI
- `src/data/collectors/rss_collector.py` - RSS стрічки
- `src/data/collectors/google_news_collector.py` - Google новини
- `src/data/collectors/fred_collector.py` - FRED дані
- `src/data/collectors/economic_calendar_collector.py` - економічний календар
- `src/data/collectors/sec_filings_collector.py` - SEC звіти
- `src/data/collectors/insider_collector.py` - інсайдерські транзакції
- `src/data/collectors/bigquery_collector.py` - BigQuery
- `src/data/collectors/free_google_trends_collector.py` - Google Trends
- `src/data/collectors/huggingface_collector.py` - HuggingFace
- `src/data/collectors/custom_csv_collector.py` - кастомні CSV
- `src/data/collectors/local_file_collector.py` - локальні файли
- `src/data/collectors/synthetic_generator.py` - синтетичні дані
- `src/data/collectors/market_data_collector.py` - ринкові дані
- `src/data/management/data_manager.py` - управління даними
- `src/processing/price_preprocessor.py` - попередня обробка
- `src/utils/rate_limiter.py` - обмеження запитів
- `src/utils/trading_calendar.py` - торговий календар
- `src/validation/data_leakage_detector.py` - перевірка витоків
- `src/scripts/data/auto_accumulator.py` - накопичення даних
- `src/integrations/infra/github_actions.py` - CI/CD
- `src/factories/model_factory.py` - фабрика моделей
- `src/models/interfaces.py` - інтерфейси моделей

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub)
- Processing модулі (cleaners, normalization)
- Analytics модулі (аналіз даних)
- Sentiment модулі (аналіз настроїв)
- Features модулі (NLP stack)
- Models модулі (18+ моделей)
- Ensembling модулі (ансамблі)
- Meta-Learning модулі (контекст, пам'ять)
- Optimization модулі (bayesian, portfolio)
- Patterns модулі (аналіз патернів)
- Predictions модулі (прогнозування)
- Training модулі (тренування)
- Targets модулі (таргети)
- Trading модулі (торгівля)
- Metrics модулі (метрики)
- Validation модулі (валідація)

### **🧹 Етап 2 - Processing**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_2_processing.py` - обробка даних
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/processing/cleaners.py` - очищення даних
- `src/processing/data_filter.py` - фільтрація даних
- `src/processing/normalization_manager.py` - нормалізація
- `src/processing/parallel_processor.py` - паралельна обробка
- `src/processing/sampling.py` - семплювання
- `src/core/analysis/news_impact.py` - аналіз впливу новин
- `src/core/analysis/context_advisor_switch.py` - контекстна оптимізація
- `src/core/analysis/adaptive_noise_filter.py` - фільтрація шуму
- `src/features/nlp/models/finbert_pipeline.py` - FinBERT sentiment
- `src/features/nlp/processors/news_processing.py` - обробка новин
- `src/sentiment/sentiment_models.py` - аналіз сентименту
- `src/utils/json_utils.py` - серіалізація даних
- `src/validation/time_series_validator.py` - валідація часових рядів
- `src/patterns/pattern_analyzer.py` - аналіз патернів

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub)
- Analytics модулі (аналіз якості)
- Features модулі (enrichers, selection)
- Models модулі (підготовка даних)
- Ensembling модулі (підготовка)
- Meta-Learning модулі (контекст)
- Optimization модулі (підготовка)
- Patterns модулі (аналіз)
- Predictions модулі (підготовка)
- Training модулі (підготовка)
- Targets модулі (підготовка)
- Trading модулі (підготовка)
- Metrics модулі (аналіз)
- Validation модулі (якість)

### **🎯 Етап 3 - Feature Engineering**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_3_feature_engineering.py` - створення фіч
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/features/feature_orchestrator.py` - оркестратор фіч
- `src/features/enrichers/context_map_enricher.py` - контекстне збагачення
- `src/features/enrichers/technical_analysis_enricher.py` - технічні індикатори
- `src/features/enrichers/sentiment_features_enricher.py` - сентимент фічі
- `src/features/enrichers/macro_features_enricher.py` - макро фічі
- `src/features/enrichers/time_features_enricher.py` - часові фічі
- `src/features/enrichers/nlp_features_enricher.py` - NLP фічі
- `src/features/enrichers/derived_features_enricher.py` - похідні фічі
- `src/features/enrichers/decay_features_enricher.py` - фічі затухання
- `src/features/enrichers/hype_enricher.py` - hype фічі
- `src/features/enrichers/macro_context_enricher.py` - макро контекст
- `src/features/enrichers/significance_features_enricher.py` - значущі події
- `src/features/selection/smart_selector.py` - вибір фіч
- `src/features/selection/enhanced_selector.py` - розширений вибір
- `src/features/selection/volatility_driver_selector.py` - волатильність
- `src/features/nlp/extractors/entity_extractor.py` - витягнення сутностей
- `src/features/nlp/scoring/news_scorer.py` - скорінг новин
- `src/sentiment/optimized_sentiment.py` - оптимізований сентимент
- `src/processing/normalization_manager.py` - підготовка фіч
- `src/targets/target_orchestrator.py` - оркестратор таргетів
- `src/targets/calculators/regression_calculator.py` - регресійні таргети
- `src/targets/calculators/classification_calculator.py` - класифікаційні таргети
- `src/targets/calculators/indicator_prediction_calculator.py` - індикатори
- `src/patterns/pattern_recognition_adjustment.py` - коригування патернів
- `src/validation/validation_protocols.py` - перевірка фіч

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub)
- Analytics модулі (аналіз фіч)
- Models модулі (підготовка до тренування)
- Ensembling модулі (підготовка)
- Meta-Learning модулі (контекст)
- Optimization модулі (підготовка)
- Patterns модулі (коригування)
- Predictions модулі (підготовка)
- Training модулі (підготовка)
- Targets модулі (створення)
- Trading модулі (підготовка)
- Metrics модулі (аналіз)
- Validation модулі (якість)

### **🤖 Етап 4 - Modeling**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_4_modeling.py` - моделювання
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/factories/model_factory.py` - фабрика моделей
- `src/models/interfaces.py` - інтерфейси моделей
- **Neural Models (9 модулів):**
  - `src/models/neural/lstm_model.py` - LSTM
  - `src/models/neural/gru_model.py` - GRU
  - `src/models/neural/cnn_model.py` - CNN
  - `src/models/neural/transformer_model.py` - Transformer
  - `src/models/neural/tabnet_model.py` - TabNet
  - `src/models/neural/mlp_model.py` - MLP
  - `src/models/neural/autoencoder_model.py` - Autoencoder
  - `src/models/neural/base_neural.py` - база нейромереж
  - `src/models/neural/neural_network_model.py` - загальна нейромережа
- **Tree Models (4 модулі):**
  - `src/models/tree/xgboost_model.py` - XGBoost
  - `src/models/tree/lightgbm_model.py` - LightGBM
  - `src/models/tree/catboost_model.py` - CatBoost
  - `src/models/tree/random_forest_model.py` - Random Forest
- **Linear Models (3 модулі):**
  - `src/models/linear/linear_model.py` - Linear Regression
  - `src/models/linear/svm_model.py` - SVM
  - `src/models/linear/knn_model.py` - KNN
- **Ensemble Models:**
  - `src/models/ensemble/ensemble_model.py` - ансамблеві моделі
- **Analytics модулі:**
  - `src/analytics/arena/arena_battle.py` - система битв
  - `src/analytics/arena/model_tournament.py` - турніри
  - `src/analytics/arena/battle_scoring.py` - скоринг
  - `src/analytics/analyzers/model_comparison_analyzer.py` - порівняння моделей
  - `src/analytics/analyzers/adaptive_confidence_analyzer.py` - адаптивна довіра
  - `src/analytics/analyzers/performance_analyzer.py` - аналіз продуктивності
- **Ensembling модулі:**
  - `src/ensembling/ensemble.py` - StackedEnsemble
  - `src/ensembling/ensemble/ensemble_model.py` - модель ансамблю
- **Meta-Learning модулі:**
  - `src/meta_learning/evolution/dual_loops.py` - подвійні цикли
  - `src/meta_learning/evolution/optimization/bayesian_optimizer.py` - Bayesian
- **Optimization модулі:**
  - `src/optimization/hyperparameters/bayesian.py` - Bayesian оптимізація
- **Patterns модулі:**
  - `src/patterns/pattern_tuning.py` - тюнинг ваг
- **Training модулі (7 модулів):**
  - `src/training/unified_training_manager.py` - уніфікований тренер
  - `src/training/adaptive_training_manager.py` - адаптивний тренер
  - `src/training/progressive_trainer.py` - прогресивний тренер
  - `src/training/batch_trainer.py` - пакетний тренер
  - `src/training/light_model_trainer.py` - легкі моделі
  - `src/training/pattern_aware_training.py` - з патернами
  - `src/training/run_training.py` - виконання
- **Processing модулі:**
  - `src/processing/parallel_processor.py` - прискорення
- **Utils модулі:**
  - `src/utils/dynamic_module_loader.py` - завантаження
- **Validation модулі:**
  - `src/validation/validators.py` - валідація моделей

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub, ML analytics)
- Analytics модулі (оцінка моделей)
- Features модулі (використання фіч)
- Models модулі (тренування)
- Ensembling модулі (тренування)
- Meta-Learning модулі (оптимізація)
- Optimization модулі (оптимізація)
- Patterns модулі (тюнинг)
- Predictions модулі (підготовка)
- Targets модулі (використання)
- Trading модулі (підготовка)
- Metrics модулі (оцінка)
- Validation модулі (якість)

### **🔮 Етап 5 - Prediction**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_5_prediction.py` - прогнозування
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/predictions/models_predict.py` - універсальний предиктор
- `src/predictions/deep_predict.py` - deep learning предиктор
- `src/analytics/analyzers/adaptive_confidence_analyzer.py` - адаптивна довіра
- `src/trading/consensus_engine.py` - консенсус двигун

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub, ML analytics)
- Features модулі (використання фіч)
- Models модулі (використання тренованих моделей)
- Ensembling модулі (використання ансамблів)
- Meta-Learning модулі (контекст для прогнозів)
- Patterns модулі (використання патернів)
- Training модулі (використання тренованих моделей)
- Targets модулі (валідація прогнозів)
- Trading модулі (підготовка сигналів)
- Metrics модулі (оцінка прогнозів)
- Validation модулі (якість прогнозів)

### **💼 Етап 6 - Trading Execution**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_6_trading_execution.py` - торгівля
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/trading/trading_orchestrator.py` - оркестратор торгівлі
- `src/trading/trader.py` - основний трейдер
- `src/trading/portfolio_manager.py` - управління портфоліо
- `src/trading/virtual_portfolio.py` - віртуальне портфоліо
- `src/trading/consensus_engine.py` - консенсус двигун
- `src/trading/post_inference_filter.py` - фільтрація сигналів
- `src/analytics/analyzers/performance_analyzer.py` - аналіз торгівлі
- `src/utils/trading_calendar.py` - торговий календар

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub, ML analytics)
- Features модулі (використання фіч для сигналів)
- Models модулі (використання прогнозів)
- Ensembling модулі (використання ансамблевих прогнозів)
- Meta-Learning модулі (контекст для торгівлі)
- Patterns модулі (використання патернів)
- Predictions модулі (використання прогнозів)
- Training модулі (використання моделей)
- Targets модулі (оцінка сигналів)
- Metrics модулі (оцінка торгівлі)
- Validation модулі (якість сигналів)

### **📈 Етап 7 - Evaluation**
**Прямі модулі:**
- `src/pipeline/orchestrator.py` - оркестратор pipeline
- `src/pipeline/stages/stage_7_evaluation.py` - оцінка
- `src/main/system_orchestrator.py` - головний оркестратор
- `src/metrics/calculator.py` - калькулятор метрик
- `src/metrics/financial/portfolio_metrics.py` - фінансові метрики
- `src/metrics/model/ml_evaluator.py` - ML метрики
- **Arena модулі:**
  - `src/analytics/arena/arena_battle.py` - фінальна битва
  - `src/analytics/arena/model_tournament.py` - турнірна таблиця
  - `src/analytics/arena/battle_scoring.py` - підсумковий скоринг
- **Analytics модулі:**
  - `src/analytics/analyzers/model_comparison_analyzer.py` - фінальне порівняння
  - `src/analytics/analyzers/performance_analyzer.py` - фінальний аналіз
  - `src/analytics/analyzers/adaptive_confidence_analyzer.py` - фінальна довіра
- **Analysis модулі:**
  - `src/core/analysis/news_impact.py` - оцінка впливу новин
  - `src/core/analysis/context_advisor_switch.py` - фінальна оптимізація
- **Ensembling модулі:**
  - `src/ensembling/ensemble.py` - фінальна оцінка ансамблю
  - `src/ensembling/ensemble/ensemble_model.py` - оцінка моделі
- **Meta-Learning модулі:**
  - `src/meta_learning/dean_integration.py` - фінальна оцінка Dean моделей
- **Patterns модулі:**
  - `src/patterns/pattern_analyzer.py` - фінальний аналіз
  - `src/patterns/pattern_recognition_adjustment.py` - фінальне коригування
  - `src/patterns/pattern_tuning.py` - фінальний тюнинг
- **Scripts модулі:**
  - `src/scripts/analysis/generate_context_rules.py` - аналіз результатів
- **Validation модулі:**
  - `src/validation/validators.py` - фінальна перевірка
  - `src/validation/validation_protocols.py` - фінальні протоколи

**Опосередковані модулі:**
- Всі Core модулі (logging, error handling, file management)
- Monitoring модулі (resource monitor, health hub, ML analytics)
- Features модулі (аналіз ефективності фіч)
- Models модулі (оцінка ефективності)
- Ensembling модулі (оцінка ансамблів)
- Meta-Learning модулі (оцінка meta-learning)
- Optimization модулі (оцінка оптимізації)
- Patterns модулі (оцінка патернів)
- Predictions модулі (оцінка якості прогнозів)
- Training модулі (оцінка якості тренування)
- Targets модулі (оцінка якості таргетів)
- Trading модулі (оцінка торгових результатів)
- Metrics модулі (розрахунок метрик)
- Validation модулі (фінальна перевірка якості)

---

## 🎯 **Резюме залученості Validation модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `validators.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `validation_protocols.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме |
| `time_series_validator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `data_leakage_detector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Dashboard модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `main_app.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Simulation модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `simulation_engine.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Experiments модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `compare_layers.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Trained Models модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `progressive/` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Utils модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `dynamic_module_loader.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `json_utils.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `rate_limiter.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `trading_calendar.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |

## 🎯 **Резюме залученості Training модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `unified_training_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `adaptive_training_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `progressive_trainer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `batch_trainer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `light_model_trainer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `pattern_aware_training.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `run_training.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Trading модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `trading_orchestrator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `trader.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `portfolio_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `virtual_portfolio.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `consensus_engine.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано |
| `post_inference_filter.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |

## 🎯 **Резюме залученості Targets модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `target_orchestrator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `calculators/regression_calculator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `calculators/classification_calculator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `calculators/indicator_prediction_calculator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Sentiment модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `sentiment_models.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `optimized_sentiment.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |

## 🎯 **Резюме залученості Scripts модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `analysis/generate_context_rules.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `colab/auto_colab_sync.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `config/ticker_config_updater.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `data/auto_accumulator.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `debug/data_merge_debugger.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `fix/data_fixer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `modeling/train_consensus_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `monitoring/run_health_check.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Processing модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `cleaners.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `price_preprocessor.py` | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `data_filter.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `normalization_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `parallel_processor.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `sampling.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Predictions модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `models_predict.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `deep_predict.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано |
| `prediction_utils.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Pipeline модулів**

### **🔄 Core Orchestrator:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `pipeline_orchestrator.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **🎯 Stages (7 етапів):**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `stages/stage_0_setup.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_1_collection.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_2_processing.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_3_feature_engineering.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_4_modeling.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_5_prediction.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_6_trading_execution.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `stages/stage_7_evaluation.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |

### **🔧 Support Modules:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `stages/base_stage.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/stage_config.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `stages/incremental_pipeline.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Patterns модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `pattern_analyzer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |
| `pattern_recognition_adjustment.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `pattern_tuning.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |

## 🎯 **Резюме залученості Optimization модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `factory.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `hyperparameters/bayesian.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `portfolio/optimizer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |

## 🎯 **Резюме залученості Monitoring модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `health_hub.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `ml_analytics.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `infrastructure/resource_monitor.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `reporting/performance_reports.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |

## 🎯 **Резюме залученості Models модулів**

### **🤖 Core & Interfaces:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `interfaces.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `factory.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме |

### **🧠 Neural Models (9 модулів):**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `neural/lstm_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/gru_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/cnn_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/transformer_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/tabnet_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/mlp_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/autoencoder_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `neural/base_neural.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `neural/neural_network_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **🌳 Tree Models (4 модулі):**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `tree/xgboost_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `tree/lightgbm_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `tree/catboost_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `tree/random_forest_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **📈 Linear Models (3 модулі):**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `linear/linear_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `linear/svm_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `linear/knn_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **🎭 Ensemble & Special:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `ensemble/ensemble_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `dean/dean_bootstrap_system.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |

## 🎯 **Резюме залученості Metrics модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `calculator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `financial/portfolio_metrics.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |
| `model/ml_evaluator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `utils/calculation_tools.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Meta-Learning модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `dean_integration.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `dean_trading_models.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `awareness/context_engine.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `evolution/dual_loops.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `evolution/optimization/bayesian_optimizer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `memory/diary_engine.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |

## 🎯 **Резюме залученості Main модулів**

### **🎮 Core Orchestrator:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `system_orchestrator.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **🎯 Operating Modes:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `modes/base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `modes/train.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `modes/predict.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано |
| `modes/backtest.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `modes/intelligent.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `modes/training_data_pipeline.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `modes/monster_test.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `modes/web_ui.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Integrations модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `base.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `data/bigquery_client.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `infra/github_actions.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Features модулів**

### **🎯 Core & Enrichers:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `feature_orchestrator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/context_map_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/technical_analysis_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/sentiment_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/macro_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/time_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/nlp_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/derived_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/decay_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/hype_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/macro_context_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `enrichers/significance_features_enricher.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

### **🧠 Selection:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `selection/smart_selector.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `selection/enhanced_selector.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `selection/volatility_driver_selector.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

### **🔤 NLP Stack:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `nlp/models/finbert_pipeline.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `nlp/processors/news_processing.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `nlp/extractors/entity_extractor.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `nlp/scoring/news_scorer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Factories модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `model_factory.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме |

## 🎯 **Резюме залученості Ensembling модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `ensemble.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `ensemble/ensemble_model.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

## 🎯 **Резюме залученості DevTools модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `system_validator.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `task_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `rule_generator.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `experimentation/run_hyperparameter_tuning.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `prototypes/live_trading_ticker_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |

## 🎯 **Резюме залученості Data модулів**

### **📊 Collectors (Етап 1):**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `yf_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `newsapi_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `rss_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `fred_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `sec_filings_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `insider_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `economic_calendar_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `google_news_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `bigquery_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `free_google_trends_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `huggingface_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `custom_csv_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `local_file_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `synthetic_generator.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `market_data_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

### **🔧 Management:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `data_manager.py` | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `asset_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `data_versioning.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |

### **🏭 Factory:**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `collector_factory.py` | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `base_collector.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

## 🎯 **Резюме залученості Core модулів**

| Модуль | Етап 0 | Етап 1 | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| `logging/logger.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `error_handling/error_handler.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `file_management/file_manager.py` | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме | ✅ Пряме |
| `validation/validators.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `cache/cache_manager.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `system/batch_processor.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `security/secure_secrets_manager.py` | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `system/version_manager.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |
| `clients/http_client_factory.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |

## 🎯 **Резюме залученості Analytics модулів**

| Модуль | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|
| `model_comparison_analyzer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `adaptive_confidence_analyzer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме |
| `performance_analyzer.py` | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ✅ Пряме |
| `arena/arena_battle.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `arena/model_tournament.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `arena/battle_scoring.py` | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |

## 🎯 **Резюме залученості Analysis модулів**

| Модуль | Етап 2 | Етап 3 | Етап 4 | Етап 5 | Етап 6 | Етап 7 |
|--------|--------|--------|--------|--------|--------|--------|
| `news_impact.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ✅ Пряме |
| `context_advisor_switch.py` | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано | ✅ Пряме | ⚪ Опосередковано |
| `adaptive_noise_filter.py` | ✅ Пряме | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано | ⚪ Опосередковано |

**Легенда:**
- ✅ **Пряме залучення** - модуль викликається безпосередньо в коді етапу
- ⚪ **Опосередковане залучення** - модуль використовується через проміжні компоненти

---

## 📝 **Примітки**
- Analysis модулі інтегровані в систему через розширення оригінальних файлів
- Багато модулів мають fallback механізми при помилках
- Система підтримує як локальне, так і Colab виконання
- Meta-learning компоненти працюють крос-етапно для адаптивної оптимізації
