# Modules Usage Report

## 📋 Dataset Builder Analysis

### NewsDatasetBuilder - Логіка відсікання новин

**Файл:** `src/features/news_dataset_builder.py`

**Метод:** `_filter_news_with_sufficient_candles` (рядки 67-117)

**Логіка відсікання:**
1. Класифікує вплив новини через `NewsImpactClassifier`
2. Отримує релевантні комбінації ticker/timeframe
3. Для кожної комбінації перевіряє:
   - Чи є дані для цього timeframe в prices_dict
   - Чи є дані для цього ticker
   - **Quick check for at least 1 candle before and after** (рядки 98-104)
4. Якщо є хоча б одна валідна комбінація, новина залишається
5. Якщо немає жодної валідної комбінації, новина відсікається

**Висновок:** ✅ Логіка відсікання новин без наступних свічок реалізована правильно

**Код:**
```python
# Quick check for at least 1 candle before and after
before = self.seeker.get_candles_before(ticker_prices, news_time, timeframe, n=1)
after = self.seeker.get_candles_after(ticker_prices, news_time, timeframe, n=1)

if before and after:
    has_valid_combination = True
    break
```

---

## 📊 Аналізатори (29 знайдено)

### Analytics Analyzers (13)
1. **adaptive_confidence_analyzer.py** - Адаптивний аналіз впевненості
2. **drift_analyzer.py** - Аналіз дрейфу даних
3. **hedge_fund_analyzer.py** - Аналіз хедж-фондів
4. **model_comparison_analyzer.py** - Порівняння моделей
5. **news_impact_analyzer.py** - Аналіз впливу новин ✅ (використовується)
6. **performance_attribution_analyzer.py** - Атрибуція продуктивності
7. **risk_decomposition_analyzer.py** - Декомпозиція ризиків
8. **shap_analyzer.py** - SHAP explainability
9. **analyzer_registry.py** - Реєстр аналізаторів

### Context Analyzers (4)
10. **macro_context_analyzer.py** - Макро контекст
11. **market_context_analyzer.py** - Ринковий контекст
12. **market_phase_analyzer.py** - Фаза ринку
13. **market_regime_analyzer.py** - Режим ринку

### Features Analysis (2)
14. **importance_stability_analyzer.py** - Стабільність важливості фіч
15. **market_conditions_analyzer.py** - Ринкові умови

### NLP (1)
16. **news_analyzer.py** - Аналіз новин

### Meta Learning (1)
17. **analyzer.py** - Контекстний аналізатор

### Models Analysis (4)
18. **model_analyzer.py** - Аналіз моделей
19. **model_health_analyzer.py** - Здоров'я моделей
20. **overfitting_detection/analyzer.py** - Виявлення overfitting
21. **regime_winner_analyzer.py** - Аналіз переможця режиму

### Models Ensemble (2)
22. **model_correlation_analyzer.py** - Кореляція моделей
23. **weight_stability/analyzer.py** - Стабільність ваг

### Models Monitoring (1)
24. **drift/analyzer.py** - Моніторинг дрейфу

### Patterns (1)
25. **pattern_analyzer.py** - Аналіз патернів

### Pipeline Evaluation (1)
26. **backtest_analyzer.py** - Backtest аналіз

### Risk Analyzers (2)
27. **concentration_analyzer.py** - Концентрація
28. **correlation_analyzer.py** - Кореляція

### Reporting (1)
29. **model_analyzer.py** - Звітність моделей

---

## 🧮 Калькулятори (22 знайдено)

### Analytics Calculators (8)
1. **advanced_econometrics_calculator.py** - Просунута економетрика
2. **drawdown_calculator.py** - Drawdown
3. **econometrics_calculator.py** - Економетрика
4. **explainability_calculator.py** - Explainability
5. **macro_score_calculator.py** - Макро score
6. **risk_reward_calculator.py** - Риск/нагорода
7. **sentiment_stats_calculator.py** - Статистика сентименту
8. **volatility_calculator.py** - Волатильність

### Meta Learning (1)
9. **contextual_weight_calculator.py** - Контекстні ваги

### Metrics (1)
10. **calculator.py** - Метрики

### Models Ensemble (1)
11. **weight_stability/calculator.py** - Стабільність ваг

### Models Monitoring (1)
12. **drift_calculator.py** - Калькулятор дрейфу

### Pipeline Evaluation (1)
13. **metrics_calculator.py** - Калькулятор метрик

### Risk (3)
14. **exposure_calculator.py** - Експозиція
15. **kill_switch/calculator.py** - Kill switch
16. **var_calculator.py** - VaR

### Targets Calculators (6)
17. **base_news_target_calculator.py** - Базовий калькулятор таргетів новин
18. **classification_calculator.py** - Класифікація
19. **indicator_prediction_calculator.py** - Прогноз індикаторів
20. **post_news_target_calculator.py** - Таргети після новин
21. **pre_news_target_calculator.py** - Таргети до новин
22. **regression_calculator.py** - Регресія

---

## 🔧 Збагачувачі (19 знайдено)

### Features Enrichers (14) - ✅ Всі включені в конфігурації
1. **advanced_analytics_enricher.py** - ✅ enabled
2. **context_map_enricher.py** - ✅ enabled
3. **decay_features_enricher.py** - ✅ enabled
4. **derived_features_enricher.py** - ✅ enabled
5. **hype_enricher.py** - ✅ enabled
6. **keyword_entity_enricher.py** - ✅ enabled
7. **macro_features_enricher.py** - ✅ enabled
8. **market_context_enricher.py** - ✅ enabled
9. **news_impact_enricher.py** - ✅ enabled
10. **news_quality_enricher.py** - ✅ enabled
11. **nlp_features_enricher.py** - ✅ enabled
12. **sentiment_features_enricher.py** - ✅ enabled
13. **significance_features_enricher.py** - ✅ enabled
14. **technical_analysis_enricher.py** - ✅ enabled

### Features Enrichers (додаткові)
15. **time_features_enricher.py** - ✅ enabled
16. **volatility_enricher.py** - Не в конфігурації
17. **volume_enricher.py** - Не в конфігурації

### Builders (2)
18. **builders/news_event/enricher.py** - NewsGlobalEnricher для dataset builder
19. **pipeline/stages/feature_engineering/enricher.py** - Enricher для feature engineering stage

---

## 🔍 Аналіз використання модулів в пайплайні

### Stage 1: Collection
**Використані модулі:**
- Колектори (8 включених)
- DataManager (для зберігання та фільтрації дублікатів)
- CollectorFactory (для створення колекторів)

### Stage 2: Processing
**Використані модулі:**
- IntelligentDataFilter (PriceFilter, NewsFilter, SocialFilter)
- NormalizationManager
- ProcessingValidator

### Stage 3: Feature Engineering
**Використані модулі:**
- ✅ Всі 14 збагачувачів з конфігурації features.yaml
- SignificanceDetector (для створення is_significant колонки)
- NewsImpactAnalyzer (для розрахунку impact scores)

### Stage 4: Modeling
**Використані модулі:**
- Модельні аналізатори (model_analyzer, model_health_analyzer)
- Ensemble аналізатори (model_correlation_analyzer)
- Overfitting detection

### Stage 5: Prediction
**Використані модулі:**
- NewsImpactClassifier (для класифікації впливу новин)
- Contextual weight calculator (для ваг моделей)

### Stage 6: Trading
**Використані модулі:**
- Risk калькулятори (exposure, var)
- Kill switch calculator

### Stage 7: Evaluation
**Використані модулі:**
- Backtest analyzer
- Metrics calculator
- Performance attribution analyzer

---

## 📋 Підсумок

### Dataset Builder
✅ Логіка відсікання новин без наступних свічок реалізована правильно

### Аналізатори (29)
- **Прямо задіяні:** ~10-15 (news_impact, model_analyzer, drift_analyzer, etc.)
- **Опосередковано задіяні:** ~5-10 (через registry або конфігурацію)
- **Не задіяні:** ~5-10 (спеціалізовані аналізатори для конкретних задач)

### Калькулятори (22)
- **Прямо задіяні:** ~8-10 (metrics, risk, targets)
- **Опосередковано задіяні:** ~5-8 (через інші модулі)
- **Не задіяні:** ~5-9 (спеціалізовані калькулятори)

### Збагачувачі (19)
- **Прямо задіяні:** 14 (всі з features.yaml)
- **Опосередковано задіяні:** 2 (для dataset builder та feature engineering stage)
- **Не задіяні:** 2 (volatility_enricher, volume_enricher)

### Рекомендації
1. Перевірити чи всі аналізатори та калькулятори працюють правильно
2. Включити volatility_enricher та volume_enricher в конфігурацію
3. Створити тести для перевірки функціональності модулів
