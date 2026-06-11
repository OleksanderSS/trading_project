# План аналізу допоміжних інструментів

## 📋 Огляд

Цей документ містить план детального аналізу всіх допоміжних інструментів пайплайну: калькуляторів, аналізаторів, алгоритмів, збагачувачів, селекторів, фільтрів, гардів та детекторів.

---

## 📊 Статистика компонентів

- **Калькулятори:** 47 результатів
- **Аналізатори:** 53 результати
- **Збагачувачі:** 43 результати
- **Селектори:** 28 результатів
- **Консенсус:** 2 результати
- **Фільтри:** 16 результатів
- **Гарди:** 13 результатів
- **Детектори:** 30 результати

**Всього:** ~232 компоненти

---

## 🎯 Пріоритети аналізу

### Високий пріоритет (Critical для роботи пайплайну)
1. **Feature Enrichers** - Збагачення фіч (17 збагачувачів)
2. **Feature Selectors** - Вибір фіч (smart selector, enhanced smart selector)
3. **Model Selectors** - Вибір моделей (adaptive, smart, fingerprint)
4. **Metrics Calculators** - Фінансові метрики
5. **Risk Calculators** - Ризик-менеджмент
6. **Pipeline Guards** - Безпека фіч

### Середній пріоритет (Важливо для якості)
7. **Analytics Analyzers** - Аналіз результатів
8. **Target Calculators** - Генерація таргетів
9. **Filters** - Фільтрація даних
10. **Detectors** - Виявлення аномалій

### Низький пріоритет (Додаткові функції)
11. **Consensus Engine** - Консенсус
12. **Other Calculators** - Інші калькулятори

---

## 📝 Детальний план аналізу

### Категорія 1: Feature Enrichers (17 збагачувачів)

#### Список:
1. **time_features_enricher** - Часові фічі
2. **technical_analysis_enricher** - Технічні індикатори (~50)
3. **derived_features_enricher** - Похідні фічі
4. **macro_features_enricher** - Макро фічі
5. **keyword_entity_enricher** - Ключові слова
6. **news_quality_enricher** - Якість новин
7. **sentiment_features_enricher** - Сентимент
8. **nlp_features_enricher** - NLP фічі
9. **news_impact_enricher** - Вплив новин
10. **hype_enricher** - Hype/attention
11. **significance_features_enricher** - Статистична значущість
12. **decay_features_enricher** - Decay функції
13. **advanced_analytics_enricher** - Advanced analytics
14. **context_map_enricher** - Context fingerprint
15. **market_context_enricher** - Macro/market indicators
16. **volatility_enricher** - Волатильність (новий)
17. **volume_enricher** - Об'єм (новий)

#### Що аналізувати:
- Правильність розрахунків
- Обробка відсутніх даних
- Ефективність
- Коректність типів даних
- Обробка крайових випадків

---

### Категорія 2: Feature Selectors (3 селектори)

#### Список:
1. **smart_selector** - Розумний вибір фіч
2. **enhanced_smart_selector** - Покращений розумний вибір
3. **volatility_driver_selector** - Вибір волатильних драйверів

#### Що аналізувати:
- Алгоритм вибору фіч
- Коректність скорингу
- Обробка кореляції
- Стабільність вибору

---

### Категорія 3: Model Selectors (3 селектори)

#### Список:
1. **adaptive_selector** - Адаптивний вибір моделей
2. **smart_selector** - Розумний вибір моделей
3. **fingerprint_selector** - Вибір за відбитком

#### Що аналізувати:
- Логіка вибору моделі
- Коректність порівняння моделей
- Обробка Pattern-Aware контексту
- Стабільність вибору

---

### Категорія 4: Metrics Calculators (8 калькуляторів)

#### Список:
1. **metrics_calculator** (pipeline evaluation) - Фінансові метрики
2. **drawdown_calculator** - Drawdown метрики
3. **risk_reward_calculator** - Risk/Reward
4. **volatility_calculator** - Волатильність
5. **econometrics_calculator** - Економетрика
6. **advanced_econometrics_calculator** - Advanced економетрика
7. **macro_score_calculator** - Macro score
8. **sentiment_stats_calculator** - Статистика сентименту

#### Що аналізувати:
- Правильність формул
- Обробка крайових випадків (наприклад, нульовий Sharpe)
- Коректність розрахунків
- Обробка відсутніх даних

---

### Категорія 5: Risk Calculators (3 калькулятори)

#### Список:
1. **var_calculator** - Value at Risk
2. **exposure_calculator** - Експозиція
3. **kill_switch/calculator** - Kill switch калькулятор

#### Що аналізувати:
- Правильність розрахунків VaR
- Коректність експозиції
- Логіка kill switch

---

### Категорія 6: Pipeline Guards (4 гарди)

#### Список:
1. **temporal_leakage_guard** - Захист від temporal leakage
2. **macro_release_timing_guard** - Таймінг макро релізів
3. **timeframe_alignment_guard** - Вирівнювання таймфреймів
4. **temporal_target_guard** - Захист таргетів

#### Що аналізувати:
- Коректність логіки захисту
- Обробка крайових випадків
- Ефективність перевірок

---

### Категорія 7: Analytics Analyzers (8 аналізаторів)

#### Список:
1. **drift_analyzer** - Виявлення drift
2. **hedge_fund_analyzer** - Оцінка через призму хедж-фондів
3. **model_comparison_analyzer** - Порівняння моделей
4. **news_impact_analyzer** - Вплив новин
5. **performance_attribution_analyzer** - Attribution
6. **risk_decomposition_analyzer** - Декомпозиція ризику
7. **shap_analyzer** - SHAP аналіз
8. **adaptive_confidence_analyzer** - Адаптивна впевненість

#### Що аналізувати:
- Коректність алгоритмів
- Обробка відсутніх даних
- Ефективність

---

### Категорія 8: Target Calculators (5 калькуляторів)

#### Список:
1. **regression_calculator** - Регресійні таргети
2. **classification_calculator** - Класифікаційні таргети
3. **pre_news_target_calculator** - Pre-news таргети
4. **post_news_target_calculator** - Post-news таргети
5. **indicator_prediction_calculator** - Прогноз індикаторів

#### Що аналізувати:
- Правильність розрахунків таргетів
- Обробка future data
- Коректність horizons

---

### Категорія 9: Filters (4 фільтри)

#### Список:
1. **price_filter** - Фільтр цін
2. **news_filter** - Фільтр новин
3. **social_filter** - Фільтр соціальних даних
4. **post_inference_filter** - Post-inference фільтр

#### Що аналізувати:
- Логіка фільтрації
- Коректність критеріїв
- Обробка крайових випадків

---

### Категорія 10: Detectors (10 детекторів - пріоритетні)

#### Список:
1. **anomaly_detector** - Виявлення аномалій
2. **critical_signal_detector** - Критичні сигнали
3. **feature_drift_detector** - Drift фіч
4. **redundancy_detector** - Надлишковість фіч
5. **regime_detector** - Виявлення режимів
6. **bias_detector** - Виявлення зміщення
7. **overfitting_detector** - Виявлення overfitting
8. **data_leakage_detector** - Виявлення data leakage
9. **missing_data_anomaly_detector** - Відсутні дані
10. **news_ticker_detector** - Виявлення тікерів в новинах

#### Що аналізувати:
- Коректність алгоритмів виявлення
- Чутливість до false positives
- Ефективність

---

### Категорія 11: Consensus Engine (1 компонент)

#### Список:
1. **consensus_engine** - Консенсус двигун

#### Що аналізувати:
- Логіка консенсусу
- Обробка конфліктів
- Коректність агрегації

---

## 🎯 Порядок аналізу

1. Feature Enrichers (17) - Високий пріоритет
2. Feature Selectors (3) - Високий пріоритет
3. Model Selectors (3) - Високий пріоритет
4. Metrics Calculators (8) - Високий пріоритет
5. Risk Calculators (3) - Високий пріоритет
6. Pipeline Guards (4) - Високий пріоритет
7. Analytics Analyzers (8) - Середній пріоритет
8. Target Calculators (5) - Середній пріоритет
9. Filters (4) - Середній пріоритет
10. Detectors (10) - Середній пріоритет
11. Consensus Engine (1) - Низький пріоритет

---

## 📝 Результати аналізу

Для кожного компонента буде створено детальний аналіз, який включає:
- Опис функціональності
- Аналіз правильності роботи
- Виявлені проблеми
- Рекомендації

Фінальний звіт буде об'єднати всі аналізи в один документ.
