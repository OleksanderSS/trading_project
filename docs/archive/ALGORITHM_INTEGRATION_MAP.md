# Система Аналітики та Алгоритмів (Аудит)

Цей документ описує повну карту інтеграції модулів у пайплайни (Stages 0-7).

## 1. Колектори (Stage 1)
| Модуль | Статус | Роль |
| :--- | :--- | :--- |
| `yf_collector.py` | Активовано | Основний: котирування. |
| `vix_collector.py` | Активовано | Волатильність (інтегровано). |
| `sec_filings_collector.py` | Активовано | Звітність компаній (інтегровано). |
| `rss_collector.py` | Активовано | Новини (RSS). |

## 2. Алгоритми (src/algorithms)
| Модуль | Статус | Де використовується | Опис |
| :--- | :--- | :--- | :--- |
| `regime_detector.py` | Активовано | Stage 3 (TechnicalAnalysisEnricher) | ML-визначення режиму ринку. |
| `adaptive_position_sizer.py`| Активовано | Stage 6 | Розрахунок позиції (VaR, Kelly). |
| `risk_parity_allocator.py` | Активовано | Stage 6 | Розподіл капіталу (ERC, HRP). |

## 3. Аналітичні калькулятори (src/analytics/calculators)
| Модуль | Статус | Де використовується | Опис |
| :--- | :--- | :--- | :--- |
| `drawdown_calculator.py` | Активовано | Stage 3, Stage 6 | Розрахунок просадок (MDD). |
| `risk_reward_calculator.py`| Активовано | Stage 3, Stage 4, Stage 6 | Sharpe, Sortino ratios. |
| `volatility_calculator.py` | Активовано | Stage 3, Stage 6 | Волатильність. |
| `fama_french_factors.py` | Активовано | Stage 3 | Факторне моделювання. |
| `macro_score_calculator.py`| Активовано | Stage 3 | Композитний індекс макро. |
| `sentiment_stats_calculator.py`| Активовано | Stage 3 | Статистики настроїв. |
| `explainability_calculator.py`| Активовано | Stage 4 | SHAP-аналіз моделей. |
| `advanced_econometrics_calculator.py` | Активовано | Stage 3 | Статистичне моделювання. |

## 4. Meta-Learning та Пам'ять (src/meta_learning)
| Модуль | Статус | Де використовується | Опис |
| :--- | :--- | :--- | :--- |
| `diary_engine.py` | Активовано | Stage 4, Stage 5, Stage 6 | Зберігає історію рішень та помилок (Experience Memory). |
| `dual_loops.py` | Активовано | Stage 4 | Еволюційний цикл навчання моделей. |
| `context_engine.py` | Активовано | Stage 3, Stage 5 | Контекстна обізнаність (News/Events). |

## 5. Важливі примітки
- **Інтеграція**: Усі калькулятори інтегровані через `TechnicalAnalysisEnricher` (Stage 3). Це означає, що всі розрахунки відбуваються автоматично під час збагачення даних.
- **Стабільність**: Помилки при розрахунках (NaN/Inf) обробляються через `.fillna(0)` на рівні Enricher, що гарантує цілісність даних для ML-моделей.
- **Масштабування**: Фабрика `ModelFactory` динамічно налаштовує розміри нейронних мереж під кількість ознак (168+), тому при додаванні нових калькуляторів код змінювати не потрібно.
