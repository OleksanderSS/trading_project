# 📰 News Dataset Structure

## Overview

News dataset містить **один рядок на кожну новину** з повним контекстом до і після публікації.

**Мета:** Передбачити реакцію ринку ПІСЛЯ публікації новини на основі:
1. Характеристик самої новини
2. Стану ринку ДО публікації
3. Реакції ринку ПІСЛЯ публікації

---

## 🔹 Row Structure

Кожен рядок містить **~43,236 колонок**, розділених на 4 блоки:

### БЛОК 1: Інформація про новину (6 колонок)

```python
news_id              # Унікальний ID: "source_20260426_153000"
news_timestamp       # Час публікації новини (datetime)
news_title           # Заголовок новини (string)
news_sentiment       # Сентимент (-1.0 to 1.0)
news_type            # Тип: "TSLA", "AMD", "general", "macro"
news_source          # Джерело: "google_news", "rss", "newsapi"
```

**Приклад:**
```
news_id: "google_news_20260426_153000"
news_timestamp: 2026-04-26 15:30:00
news_title: "Tesla announces record Q1 deliveries"
news_sentiment: 0.85
news_type: "TSLA"
news_source: "google_news"
```

---

### БЛОК 2: Макро контекст на момент новини (~30 колонок)

```python
# Макроекономічні показники
fed_funds_rate           # Federal Funds Rate
treasury_10y             # 10-Year Treasury Yield
treasury_2y              # 2-Year Treasury Yield
yield_curve_slope        # 10Y - 2Y spread
yield_curve_inverted     # Inversion flag (0/1)
cpi                      # Consumer Price Index
unemployment_rate        # Unemployment Rate
vix                      # Volatility Index
dollar_strength          # DXY Index

# Ринковий сентимент
fear_greed_index         # Fear & Greed Index (0-100)
market_phase             # Bull/Bear/Neutral

# Часові фічі
hour_of_day              # 0-23
day_of_week              # 0-6 (Monday=0)
is_trading_hours         # Boolean
is_market_open           # Boolean
```

**Приклад:**
```
fed_funds_rate: 5.25
treasury_10y: 4.15
vix: 18.5
fear_greed_index: 65
hour_of_day: 15
day_of_week: 2  # Wednesday
is_trading_hours: True
```

---

### БЛОК 3: Контекст ДО новини (18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей = ~21,600 колонок)

Для кожного тікера × кожного таймфрейму: **2 свічки ДО публікації + ВСІ їх фічі**

**Структура для одного тікера/таймфрейму:**

```python
# Свічка 1 ДО новини
{ticker}_{timeframe}_before_1_datetime       # Timestamp свічки
{ticker}_{timeframe}_before_1_open           # OHLCV
{ticker}_{timeframe}_before_1_high
{ticker}_{timeframe}_before_1_low
{ticker}_{timeframe}_before_1_close
{ticker}_{timeframe}_before_1_volume
{ticker}_{timeframe}_before_1_sma_5          # Технічні індикатори
{ticker}_{timeframe}_before_1_sma_10
{ticker}_{timeframe}_before_1_rsi_14
{ticker}_{timeframe}_before_1_macd
{ticker}_{timeframe}_before_1_bb_upper
{ticker}_{timeframe}_before_1_volatility_5d  # Волатильність
{ticker}_{timeframe}_before_1_trend_5d       # Тренд
{ticker}_{timeframe}_before_1_sentiment_score # Сентимент
{ticker}_{timeframe}_before_1_context_fingerprint # Контекст
... (~200 фічей)

# Свічка 2 ДО новини
{ticker}_{timeframe}_before_2_datetime
{ticker}_{timeframe}_before_2_open
... (~200 фічей)
```

**Приклад для AMD 15m:**
```
AMD_15m_before_1_datetime: 2026-04-26 15:15:00
AMD_15m_before_1_open: 145.20
AMD_15m_before_1_close: 145.50
AMD_15m_before_1_volume: 125000
AMD_15m_before_1_rsi_14: 58.3
AMD_15m_before_1_sma_20: 144.80
AMD_15m_before_1_volatility_5d: 0.025

AMD_15m_before_2_datetime: 2026-04-26 15:00:00
AMD_15m_before_2_open: 144.90
AMD_15m_before_2_close: 145.20
...
```

**Всі комбінації:**
- 18 тікерів: AMD, NVDA, TSLA, AAPL, GOOGL, MSFT, AMZN, META, NFLX, INTC, QCOM, AVGO, TXN, AMAT, LRCX, KLAC, ASML, MU
- 3 таймфрейми: 15m, 60m, 1d
- 2 свічки ДО
- ~200 фічей на свічку

**Всього: 18 × 3 × 2 × 200 = 21,600 колонок**

---

### БЛОК 4: Реакція ПІСЛЯ новини (18 тікерів × 3 таймфрейми × 2 свічки × ~200 фічей = ~21,600 колонок)

Аналогічно БЛОКУ 3, але для свічок ПІСЛЯ публікації новини.

**Структура для одного тікера/таймфрейму:**

```python
# Свічка 1 ПІСЛЯ новини
{ticker}_{timeframe}_after_1_datetime
{ticker}_{timeframe}_after_1_open
{ticker}_{timeframe}_after_1_high
{ticker}_{timeframe}_after_1_low
{ticker}_{timeframe}_after_1_close
{ticker}_{timeframe}_after_1_volume
{ticker}_{timeframe}_after_1_sma_5
{ticker}_{timeframe}_after_1_rsi_14
... (~200 фічей)

# Свічка 2 ПІСЛЯ новини
{ticker}_{timeframe}_after_2_datetime
{ticker}_{timeframe}_after_2_open
... (~200 фічей)
```

**Приклад для AMD 15m:**
```
AMD_15m_after_1_datetime: 2026-04-26 15:30:00  # Одразу після новини
AMD_15m_after_1_open: 145.50
AMD_15m_after_1_close: 146.20  # Зріс на 0.70 (+0.48%)
AMD_15m_after_1_volume: 250000  # Обсяг подвоївся
AMD_15m_after_1_rsi_14: 62.5   # RSI зріс

AMD_15m_after_2_datetime: 2026-04-26 15:45:00
AMD_15m_after_2_open: 146.20
AMD_15m_after_2_close: 146.80  # Продовжує рости
...
```

**Всього: 18 × 3 × 2 × 200 = 21,600 колонок**

---

## 📊 Summary

```
БЛОК 1: Новина (6 колонок)
    ↓
БЛОК 2: Макро контекст (30 колонок)
    ↓
БЛОК 3: Контекст ДО (21,600 колонок)
    - 18 тікерів × 3 таймфрейми × 2 свічки × 200 фічей
    ↓
БЛОК 4: Реакція ПІСЛЯ (21,600 колонок)
    - 18 тікерів × 3 таймфрейми × 2 свічки × 200 фічей

ВСЬОГО: ~43,236 колонок на рядок
```

---

## 🎯 ML Pipeline Usage

### Що модель бачить:

1. **Input Features (X):**
   - БЛОК 1: Характеристики новини
   - БЛОК 2: Макро контекст
   - БЛОК 3: Стан ринку ДО новини

2. **Target (y):**
   - БЛОК 4: Реакція ринку ПІСЛЯ новини
   - Можна розрахувати різні таргети:
     - Price change: `after_1_close - before_2_close`
     - Volume spike: `after_1_volume / before_2_volume`
     - Volatility change: `after_1_volatility - before_2_volatility`

### Feature Selection:

SmartFeatureSelector автоматично вибере найважливіші фічі з ~43,000 колонок:
- Які тікери найбільше реагують на новини?
- Які таймфрейми найбільш предиктивні?
- Які технічні індикатори найважливіші?
- Які макро показники найбільш релевантні?

---

## ✅ Data Quality

### Фільтрація новин:

Новини фільтруються перед додаванням до датасету:

1. **Sufficient candles check:**
   - Новина має мати 2+ свічки ПІСЛЯ публікації
   - Інакше не можемо оцінити вплив

2. **News clustering:**
   - Схожі новини кластеризуються (similarity > 0.85)
   - Зменшує обсяг даних на 70-84%

3. **Sentiment filtering:**
   - Тільки новини з confidence > 0.75

### Event-series format:

- Sorted by `news_timestamp`
- No duplicate news_id
- All timestamps aligned to trading hours

---

## 📁 Output

**File:** `data/batches/{batch_name}/news_features.parquet`

**Format:** Parquet (compressed)

**Size:** ~500MB - 2GB (залежить від кількості новин)

**Rows:** ~100-1000 новин (після фільтрації та кластеризації)

**Columns:** ~43,236

---

## 🔧 Implementation

**Module:** `src/features/news_dataset_builder.py`

**Class:** `NewsContextDatasetBuilder`

**Key Methods:**
- `build_dataset()` - Головний метод
- `_filter_news_with_sufficient_candles()` - Фільтрація новин
- `_build_news_row()` - Побудова одного рядка
- `_get_macro_context()` - Макро контекст
- `_get_market_sentiment_context()` - Ринковий сентимент
- `_get_temporal_features()` - Часові фічі

**Usage:**
```python
from src.features.news_dataset_builder import NewsContextDatasetBuilder

builder = NewsContextDatasetBuilder(config_manager)

news_dataset = builder.build_dataset(
    news_df=news_df,
    prices_dict=enriched_prices,  # Dict[str, pd.DataFrame] - 15m, 60m, 1d
    macro_df=macro_df,
    market_sentiment_df=sentiment_df
)

builder.save_dataset(news_dataset, output_path)
```

---

## 🚀 Next Steps

1. **Generate news dataset** - Run Stage 3 with news data
2. **Feature selection** - SmartFeatureSelector вибере найважливіші фічі
3. **Train models** - Навчити моделі передбачати реакцію ринку
4. **Evaluate** - Оцінити на реальних даних

---

## 📚 References

- **Code:** `src/features/news_dataset_builder.py`
- **Config:** `src/config/features.yaml`
- **Pipeline:** `src/pipeline/stages/stage_3_feature_engineering.py`
- **Data Strategy:** `scripts/data_accumulation_strategy.md`
