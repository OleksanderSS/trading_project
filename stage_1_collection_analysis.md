# Stage 1: Collection - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_1_collection.py`  
**Конфігурація:** `src/config/collectors.yaml`  
**Призначення:** Збір даних з різних джерел

---

## 🔧 Архітектура

### Ключові компоненти:

1. **CollectorFactory** - Фабрика для створення колекторів
2. **CollectionManager** - Менеджер для координації збору даних
3. **DataManager** - Менеджер бази даних для збереження
4. **DataSchemaMapper** - Маппер для приведення даних до схеми
5. **TemporalAlignmentChecker** - Перевірка часової узгодженості новин

---

## 📊 Включені колектори

### ✅ Активні колектори (8):

1. **yahoo_finance**
   - **Тип:** Цінові дані
   - **Критичний:** ✅ Так
   - **Таблиця:** market_data_raw
   - **Кеш:** 15 хвилин
   - **Timeframes:** 15m, 60m, 1d
   - **Період:** 60 днів

2. **fred**
   - **Тип:** Макроекономічні дані
   - **Критичний:** ❌ Ні
   - **Таблиця:** fred_data
   - **Кеш:** 24 години
   - **Series IDs:** 28 серій (FEDFUNDS, DGS10, CPIAUCSL, etc.)

3. **google_news**
   - **Тип:** Новини
   - **Критичний:** ❌ Ні
   - **Таблиця:** google_news
   - **Кеш:** 1 година
   - **Макс. результатів:** 10
   - **Фільтр:** Виключає horoscope, celebrity, sports score, weather, recipe

4. **rss**
   - **Тип:** Новини
   - **Критичний:** ❌ Ні
   - **Таблиця:** rss_news
   - **Кеш:** 1 година
   - **Ліміт на фід:** 20
   - **Фільтр:** Виключає horoscope, celebrity, sports

5. **newsapi**
   - **Тип:** Новини
   - **Критичний:** ❌ Ні
   - **Таблиця:** newsapi_articles
   - **Кеш:** 1 година
   - **API Key:** NEWS_API_KEY
   - **Мін. якість джерела:** 0.4
   - **Фільтр:** Виключає horoscope, celebrity

6. **hugging_face**
   - **Тип:** Новини (датасет)
   - **Критичний:** ❌ Ні
   - **Таблиця:** huggingface_data
   - **Кеш:** 7 днів
   - **Датасет:** m-ric/financial-news-2024
   - **HF_KEY:** ✅ Підключено

7. **alternative_me**
   - **Тип:** Альтернативні дані (Fear & Greed)
   - **Критичний:** ❌ Ні
   - **Таблиця:** fear_greed_data
   - **Кеш:** 1 година
   - **Ліміт:** 100 записів

8. **vix**
   - **Тип:** Альтернативні дані (VIX)
   - **Критичний:** ❌ Ні
   - **Таблиця:** vix_data
   - **Кеш:** 1 година
   - **Тікер:** ^VIX
   - **Період:** 60 днів

### ❌ Вимкнені колектори (10):

1. **economic_calendar** - Економічний календар
2. **free_google_trends** - Google Trends
3. **sec_filings** - SEC filings
4. **insider** - Insider trades
5. **bigquery** - BigQuery
6. **custom_csv** - Custom CSV
7. **put_call_ratio** - Put/Call Ratio (вимкнено через помилку CBOE)
8. **cftc** - CFTC (вимкнено через застарілі URL)
9. **aaii_sentiment** - AAII Sentiment
10. **fear_greed** - Fear & Greed (дублює alternative_me)
11. **reddit_sentiment** - Reddit Sentiment

---

## 🔄 Процес збору даних

### Крок 1: Ініціалізація
```python
collector_configs = self.config_manager.get_config('collectors')
factory = CollectorFactory(configs=collector_configs, ...)
self.collection_manager = CollectionManager(factory, ...)
```

### Крок 2: Визначення тікерів
```python
tickers = kwargs.get('tickers') or preset_config.get('tickers', ['TSLA', 'NVDA', 'SPY', 'QQQ', 'AMD'])
```

### Крок 3: Підготовка до збору
```python
self._prepare_collection()  # FORCING data collection (temporary fix)
```

### Крок 4: Збір даних
```python
keywords = ['earnings', 'fed', 'inflation', 'market', 'trading']
raw_data = await self.collection_manager.fetch_all(self._tickers, keywords)
```

### Крок 5: Завантаження з бази
```python
db_data = self.fetch_all_data_from_db()
raw_data.update(db_data)
```

### Крок 6: Мапінг до схеми
```python
mapped_data = self.schema_mapper.map_to_schema(raw_data)
return mapped_data
```

---

## 🔒 Дедуплікація та кешування

### Дедуплікація:
- **Hash keys:** Унікальні ключі для кожного колектора
- **Hash column:** Автоматично додається якщо є в DataFrame
- **Link column:** Додається для новин
- **Filter new records:** `DataManager.filter_new_records()` фільтрує дублікати

### Кешування:
- **Cache TTL:** Час життя кешу для кожного колектора
- **Cache duration:** Тривалість кешування в хвилинах
- **Cache metadata:** Зберігається в таблиці `cache_metadata`

---

## 📰 Обробка новин

### Комбінація новин:
```python
all_news_dfs = []  # Збирає всі новинні DataFrame
news_df = pd.concat(all_news_dfs, ignore_index=True)
news_df = self._remove_news_duplicates(news_df)
```

### Temporal Alignment:
```python
news_df = self._check_news_temporal_alignment(news_df, raw_data)
```
- Перевіряє чи новини не датовані майбутнім
- Фільтрує future-dated news
- Використовує TemporalAlignmentChecker

### Фільтрація дублікатів:
```python
hashable_cols = self._get_hashable_columns(news_df)
news_df = news_df.drop_duplicates(subset=hashable_cols)
```

---

## 💾 Збереження в базу

### Upsert процес:
```python
def _upsert_dataframe(self, table_name: str, df: pd.DataFrame, unique_on: list[str]):
    if not self.db_manager.table_exists(table_name):
        self.db_manager.upsert(table_name=table_name, df=df, unique_on=unique_on)
    else:
        new_df = self.db_manager.filter_new_records(table_name, df)
        if not new_df.empty:
            self.db_manager.upsert(table_name=table_name, df=new_df, unique_on=unique_on)
```

### Конвертація дат:
```python
date_col = self._find_date_column_in_df(df)
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
```

---

## 🎯 Ключові функції

### 1. `fetch_all_data_from_db()`
Завантажує всі дані з бази для наступного етапу:
- Пропускає `cache_metadata`
- Об'єднує всі новинні джерела в один DataFrame
- Логує підсумок завантажених даних

### 2. `_check_news_temporal_alignment()`
Перевіряє часову узгодженість новин:
- Знаходить timestamp колонки в news та market data
- Використовує TemporalAlignmentChecker
- Фільтрує future-dated news
- Логує кількість відфільтрованих записів

### 3. `_remove_news_duplicates()`
Видаляє дублікати з новин:
- Знаходить hashable колонки
- Використовує drop_duplicates()

---

## 📈 Результати роботи

### Очікувані дані на виході:
1. **market_data_raw** - Цінові дані з Yahoo Finance
2. **fred_data** - Макроекономічні дані з FRED
3. **news** - Об'єднані новини з усіх джерел
4. **fear_greed_data** - Fear & Greed Index
5. **vix_data** - VIX дані
6. **huggingface_data** - Новини з HuggingFace датасету

### Логування:
- Кількість тікерів
- Кількість записів від кожного колектора
- Кількість нових записів після фільтрації
- Підсумок загальної кількості записів

---

## ⚠️ Потенційні проблеми

### 1. **FORCING data collection**
```python
self.logger.info('🔄 FORCING data collection for all tickers (temporary fix)')
```
- Це тимчасове рішення
- Може призвести до зайвих запитів до API

### 2. **Вимкнені колектори**
- put_call_ratio вимкнено через помилку CBOE
- cftc вимкнено через застарілі URL
- economic_calendar, google_trends, sec_filings, insider - вимкнені без причини

### 3. **Temporal Alignment**
- Перевіряється тільки якщо є market_data
- Може пропускати фільтрацію якщо немає даних

---

## ✅ Статус Stage 1

**Загальний статус:** ✅ Працює коректно

**Активні колектори:** 8/8 працюють
- ✅ yahoo_finance - критичний, працює
- ✅ fred - працює
- ✅ google_news - працює
- ✅ rss - працює
- ✅ newsapi - працює з API key
- ✅ hugging_face - працює з HF_KEY
- ✅ alternative_me - працює
- ✅ vix - працює

**Дедуплікація:** ✅ Працює через hash_keys
**Кешування:** ✅ Працює через cache_ttl
**Temporal Alignment:** ✅ Працює для новин

**Рекомендації:**
1. Видалити "FORCING data collection" temporary fix
2. Перевірити чому put_call_ratio не працює
3. Перевірити чому cftc не працює
4. Розглянути включення economic_calendar
