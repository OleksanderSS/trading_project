# Інструкції для запуску пайплайну в Google Colab

## Правильний потік даних

**Важливо**: Пайплайн розбитий на дві частини - локальну та Colab:

1. **Локально (етапи 0-3)**: Збір даних, обробка, feature engineering
2. **Colab (етапи 4-5)**: Вибір фіч, тренування важких моделей
3. **Локально (етапи 6-7)**: Тренування легких моделей, порівняння результатів

## Частина 1: Локально - Етапи 0-3

### Запуск локального пайплайну

```bash
python run_hybrid_pipeline.py --mode prepare --batch-name main_database
```

### Що робить цей режим:
1. **Stage 0 - CollectionStage**: Збір даних з різних джерел (Yahoo Finance, FRED, новини)
2. **Stage 1 - ProcessingStage**: Обробка та очищення даних
3. **Stage 2 - FeatureEngineeringStage**: Створення технічних індикаторів та фічів
4. **Stage 3 - TargetGenerationStage**: Генерація таргетів для навчання

### Результат:
- Фічі збережені в: `data/colab/accumulated/main_database/features.parquet`
- Таргети збережені в: `data/colab/accumulated/main_database/targets.parquet`
- Пайплайн зупиняється з повідомленням: "PAUSED: Colab training required"

## Частина 2: Colab - Проміжний етап (Важкі моделі + Feature Selection)

### Підготовка в Colab

```python
# 1. Підключення Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Перехід в папку проекту
import os
os.chdir('/content/drive/MyDrive/trading_project')

# 3. Встановлення залежностей (якщо потрібно)
!pip install -r requirements.txt
```

### Запуск тренування важких моделей

```python
!python scripts/colab/colab_clean_cell.py
```

### Що робить цей скрипт:
1. Завантажує features.parquet та targets.parquet з локального пайплайну
2. **Обрахунок фіч ансамблевим способом** через SmartFeatureSelector
3. Тренує **важкі нейронні моделі**:
   - MLP (Multi-Layer Perceptron)
   - CNN (Convolutional Neural Network)
   - LSTM (Long Short-Term Memory)
   - GRU (Gated Recurrent Unit)
   - Transformer
   - TabNet
   - Autoencoder
4. Зберігає результати у форматі colab_results.json
5. Зберігає selected_features_*.json для кожної моделі

### Результат:
- `colab_results.json` - результати тренування важких моделей
- `selected_features_*.json` - вибрані фічі для кожної моделі (ансамблевий спосіб)
- `model_*.keras` / `model_*.pkl` / `model_*.zip` - збережені нейронні моделі

## Частина 3: Локально - Етапи 6-7 (Легкі моделі + Порівняння)

### Запуск продовження пайплайну

```bash
python run_hybrid_pipeline.py --mode continue --batch-name main_database
```

### Що робить цей режим:
1. Завантажує colab_results.json з Colab
2. Тренує легкі моделі локально
3. Порівнює результати важких та легких моделей
4. Генерує фінальний звіт

### Результат:
- Фінальні результати в: `results/`
- Порівняння моделей в: `models/results/`
- Метрики якості в: `results/`

## Повний приклад запуску

### Локально (етапи 0-3):
```bash
python run_hybrid_pipeline.py --mode prepare --batch-name main_database
```

### В Colab (проміжний етап - важкі моделі + feature selection):
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/trading_project')

!python scripts/colab/colab_clean_cell.py
```

### Локально (етапи 6-7 - легкі моделі + порівняння):
```bash
python run_hybrid_pipeline.py --mode continue --batch-name main_database
```

## Перевірка результатів

### Перевірка фічів (після етапу 0-3)

```python
import pandas as pd

# Завантаження фічів
features = pd.read_parquet('data/colab/accumulated/colab_batch/features.parquet')
print(f"Фічі: {features.shape}")
print(f"Колонки: {features.columns.tolist()[:20]}")

# Перевірка макро фічів
macro_cols = [col for col in features.columns if any(keyword in col.lower() for keyword in ['fed', 'treasury', 'cpi', 'unemployment', 'vix'])]
print(f"Макро колонки: {macro_cols}")
```

### Перевірка результатів тренування (після етапу 4-7)

```python
import pandas as pd

# Завантаження результатів
results = pd.read_parquet('results/colab_heavy_results_*.parquet')
print(f"Результати: {results.shape}")
print(results.head())
```

## Увага

1. **Пам'ять**: Colab має обмеження по пам'яті. Якщо отримуєте помилку "Out of memory", зменшіть кількість тікерів або днів.
2. **Час**: Етапи 0-3 можуть займати 10-20 хвилин, етапи 4-7 - 30-60 хвилин залежно від налаштувань.
3. **GPU**: Для етапів 4-7 рекомендується використовувати GPU runtime в Colab.
4. **API ключі**: Переконайтеся, що FRED API ключ налаштований в `.env` файлі.

## Вирішення проблем

### Помилка "Module not found"
```python
!pip install -r requirements.txt
```

### Помилка "Out of memory"
- Зменшіть кількість тікерів: `--tickers AMD NVDA`
- Зменшіть кількість днів: `--days 15`

### Помилка "API key not found"
- Перевірте `.env` файл
- Додайте FRED API ключ: `FRED_API_KEY=your_key_here`

## Структура файлів

```
/content/drive/MyDrive/trading_project/
├── scripts/
│   └── colab/
│       ├── colab_stages_0_3.py  # Етапи 0-3
│       └── colab_stages_4_7.py  # Етапи 4-7
├── data/
│   └── colab/
│       └── accumulated/
│           └── colab_batch/
│               ├── features.parquet  # Фічі
│               └── targets.parquet   # Таргети
├── models/
│   ├── trained/          # Навчені моделі
│   └── results/          # Результати
└── results/              # Фінальні результати
```
