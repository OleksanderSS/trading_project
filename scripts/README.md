# 📋 Scripts Directory - Скрипти для аналізу, дебагінгу та тестування

## 📂 Структура папки:

```
scripts/
├── check/          # Перевірка даних та стану системи
│   ├── check_data.py      # Уніфікована перевірка всіх даних
│   ├── check_database.py  # Перевірка структури бази даних
│   └── check_gaps.py      # Перевірка гепів
│
├── analyze/        # Глибокий аналіз проблем та метрик
│   ├── analyze_pipeline.py  # Аналіз логіки пайплайну
│   ├── analyze_data_quality.py  # Аналіз якості даних
│   └── analyze_missing.py   # Аналіз відсутніх даних
│
├── debug/          # Дебагінг конкретних проблем
│   ├── debug_merge.py     # Дебагінг об'єднання даних
│   ├── debug_gaps.py      # Дебагінг гепів
│   └── debug_indicators.py # Дебагінг індикаторів
│
├── fix/            # Виправлення проблем
│   ├── fix_data.py       # Виправлення всіх даних
│   ├── fix_gaps.py       # Виправлення фільтрації гепів
│   └── fix_indicators.py # Виправлення індикаторів
│
└── test/           # Тестування функціональності
│   ├── test_runner.py    # Центральний тестовий раннер
│   ├── test_data.py      # Тести даних
│   ├── test_merge.py     # Тести об'єднання
│   └── test_indicators.py # Тести індикаторів
```

---

## 🚀 Використання:

### **📋 Перевірка даних:**
```bash
# Запуск всіх перевірок
python scripts/check/check_data.py

# Запуск конкретної перевірки
python scripts/check/check_database.py
python scripts/check/gaps.py
```

### **🔍 Аналіз проблем:**
```bash
# Запуск всіх аналізів
python scripts/analyze/analyze_pipeline.py

# Запуск конкретного аналізу
python scripts/analyze/analyze_data_quality.py
python scripts/analyze/analyze_missing.py
```

### **🐛 Дебагінг:**
```bash
# Запуск всіх дебагівгів
python scripts/debug/debug_merge.py

# Запуск конкретного дебагінгу
python scripts/debug/debug_gaps.py
python scripts/debug/debug_indicators.py
```

### **🔧 Виправлення:**
```bash
# Запуск всіх виправлень
python scripts/fix/fix_data.py

# Запуск конкретного виправлення
python scripts/fix/fix_gaps.py
python scripts/fix/fix_indicators.py
```

### **🧪 Тестування:**
```bash
# Запуск всіх тестів
python scripts/test/test_runner.py

# Запуск конкретних тестів
python -m unittest scripts.test.test_data.TestDataIntegrity
python -m unittest scripts.test.test_merge.TestMergeLogic
```

---

## 📋 Документація:

### **check/** - Перевірка даних:**
- **check_data.py** - Уніфікована перевірка всіх аспектів даних
- **check_database.py** - Перевірка структури бази даних
- **check_gaps.py** - Перевірка всіх типів гепів

### **analyze/** - Аналіз проблем:**
- **analyze_pipeline.py** - Комплексний аналіз пайплайну
- **analyze_data_quality.py** - Аналіз якості та повноти даних
- **analyze_missing_data.py** - Аналіз відсутніх даних

### **debug/** - Дебагінг:**
- **debug_merge.py** - Детальний дебагінг об'єднання
- **debug_gaps.py** - Дебагінг логіки гепів
- **debug_indicators.py** - Дебагінг технічних індикаторів

### **fix/** - Виправлення:**
- **fix_data.py** - Виправлення всіх проблем з даними
- **fix_gaps.py** - Виправлення фільтрації гепів
- **fix_indicators.py** - Виправлення технічних індикаторів

### **test/** - Тестування:**
- **test_runner.py** - Центральний тестовий раннер
- **test_data.py** - Тести цілісності даних
- **test_merge.py** - Тести логіки об'єднання
- **test_indicators.py** - Тести індикаторів

---

## 🎯 Переваги:

### **✅ Централізація:**
- **Єдиний** набір скриптів для кожної категорії
- **Уніфіковані** функції з кількох файлів
- **Єдиний** стандарт іменування

### **🔧 Надійність:**
- **Fallback** механізми для відсутніх файлів
- **Обробка помилок** з детальним логуванням
- **Створення** бекапів перед виправленнями

### **📊 Документація:**
- **Чітка** структура кожного скрипту
- **Детальний** опис функціональності
- **Приклади** використання

---

## 🔧 Розширення:

### **Додавання нових скриптів:**
1. Створіть новий файл у відповідній папці
2. Додайте функціональність
3. Додайте документацію в README.md
4. Додайте імпорт в відповідний уніфікований файл

### **Інтеграція з CI/CD:**
```bash
# В CI/CD pipeline
python scripts/check/check_data.py
python scripts/test/test_runner.py
python scripts/fix/fix_data.py
```

### **Планування завдань:**
```bash
# Запуск перевірок
python scripts/check/check_data.py

# Запуск аналізу
python scripts/analyze/analyze_pipeline.py

# Запуск виправлень
python scripts/fix/fix_data.py

# Запуск тестів
python scripts/test/test_runner.py
```

---

## 📈 Історія змін:

- **v1.0** - Створено базову структуру
- **v1.1** - Додано уніфіковані скрипти
- **v1.2** - Додано центральні тестові раннери

---

## ✅ Статус: ГОТОВО ДО ВИКОРИСТАННЯ
