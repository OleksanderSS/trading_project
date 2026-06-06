# План впровадження рекомендацій для Paper Trading

## 📋 Огляд

Цей документ містить план впровадження рекомендацій з аналізу пайплайну для paper trading.

---

## 🎯 Пріоритети впровадження

### Високий пріоритет (High Priority)
1. **Stage 1:** Видалити 'FORCING data collection' temporary fix
2. **Stage 2:** Додати очищення для macro_data та news

### Середній пріоритет (Medium Priority)
3. **Stage 3:** Додати таргети для інших таймфреймів
4. **Stage 4:** Додати cross-validation
5. **Stage 5:** Додати адаптивні пороги для champion-bias adjustment
6. **Stage 6:** Увімкнути консенсус (EnhancedConsensusEngine)

### Низький пріоритет (Low Priority)
7. **Stage 7:** Додати stress testing

---

## 📝 Детальний план

### Stage 1: Видалити 'FORCING data collection' temporary fix

**Файл:** `src/pipeline/stages/stage_1_collection.py`  
**Рядок:** 55-59

**Поточний код:**
```python
def _prepare_collection(self):
    """Prepare for data collection."""
    self.logger.info(
        '🔄 FORCING data collection for all tickers (temporary fix)')
    self.logger.info('Collection stage finished.')
```

**Зміна:** Видалити логування про FORCING data collection

---

### Stage 2: Додати очищення для macro_data та news

**Файл:** `src/pipeline/stages/processing/orchestrator.py`  
**Метод:** `_process_all_data_types`

**Поточний код:**
```python
# ✅ Pass macro_data from Stage 1 (FredCollector) to Feature Engineering
if 'macro_data' in raw_data and isinstance(raw_data['macro_data'], __import__('pandas').DataFrame):
    cleaned_data_map['macro_data'] = raw_data['macro_data']

# Pass news data
if 'news' in raw_data:
    cleaned_data_map['news'] = raw_data['news']
```

**Зміна:** Додати очищення через DataCleaner перед передачею

---

### Stage 3: Додати таргети для інших таймфреймів

**Файл:** `src/pipeline/stages/feature_engineering/orchestrator.py`  
**Метод:** `run`

**Поточний код:**
```python
# 2. Target Generation (usually on 1d)
if tf == '1d':
    targets_df = self.target_gen.generate_targets(enriched_df)
    all_targets[tf] = targets_df
```

**Зміна:** Додати генерацію таргетів для інших таймфреймів (15m, 60m)

---

### Stage 4: Додати cross-validation

**Файл:** `src/pipeline/stages/stage_4_modeling.py`  
**Метод:** `_process_ticker_with_async`

**Зміна:** Додати k-fold cross-validation замість простого train/test split

---

### Stage 5: Додати адаптивні пороги для champion-bias adjustment

**Файл:** `src/pipeline/stages/stage_5_prediction.py`  
**Метод:** `_process_single_context`

**Поточний код:**
```python
if velocity > 0.7:
    pred['confidence'] *= 0.5  # Штраф 50%
if velocity > 0.85:
    pred['confidence'] = 0.0  # Повністю анулюємо сигнал
```

**Зміна:** Зробити пороги адаптивними на основі історичних даних

---

### Stage 6: Увімкнути консенсус (EnhancedConsensusEngine)

**Файл:** `src/pipeline/stages/stage_6_trading_execution.py`  
**Метод:** `_initialize_trading_stack`

**Поточний код:**
```python
self.trading_orchestrator = TradingOrchestrator(
    consensus_engine=None,  # Вимкнено
    ...
)
```

**Зміна:** Увімкнути consensus_engine=self.enhanced_consensus

---

### Stage 7: Додати stress testing

**Файл:** `src/pipeline/stages/stage_7_evaluation.py`  
**Метод:** `_run_comprehensive_evaluation`

**Зміна:** Додати stress testing сценарії (crisis, high volatility, etc.)

---

## ✅ Статус впровадження

- [ ] Stage 1: Видалити 'FORCING data collection' temporary fix
- [ ] Stage 2: Додати очищення для macro_data та news
- [ ] Stage 3: Додати таргети для інших таймфреймів
- [ ] Stage 4: Додати cross-validation
- [ ] Stage 5: Додати адаптивні пороги для champion-bias adjustment
- [ ] Stage 6: Увімкнути консенсус (EnhancedConsensusEngine)
- [ ] Stage 7: Додати stress testing

---

## 🎯 Обмеження для Paper Trading

- **Тільки paper trading** - не впроваджувати інтеграцію з реальним брокером
- **Без stop-loss/take-profit** - це вимагає реального брокера
- **Без real-time execution** - це вимагає реального брокера
- **Фокус на покращення якості сигналів** - для paper trading
