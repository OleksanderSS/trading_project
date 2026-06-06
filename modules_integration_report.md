# Modules Integration Report

## 🔍 Детальний аналіз інтеграції модулів

**Дата:** 2026-06-06  
**Мета:** Зрозуміти де і як підключати модулі, чому не задіяні

---

## 📡 Колектори - HF_KEY підключено

### HuggingFace Collector
**Файл:** `src/data/collectors/huggingface_collector.py`  
**Зміни:** ✅ HF_KEY підключено

**Додано:**
```python
import os
# ...
# Get HF_KEY from environment
self.hf_key = os.getenv('HF_KEY')
if self.hf_key:
    self.logger.info('[HuggingFace] HF_KEY found in environment')
else:
    self.logger.warning('[HuggingFace] HF_KEY not found in environment')
```

**Статус:** ✅ HF_KEY тепер підключено

---

## 🧊 Аналізатори - де і як підключати

### Hedge Fund Analyzer
**Файл:** `src/analytics/analyzers/hedge_fund_analyzer.py`  
**Статус:** ✅ Вже налаштований в конфігураціях

**Де налаштовано:**
1. `src/analytics/analyzer_registry.py` - зареєстрований як "hedge_fund"
2. `src/config/unified_config.yaml` - налаштований в analysis section
3. `src/config/analysis.yaml` - налаштований з data_mapping

**Як викликається:**
```python
# В UnifiedAnalyticsEngine через analyzer_registry
from src.analytics.analyzer_registry import get_analyzer
analyzer = get_analyzer("hedge_fund", config={})
result = analyzer.analyze(data_map)
```

**Чому не задіяний в основному пайплайні:**
- Це спеціалізований аналізатор для інституційних інвесторів
- Використовується в Stage 7 (Evaluation) через UnifiedAnalyticsEngine
- Не потрібен для основного пайплайну (Stages 1-6)

**Як підключити:**
```yaml
# В src/config/analysis.yaml
- name: "hedge_fund_style"
  module: "src.analytics.analyzers.hedge_fund_analyzer"
  class: "HedgeFundAnalyzer"
  data_mapping: ["portfolio_data", "market_data"]
```

---

### SHAP Analyzer
**Файл:** `src/analytics/analyzers/shap_analyzer.py`  
**Статус:** ✅ Вже налаштований в конфігураціях

**Де налаштовано:**
1. `src/analytics/analyzer_registry.py` - зареєстрований як "shap"

**Як викликається:**
```python
# В UnifiedAnalyticsEngine через analyzer_registry
from src.analytics.analyzer_registry import get_analyzer
analyzer = get_analyzer("shap", config={})
result = analyzer.analyze(data_map)
```

**Чому не задіяний в основному пайплайні:**
- Це спеціалізований аналізатор для explainability моделей
- Використовується в Stage 7 (Evaluation) через UnifiedAnalyticsEngine
- Не потрібен для основного пайплайну (Stages 1-6)

**Як підключити:**
```yaml
# В src/config/analysis.yaml
- name: "shap_analysis"
  module: "src.analytics.analyzers.shap_analyzer"
  class: "ShapAnalyzer"
  data_mapping: ["model", "features"]
```

---

### Pattern Analyzer
**Файл:** `src/patterns/pattern_analyzer.py`  
**Статус:** ✅ Вже налаштований в конфігураціях

**Де налаштовано:**
1. `src/config/analysis.yaml` - налаштований з data_mapping

**Як викликається:**
```python
# В UnifiedAnalyticsEngine через analysis.yaml
# Автоматично завантажується та викликається
```

**Чому не задіяний в основному пайплайні:**
- Це спеціалізований аналізатор для патернів
- Використовується в Stage 7 (Evaluation) через UnifiedAnalyticsEngine
- Не потрібен для основного пайплайну (Stages 1-6)

**Як підключити:**
```yaml
# В src/config/analysis.yaml
- name: "pattern_analysis"
  module: "src.patterns.pattern_analyzer"
  class: "PatternAnalyzer"
  data_mapping: ["price_data"]
  params:
    fractal_window: 20
```

---

## 🧮 Калькулятори - де і як підключати

### Advanced Econometrics Calculator
**Файл:** `src/analytics/calculators/advanced_econometrics_calculator.py`  
**Статус:** ❌ Не в analyzer_registry

**Де налаштовано:**
- Немає в analyzer_registry.py
- Немає в unified_config.yaml
- Немає в analysis.yaml

**Чому не задіяний:**
- Це спеціалізований калькулятор для глибокого економічного аналізу
- Не інтегрований в UnifiedAnalyticsEngine
- Використовується як утиліта для специфічних задач

**Як підключити:**
```python
# Прямий виклик
from src.analytics.calculators.advanced_econometrics_calculator import AdvancedEconometricsCalculator

calculator = AdvancedEconometricsCalculator()
result = calculator.test_granger_causality(df, target_col, predictor_cols, maxlag=5)
```

**Або додати в analyzer_registry:**
```python
# В src/analytics/analyzer_registry.py
from src.analytics.calculators.advanced_econometrics_calculator import AdvancedEconometricsCalculator

ANALYZER_REGISTRY: dict[str, type[IAnalyzer]] = {
    # ...
    "advanced_econometrics": AdvancedEconometricsCalculator,
}
```

---

### Explainability Calculator
**Файл:** `src/analytics/calculators/explainability_calculator.py`  
**Статус:** ✅ Вже налаштований в конфігураціях

**Де налаштовано:**
1. `src/features/enrichers/technical_analysis_enricher.py` - використовується всередині
2. `src/config/analysis.yaml` - налаштований в explainability section

**Як викликається:**
```python
# В technical_analysis_enricher.py
from src.analytics.calculators.explainability_calculator import ExplainabilityCalculator

self.ExplainabilityCalculator = ExplainabilityCalculator()
result = self.ExplainabilityCalculator.analyze_feature_importance(model, data, feature_names)
```

**Чому не задіяний в основному пайплайні:**
- Це спеціалізований калькулятор для explainability
- Використовується всередині technical_analysis_enricher
- Не потрібен для основного пайплайну (Stages 1-6)

**Як підключити:**
```yaml
# В src/config/analysis.yaml
explainability:
  module: "src.analytics.calculators.explainability_calculator"
  class: "ExplainabilityCalculator"
```

---

## 🔧 Збагачувачі - Volatility та Volume

### Volatility Enricher
**Файл:** `src/features/enrichers/volatility_enricher.py`  
**Статус:** ❌ Не в конфігурації features.yaml

**Функціональність:**
- Historical Volatility (5, 10, 20 days)
- Average True Range (ATR)
- Garman-Klass Volatility
- Volatility Regime (low, normal, high, extreme)
- Всього 8 індикаторів

**Чому не в конфігурації:**
- Technical analysis enricher вже розраховує ATR
- Може бути дублюванням функціональності
- Не включено в конфігурацію

**Чи дублює функціональність:**
- Частково - ATR вже в technical_analysis_enricher
- Але volatility_enricher має більше індикаторів (volatility_5, volatility_10, volatility_20, gk_volatility, volatility_regime)
- Це додаткові індикатори, не дублювання

**Як підключити:**
```yaml
# В src/config/features.yaml
features:
  enabled_enrichers:
    # ... інші збагачувачі
    volatility: true  # Додати цей рядок
```

**Рекомендація:** ✅ Включити в конфігурацію - це корисні додаткові індикатори волатильності

---

### Volume Enricher
**Файл:** `src/features/enrichers/volume_enricher.py`  
**Статус:** ❌ Не в конфігурації features.yaml

**Функціональність:**
- Volume Moving Averages (5, 10 days)
- Volume Rate of Change
- Price-Volume Trend
- On-Balance Volume (OBV)
- Volume Relative Strength
- Всього 6 індикаторів

**Чому не в конфігурації:**
- Не включено в конфігурацію
- Причина невідома

**Чи дублює функціональність:**
- Не дублює - це унікальні об'ємні індикатори
- Немає аналогів в інших збагачувачах

**Як підключити:**
```yaml
# В src/config/features.yaml
features:
  enabled_enrichers:
    # ... інші збагачувачі
    volume: true  # Додати цей рядок
```

**Рекомендація:** ✅ Включити в конфігурацію - це корисні об'ємні індикатори

---

## 📋 Підсумок інтеграції

### Аналізатори - всі вже налаштовані
- **Hedge Fund Analyzer:** ✅ В analyzer_registry та unified_config.yaml
- **SHAP Analyzer:** ✅ В analyzer_registry
- **Pattern Analyzer:** ✅ В analysis.yaml

**Як вони працюють:**
- Всі аналізатори викликаються через UnifiedAnalyticsEngine
- UnifiedAnalyticsEngine використовується в Stage 7 (Evaluation)
- Аналізатори не потрібні для основного пайплайну (Stages 1-6)
- Вони виконують спеціалізовані задачі після завершення пайплайну

### Калькулятори - Explainability налаштований, Advanced Econometrics ні
- **Explainability Calculator:** ✅ В technical_analysis_enricher та analysis.yaml
- **Advanced Econometrics Calculator:** ❌ Не в analyzer_registry

**Як вони працюють:**
- Explainability Calculator використовується всередині technical_analysis_enricher
- Advanced Econometrics Calculator - це утиліта для специфічних задач
- Обидва не потрібні для основного пайплайну

### Збагачувачі - Volatility та Volume не в конфігурації
- **Volatility Enricher:** ❌ Не в features.yaml, але корисний
- **Volume Enricher:** ❌ Не в features.yaml, але корисний

**Рекомендація:** Включити обидва в конфігурацію features.yaml

---

## 🎯 Дії для виконання

### 1. Включити Volatility Enricher в конфігурацію
```yaml
# В src/config/features.yaml
features:
  enabled_enrichers:
    # ... інші збагачувачі
    volatility: true
```

### 2. Включити Volume Enricher в конфігурацію
```yaml
# В src/config/features.yaml
features:
  enabled_enrichers:
    # ... інші збагачувачі
    volume: true
```

### 3. Advanced Econometrics Calculator - залишити як утиліту
- Не додавати в основний пайплайн
- Використовувати для специфічних задач економічного аналізу

---

## 📄 Висновок

**Ми це все для чогось створювали:**
- ✅ Аналізатори (Hedge Fund, SHAP, Pattern) - для спеціалізованого аналізу в Stage 7
- ✅ Калькулятори (Explainability) - для explainability всередині technical_analysis_enricher
- ✅ Калькулятори (Advanced Econometrics) - для специфічних економічних задач
- ✅ Збагачувачі (Volatility, Volume) - для додаткових індикаторів в Stage 3

**Чому не задіяні в основному пайплайні:**
- Аналізатори та калькулятори - це спеціалізовані інструменти для аналізу результатів
- Вони викликаються через UnifiedAnalyticsEngine в Stage 7 (Evaluation)
- Не потрібні для основного пайплайну (Stages 1-6)

**Що треба зробити:**
1. ✅ HF_KEY підключено в huggingface_collector.py
2. Включити volatility_enricher в features.yaml
3. Включити volume_enricher в features.yaml
4. Залишити Advanced Econometrics Calculator як утиліту
