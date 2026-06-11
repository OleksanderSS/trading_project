# Stage 7: Evaluation - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_7_evaluation.py`  
**Конфігурація:** `src/config/evaluation.yaml`  
**Призначення:** Оцінка стратегії, бектестинг та глибокий аналіз

---

## 🔧 Архітектура

### Ключові компоненти:

1. **EvaluationStage** - Оркестратор оцінки
2. **AdvancedBacktestEngine** - Двигун бектестингу
3. **UnifiedAnalyticsEngine** - Уніфікований аналітичний двигун
4. **UniversalNotifier** - Універсальний нотифікатор
5. **RealTimeLearning** - Реальне навчання
6. **MetricsCalculator** - Калькулятор фінансових метрик
7. **ReportGenerator** - Генератор звітів
8. **BacktestAnalyzer** - Аналізатор бектестингу

---

## 🔄 Процес оцінки

### Крок 1: Ініціалізація
```python
self.results_dir = Path('data/results')
self.reports_dir = Path('reports/evaluation')
self.results_dir.mkdir(parents=True, exist_ok=True)
self.reports_dir.mkdir(parents=True, exist_ok=True)

self.backtester = AdvancedBacktestEngine(self.config_manager)
self.analytics_engine = UnifiedAnalyticsEngine(self.config_manager)
self.notifier = UniversalNotifier(config_manager)
self.real_time_learning = RealTimeLearning(config_manager)

self.metrics_calc = get_metrics_calculator()
self.report_gen = get_report_generator(self.reports_dir)
self.backtest_analyzer = get_backtest_analyzer(self.backtester)
```

**Компоненти:**
- **AdvancedBacktestEngine** - Реалістичний бектестинг
- **UnifiedAnalyticsEngine** - Глибокий аналіз
- **UniversalNotifier** - Нотифікації
- **RealTimeLearning** - Адаптивне навчання
- **MetricsCalculator** - Фінансові метрики
- **ReportGenerator** - Генерація звітів
- **BacktestAnalyzer** - Аналіз бектестингу

### Крок 2: Завантаження сигналів
```python
signals_data = self._load_signals_data(**kwargs)
```

**Джерела:**
- З kwargs (прямо з Stage 6)
- Fallback на диску

**Дані:**
- signals - торгові сигнали
- trading_activity - історія торгів
- portfolio_summary - підсумок портфеля

### Крок 3: Підготовка сигналів
```python
signals_df = self._prepare_signals_df(signals_data['signals'])
```

**Стандартизація:**
- Конвертація в DataFrame
- Додавання колонки 'signal'
- Додавання колонки 'price'
- Конвертація прогнозів в сигнали (BUY/SELL/HOLD)

**Prediction to Signal:**
```python
def _prediction_to_signal(self, pred) -> str:
    val = pred[-1] if isinstance(pred, (list, tuple, np.ndarray)) and len(pred) > 0 else pred
    if val > 0:
        return 'BUY'
    if val < 0:
        return 'SELL'
    return 'HOLD'
```

### Крок 4: Перевірка чи можна запустити бектест
```python
if not self.backtest_analyzer.can_run_backtest(signals_df):
    self.logger.warning('⚠️ Insufficient numeric price data for backtest. Using basic evaluation.')
    return self._create_basic_evaluation(signals_df, signals_data)
```

**Перевірки:**
- Наявність числових даних про ціни
- Достатня кількість даних
- Валідність сигналів

**Fallback:**
- Якщо бектест неможливий - базова оцінка
- Включає: total_signals, trades_executed, portfolio_value

### Крок 5: Запуск комплексної оцінки
```python
return asyncio.run(self._run_comprehensive_evaluation(signals_df, signals_data))
```

**Асинхронне виконання:**
- Бектестинг
- Розрахунок метрик
- Глибокий аналіз
- Генерація звітів

### Крок 6: Бектестинг
```python
backtest_results = await self.backtest_analyzer.run_backtest(signals_df)
```

**BacktestAnalyzer:**
- Реалістичне виконання торгів
- Урахування комісій
- Slippage
- Позиціонний розмір
- Ризик-менеджмент

**Результати:**
- portfolio_history - історія портфеля
- performance - метрики продуктивності
- trades - історія торгів

### Крок 7: Розрахунок фінансових метрик
```python
financial_metrics = self.metrics_calc.calculate_financial_metrics(portfolio_history)
```

**MetricsCalculator:**
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
- Information Ratio

### Крок 8: Глибокий аналіз
```python
analysis_results = self._run_deep_analysis(signals_df, portfolio_history)
```

**UnifiedAnalyticsEngine:**
- Використовує аналізатори з analyzer_registry
- Включає: drift, hedge_fund, causal_event, shap, drawdown, volatility, fama_french, ensemble_selector
- Глибокий аналіз стратегії

**Data Map:**
```python
data_map = {
    'price_data': signals_df[['price']],
    'portfolio_data': portfolio_history,
    'signals': signals_df['signal']
}
```

### Крок 9: Генерація підсумку
```python
final_summary = self.report_gen.create_evaluation_summary(
    financial_metrics, backtest_results, analysis_results, signals_df
)
```

**ReportGenerator:**
- Об'єднує всі результати
- Створює структурований підсумок
- Включає метрики, графіки, аналіз

### Крок 10: Real-time Learning Adaptation
```python
if signals_data['trading_activity']:
    final_summary['learning_adaptation'] = self.real_time_learning.update_and_adapt(signals_data['trading_activity'])
```

**RealTimeLearning:**
- Оновлення моделей на основі реальних результатів
- Адаптація параметрів
- Meta-learning

### Крок 11: Збереження та побудова графіків
```python
self.report_gen.save_summary(final_summary, self.results_dir)
equity_path = self.report_gen.plot_equity_curve(portfolio_history, financial_metrics)
```

**Збереження:**
- `data/results/evaluation_summary.json` - підсумок оцінки
- `reports/evaluation/equity_curve.png` - графік equity curve

### Крок 12: Нотифікація
```python
msg = self.report_gen.generate_notification_message(financial_metrics)
await self.notifier.send_report(msg, image_path=equity_path)
```

**UniversalNotifier:**
- Відправка звіту
- Підтримка різних каналів (email, slack, telegram)
- Включає графіки

---

## 📊 Financial Metrics

### MetricsCalculator:
- **Total Return** - Загальний дохід
- **Sharpe Ratio** - Коефіцієнт Шарпа (ризик-адаптований дохід)
- **Sortino Ratio** - Коефіцієнт Сортино (ризик-адаптований дохід з урахуванням тільки downside)
- **Max Drawdown** - Максимальне просідання
- **Win Rate** - Відсоток виграшних торгів
- **Profit Factor** - Коефіцієнт прибутку (gross profit / gross loss)
- **Calmar Ratio** - Коефіцієнт Кальмара (annual return / max drawdown)
- **Information Ratio** - Коефіцієнт інформації (excess return / tracking error)

---

## 🔍 Deep Analysis

### UnifiedAnalyticsEngine:
Використовує аналізатори з analyzer_registry:

1. **DriftAnalyzer** - Виявлення drift у фічах
2. **HedgeFundAnalyzer** - Оцінка через призму хедж-фондів
3. **CausalEventFinder** - Виявлення причинно-наслідкових зв'язків
4. **ShapAnalyzer** - SHAP аналіз важливості фіч
5. **DrawdownAnalyzer** - Аналіз просідань
6. **VolatilityAnalyzer** - Аналіз волатильності
7. **FamaFrenchAnalyzer** - Fama-French факторний аналіз
8. **EnsembleSelector** - Вибір ансамблю

---

## 📝 Report Generator

### create_evaluation_summary:
```python
final_summary = {
    'metrics': financial_metrics,
    'backtest_stats': backtest_results.get('performance', {}),
    'analysis': analysis_results,
    'timestamp': pd.Timestamp.now().isoformat()
}
```

### plot_equity_curve:
- Побудова графіку equity curve
- Побудова графіку drawdown
- Побудова графіку P&L
- Збереження в PNG

### generate_notification_message:
- Формування повідомлення
- Включає ключові метрики
- Зрозумілий формат

---

## 🎯 Real-Time Learning

### RealTimeLearning:
- **Update** - Оновлення моделей
- **Adapt** - Адаптація параметрів
- **Meta-Learning** - Навчання на досвіді

**Адаптація:**
- Оновлення ваг моделей
- Коригування параметрів
- Збереження в щоденник

---

## 📈 Результати роботи

### Очікувані дані на виході:
1. **evaluation_summary** - Підсумок оцінки
2. **metrics** - Фінансові метрики
3. **backtest_stats** - Статистика бектестингу
4. **analysis** - Результати глибокого аналізу
5. **learning_adaptation** - Результати адаптації

### Evaluation Summary:
```python
{
    'metrics': {
        'total_return_pct': 15.5,
        'sharpe_ratio': 1.2,
        'sortino_ratio': 1.5,
        'max_drawdown_pct': -8.5,
        'win_rate': 0.65,
        'profit_factor': 2.1,
        'calmar_ratio': 1.8,
        'information_ratio': 0.9
    },
    'backtest_stats': {
        'total_trades': 45,
        'winning_trades': 29,
        'losing_trades': 16,
        'avg_win': 2.5,
        'avg_loss': -1.8
    },
    'analysis': {
        'drift': {...},
        'hedge_fund': {...},
        'shap': {...},
        ...
    },
    'timestamp': '2026-06-06T10:00:00'
}
```

---

## ⚠️ Потенціальні проблеми

### 1. **Fallback на базову оцінку**
```python
if not self.backtest_analyzer.can_run_backtest(signals_df):
    return self._create_basic_evaluation(signals_df, signals_data)
```
- Якщо бектест неможливий - базова оцінка
- Втрачається детальність
- Може бути не інформативною

### 2. **Асинхронне виконання**
```python
return asyncio.run(self._run_comprehensive_evaluation(signals_df, signals_data))
```
- Використовує asyncio.run()
- Може бути проблематично в певних контекстах
- Потрібен proper async handling

### 3. **Відсутність cross-validation**
- Тільки один бектест
- Немає k-fold cross-validation
- Менш надійні оцінки

### 4. **Фіксовані метрики**
- Фіксований набір метрик
- Немає кастомізації
- Може бути не оптимальним для всіх стратегій

---

## ✅ Статус Stage 7

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ EvaluationStage - оркеструє оцінку
- ✅ AdvancedBacktestEngine - реалістичний бектестинг
- ✅ UnifiedAnalyticsEngine - глибокий аналіз
- ✅ UniversalNotifier - нотифікації
- ✅ RealTimeLearning - адаптивне навчання
- ✅ MetricsCalculator - фінансові метрики
- ✅ ReportGenerator - генератор звітів
- ✅ BacktestAnalyzer - аналізатор бектестингу

**Бектестинг:** ✅ Працює
- Реалістичне виконання торгів
- Урахування комісій
- Slippage
- Позиціонний розмір
- Ризик-менеджмент

**Фінансові метрики:** ✅ Працюють
- Total Return
- Sharpe Ratio
- Sortino Ratio
- Max Drawdown
- Win Rate
- Profit Factor
- Calmar Ratio
- Information Ratio

**Глибокий аналіз:** ✅ Працює
- 8 аналізаторів з analyzer_registry
- Drift, Hedge Fund, Causal Event, SHAP
- Drawdown, Volatility, Fama-French, Ensemble

**Real-Time Learning:** ✅ Працює
- Оновлення моделей
- Адаптація параметрів
- Meta-learning

**Звіти:** ✅ Працюють
- JSON підсумок
- Equity curve графік
- Нотифікації

**Рекомендації:**
1. Додати cross-validation
2. Покращити fallback логіку
3. Додати кастомізацію метрик
4. Додати stress testing
5. Додати scenario analysis
6. Покращити async handling
