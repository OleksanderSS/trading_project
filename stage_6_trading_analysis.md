# Stage 6: Trading - Детальний аналіз

## 📋 Огляд етапу

**Файл:** `src/pipeline/stages/stage_6_trading_execution.py`  
**Конфігурація:** `src/config/trading.yaml`  
**Призначення:** Виконання торгівлі на основі прогнозів з контекстно-орієнтованим підходом

---

## 🔧 Архітектура

### Ключові компоненти:

1. **TradingExecutionStage** - Оркестратор торгівлі
2. **VirtualPortfolio** - Віртуальний портфель
3. **PostInferenceFilter** - Фільтр після інференсу
4. **DiaryEngine** - Щоденник рішень
5. **EnhancedConsensusEngine** - Покращений консенсус
6. **EliteRiskSizer** - Розрахунок розміру позиції
7. **EliteRiskMetrics** - Метрики ризику
8. **AdaptiveParameterManager** - Адаптивний менеджер параметрів
9. **MaxExposureMonitor** - Монітор максимальної експозиції
10. **PortfolioManager** - Менеджер портфеля
11. **Trader** - Трейдер (paper trading)
12. **TradingOrchestrator** - Оркестратор торгівлі

---

## 🔄 Процес торгівлі

### Крок 1: Ініціалізація торгового стеку
```python
def _initialize_trading_stack(self):
    self.portfolio = VirtualPortfolio()
    self.post_inference_filter = PostInferenceFilter()
    self.diary_engine = DiaryEngine()
    self.enhanced_consensus = EnhancedConsensusEngine()
    self.risk_sizer = EliteRiskSizer(logger=self.logger)
    self.risk_metrics = EliteRiskMetrics(logger=self.logger)
    self.param_manager = AdaptiveParameterManager(logger=self.logger)
    self.exposure_monitor = MaxExposureMonitor(config=self.config_manager.get('strategy.risk_management', {}))
    
    self.portfolio_manager = PortfolioManager(
        virtual_portfolio=self.portfolio,
        elite_risk_sizer=self.risk_sizer,
        config=self.config_manager.get('strategy.risk_management', {})
    )
    self.trader = Trader(paper_trading=True)
    
    self.trading_orchestrator = TradingOrchestrator(
        consensus_engine=None,
        portfolio_manager=self.portfolio_manager,
        virtual_portfolio=self.portfolio,
        trader=self.trader,
        post_inference_filter=self.post_inference_filter
    )
```

**Компоненти:**
- **VirtualPortfolio** - Віртуальний портфель для paper trading
- **EliteRiskSizer** - Розрахунок розміру позиції з урахуванням ризику
- **MaxExposureMonitor** - Моніторинг максимальної експозиції
- **PortfolioManager** - Управління портфелем
- **Trader** - Виконання торгів (paper trading)
- **TradingOrchestrator** - Оркестрація торгівлі

### Крок 2: Завантаження прогнозів
```python
predictions, current_prices = await self._load_or_extract_data(kwargs)
```

**Джерела:**
- З kwargs (прямо з Stage 5)
- З диску (stage_5_results.json)
- Fallback на останній batch

### Крок 3: Застосування контекстних правил
```python
processed_signals = self._apply_context_rules(predictions)
```

**Anxiety Kill-Switch:**
```python
if velocity > 0.7:
    pred['confidence'] *= 0.5  # Штраф 50%
    self.logger.warning(f"🚨 High Context Velocity ({velocity:.2f}) for {ticker}. Reducing exposure.")
```

**Panic Block:**
```python
if velocity > 0.85:
    if (self._extract_model_prediction(pred) or 0.0) > 0:
        pred['confidence'] = 0.0  # Повністю анулюємо сигнал
        self.logger.error(f"🛑 CRITICAL ANXIETY for {ticker}. Blocking BUY signal.")
```

**Context Velocity:**
- **< 0.7** - Нормальний режим
- **0.7 - 0.85** - Висока тривожність (штраф 50%)
- **> 0.85** - Критична тривожність (блок BUY)

### Крок 4: Обробка сигналів
```python
self.trading_orchestrator.process_signals(
    raw_predictions=processed_signals,
    current_prices=current_prices
)
```

**TradingOrchestrator:**
- Фільтрація сигналів
- Консенсус (якщо увімкнено)
- Розрахунок розміру позиції
- Виконання торгів
- Управління портфелем

### Крок 5: Запис транзакцій в щоденник
```python
new_transactions = getattr(self.portfolio, 'transactions', [])[existing_tx_count:]
diary_records_written = self._record_transactions_to_diary(new_transactions, processed_signals)
```

**DiaryEngine:**
- Записує рішення в щоденник
- Включає контекст
- Включає результат
- Використовується для meta-learning

### Крок 6: Фіналізація результатів
```python
portfolio_summary = self.portfolio.get_portfolio_summary(current_prices)
trade_history = getattr(self.portfolio, 'transactions', [])

return {
    'trading_activity': trade_history[-5:],
    'portfolio_summary': portfolio_summary,
    'signals': predictions
}
```

---

## 🎯 Context-Aware Execution

### Концепція:
- **Anxiety Kill-Switch** - Вимикач тривожності
- **Context Velocity** - Швидкість зміни контексту
- **Panic Block** - Блок паніки
- **Pattern-Aware** - Врахування патернів

### Context Velocity:
```python
velocity = pred.get('context_velocity')  # 0.0 - 1.0
```

**Інтерпретація:**
- **0.0 - 0.7** - Нормальний режим
- **0.7 - 0.85** - Висока тривожність
- **> 0.85** - Критична тривожність

### Anxiety Kill-Switch:
- **Velocity > 0.7** - Штраф 50% впевненості
- **Velocity > 0.85** - Блок BUY сигналів
- Запобігає торгівлі в хаотичних ринках

---

## 📊 Risk Management

### EliteRiskSizer:
- **Position Sizing** - Розрахунок розміру позиції
- **Risk-Based** - На основі ризику
- **Kelly Criterion** - Критерій Келлі
- **Volatility-Adjusted** - З урахуванням волатильності

### MaxExposureMonitor:
- **Multi-Layer** - Багатошаровий моніторинг
- **Position Limits** - Ліміти позицій
- **Sector Limits** - Ліміти секторів
- **Total Exposure** - Загальна експозиція

### EliteRiskMetrics:
- **VaR** - Value at Risk
- **Sharpe Ratio** - Коефіцієнт Шарпа
- **Max Drawdown** - Максимальний просідання
- **Win Rate** - Відсоток виграшних торгів

---

## 📝 Diary Engine

### DecisionRecord:
```python
DecisionRecord(
    agent_id=str(model_name),
    ticker=ticker,
    decision_type=DecisionType.BUY or DecisionType.SELL,
    reasoning=str(transaction.get('reason')),
    market_context=market_context,
    context_fingerprint=str(prediction.get('context_fingerprint')),
    context_pattern_seq=prediction.get('context_pattern_seq'),
    model_prediction=model_prediction,
    model_confidence=confidence,
    entry_price=entry_price,
    exit_price=exit_price,
    outcome=DecisionOutcome.PROFITABLE/UNPROFITABLE/BREAK_EVEN/PENDING,
    profit_loss=profit_loss,
    decision_timestamp=timestamp
)
```

**Market Context:**
- transaction_type
- reason
- quantity
- price
- trade_value
- pnl
- pnl_pct
- context_pattern_id
- context_pattern_seq
- context_velocity
- confidence
- raw_forecast
- selected_primary_model

---

## 🎯 Trading Orchestrator

### Процес:
1. **Signal Filtering** - Фільтрація сигналів
2. **Consensus** - Консенсус (якщо увімкнено)
3. **Position Sizing** - Розрахунок розміру позиції
4. **Risk Check** - Перевірка ризику
5. **Execution** - Виконання торгів
6. **Portfolio Update** - Оновлення портфеля

### PostInferenceFilter:
- Фільтрація після інференсу
- Перевірка якості сигналів
- Виключення слабких сигналів

---

## 💾 Virtual Portfolio

### Функціональність:
- **Paper Trading** - Паперові торги
- **Position Tracking** - Відстеження позицій
- **P&L Calculation** - Розрахунок P&L
- **Transaction History** - Історія транзакцій

### Portfolio Summary:
```python
{
    'total_value': 100000.0,
    'cash': 50000.0,
    'positions_value': 50000.0,
    'positions': {...},
    'pnl': 5000.0,
    'pnl_pct': 5.0
}
```

---

## 📈 Результати роботи

### Очікувані дані на виході:
1. **trading_activity** - Останні 5 транзакцій
2. **portfolio_summary** - Підсумок портфеля
3. **signals** - Оброблені сигнали
4. **diary_records_written** - Кількість записів в щоденник

### Trading Activity:
```python
[
    {
        'ticker': 'TSLA',
        'type': 'BUY',
        'quantity': 10,
        'price': 185.50,
        'reason': 'Strong buy signal with high confidence',
        'timestamp': '2026-06-06T10:00:00'
    },
    ...
]
```

### Portfolio Summary:
```python
{
    'total_value': 100000.0,
    'cash': 50000.0,
    'positions_value': 50000.0,
    'positions': {
        'TSLA': {'quantity': 10, 'avg_price': 180.0, 'current_price': 185.50, 'pnl': 550.0}
    },
    'pnl': 5000.0,
    'pnl_pct': 5.0
}
```

---

## ⚠️ Потенціальні проблеми

### 1. **Фіксовані пороги тривожності**
```python
if velocity > 0.7:
    pred['confidence'] *= 0.5  # Штраф 50%
if velocity > 0.85:
    pred['confidence'] = 0.0  # Блок BUY
```
- Фіксовані пороги
- Може бути не оптимальним для всіх ринків
- Немає адаптивності

### 2. **Paper Trading**
```python
self.trader = Trader(paper_trading=True)
```
- Тільки paper trading
- Немає реальних торгів
- Потрібна інтеграція з реальним брокером

### 3. **Консенсус вимкнено**
```python
self.trading_orchestrator = TradingOrchestrator(
    consensus_engine=None,  # Вимкнено
    ...
)
```
- EnhancedConsensusEngine ініціалізовано але не використовується
- Може покращити якість сигналів

### 4. **Відсутність stop-loss**
- Немає явного stop-loss
- Може призвести до великих втрат
- Потрібна інтеграція stop-loss

---

## ✅ Статус Stage 6

**Загальний статус:** ✅ Працює коректно

**Компоненти:**
- ✅ TradingExecutionStage - оркеструє торгівлю
- ✅ VirtualPortfolio - віртуальний портфель
- ✅ PostInferenceFilter - фільтрує сигнали
- ✅ DiaryEngine - записує рішення
- ✅ EliteRiskSizer - розраховує розмір позиції
- ✅ EliteRiskMetrics - метрики ризику
- ✅ AdaptiveParameterManager - адаптивні параметри
- ✅ MaxExposureMonitor - монітор експозиції
- ✅ PortfolioManager - управляє портфелем
- ✅ Trader - виконує торги (paper trading)
- ✅ TradingOrchestrator - оркеструє торгівлю

**Context-Aware Execution:** ✅ Працює
- Anxiety Kill-Switch
- Panic Block
- Context Velocity
- Pattern-Aware

**Risk Management:** ✅ Працює
- EliteRiskSizer
- MaxExposureMonitor
- EliteRiskMetrics
- Multi-layer exposure monitoring

**Diary Engine:** ✅ Працює
- Записує рішення
- Включає контекст
- Включає результат
- Meta-learning ready

**Virtual Portfolio:** ✅ Працює
- Paper trading
- Position tracking
- P&L calculation
- Transaction history

**Trading Orchestrator:** ✅ Працює
- Signal filtering
- Position sizing
- Risk check
- Execution
- Portfolio update

**Рекомендації:**
1. Додати адаптивні пороги тривожності
2. Інтегрувати реальний брокер
3. Увімкнути консенсус
4. Додати stop-loss
5. Додати take-profit
6. Покращити position sizing
