# Project Integration & Architecture Map

Цей документ містить актуальну мапу інтеграції ключових аналітичних компонентів проекту.

## 1. Реєстр аналізаторів (`src/analytics/analyzer_registry.py`)
Центральна точка доступу до всіх інструментів аналітики. Використовує шаблон Factory.
**Виклик:** `from src.analytics.analyzer_registry import get_analyzer`

### Зареєстровані компоненти:
- `drift`: `DriftAnalyzer`
- `hedge_fund`: `HedgeFundAnalyzer`
- `causal_event`: `CausalEventFinder`
- `shap`: `ShapAnalyzer`
- `drawdown`: `DrawdownAnalyzer`
- `volatility`: `VolatilityAnalyzer`
- `fama_french`: `FamaFrenchAnalyzer`
- `causal_wrapper`: `CausalAnalyzer`
- `ensemble_selector`: `EnsembleSelector`

## 2. Цикл автономного навчання (Feedback Loop)
Логіка інтеграції між `PortfolioManager` та `TradingModelArena`.

- **Сигналізація:** `PortfolioManager.check_risk_exits` автоматично викликає `arena.track_model_failure` при спрацюванні Stop-Loss.
- **Дія:** `TradingModelArena` переводить модель у статус `COOLDOWN` при перевищенні порогу невдач (`failure_threshold = 3`).
- **Захист:** `PortfolioManager._create_buy_order` блокує торгівлю для моделей у статусі `COOLDOWN`.

## 3. Автономне відновлення (`scripts/manage_arena.py`)
Скрипт для фонового обслуговування системи.
- **Дія:** `run_recovery_service` перевіряє статус моделей та час виходу з `COOLDOWN`.
- **Валідація:** Перед активацією (`status: ACTIVE`) модель проходить примусовий `run_blind_challenge` (Arena-test).

## 5. Двигун ознак (Feature Engine)
Розташований у `src/analytics/context/market_context_analyzer.py`.
- **Принцип роботи:** Використовує динамічний виклик методів (`getattr`) на основі списку `context_features`.
- **Інтеграція:** Додано у `vulture_whitelist.py` для уникнення помилок стат-аналізу.
- **Тестування:** `tests/test_feature_engine.py` гарантує, що всі динамічні методи працюють коректно.
