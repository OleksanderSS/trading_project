# DEAN-OS v4.2 — Фінальна архітектура мультиагентної системи

Документ підсумовує брейншторм і фіксує остаточну архітектуру.
Базується на v4.1, виправлені архітектурні помилки, додані відсутні елементи.

> Статус документа: це цільова архітектура й набір інваріантів, а не точний
> опис поточного коду. Перед реалізацією кожного наступного кроку її потрібно
> звіряти з активним pipeline, `dean_os/IMPLEMENTATION_STATUS.md`,
> `dean_os/NEXT_CHAT_HANDOFF.md` і актуальними review-артефактами. Якщо документ
> суперечить перевіреному коду або causal/data contract, пріоритет має
> перевірений контракт, а архітектурний документ оновлюється.

> Поточне уточнення (2026-06-28): аналітичний шар Stage 7 має єдине джерело
> конфігурації в `src/config/analysis.yaml`. Активні лише перевірені
> `market_regime` і `critical_signals`; інші модулі є каталогом можливостей, а
> не активною системою. Кожен запуск фіксує executed/skipped/failed/disabled,
> розділяє ціни за ticker+timeframe і повертає тільки supporting review context.
> Цей шар не може підмінити locked evidence, consensus, promotion або execution.
> У DEAN цей контекст проходить через `dean_stage7_analyzer_review_v1`.
> `ModelPerformanceAgent` не витягує метрики з довільних вкладених полів:
> pipeline-метрики дозволені лише з `evaluation_summary.metrics`.
> Явний model-performance файл приймається лише як verified
> `locked_model_evaluation`, зі source SHA, joined lineage та as-of часом із
> evaluation window. Навіть валідний locked-файл не дає `clear`, якщо повний
> real-metric evidence chain заблокований.

---

## 0. Принципи, що не змінюються

```
Математика керує діями.
LLM керує розумінням.
Агенти — аналітичний штаб, не автономні трейдери.
```

Конкретно:
- Числа завжди з детерміністичного коду (pandas, sklearn, backtest engine).
- LLM тільки пояснює вже прийняте рішення, не приймає його.
- Жоден агент не змінює production config самостійно.
- Жоден агент не підтверджує власні пропозиції.
- За замовчуванням активний pipeline є review-only: Stage 5 переходить прямо до Stage 7; paper і live execution вимкнені.

---

## 1. Загальна топологія

```
                    ┌─────────────────────────┐
                    │      DEAN-Orchestrator   │
                    │    (state machine, sync)  │
                    └────────────┬────────────┘
                                 │
               ┌─────────────────┴─────────────────┐
               │                                   │
               ▼                                   ▼
   ┌───────────────────────┐         ┌───────────────────────┐
   │   PIPELINE BRANCH     │         │  ANALYTICAL BRANCH    │
   │  Deterministic / Sync │         │  Heuristic / Parallel │
   │                       │         │                       │
   │  PipelineAuditAgent   │         │  MacroHistoryAgent    │
   │    [Hard Veto]        │         │    [Context only]     │
   │  DataQualityAgent     │         │  SectorAgents         │
   │    [Hard Veto]        │         │    [Context / P2]     │
   │  RiskAgent            │         │  GrahamAgent          │
   │    [Hard Veto]        │         │    [Context / P3]     │
   │  ModelPerformanceAgent│         │  BuffettAgent         │
   │    [Soft]             │         │    [Context / P3]     │
   │  TuningAgent          │         │                       │
   │    [Proposal Only]    │         │  → виводять           │
   │  RegimeAgent          │         │    PositionBias,      │
   │    [Soft]             │         │    не ExecutionSignal │
   └───────────┬───────────┘         └───────────┬───────────┘
               │                                 │
               └─────────────┬───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ConsensusEngine │
                    │  (deterministic)│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ ConsensusDecision│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ DecisionLogger  │  ← SHA-256 hashes всього
                    └────────┬────────┘
                             │
               ┌─────────────┴─────────────┐
               │                           │
               ▼                           ▼
   ┌───────────────────────┐   ┌───────────────────────┐
   │  Paper Trading /      │   │   Narrative Log        │
   │  Human Review Layer   │   │   (LLM explanation)    │
   └───────────┬───────────┘   └───────────────────────┘
               │
               ▼
   ┌───────────────────────┐
   │  Live Execution Policy│  ← вимкнено за замовчуванням
   └───────────────────────┘
```

**Виправлена помилка v4.1:** Agent Registry — це конфігурація, що читається при старті для інстанціювання агентів. Він не є runtime-компонентом між агентами і ConsensusEngine. Registry живе поза топологією.

---

## 2. Інтеграція з існуючим DEAN pipeline

Це питання, якого у v4.1 не було. DEAN-OS огортає HybridOrchestrator, а не замінює його.

```python
class DEANOrchestrator:
    """
    DEAN-OS є зовнішнім шаром поверх існуючого DEAN pipeline.
    Він не замінює HybridOrchestrator — він вирішує, чи можна
    йому довіряти, і що робити з його результатами.
    """

    def __init__(self, config_manager, agent_registry):
        self.pipeline = HybridOrchestrator(config_manager)  # існуючий
        self.agents   = agent_registry.load_all()

    async def run(self, context: MarketContext) -> OrchestratorDecision:
        # 1. Pipeline branch — Guardian агенти спочатку
        pipeline_reports = await self.pipeline_branch.run(context)

        # 2. Hard veto — якщо є блокер, аналітики не запускаються
        if veto := self._find_hard_veto(pipeline_reports):
            return self._blocked_decision(veto)

        # 3. Запускаємо існуючий DEAN pipeline
        pipeline_result = await self.pipeline.run(
            tickers=context.tickers,
            timeframes=context.timeframes
        )

        # 4. Analytical branch — паралельно, незалежно
        analytical_reports = await self.analytical_branch.run_parallel(context)

        # 5. Consensus
        decision = self.consensus.combine(
            pipeline_reports, pipeline_result, analytical_reports
        )

        # 6. Log все
        self.decision_logger.log(decision, pipeline_reports, analytical_reports)

        return decision
```

---

## 3. Agent Registry

Registry — конфігурація, не runtime-компонент. Читається один раз при старті.

```yaml
# config/agent_registry.yaml

agents:
  pipeline_audit:
    class_path: dean_os.agents.pipeline_audit:PipelineAuditAgent
    branch: pipeline
    veto_level: hard
    enabled: true
    error_behavior: block       # якщо впав — блокує pipeline
    timeout_seconds: 5
    required_inputs:
      - audit_reports/findings.json
      - audit_reports/triage_P0_P1.md
    max_findings_age_hours: 24  # ← стale findings блокують

  data_quality:
    class_path: dean_os.agents.data_quality:DataQualityAgent
    branch: pipeline
    veto_level: hard
    enabled: true
    error_behavior: block
    timeout_seconds: 10

  risk:
    class_path: dean_os.agents.risk:RiskAgent
    branch: pipeline
    veto_level: hard
    enabled: true
    error_behavior: block
    timeout_seconds: 10

  model_performance:
    class_path: dean_os.agents.model_performance:ModelPerformanceAgent
    branch: pipeline
    veto_level: soft
    enabled: true
    error_behavior: skip        # якщо впав — пропускається з попередженням
    timeout_seconds: 15

  regime:
    class_path: dean_os.agents.regime:RegimeAgent
    branch: pipeline
    veto_level: soft
    enabled: true
    error_behavior: skip
    timeout_seconds: 10

  tuning:
    class_path: dean_os.agents.tuning:TuningAgent
    branch: pipeline
    veto_level: none
    enabled: false              # вмикається після Guardian MVP стабільний
    proposal_only: true
    error_behavior: warn
    timeout_seconds: 300        # Optuna може довго бігти

  macro_history:
    class_path: dean_os.agents.macro_history:MacroHistoryAgent
    branch: analytical
    veto_level: none
    enabled: false              # P1: після RSS/news pipeline стабільний
    error_behavior: skip
    timeout_seconds: 30

  sector:
    class_path: dean_os.agents.sector:SectorAgent
    branch: analytical
    veto_level: none
    enabled: false              # P2: після sector ETF data
    error_behavior: skip
    timeout_seconds: 20

  graham:
    class_path: dean_os.agents.graham:GrahamAgent
    branch: analytical
    veto_level: none
    enabled: false              # P3: після fundamental data feed
    error_behavior: skip
    timeout_seconds: 20

  buffett:
    class_path: dean_os.agents.buffett:BuffettAgent
    branch: analytical
    veto_level: none
    enabled: false              # P3: після fundamental data feed
    error_behavior: skip
    timeout_seconds: 20
```

```python
# registry.py
class AgentRegistry:
    def load_all(self, context: RuntimeContext) -> list[BaseAgent]:
        agents = []
        for name, cfg in self._config.items():
            if not cfg.get('enabled', False):
                continue
            agent = self._instantiate(cfg)
            if not agent.check_prerequisites(context):  # runtime check
                logger.warning(f"{name}: prerequisites not met, skipping")
                continue
            agents.append(agent)
        return agents
```

---

## 4. Agent Capabilities

```python
from pydantic import BaseModel
from typing import Literal

class AgentCapabilities(BaseModel):
    can_veto:                  bool = False
    can_modify_pipeline:       bool = False  # ніколи True для агентів
    can_generate_trade_signal: bool = False  # тільки ConsensusEngine
    can_access_network:        bool = False
    can_use_llm:               bool = False
    requires_human_review:     bool = True
    timeout_seconds:           int  = 10
    error_behavior:            Literal["block", "skip", "warn"] = "skip"
```

Правило, що не змінюється:
```
Жоден агент не може:
- самостійно змінювати production config
- самостійно запускати live trade
- самостійно перезаписувати model artifact
- самостійно змінювати test set
- підтверджувати власні пропозиції
```

---

## 5. Схеми (виправлені)

```python
from pydantic import BaseModel, Field, model_validator
from typing import Any, Literal, Optional, Union

# ── Типи ─────────────────────────────────────────────────────────────────────

Verdict = Literal[
    "clear", "caution", "blocked",
    "bullish", "bearish", "neutral",
    "undervalued", "overvalued", "needs_more_data",
]

DecisionType = Literal[
    "blocked",
    "no_trade",
    "watchlist",
    "paper_trade_only",
    "candidate_long",
    "candidate_short",
    "reduce_position",
    "exit_position",
    "needs_more_data",
]

# ── EvidenceItem ──────────────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    source_type: Literal[
        "metric", "file", "news", "mlflow",
        "audit_finding", "dataframe_check", "config"
    ]
    source:    str
    key:       str
    # ВИПРАВЛЕНО v4.1: Any → JSON-serializable types
    value:     Union[str, int, float, bool, list, dict, None]
    timestamp: Optional[str] = None

# ── BaseAgentReport ───────────────────────────────────────────────────────────

class BaseAgentReport(BaseModel):
    agent_name:    str
    agent_version: str
    branch:        Literal["pipeline", "analytical"]
    verdict:       Verdict

    # ВИПРАВЛЕНО v4.1: розділено на три поля
    confidence:         float = Field(..., ge=0.0, le=1.0)
    data_quality_score: float = Field(..., ge=0.0, le=1.0)
    signal_strength:    Optional[float] = Field(None, ge=-1.0, le=1.0)

    reasons:     list[str]
    risks:       list[str]
    blind_spots: list[str]  # обов'язково — чого агент НЕ знає
    evidence:    list[EvidenceItem]  # обов'язково — чим підтверджується

    input_hash:  Optional[str] = None
    config_hash: Optional[str] = None
    timestamp:   str

# ── Pipeline агенти ───────────────────────────────────────────────────────────

class PipelineReport(BaseAgentReport):
    branch: Literal["pipeline"] = "pipeline"
    metrics_snapshot: dict[str, Any]

    @model_validator(mode='after')
    def hard_veto_needs_evidence(self) -> 'PipelineReport':
        """Hard veto без evidence — неприпустимо."""
        # (veto_level визначається registry, але перевіряємо тут)
        if self.verdict == "blocked" and len(self.evidence) == 0:
            raise ValueError("Blocked verdict requires at least one EvidenceItem")
        return self

# ── Analytical агенти ─────────────────────────────────────────────────────────

class AnalyticalReport(BaseAgentReport):
    branch: Literal["analytical"] = "analytical"
    ticker:       str
    horizon_years: float

    thesis:       str  # одне речення — головна ідея
    data_quality: Literal["strong", "partial", "weak"]

    # ДОДАНО: явний position bias, окремо від verdict
    position_bias: Literal["bullish", "bearish", "neutral", "insufficient_data"]

# ── ConsensusDecision ─────────────────────────────────────────────────────────

class ConsensusDecision(BaseModel):
    decision_id: str
    decision:    DecisionType

    # ВИПРАВЛЕНО v4.1: trade_allowed — derived property, не окреме поле
    @property
    def trade_allowed(self) -> bool:
        return self.decision in (
            "candidate_long", "candidate_short", "paper_trade_only"
        )

    requires_human_approval: bool = True  # True за замовчуванням

    final_score: float = Field(..., ge=-1.0, le=1.0)
    confidence:  float = Field(..., ge=0.0, le=1.0)

    blocking_agents:  list[str]
    supporting_agents: list[str]
    opposing_agents:  list[str]

    reasons:     list[str]
    risks:       list[str]
    blind_spots: list[str]
    evidence:    list[EvidenceItem]

    # ВИПРАВЛЕНО v4.1: position sizing — відповідальність RiskAgent, не тут
    # ConsensusDecision тільки передає що RiskAgent порахував:
    risk_context: Optional[dict[str, Any]] = None  # VaR, drawdown, position cap

    agent_report_hashes: dict[str, str]
    config_hash: str
    timestamp:   str
```

---

## 6. Pipeline branch — логіка виконання

```python
class PipelineBranch:
    HARD_VETO_AGENTS = {"pipeline_audit", "data_quality", "risk"}

    async def run(self, context) -> list[PipelineReport]:
        reports = []

        for agent in self.agents:
            try:
                async with asyncio.timeout(agent.capabilities.timeout_seconds):
                    report = await agent.run(context)
                reports.append(report)

                # Зупинка після першого hard veto
                if (agent.name in self.HARD_VETO_AGENTS
                        and report.verdict == "blocked"):
                    break

            except TimeoutError:
                self._handle_timeout(agent, reports)
            except Exception as e:
                self._handle_error(agent, e, reports)

        return reports

    def _handle_error(self, agent, exc, reports):
        behavior = agent.capabilities.error_behavior
        if behavior == "block":
            reports.append(self._error_report(agent, exc, verdict="blocked"))
        elif behavior == "skip":
            logger.warning(f"{agent.name} failed, skipping: {exc}")
        elif behavior == "warn":
            logger.error(f"{agent.name} failed: {exc}")
```

---

## 7. Analytical branch — Position Bias, не Execution Signal

Аналітичні агенти ніколи не кажуть buy/sell/execute. Вони дають:

```
bullish_context     → position_bias = "bullish"
bearish_context     → position_bias = "bearish"
valuation_risk      → risks += [...]
sector_tailwind     → reasons += [...]
historical_analogy  → thesis = "..."
watchlist_priority  → confidence > threshold
```

ConsensusEngine перетворює це у:

```python
analytical_modifier = 1.0
for report in analytical_reports:
    if report.data_quality == "weak":
        continue  # агент без даних не впливає

    discount = _horizon_discount(report.horizon_years, signal_timeframe)
    weight   = report.confidence * discount

    if report.position_bias == "bullish":
        analytical_modifier += ANALYTICAL_STEP * weight
    elif report.position_bias == "bearish":
        analytical_modifier -= ANALYTICAL_STEP * weight

# Clamp: аналітика може змінити score максимум на ±20%
analytical_modifier = max(0.80, min(1.20, analytical_modifier))
```

---

## 8. Horizon Discounting — policy-based (виправлено)

```yaml
# config/horizon_policy.yaml

horizon_discount:
  intraday_execution:        # 15m / 1h signals
    max_analytical_modifier: 0.05   # майже нульовий вплив
    min_discount_factor:     0.0

  swing_trading:             # 1d signals
    max_analytical_modifier: 0.15
    min_discount_factor:     0.03

  portfolio_allocation:      # тижневий / місячний горизонт
    max_analytical_modifier: 0.25
    min_discount_factor:     0.05
```

```python
def _horizon_discount(agent_horizon_years: float,
                      signal_timeframe: str) -> float:
    policy = HORIZON_POLICY[_map_timeframe(signal_timeframe)]
    # Логарифмічне затухання: довший горизонт агента = менший вплив на короткий сигнал
    discount = 1.0 / (1.0 + math.log(1.0 + agent_horizon_years))
    return max(policy["min_discount_factor"],
               min(1.0, discount))
```

---

## 9. ConsensusEngine

```python
class ConsensusEngine:
    def combine(
        self,
        pipeline_reports:    list[PipelineReport],
        pipeline_result:     dict,
        analytical_reports:  list[AnalyticalReport]
    ) -> ConsensusDecision:

        # ── Рівень 1: Hard veto ───────────────────────────────────────────────
        for r in pipeline_reports:
            if r.verdict == "blocked" and r.agent_name in HARD_VETO_AGENTS:
                return ConsensusDecision(
                    decision="blocked",
                    final_score=-1.0,
                    confidence=r.confidence,
                    blocking_agents=[r.agent_name],
                    reasons=r.reasons,
                    evidence=r.evidence,
                    ...
                )

        # ── Рівень 2: Pipeline score (детерміністичний) ───────────────────────
        model_score    = self._extract_model_score(pipeline_result)
        regime_score   = self._extract_regime_score(pipeline_reports)
        risk_score     = self._extract_risk_score(pipeline_reports)

        pipeline_score = (
            0.40 * model_score +
            0.35 * risk_score +
            0.25 * regime_score
        )

        # ── Рівень 3: Analytical modifier (±20% max) ─────────────────────────
        analytical_modifier = self._compute_analytical_modifier(
            analytical_reports, pipeline_result.get("timeframe")
        )

        final_score = pipeline_score * analytical_modifier

        # ── Рівень 4: Decision type ───────────────────────────────────────────
        decision = self._map_score_to_decision(final_score, pipeline_reports)

        # ── LLM тільки для пояснення — числа вже є ───────────────────────────
        narrative = self.llm.explain(
            pipeline_reports, analytical_reports, final_score
        ) if self.llm_enabled else ""

        return ConsensusDecision(
            decision=decision,
            final_score=final_score,
            confidence=self._compute_confidence(pipeline_reports),
            requires_human_approval=True,  # завжди True для MVP
            ...
        )

    def _map_score_to_decision(self, score, pipeline_reports) -> DecisionType:
        # Перевіряємо спочатку чи є soft veto сигнали
        if any(r.verdict == "caution" for r in pipeline_reports):
            return "watchlist"

        if score > 0.70:   return "candidate_long"
        if score < -0.70:  return "candidate_short"
        if score > 0.40:   return "watchlist"
        if score < -0.40:  return "watchlist"
        return "no_trade"
```

---

## 10. TuningAgent — proposal lifecycle

TuningAgent створює proposals, не змінює production.

```python
# Multi-objective reward function — не "максимальний прибуток"
def tuning_objective(trial, data):
    params = suggest_params(trial)
    results = walk_forward_evaluation(params, data, n_splits=5)

    # Hard constraints — порушення = -inf
    if results.max_drawdown > RISK_AGENT_THRESHOLD:
        return float('-inf')
    if has_unresolved_p0_findings():
        return float('-inf')
    if used_synthetic_in_validation(results):
        return float('-inf')

    # Multi-objective
    return (
        results.mean_sharpe
        - 0.3 * results.sharpe_std_across_regimes   # стабільність
        - 0.2 * results.max_drawdown
        - 0.1 * results.turnover_penalty
        - 0.1 * results.transaction_cost_estimate
        - 0.05 * results.complexity_score
    )
```

Lifecycle proposals:

```python
class TuningProposal(BaseModel):
    proposal_id:       str
    status:            Literal["pending", "approved", "rejected", "expired"]
    proposed_change:   str
    expected_effect:   str
    backtest_results:  dict[str, Any]
    created_at:        str
    expires_at:        str       # TTL: якщо не розглянуто за N днів → expired
    requires_backtest: bool = True
    requires_human_review: bool = True
    allowed_for_production: bool = False  # завжди False до approval
```

---

## 11. Decision Logging + Replay

Decision Logging — Етап 1.5, найважливіший окремий елемент.

```python
class DecisionLogger:
    def log(
        self,
        decision:           ConsensusDecision,
        pipeline_reports:   list[PipelineReport],
        analytical_reports: list[AnalyticalReport],
        input_snapshot:     dict,   # ← ВАЖЛИВО: не тільки hash, а snapshot
        config:             dict,
        pipeline_git_commit: str,
    ) -> str:
        """Повертає decision_id."""

        entry = {
            "event_type":    "decision",
            "decision_id":   decision.decision_id,
            "timestamp":     decision.timestamp,

            # Hashes для верифікації цілісності
            "input_hash":    sha256(input_snapshot),
            "config_hash":   sha256(config),
            "pipeline_commit": pipeline_git_commit,
            "risk_policy_version": RISK_POLICY_VERSION,
            "audit_baseline_hash": sha256(findings_json),

            # Snapshot вхідних даних — потрібен для реального replay
            "input_snapshot": input_snapshot,  # ← v4.1 це пропустила

            # Report hashes для перевірки
            "agent_report_hashes": {
                r.agent_name: sha256(r.model_dump())
                for r in pipeline_reports + analytical_reports
            },

            "final_decision": decision.decision,
            "final_score":    decision.final_score,

            # Narrative (LLM) — окремо, щоб не засмічував структуровані дані
            "narrative_log_id": self._save_narrative(decision),
        }

        self._append_to_log(entry)
        return decision.decision_id
```

```bash
# replay_decision.py
python dean_os/replay_decision.py --decision-id 2026-05-14_NVDA_001

# Показує:
# - які inputs були (з snapshot)
# - які агенти спрацювали
# - хто заблокував і чому
# - metrics snapshot
# - config що діяв
# - git commit pipeline
# - чому фінальне рішення саме таке
```

Retention policy:
```yaml
# config/logging_policy.yaml
decision_log:
  retention_days: 365
  archive_after_days: 90
  compress_snapshots: true
```

---

## 12. Human Approval / Paper Trading Layer

```python
class ExecutionGateway:
    """
    Між ConsensusDecision і реальним ринком.
    Поки немає paper-trading history і стабільних P0 tests:
        live_execution_enabled = False
    """
    def __init__(self, policy: ExecutionPolicy):
        self.policy = policy

    def process(self, decision: ConsensusDecision) -> ExecutionOutcome:
        # 1. Paper trading gate
        if not self.policy.live_execution_enabled:
            return self._paper_trade(decision)

        # 2. Human approval gate
        if decision.requires_human_approval:
            return self._queue_for_human_review(decision)

        # 3. Live execution (тільки після явного вмикання в policy)
        return self._execute_live(decision)
```

Paper trading потребує явної моделі виконання:
```yaml
# config/paper_trading.yaml
execution_model:
  price_model:  "last_close"    # або "vwap", "mid_quote"
  slippage_bps: 5               # 5 basis points
  commission_per_trade: 1.0     # USD
  market_impact_model: "linear" # для великих позицій
```

---

## 13. Agent Memory — тільки event log

```python
# Дозволено зберігати:
ALLOWED_MEMORY_TYPES = {
    "decision",          # past decisions
    "agent_report",      # agent report snapshots
    "model_version",     # which model was used
    "data_range",        # what data was used
    "known_incident",    # documented failures
    "suppression",       # known false positives
    "post_mortem",       # incident analysis
    "tuning_proposal",   # TuningAgent proposals
}

# Заборонено:
FORBIDDEN_MEMORY_TYPES = {
    "llm_free_reasoning",    # вільні LLM думки без підтвердження
    "unverified_opinion",    # ринкові думки без evidence
    "claim_without_evidence",
}
```

Retention:
```yaml
memory:
  ttl_days: 90
  archive_after_days: 30
  max_entries_per_type: 1000
```

---

## 14. Схема агентів — що потрібно для даних

| Агент | Мінімальні дані | Наявність у DEAN | Пріоритет |
|-------|-----------------|------------------|-----------|
| PipelineAuditAgent | findings.json | ✅ є (audit kit) | P0 |
| DataQualityAgent | DataFrame stats | ✅ є | P0 |
| RiskAgent | Returns, positions | ✅ є | P0 |
| ModelPerformanceAgent | MLflow / Arena results | ✅ є | P1 |
| RegimeAgent | OHLCV + indicators | ✅ є | P1 |
| TuningAgent | Walk-forward results | ✅ є | P1 |
| MacroHistoryAgent | RSS / news | ✅ є | P1 |
| SectorAgents | Sector ETF data | ⚠️ частково | P2 |
| GrahamAgent | P/E, P/B, D/E, CR | ❌ потрібен feed | P3 |
| BuffettAgent | ROE, FCF, revenue history | ❌ потрібен feed | P3 |

Graham/Buffett без фундаментальних даних → `data_quality: weak` → автоматично ігноруються ConsensusEngine. Правильна поведінка.

---

## 15. Файлова структура MVP

```
dean_os/
├── schemas.py              ← всі pydantic моделі
├── base.py                 ← BaseAgent ABC з timeout + error_behavior
├── registry.py             ← читає YAML, instantiates, перевіряє prerequisites
├── consensus.py            ← ConsensusEngine (без LLM спочатку)
├── decision_logger.py      ← SHA-256 hashes + input snapshots
├── replay.py               ← replay_decision.py
├── execution_gateway.py    ← Paper Trading / Human Review Layer
│
├── agents/
│   ├── __init__.py
│   ├── pipeline_audit.py   ← читає findings.json, перевіряє свіжість
│   ├── data_quality.py     ← NaN, synthetic, stale, bfill без groupby
│   ├── risk.py             ← VaR, drawdown (огортає існуючий risk module)
│   ├── model_performance.py
│   ├── regime.py           ← огортає існуючий regime_detector.py
│   ├── tuning.py           ← Optuna wrapper, proposal_only
│   ├── macro_history.py
│   ├── sector.py
│   ├── graham.py
│   └── buffett.py
│
├── config/
│   ├── agent_registry.yaml
│   ├── horizon_policy.yaml
│   ├── paper_trading.yaml
│   └── logging_policy.yaml
│
└── tests/
    ├── test_schemas.py
    ├── test_pipeline_audit.py   ← offline, fixture findings.json
    ├── test_data_quality.py     ← offline, synthetic DataFrame
    ├── test_risk.py             ← offline, synthetic returns
    ├── test_consensus.py        ← offline, mock reports
    ├── test_decision_logger.py
    └── test_replay.py
```

---

## 16. Порядок реалізації

```
[ Зараз ]
  Закрити 4 критичних з аудиту v3:
  - train_test_split без shuffle=False (10 місць)
  - cov() без dropna (optimizer.py)
  - Sharpe std=0 guard (2 файли)
  - result_queue.get_nowait() без try/except

      ↓

[ ЕТАП 1: Guardian MVP ]
  1. schemas.py
  2. agent_registry.yaml
  3. base.py (BaseAgent з timeout, error_behavior)
  4. agents/pipeline_audit.py
  5. agents/data_quality.py
  6. agents/risk.py
  7. consensus.py (без LLM)
  8. execution_gateway.py (paper_trade_only)
  9. Тести для кожного (offline, без мережі)

      ↓

[ ЕТАП 1.5: Decision Logging + Replay ]
  1. decision_logger.py
  2. replay.py
  3. Тести для logging та replay
  4. Перші paper-trade runs з логуванням

      ↓

[ ЕТАП 2: Alpha & Optimization Layer ]
  1. agents/model_performance.py
  2. agents/regime.py
  3. agents/tuning.py (proposal_only, multi-objective)
  4. agents/macro_history.py (на RSS що вже є)
  5. horizon_policy.yaml та Horizon Discounting
  6. LLM пояснення у ConsensusEngine

      ↓

[ ЕТАП 3: Fundamental Depth ]
  1. Fundamental data feed (Yahoo Finance fundamentals / SEC)
  2. agents/sector.py
  3. agents/graham.py
  4. agents/buffett.py
  5. Horizon-discounted portfolio context

```

---

## 17. Ризики та як їх уникнути

| Ризик | Як уникнути |
|-------|-------------|
| Агенти пояснюють leakage впевнено | PipelineAuditAgent блокує до закриття P0 |
| TuningAgent overfits до history | Multi-objective з walk-forward, proposal_only |
| Graham/Buffett без даних галюцинують | `data_quality: weak` → ігноруються автоматично |
| Replay не відтворює результат | Input snapshots, не тільки хеші |
| Agent Memory стає сміттям | Event log тільки, TTL 90 днів |
| Занадто багато decision types → dead code | Починати з 4: blocked/no_trade/watchlist/candidate |
| Paper trading без slippage → оптимізм | Явна модель виконання в paper_trading.yaml |
| Аналітики впливають на intraday | Horizon discount max 0.05 для intraday |
| Система стає overcomplicated | Guardian MVP перший, решта поетапно |

---

## 18. Що НЕ робити

- **Graham/Buffett у MVP** — без fundamental data це театр.
- **MacroHistoryAgent як прогнозист** — тільки context warning, не прогноз.
- **LLM у ConsensusEngine для числових рішень** — LLM тільки для narrative.
- **Live execution до paper-trading history** — вимкнено за замовчуванням.
- **TuningAgent з правом змінювати production** — proposal_only завжди.
- **Agent Registry як runtime-компонент** — це конфігурація, читається при старті.
- **Free-form LLM memory** — тільки structured event log.
- **Всі 9 decision types одразу** — починати з 4, розширювати за потребою.

Implementation note 2026-06-14:

- Current DEAN-OS work is in the replay/calibration lane, not live execution.
- Repaired price replay is clean enough for diagnostics, but analyst calibration remains blocked by evidence coverage and neutral/inconclusive research.
- `ReplayEvidenceWindowSelector` now selects windows where repaired prices, future outcome horizon, and pre-`as_of` evidence overlap.
- Latest selected replay windows are `2026-03-04` through `2026-04-01`; after the stance-rule fix selected research replay returns `constructive=4`, `mixed=1`.
- `TickerSpecificAttributionAudit` now shows the next blocker clearly: 0/5 selected replay runs are ticker-ready, all 5 selected notes are still basket notes, and 2 early windows have weak direct ticker evidence.
- `TickerFocusedResearchNoteBuilder` now builds separate ticker-focused note candidates from the same pre-`as_of` evidence: 3/5 selected windows are focused-note-ready, while 2 early `TSM` windows still require direct evidence backfill.
- `TickerFocusedReplayExamBridge` now compares original basket-note exams with ticker-focused overlays: 3/5 are overlay-ready, 2 early `TSM` windows remain blocked, and `AMD` on `2026-04-01` remains neutral instead of being forced bullish.
- Optional focused overlay integration is now implemented in historical research replay; default behavior is unchanged, original basket-note exams are preserved, and applied focused overlays make 3/5 runs ticker-ready while 2 early `TSM` windows remain blocked by weak direct evidence.
- Sector/industry agents are still valid and important, but their output is a sector thesis, not automatically a ticker thesis. A sector thesis can map to a basket or candidate list only through a separate attribution/allocation layer that checks direct ticker evidence, valuation/risk, and portfolio constraints.
- Next architecture-safe step is focused-overlay evidence expansion or a sector-thesis-to-ticker-basket contract before adding more specialist profiles or changing analyst weights.

Implementation note 2026-06-18:

- `SectorThesisToTickerBasketBridge` is now implemented as the first concrete sector-to-ticker contract.
- Current real run: `partial_basket_ready` for `semiconductor_ai_infrastructure`, with `AMD` and `TSM` as reviewed ticker candidates.
- The bridge deliberately keeps `sector_thesis`, `ticker_candidate_map`, and `direct_ticker_thesis` as separate levels.
- `TSM` remains evidence-limited because 2 early replay windows are still blocked by weak direct evidence; the bridge therefore must not be treated as calibration-ready.
- Latest full DEAN-OS tests: `132 passed`.
- Recommended next local module: `SectorToTickerReviewPacket` / `DomainSpecialistReviewPacket`, JSON/Markdown-only, no learning writes, no analyst weight changes, no config writes, no recommendations, and no trading.
- Recommended architecture path: stabilize one high-quality domain specialist pattern first, then clone that contract into other sectors.
- Assistant workbench package in `dean_os/draft/dean_os_after_245_full_context_bundle` stays staged-only/review-only. Its next block is `246_review_only_real_source_normalized_packet_validation_gate_v1`; do not promote draft fixtures into DEAN-OS as facts.

Implementation note 2026-06-18, domain-first correction:

- Web/draft AMD work is a pilot fixture, not the architecture axis.
- Domain/sector agents analyze their economic sphere first: sources, claims, events, sectors, topics, supply-chain exposure, macro/industry context.
- Tickers come from the asset/universe dictionary as candidate entities and exposure nodes.
- `DomainSpecialistReviewPacket` is now the correct standardization candidate: it starts with `domain_thesis`, `source_evidence_context`, `claims_events_entities`, and `sector_exposure_map`.
- `SectorToTickerReviewPacket` remains a lower bridge/gate for direct ticker evidence only.
- A domain thesis may be manually reviewable even when ticker candidate review is blocked.
- No sector/domain packet is a recommendation, price target, allocation, learning promotion, paper trade, or live trade.

Implementation note 2026-06-18, source evidence gate:

- The useful part of draft block 245 is the validation boundary, not a replacement ingestion stack.
- `SourceEvidenceValidationGate` now validates local source artifacts before domain-specialist review.
- Draft normalized packet fixtures are accepted only as staged contract material and are explicitly blocked from evidence promotion.
- Local `AnalystEvidencePack` artifacts can enter manual domain research when shape checks pass; current real pack has timestamp warnings but no failures.
- Claim/event/entity extraction remains a separate later contract and is not performed by this gate.
- Recommended order is source evidence validation, domain-specialist review, then sector-to-ticker bridge only where direct ticker evidence exists.

Implementation note 2026-06-18, extraction contract:

- Draft block 247 is now represented locally by `SourceExtractionReviewPacket`.
- It defines the review-only contract for future candidate claims, events, entity mentions, topics, sectors, assets, financial implication candidates, and source anchors.
- It does not execute extraction and does not emit claims/events/entities.
- Current real packet is `extraction_contract_ready_with_warnings`: source anchors exist, but 111 source units lack `published_at` timestamps and event chronology must stay limited.
- This contract must be reviewed before any fixture-only extraction stage is built.
- Future extraction output must remain staged/review-only until a separate promotion gate exists.

Implementation note 2026-06-18, extraction fixture:

- Draft block 248 is now represented locally by `SourceExtractionFixturePacket`.
- It materializes candidate output shapes over a small anchor subset, but still performs no real semantic extraction.
- Current real fixture has 12 claim fixtures, 12 event fixtures, 12 entity fixtures, and 12 financial implication fixtures.
- The selected anchors are entity-bearing but timestamp-limited, so the packet remains `extraction_fixture_ready_with_warnings`.
- Candidate fixtures are not evidence, resolved entities, financial implications, recommendations, or trade signals.
- The next safe work is manual review of fixture shape and timestamp repair strategy, not promotion or trading.

Implementation note 2026-06-18, fixture review gate:

- `SourceExtractionFixtureReviewGate` now reviews the 248 fixture shape before any real extractor implementation.
- Current real gate result is `fixture_review_ready_with_warnings`: candidate groups and anchor links are valid, evidence/downstream boundaries are disabled, but selected anchors lack timestamps.
- `can_standardize_fixture_shape=false` until timestamp limitations are repaired or explicitly accepted.
- This keeps fixture shape review separate from extraction execution, evidence promotion, learning writes, recommendations, allocation, and trading.

Implementation note 2026-06-20, pipeline-control real metric evidence:

- Synthetic pipeline-control fixture validation remains useful only as a control-flow proof.
- `PipelineControlRealMetricEvidenceRun` is now the real-evidence counterpart.
- It accepts non-synthetic saved/past/locked `model_evaluation_json` plus `feature_stability_report` and runs readiness -> surface -> instance -> caution review.
- It rejects synthetic/fixture artifacts even if their fields would clear the metric planes.
- Current run without supplied real model/feature artifacts is `real_metric_evidence_rejected`.
- Real cautions remain `risk`, `validation`, and `feature_stability` until locked drawdown, holdout/sample metrics, and feature-stability evidence are supplied.
- No collectors, replay reruns, training, autonomous tuning, config writes, learning writes, recommendations, paper trades, or live trades are enabled by this runner.

Implementation note 2026-06-20, staged workbench integration review:

- `StagedWorkbenchIntegrationReview` now inspects the web-bot staged bundle under `dean_os/draft/dean_os_after_245_full_context_bundle`.
- The review classifies staged blocks as integrate candidates, documentation-only, audit-history-only, redundant metadata ladders, defective/superseded, or manual-review-needed.
- Blocks 243-245 are the immediate integration direction, but only through existing main repo real-source modules.
- Strict staged file candidates are limited to four test-intent files around blocks 243-245; canonical snapshot code remains manual-diff/history.
- Blocks 216-238 are mostly repeated contract -> fixture -> validation ladders and should remain docs/audit history until there is a real fundamental-feed input.
- The first offline vertical slice is structurally close but not yet viable: `docs/research` needs one operator source file and the repo needs an explicit normalized-packet -> evidence-pack/read-model projection preview.
- No live fetch, external API call, claim/event/entity extraction, recommendation, valuation, autonomous loop, dashboard publication, order generation, broker routing, paper trade, or live trade is enabled by this review.

Implementation note 2026-06-20, domain analyst vertical slice:

- `DomainAnalystVerticalSliceRun` now runs the full analyst branch for one reusable domain candidate.
- The current semiconductor analyst run is built from local parquet data: `data/processed/features/news_data.parquet` and `data/processed/features/macro_data.parquet`.
- Current status is `domain_analyst_candidate_complete_pending_manual_acceptance`.
- Evidence now uses the supplied local strict evidence pack with 144 documents and 144 analyst evidence items; synthetic and fixture markers are false, while the smoke label is true as a caution label only.
- The template is ready for manual accept/reject, but no acceptance decision is recorded by the runner.
- Remaining cautions are real review limitations: 3 dropped source rows and 0 direct ticker evidence.
- Domain scaling, sector-to-ticker bridge, learning writes, recommendations, allocation, paper trading, and live trading remain disabled until separate gates and manual decisions exist.

Implementation note 2026-06-20, domain analyst portability:

- `DomainAnalystPortabilityReview` now checks whether the completed semiconductor analyst candidate can be reused across economic domains.
- All 5 configured domain profiles are structurally portable and have required evidence aliases.
- Reuse is slot-based, not copy-paste: domain id, core questions, required/useful evidence types, sector keywords, ticker universe hints, contradiction rules, direct ticker rules, blockers, and local source paths.
- GPT and FinBERT are optional enrichment adapters only; they are not required for the MVP analyst.
- GPT may later summarize/draft from cited evidence only; FinBERT may later add local-only sentiment annotations. Neither can accept templates, clone domains, create ticker theses, recommend, allocate, write config/learning, or trade.
- `can_clone_domain_profiles_now=false` until the source semiconductor analyst template is manually accepted.

Implementation note 2026-06-20, domain analyst forecast review:

- `DomainAnalystForecastReviewPacket` is now the expectation ledger between domain thesis review and future learning/outcome review.
- Manual accept/reject of the semiconductor analyst template means accepting the reusable process, not declaring the thesis true.
- Analyst outputs should be named `thesis_expectation_or_forecast_candidate`, not investment recommendations.
- Every expectation keeps horizon, confidence, evidence ids, assumptions, contradiction context, invalidation triggers, and future outcome-review protocol.
- Outcome review separates `correct_for_stated_reasons`, `correct_but_lucky_or_wrong_reason`, `incorrect_forecast`, `inconclusive_or_not_mature`, `unfalsifiable_or_underspecified`, and `data_unavailable`.
- The analyst may later summarize why it was right/wrong and propose improvements, but it cannot self-apply changes, write learning memory, change weights/config, recommend, allocate, paper trade, or live trade.
- `DomainAnalystVerticalSliceRun` now creates `forecast_review_json`; `DomainAnalystPortabilityReview` keeps this packet in the fixed non-portable contract for cloned domains.
- `DomainAnalystCaseRegistryPacket` now accepts `forecast_review_json` and registers the frozen expectation as `pending_expectation_outcome`, preserving the lucky-hit vs correct-for-reasons taxonomy before any learning promotion.

Implementation note 2026-06-25, domain analyst template decision:

- `DomainAnalystTemplateDecisionPacket` is now the explicit manual accept/reject gate for the semiconductor analyst template.
- This gate records a decision about the reusable analyst process/template only. It does not declare the thesis true and does not score the forecast outcome.
- Review-only analyst recommendations are allowed: research recommendations, scenario priorities, evidence requests, causal postmortems, and self-improvement proposals.
- Execution/investment recommendations remain blocked: no buy/sell/hold, sizing, allocation, order routing, paper trade, or live trade.
- Current real run is `manual_template_decision_pending`: template accepted is false, one-domain clone candidate is disabled, checks are pass=18, warn=1, fail=0.
- `CurrentArchitectureMap` version is now `2026-06-25-domain-context-sliced-event-v1` and places `DomainAnalystTemplateDecisionPacket`, `DomainAnalystProfilePolicyPacket`, `DomainAnalystEventInterpretationPacket`, and `DomainAnalystFeedbackLoopPacket` before any clone/learning attempt.
- If the template is later explicitly accepted with rationale, only one next-domain clone candidate may be prepared through portable profile slots and local source paths. Learning writes, config writes, sector-to-ticker bridge, execution recommendations, and trading remain separately gated.
- After-385 draft kits under `dean_os/draft` should be harvested selectively. The next useful executable layer is domain/source/evidence policy for modular analyst profiles, not another broad template ladder.

Implementation note 2026-06-25, after-385 profile policy slot harvest:

- The useful after-385 domain-learning draft ideas were integrated into the existing profile system, not copied as standalone production templates.
- `DomainProfile` now includes `source_registry_policy`, `ingestion_filter_policy`, `evidence_scoring_policy`, `review_output_policy`, and `feedback_label_policy`.
- `DomainAnalystIntakePacket` snapshots those policies; `DomainAnalystInstanceContract` carries them as portable template slots; `DomainAnalystTemplateStandardizationPacket` includes them in template scope.
- `DomainAnalystPortabilityReview` now checks those policies for all 5 configured profiles, and `CurrentArchitectureMap` exports their policy ids.
- `DomainAnalystProfilePolicyPacket` is now the executable policy readiness artifact: current run reviewed 5 profiles, 5 are policy-ready, pass=6, warn=0, fail=0.
- Forecast and case-registry CLI summaries now separate review-only analyst recommendations from execution recommendations.
- Current real reruns: profile policy pass=6/warn=0/fail=0; forecast review pass=24/warn=2/fail=0; case registry pass=23/warn=1/fail=0; template decision remains pending pass=18/warn=1/fail=0.
- This does not enable news event extraction, daily automation, GPT, FinBERT, learning writes, config writes, sector-to-ticker bridge, execution recommendations, or trading.

Implementation note 2026-06-25, domain analyst feedback loop:

- `DomainAnalystFeedbackLoopPacket` is now the review-only bridge from human analyst-report feedback to proposal-only learning candidates.
- It consumes the current case registry, forecast review, profile policy packet, template decision packet, and optional manual feedback JSON.
- Manual feedback can label analysis quality, data quality, causal quality, outcome review, process review, profile issues, and proposed learning actions.
- It explicitly preserves the distinction between correct-for-stated-reasons and correct-but-lucky-or-wrong-reason.
- Valid feedback can create `proposal_only_pending_human_approval` learning candidates; invalid feedback blocks the packet if it uses unknown labels, unknown targets, requests execution, requests learning apply, or requests config writes.
- Current real run is `domain_analyst_feedback_loop_ready_pending_manual_feedback`: feedback targets=4, manual feedback records=0, learning candidates=0, pass=9, warn=1, fail=0. The only warning is expected: manual feedback has not been supplied yet.
- The packet can capture manual feedback and create analyst self-improvement proposals, but cannot apply learning, write learning memory, update prompts, update source registry, update pattern memory, write production config, create execution recommendations, allocate, paper trade, or live trade.
- Next correct analyst work is either attach real manual feedback/outcomes or explicitly record template accept/reject. Do not add another broad template ladder.

Implementation note 2026-06-25, domain analyst event interpretation:

- `DomainAnalystEventInterpretationPacket` is now the offline review-only layer for detailed news/data analysis.
- This clarifies the boundary: detailed analysis is allowed; execution/investment action is blocked.
- Allowed analyst outputs now include event interpretation, mechanism hypotheses, value-chain mapping, watch metric requests, contradiction review, data-quality notes, evidence gaps, and review queue items.
- The packet adapts after-385 `NEWS_EVENT_INTERPRETATION_SCHEMA_TEMPLATE.json`, `ANALYST_NEWS_INTERPRETATION_PROMPT_TEMPLATE.md`, `CAUSAL_PATTERN_SCHEMA_TEMPLATE.yaml`, and `SAFE_AUTOMATION_BOUNDARY_TEMPLATE.yaml`.
- Current real run on the semiconductor evidence pack processed 144 source documents into 80 event interpretation packets, with 53 high-materiality/review-required items.
- Event packets include source anchors, event type, directness, sentiment as weak context only, causal patterns, mechanism chain, affected value chain, intermediate variables, counterforces, evidence gaps, next collection tasks, materiality, confidence, and horizon.
- Event packets now also include `context_conditioned_interpretation`: growth, inflation/rates/credit, war/geopolitical, commodity/energy, market/risk appetite, technology capex, and narrative context slices change how the news is interpreted.
- Example rule: the same capex/demand news is not read the same way under low rates/growth expansion, high inflation/rate pressure, or war/sanctions/security-of-supply regimes.
- Context slices create amplifiers, dampeners, watch metrics, and review flags. They are review scaffolding, not final macro truth or trading signals.
- `DomainAnalystVerticalSliceRun` now creates `event_interpretation_json` as part of the full analyst slice: evidence pack -> source gate -> event interpretation -> intake -> instance -> thesis review -> forecast review -> template standardization.
- Current vertical slice remains `domain_analyst_candidate_complete_pending_manual_acceptance`; it has 144 documents, 144 evidence items, 80 event packets, 53 review-required items, synthetic=false, fixture=false, can_scale=false, can_trade=false.
- This does not enable daily automation, live fetch, GPT, FinBERT, final thesis truth, price targets, buy/sell/hold, sizing, allocation, orders, broker calls, paper/live trades, learning writes, or production config writes.

Implementation note 2026-06-28, development walk-forward:

- The active pipeline now has a purged expanding walk-forward train/validation evaluator with causal Stage 3 timeframe lineage and a review-only Stage 4 seam.
- Its artifacts are development-only supporting evidence. They cannot be promoted or treated as locked test/model evidence.
- The first NVDA/15m candidate failed predictive and stability gates. Do not iterate variants on the same folds; accumulate new forward observations before a virgin holdout is defined.
- `PipelineControlForwardDataAccrualPlan` now registers the first-seen boundary for those observations without loading data. New rows must arrive in a new immutable artifact after registration and remain development-refresh data, not a virgin holdout.
- `PipelineControlForwardDataAccrualGate` enforces that boundary before Stage 3: it rejects pre-registration or seen files, target-contaminated inputs, insufficient post-watermark rows, invalid OHLCV/cadence/returns, duplicates, and cross-ticker copies. The existing June 25 source is blocked.
- The active walk-forward runner accepts a forward source only through a passing accrual-gate JSON, rechecks its provenance, preserves a separate development partition, and derives causal higher-timeframe context before Stage 3.

Implementation note 2026-06-28, active normal model-evidence path:

- Active Stage 4 now adapts nested prepared splits to the unified trainer and uses validation for model selection while reserving the prepared holdout.
- Candidate models are persisted separately; only the actual winner is promoted to the stable champion file.
- Stage 4 emits honest partial/measured training candidates, and Stage 5 propagates target/model/timeframe/context lineage so Stage 7 can produce joinable single-context evaluation evidence.
- This repair was contract-tested without running normal training or the heavy pipeline.

Implementation note 2026-06-28, active execution boundary:

- Normal final-stage orchestration now runs Stage 5 -> Stage 7. Stage 6 is excluded unless explicitly requested.
- Explicit Stage 6 calls remain review-only: they pass prediction signals onward but do not initialize the virtual portfolio, create orders, write the decision diary, or mutate learning state.
- Paper requests are blocked in the active pipeline and routed conceptually to the existing review receipt -> paper simulation plan -> isolated external executor -> post-paper review workflow.
- Live `Trader` initialization is rejected. No broker adapter, paper transaction, live order, or trading-memory write is authorized by this integration.
- Stage 7 no longer invokes real-time adaptation from caller-supplied trading activity. It emits a proposal-only learning-review candidate with every learning/config mutation flag false.
- Telegram/Discord evaluation delivery is off by default and requires explicit authorization in the individual final-stage request.

Implementation note 2026-06-28, DEAN orchestrator review phases:

- `DEANOrchestrator` now runs pipeline hard-veto agents as preflight, runs the explicitly selected pipeline adapter only if preflight is clear, runs analytical agents on post-pipeline context, then repeats pipeline data/risk safety review in `pre_trade`.
- Post-pipeline reports take precedence over preflight reports for the same agent, so consensus sees checks performed on actual pipeline outputs.
- Missing prerequisites for hard/block agents now produce valid evidence-backed synthetic block reports and stop the pipeline runner.
- Default consensus no longer emits `candidate_long` or `candidate_short`; high scores become review-only `watchlist` decisions.
- `HybridPipelineAdapter` now attaches a compact `dean_pipeline_review_contract_v1` to the DEAN context: pipeline status, Stage 4 manifests, Stage 7 artifacts, execution boundary, learning-review status, and explicit no-trade/no-config flags.
- Realized returns now outrank `target_return_*`. If only a supervised target label is available, it remains offline-only and RiskAgent blocks it in pre-trade review.
