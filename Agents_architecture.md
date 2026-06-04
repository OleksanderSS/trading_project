# DEAN-OS v4.2 — Фінальна архітектура мультиагентної системи

Документ підсумовує брейншторм і фіксує остаточну архітектуру.
Базується на v4.1, виправлені архітектурні помилки, додані відсутні елементи.

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
- За замовчуванням: paper trading тільки, live execution вимкнено.

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
