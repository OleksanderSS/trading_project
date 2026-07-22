# Existing Pipeline Integration Map

Purpose: map current repository modules to the after-385 kits and intended integration roles.

## Known current modules and intended mapping

| Existing module/path | Related kit | Intended role | Add / preserve |
|---|---|---|---|
| `src/data/collectors/google_news_collector.py` | Domain Learning Kit v3 + Automation Governance Kit v2 | Source packet / news event input | Keep as collector; do not make it govern the run. |
| `src/data/collectors/newsapi_collector.py` | Domain Learning Kit v3 | News source input | Add source hash, source quality, dedupe metadata. |
| `src/data/collectors/rss_collector.py` | Domain Learning Kit v3 | Allowlisted source ingestion | Add publication/event/ingestion date handling. |
| `src/data/collectors/sec_filings_collector.py` | Domain Learning Kit v3 + Eval/Audit Kit | Company filing source | Treat as higher-trust source for company facts. |
| `src/data/collectors/fred_collector.py` | Macro Regime Kit v3 | Macro indicator input | Feed macro regime snapshots and indicator crosswalk. |
| `src/data/collectors/fear_greed_collector.py` | Macro Regime Kit v3 | Modern risk appetite proxy | Use only as modern enriched indicator, not long-run comparable variable. |
| `src/data/collectors/vix_collector.py` | Macro Regime Kit v3 | Volatility/risk proxy | Do not compare directly with pre-VIX history. |
| `src/features/news_impact_classifier.py` | Domain Learning Kit v3 + Macro Regime Kit v3 | Event tagger | Analyst output should be tag-only; no final probabilities from LLM. |
| `src/features/enrichers/news_impact_enricher.py` | Domain Learning Kit v3 | Event enrichment | Add sector routing, evidence gaps, direct/indirect/contextual impact. |
| `src/analytics/analyzers/news_impact_analyzer.py` | Domain Learning Kit v3 + Macro Kit v3 | News/event interpretation | Add expectation gap and risk archetype classification. |
| `src/analytics/context/macro_context_analyzer.py` | Macro Regime Kit v3 | Macro regime snapshot / archetype mapping | Expand into historical regime context and indicator crosswalk. |
| `src/analytics/context/market_regime_analyzer.py` | Macro Regime Kit v3 | Current market regime state | Feed narrative half-life and regime-changing event assessment. |
| `src/analytics/context/difference_in_differences_methods.py` | Macro Regime Kit v3 | Affected vs baseline sector analysis | Use cross-sectional baseline with contamination-risk labels. |
| `src/analytics/context/synthetic_control_methods.py` | Macro Regime Kit v3 | Synthetic control for policy/event impact | Use for outcome review, not direct trading signal. |
| `src/analytics/context/counterfactual_generator.py` | Macro Regime Kit v3 | Counterfactual scenario support | Attach to open hypothesis tokens and outcome review. |
| `src/pipeline/guards/temporal_leakage_guard.py` | Eval/Observability/Audit Kit | Time leakage protection | Preserve and extend for replay/as-of validation. |
| `src/pipeline/guards/timeframe_alignment_guard.py` | Eval/Audit + Macro Kit v3 | Timeframe alignment | Extend with data-frequency alignment template. |
| `src/pipeline/stages/evaluation/` | Eval/Observability/Audit Kit | Outcome review / regression eval | Add hypothesis outcome review and failed-case regression. |
| `src/pipeline/stages/training/` | Automation Governance Kit v2 | Model lifecycle | Add model promotion gate and champion/challenger governance. |
| `src/pipeline/stages/prediction/` | Automation Governance + Portfolio Governance | Prediction stage | Keep separate from order execution; require risk gates. |
| `src/trading/` | Automation Governance + Portfolio Governance | Later execution layer | Wrap behind paper/shadow/supervised/constrained execution gateway. |

## Codex rule

Do not blindly graft agents onto collectors or trading modules.

Correct flow:

```text
Collectors → Pipeline Controller daily governor → Normalized event packets → Analyst routing
→ Hypothesis/evidence gap → Eval/audit → Replay/paper/shadow gates → Risk/execution gateway.
```
