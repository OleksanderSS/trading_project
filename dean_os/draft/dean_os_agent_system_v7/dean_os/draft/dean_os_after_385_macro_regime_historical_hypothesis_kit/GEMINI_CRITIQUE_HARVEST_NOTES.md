# Gemini Critique Harvest Notes

This file preserves the useful additions extracted from the Gemini cross-check.

## Useful additions accepted

1. Risk archetypes instead of naive year-matching  
   The analyst should tag mechanism states such as CREDIT_BUBBLE_BURST, LIQUIDITY_CRISIS,
   SUPPLY_CHAIN_SHOCK, INFLATION_SPIKE, REGULATORY_CRACKDOWN, TECHNOLOGY_BUBBLE_EUPHORIA,
   or FIRST_CRACKS_IN_BUBBLE. Historical years are context, not direct templates.

2. Analyst as classifier, not probability generator  
   The analyst should output structured tags, stages, affected sectors, mechanisms, counterforces,
   and evidence gaps. It should not invent exact percentages. Probabilities require historical base
   rates, market-implied data, or reviewed priors.

3. Cross-sectional baseline instead of perfect control group  
   A perfectly unaffected sector rarely exists. The system should use target/secondary/baseline
   baskets with contamination-risk labels and compare relative/abnormal moves.

4. Data alignment across frequencies  
   Historical data may be monthly/quarterly while modern data can be daily/intraday. The system must
   downsample/align carefully and label staleness, publication lag, and interpolation.

5. Leading risk monitors for Pipeline Controller  
   PnL and train/test metrics are lagging. Add leading monitors for regime shift, feature drift,
   correlation breakdown, volatility regime break, liquidity stress, and analyst-pipeline disagreement.

6. Decision lineage for debugging  
   Multi-agent systems require traceability. Record analyst, pipeline controller, and orchestrator
   contributions so failures can be attributed and regression tests updated.

## Useful criticism accepted as constraints

- Avoid stationarity fallacy.
- Avoid LLM probability hallucination.
- Avoid assuming clean control groups.
- Avoid relying only on lagging PnL/train-test metrics.
- Avoid untraceable multi-agent decisions.

## Boundary

All additions remain review-only and non-trading.
