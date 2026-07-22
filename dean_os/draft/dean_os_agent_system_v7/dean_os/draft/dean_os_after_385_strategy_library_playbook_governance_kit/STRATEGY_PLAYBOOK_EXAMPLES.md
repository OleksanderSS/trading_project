# Strategy Playbook Examples

These are illustrative governance examples, not active trading instructions.

## Example 1 — Event-driven relative impact research playbook

Purpose:
Analyze whether a major event affected a target sector more than a baseline basket.

Inputs:
- normalized event packet;
- source provenance;
- affected sector;
- secondary sectors;
- baseline basket with contamination risk;
- macro regime snapshot;
- expectation gap;
- difference-in-differences result.

Allowed mode:
- research;
- replay;
- paper only after gates.

Forbidden:
- direct LLM signal;
- single-stock price-only validation;
- live execution without gateway.

## Example 2 — Macro regime defensive allocation research playbook

Purpose:
Reduce exposure in replay/paper when liquidity crisis or regime shift candidate is detected.

Inputs:
- macro regime snapshot;
- liquidity stress monitor;
- volatility regime monitor;
- pipeline controller state;
- portfolio risk state.

Allowed mode:
- replay/paper/shadow first.

Forbidden:
- automatic live risk increase;
- ignoring kill switch.

## Example 3 — Narrative crowding reversal research playbook

Purpose:
Study crowded narrative reversal risk, such as "AI everything" or "soft landing".

Inputs:
- narrative half-life;
- positioning/crowdedness;
- expectation gap;
- earnings revisions;
- sector relative performance;
- analyst debate packet.

Forbidden:
- price target;
- buy/sell/hold;
- final probability without base rates/market-implied data/review.
