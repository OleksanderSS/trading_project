# Codex Integration Prompt — Data Pipeline vs Agentic Analyst Layer

Use this design note to keep DEAN-OS architecture clean.

Implement the boundary:
- data/feature pipeline parses, normalizes, calculates, enriches, and stores datasets;
- agentic analyst layer consumes those structured packets and performs deeper reasoning;
- evaluation layer tracks outcomes and calibration;
- review layer handles human correction and audit.

Do not merge all responsibilities into one predictor or one prompt.

Implement first:
1. NewsEvidencePacket.
2. MarketFeatureSnapshot.
3. AnalysisRequest.
4. AnalystOutputPacket.
5. Explicit data_plane / analysis_plane / evaluation_plane / review_plane package boundaries.
6. Tests ensuring analyst outputs preserve as_of_date and do not produce live trading instructions.

Strictly review-only:
- no live order;
- no buy/sell/hold;
- no position sizing;
- no broker routing;
- no autonomous execution;
- no production price target.
