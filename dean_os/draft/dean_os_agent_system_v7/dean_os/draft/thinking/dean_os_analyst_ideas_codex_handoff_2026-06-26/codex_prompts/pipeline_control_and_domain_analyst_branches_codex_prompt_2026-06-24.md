# Codex Integration Prompt — Pipeline Control Branch + Domain Analyst Branch

Use this package to update DEAN-OS architecture.

Implement the split:
1. Data / ML Pipeline Branch:
   - parses, enriches, builds datasets, trains, compares, evaluates.
2. Pipeline Controller Agent Branch:
   - reads pipeline metrics and proposes or applies only guarded, allowed parameter changes.
3. Domain Analyst Agent Branch:
   - specialized analysts perform qualitative news/regime/sector analysis using economics, history, politics/geopolitics, and domain knowledge.
4. Evaluation Plane:
   - tracks outcomes, calibration, replay, and feature value.
5. Review Plane:
   - human correction, approval, audit trail.

Strict rules:
- Pipeline controller must not optimize only PnL.
- No ungated production config writes.
- No model promotion without gates.
- Domain analysts must not produce live trading instructions.
- All outputs remain review-only unless later gates explicitly allow controlled paper/replay behavior.

First implementation target:
- Define packets:
  PipelineMetricPacket
  ParameterChangeProposal
  GuardrailValidationReport
  DomainAnalystReport
  AnalystSynthesisPacket
- Define plane boundaries:
  data_plane
  control_plane
  analysis_plane
  evaluation_plane
  review_plane
- Implement review-only controller proposal flow.
- Implement analyst orchestrator routing for Macro, Energy, and Semiconductor/AI Infrastructure analysts.
