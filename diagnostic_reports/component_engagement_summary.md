# Component Engagement Audit

## Status counts
- **ACTIVE_RISKY**: 400
- **ACTIVE_WITH_TEST_REFERENCE**: 243
- **ACTIVE_NEEDS_RUNTIME_CONFIRMATION**: 192
- **ORPHAN_LOW_SIGNAL**: 111
- **UNUSED_POTENTIALLY_VALUABLE**: 41
- **ACTIVE_OUTPUT_UNTESTED**: 18

## Category counts
- **utility_or_unknown**: 370
- **model**: 142
- **pipeline_stage**: 110
- **validator**: 63
- **analyzer**: 58
- **calculator**: 48
- **context_map**: 42
- **enricher**: 37
- **collector**: 28
- **algorithm**: 26
- **selector**: 25
- **trading**: 22
- **risk**: 17
- **detector**: 9
- **factory_registry**: 8

## How to use
1. Review `ACTIVE_RISKY` first.
2. Review `ACTIVE_OUTPUT_UNTESTED` for enrichers/calculators/analyzers.
3. Review `UNUSED_POTENTIALLY_VALUABLE` before deleting anything.
4. Add runtime lineage tracking to prove output reaches model/evaluation.