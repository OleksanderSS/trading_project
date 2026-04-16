# 📋 Optional Follow-Up Tasks

**Status**: Not blocking; optional improvements for future sessions

---

## Task 1: Root-Level Script Cleanup

### Current State
13 temporary Python scripts in repository root:
- `audit_session2.py` — previous audit
- `fix_timezone_issue.py` — one-time fix
- `force_fix_targets.py` — one-time fix
- `inspect_batch_data.py` — data inspection
- `migration_guide.py` — historical reference
- `prepare_features_for_training.py` — legacy prep
- `smart_timezone_fix.py` — one-time fix
- `split_notebook.py` — notebook splitting
- `test_config.py` — config testing
- Plus 4 more similar temporary helpers

### Recommendation
Option A (Minimal): Ignore for now—they don't affect runtime  
Option B (Recommended): Move to `scripts/archive/` and document purpose  
Option C (Aggressive): Delete from git history (complex if old commits exist)

**If Chosen**: Create `scripts/archive/README.md` with purpose of each script before moving.

---

## Task 2: Final Path Audit Sweep

### Current State
Main hardcoded paths identified and fixed. Edge cases may remain.

### Sweep Commands
```bash
# Check for any remaining src/trained_models references
grep -r "src/trained_models" --include="*.py" src/

# Check for any remaining src/config/runtime references  
grep -r "src/config/runtime" --include="*.py" src/

# Check for hardcoded data/colab references
grep -r "data/colab" --include="*.py" src/
```

### If Issues Found
- Update files to use `config["system.accumulation.output_dir"]` instead
- OR add to `UnifiedConfigManager.get_runtime_params_path()` fallback chain

---

## Task 3: Stage 5 Prediction Robustness

### Current Issue
`src/pipeline/stages/stage_5_prediction.py` has complex model-loading fallback logic (lines 256-284).

### Symptoms
- Tries 4 different paths to find model
- Hard to debug which path is being used
- Could be simplified

### Proposed Fix
```python
# Instead of:
# if exists(path1): use path1
# elif exists(path2): use path2
# ...

# Refactor to:
model_path = config.get_resolved_model_path(
    batch_name=self.batch_name,
    fallback_to_consensus=True
)
model = load_model(model_path)
```

### Effort: ~30 mins refactoring + testing

---

## Task 4: .gitignore Enhancement

### Current State
`.gitignore` covers most runtime paths but could be more explicit.

### Recommended Addition
```gitignore
# Runtime parameters (centralized)
data/runtime/runtime_params.json
data/runtime/batch_*.json

# Feature selection cache
data/cache/selected_features*.json

# Prediction cache
data/predictions/cache/

# Training logs
logs/training/**
logs/predictions/**
```

### Action
```bash
# Review current .gitignore
cat .gitignore | grep -E "data/|src/config"

# Add missing entries if needed
```

---

## Task 5: Architecture Documentation

### Create New File: `ARCHITECTURE_RUNTIME_CONFIG.md`

Should document:
1. **Path Resolution Chain**
   - Where runtime params are stored
   - Where cache is stored
   - Where models are stored

2. **Config Priority Order**
   - Environment variables
   - `runtime_params.json` (centralized)
   - `config/production.yaml`
   - Hardcoded defaults

3. **Adding New Runtime Paths**
   - Step-by-step guide for developers
   - How to add fallback logic
   - Where to document new paths

4. **Migration from Legacy Paths**
   - How to move from `src/config/` to `data/`
   - Backward compatibility guarantees
   - Deprecation timeline

---

## Summary Table

| Task | Priority | Effort | Blocking |
|------|----------|--------|----------|
| Root cleanup | Low | 30 min | No |
| Path audit sweep | Low | 15 min | No |
| Stage 5 refactor | Medium | 1 hr | No |
| .gitignore enhance | Low | 10 min | No |
| RuntimeConfig docs | Low | 45 min | No |

---

## Recommendation

**Core audit is COMPLETE and COMMITTED.** ✅

Remaining tasks are **nice-to-have** improvements that don't affect current functionality. Suggest:

1. ✅ Keep current state (audit is done)
2. ⏭️ Schedule Task 1 (root cleanup) for next session if repo feels cluttered
3. ⏭️ Schedule Task 3 (Stage 5 refactor) if prediction errors occur
4. ⏭️ Task 5 (documentation) is valuable for onboarding new developers

**No urgent action required.**

