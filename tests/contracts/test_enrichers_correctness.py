
from pathlib import Path

def test_feature_enricher_modules_do_not_emit_target_columns_by_source_scan():
    root=Path('src/features/enrichers')
    if not root.exists(): return
    offenders=[str(p) for p in root.rglob('*.py') if ('target_' in p.read_text(encoding='utf-8',errors='ignore') or 'target_forward' in p.read_text(encoding='utf-8',errors='ignore'))]
    assert not offenders, 'Feature enrichers must not create target_* columns. Offenders: '+str(offenders[:10])
def test_feature_enricher_modules_do_not_use_bfill_by_source_scan():
    root=Path('src/features')
    if not root.exists(): return
    offenders=[]
    for p in root.rglob('*.py'):
        text=p.read_text(encoding='utf-8',errors='ignore')
        if '.bfill(' in text or "method='bfill'" in text or 'method="bfill"' in text: offenders.append(str(p))
    assert not offenders, 'bfill is forbidden in causal feature paths. Offenders: '+str(offenders[:10])
