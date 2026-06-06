
from pathlib import Path

def test_calibration_does_not_mix_synthetic_into_primary_score_by_source_scan():
    path=Path('src/calibration/calibration_engine.py')
    if not path.exists(): return
    text=path.read_text(encoding='utf-8',errors='ignore')
    suspicious=['combined_metric = 0.7 * real_metric + 0.3 * synthetic_metric','0.3 * synthetic_metric','synthetic_metric * 0.3']
    assert not any(s in text for s in suspicious), 'Synthetic metric must not affect primary calibration score by default.'
def test_sample_fallback_requires_opt_in_by_source_scan():
    root=Path('src'); offenders=[]
    for p in root.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        text=p.read_text(encoding='utf-8',errors='ignore')
        if ('create_sample' in text or '_create_sample' in text or 'sample data' in text.lower()) and not any(x in text for x in ['allow_sample','allow_synthetic','eligible_for_training']): offenders.append(str(p))
    assert not offenders, 'Sample/demo fallback paths should require explicit opt-in. Review: '+str(offenders[:20])
