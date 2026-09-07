
import re
from pathlib import Path

def test_calibration_does_not_mix_synthetic_into_primary_score_by_source_scan():
    path=Path('src/calibration/calibration_engine.py')
    if not path.exists(): return
    text=path.read_text(encoding='utf-8',errors='ignore')
    suspicious=['combined_metric = 0.7 * real_metric + 0.3 * synthetic_metric','0.3 * synthetic_metric','synthetic_metric * 0.3']
    assert not any(s in text for s in suspicious), 'Synthetic metric must not affect primary calibration score by default.'
def test_sample_fallback_requires_opt_in_by_source_scan():
    """
    A module that can fabricate stand-in data must gate it behind an explicit
    opt-in, so fabricated rows never silently stand in for real ones.

    Matched on the ``create_sample`` / ``_create_sample`` factory naming only.
    The old scan also matched the phrase "sample data" anywhere in the file, so
    it flagged the docstring "Evaluate out-of-sample data using return, Sharpe,
    and max drawdown" — out-of-sample data is held-out real data, the opposite
    of fabricated — and a comment about subsampling rows with np.random.choice.
    """
    opt_in_markers=['allow_sample','allow_synthetic','eligible_for_training']
    factory=re.compile(r'\b_?create_sample\w*')
    root=Path('src'); offenders=[]
    for p in root.rglob('*.py'):
        if '__pycache__' in p.parts: continue
        text=p.read_text(encoding='utf-8',errors='ignore')
        if factory.search(text) and not any(x in text for x in opt_in_markers):
            offenders.append(str(p))
    assert not offenders, 'Sample/demo fallback paths should require explicit opt-in. Review: '+str(offenders[:20])
