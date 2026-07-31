
import re
from pathlib import Path

#: Markers that make a sample/demo fallback an explicit, reviewed opt-in.
OPT_IN_MARKERS = ('allow_sample', 'allow_synthetic', 'eligible_for_training')

#: Identifier-level signs that a module FABRICATES sample/demo data.
#: Prose is deliberately excluded: `correlation_engine.py` carries the comment
#: "# Sample data if requested" for `np.random.choice(len(X), sample_size)`,
#: which subsamples REAL rows. Matching the phrase "sample data" flagged that
#: as synthetic-data contamination, which it is not.
_FABRICATION = re.compile(
    r'\b(_?create_sample\w*|generate_sample\w*|_?sample_fallback\w*|'
    r'_?create_demo\w*|generate_demo\w*|_?fake_data\w*)\b'
)


def fabricates_sample_data(text: str) -> list[str]:
    return sorted(set(_FABRICATION.findall(text)))


def test_calibration_does_not_mix_synthetic_into_primary_score_by_source_scan():
    path = Path('src/calibration/calibration_engine.py')
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8', errors='ignore')
    suspicious = [
        'combined_metric = 0.7 * real_metric + 0.3 * synthetic_metric',
        '0.3 * synthetic_metric',
        'synthetic_metric * 0.3',
    ]
    assert not any(s in text for s in suspicious), \
        'Synthetic metric must not affect primary calibration score by default.'


def test_sample_fallback_requires_opt_in_by_source_scan():
    """Any module that fabricates sample/demo data must gate it behind an
    explicit opt-in flag or mark the rows as not trainable.

    `src/archive/` is excluded: this project archives dead code with `git mv`
    rather than deleting it, so archived modules are by definition off the
    live path and their fallbacks cannot contaminate a run.
    """
    root = Path('src')
    offenders = []
    for p in root.rglob('*.py'):
        if '__pycache__' in p.parts or 'archive' in p.parts:
            continue
        text = p.read_text(encoding='utf-8', errors='ignore')
        if fabricates_sample_data(text) and not any(x in text for x in OPT_IN_MARKERS):
            offenders.append(str(p))
    assert not offenders, \
        'Sample/demo fallback paths should require explicit opt-in. Review: ' + str(offenders[:20])


def test_the_sample_fallback_scan_still_catches_a_real_violation():
    """Guard the guard: the previous version matched the prose "sample data",
    which fired on ordinary subsampling of real rows."""
    assert fabricates_sample_data('def _create_sample_cftc_data(self): ...')
    assert fabricates_sample_data('rows = generate_sample_prices()')
    assert fabricates_sample_data('def sample_fallback(): ...')
    # ...and does not fire on subsampling real data, or on prose.
    assert not fabricates_sample_data(
        '# Sample data if requested\n'
        'indices = np.random.choice(len(X), sample_size, replace=False)'
    )
    assert not fabricates_sample_data('sample_size: int | None = None')
