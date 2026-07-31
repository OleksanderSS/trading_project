
from pathlib import Path

from _target_column_scan import scan_tree, target_column_creations


def test_feature_enricher_modules_do_not_emit_target_columns_by_source_scan():
    root = Path('src/features/enrichers')
    if not root.exists():
        return
    offenders = scan_tree(root, honour_audit_ignore=False)
    assert not offenders, (
        'Feature enrichers must not create target_* columns. Offenders: '
        + str({k: v for k, v in list(offenders.items())[:10]})
    )


def test_the_scan_still_catches_a_real_violation():
    """Guard the guard: the check above was previously a bare substring scan
    that flagged its own exclusion logic, so it is worth proving the
    replacement still fires on genuine target-column creation."""
    assert target_column_creations("df['target_up_1d'] = returns > 0")
    assert target_column_creations('frame[f"target_{name}"] = values')
    assert target_column_creations("out = df.assign(target_up_1d=labels)")
    # ...and stays quiet on the patterns that caused the false positives.
    assert not target_column_creations(
        "[c for c in cols if not c.startswith('target_')]"
    )
    assert not target_column_creations(
        "def __init__(self, target_column: str = 'close'): self.target_column = target_column"
    )
    assert not target_column_creations("df[f'LAG_{lag}'] = df[target_col].shift(lag)")


def test_feature_enricher_modules_do_not_use_bfill_by_source_scan():
    root = Path('src/features')
    if not root.exists():
        return
    offenders = []
    for p in root.rglob('*.py'):
        text = p.read_text(encoding='utf-8', errors='ignore')
        if '.bfill(' in text or "method='bfill'" in text or 'method="bfill"' in text:
            offenders.append(str(p))
    assert not offenders, 'bfill is forbidden in causal feature paths. Offenders: ' + str(offenders[:10])
