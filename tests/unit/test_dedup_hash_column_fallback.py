"""A collector delivered rows and saved none of them.

`filter_new_records` resolves the hash column as `unique_cols[0]` or 'hash'.
The table side already had a fallback for a differently-spelled column; the
DataFrame side returned early instead:

    Hash column 'hash' not found in DataFrame for table
    'wikipedia_attention_data'. Cannot filter new records.

wikimedia_attention carries 'record_hash'. Its own `run()` passes
unique_cols=['record_hash'], but the orchestrator's save path calls this
without them, so 'hash' was requested from a frame that has never had it.
Deduplication was skipped, and the insert then died on the unique index:

    Constraint Error: Duplicate key
    "record_hash: 3dfe545bb042a8ff4a2615c1777fbe4b46d0e84c956bf46a477043810f689e20"

The shape is worth naming: the guard diagnosed the problem correctly and
then made it worse, because "cannot filter" was treated as "carry on".

Verified against the live database after the fix: a repeat collection returns
0 new rows and raises nothing, where it previously raised ConstraintException.
"""
import pandas as pd
import pytest

from src.config.unified_config_manager import UnifiedConfigManager
from src.data.management.data_manager import DataManager


@pytest.fixture
def manager(tmp_path, monkeypatch):
    config = UnifiedConfigManager()
    monkeypatch.setattr(
        config, 'get',
        lambda key, default=None: (
            str(tmp_path / 'test.duckdb') if key == 'paths.raw_db'
            else UnifiedConfigManager.get(config, key, default)
        ),
    )
    return DataManager(config)


def _seed(manager, table, hash_column):
    frame = pd.DataFrame({
        hash_column: ['aaa', 'bbb'],
        'value': [1.0, 2.0],
    })
    manager.con.execute(
        f'CREATE TABLE "{table}" ("{hash_column}" VARCHAR, value DOUBLE)'
    )
    manager.con.register('seed', frame)
    manager.con.execute(f'INSERT INTO "{table}" SELECT * FROM seed')
    manager.con.unregister('seed')


def test_record_hash_is_used_when_the_caller_asked_for_hash(manager):
    """The exact wikimedia case: caller says 'hash', both sides have
    'record_hash', and nothing else is shared."""
    _seed(manager, 'attention', 'record_hash')
    incoming = pd.DataFrame({
        'record_hash': ['aaa', 'ccc'],
        'value': [1.0, 3.0],
    })

    filtered = manager.filter_new_records('attention', incoming)

    assert len(filtered) == 1, "the row already stored must be filtered out"
    assert filtered['record_hash'].tolist() == ['ccc']


def test_an_explicit_unique_col_is_honoured(manager):
    _seed(manager, 'attention', 'record_hash')
    incoming = pd.DataFrame({'record_hash': ['bbb'], 'value': [2.0]})

    filtered = manager.filter_new_records(
        'attention', incoming, unique_cols=['record_hash']
    )

    assert filtered.empty


def test_nothing_shared_is_reported_and_passed_through(manager, caplog):
    """Refusing to filter is safe only if it is said out loud: the rows then
    depend on the unique index, which is what raised the ConstraintException."""
    import logging

    _seed(manager, 'attention', 'record_hash')
    incoming = pd.DataFrame({'some_other_id': ['x'], 'value': [9.0]})

    with caplog.at_level(logging.WARNING):
        filtered = manager.filter_new_records('attention', incoming)

    assert len(filtered) == 1
    assert any('no hash column shared' in r.message.lower() for r in caplog.records)


def test_a_frame_whose_hash_matches_both_sides_still_works(manager):
    """The ordinary case must not regress while fixing the odd one."""
    _seed(manager, 'plain', 'hash')
    incoming = pd.DataFrame({'hash': ['aaa', 'zzz'], 'value': [1.0, 26.0]})

    filtered = manager.filter_new_records('plain', incoming)

    assert filtered['hash'].tolist() == ['zzz']
