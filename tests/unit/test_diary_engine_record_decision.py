from datetime import UTC, datetime

from src.meta_learning.memory.diary_engine import (
    DecisionOutcome,
    DecisionRecord,
    DecisionType,
    DiaryEngine,
)


class FakeDataManager:
    def __init__(self):
        self.upserted = []

    def execute_query(self, query, params=None):
        return None

    def get_table_schema(self, table_name):
        return {"columns": []}

    def upsert(self, table_name, df, unique_on=None):
        self.upserted.append((table_name, df, unique_on))


def test_record_decision_persists_the_stable_decision_id():
    """record_decision previously ignored decision.decision_id (a stable
    UUID string, per the dataclass's own comment) and invented a fresh
    uuid4().int & 0x7FFFFFFF truncated 31-bit int instead - dead weight
    on decision_id and a collision-prone substitute id."""
    fake_dm = FakeDataManager()
    engine = DiaryEngine(data_manager=fake_dm)

    record = DecisionRecord(
        agent_id="champion",
        ticker="AAPL",
        decision_type=DecisionType.BUY,
        reasoning="test",
        market_context={},
        context_fingerprint="fp",
        outcome=DecisionOutcome.PENDING,
        decision_timestamp=int(datetime.now(UTC).timestamp()),
    )

    engine.record_decision(record)

    assert len(fake_dm.upserted) == 1
    _, df, _ = fake_dm.upserted[0]
    assert df.iloc[0]["id"] == record.decision_id
