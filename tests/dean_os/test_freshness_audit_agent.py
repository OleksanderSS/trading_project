from __future__ import annotations

import asyncio

from dean_os.agents.freshness_audit import _parse_ts, FRESHNESS_THRESHOLDS, FreshnessAuditAgent
from dean_os.schemas import MarketContext


def test_parse_ts_none():
    assert _parse_ts(None) is None
    assert _parse_ts("") is None


def test_parse_ts_datetime():
    from datetime import UTC, datetime
    dt = datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)
    parsed = _parse_ts(dt)
    assert parsed == dt


def test_parse_ts_iso():
    from datetime import UTC, datetime
    parsed = _parse_ts("2026-06-30T12:00:00+00:00")
    assert parsed == datetime(2026, 6, 30, 12, 0, 0, tzinfo=UTC)


def test_parse_ts_with_z():
    parsed = _parse_ts("2026-06-30T12:00:00Z")
    assert parsed is not None
    assert parsed.hour == 12


def test_parse_ts_naive_iso():
    parsed = _parse_ts("2026-06-30T12:00:00")
    assert parsed is not None
    assert parsed.tzinfo is not None  # UTC assumed


def test_freshness_thresholds_defined():
    assert "news" in FRESHNESS_THRESHOLDS
    assert "macro" in FRESHNESS_THRESHOLDS
    assert "prices" in FRESHNESS_THRESHOLDS
    assert "fundamentals" in FRESHNESS_THRESHOLDS
    assert all(v > 0 for v in FRESHNESS_THRESHOLDS.values())


def test_freshness_audit_no_data():
    agent = FreshnessAuditAgent("freshness_audit", {})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of="2026-07-09T09:00:00+00:00",
                dataframes={},
            )
        )
    )
    assert report.agent_name == "freshness_audit"
    assert report.verdict in ("pass", "neutral", "caution")


def test_freshness_audit_recent_news():
    from datetime import UTC, datetime, timedelta
    now = datetime(2026, 7, 9, 9, 0, 0, tzinfo=UTC)
    recent = now - timedelta(hours=1)
    agent = FreshnessAuditAgent("freshness_audit", {})
    report = asyncio.run(
        agent.run(
            MarketContext(
                as_of=now.isoformat(),
                news=[{"published_at": recent.isoformat(), "title": "test"}],
            )
        )
    )
    # 1 healthy, 0 stale, 0 missing -> freshness_score = 1.0 -> neutral (no stale)
    assert report.verdict == "neutral"
