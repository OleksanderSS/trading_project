from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

_DEFAULT_DB = Path("data/dean_os/agent_stats.sqlite")


class AgentRun(BaseModel):
    run_id: str = ""
    agent_name: str = ""
    agent_version: str = ""
    branch: str = ""
    verdict: str = ""
    confidence: float = 0.0
    data_quality_score: float = 0.0
    duration_ms: float = 0.0
    ticker: str = ""
    timestamp: str = ""


class AgentStatsStore:
    """SQLite store for agent run statistics."""

    def __init__(self, db_path: str | Path = _DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as con:
            con.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    agent_name TEXT,
                    agent_version TEXT,
                    branch TEXT,
                    verdict TEXT,
                    confidence REAL,
                    data_quality_score REAL,
                    duration_ms REAL,
                    ticker TEXT,
                    timestamp TEXT
                );
                CREATE TABLE IF NOT EXISTS orchestrator_runs (
                    run_id TEXT PRIMARY KEY,
                    agent_count INTEGER,
                    decision TEXT,
                    confidence REAL,
                    timestamp TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_runs_agent ON runs(agent_name);
                CREATE INDEX IF NOT EXISTS idx_runs_ts ON runs(timestamp);
            """)

    def log_run(
        self,
        agent_name: str,
        agent_version: str,
        branch: str,
        verdict: str,
        confidence: float,
        data_quality_score: float,
        duration_ms: float,
        ticker: str = "",
    ) -> str:
        run_id = f"{agent_name}_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                """INSERT INTO runs
                   (run_id, agent_name, agent_version, branch, verdict, confidence,
                    data_quality_score, duration_ms, ticker, timestamp)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (run_id, agent_name, agent_version, branch, verdict,
                 confidence, data_quality_score, duration_ms, ticker, now),
            )
        return run_id

    def log_orchestrator_run(self, agent_count: int, decision: str, confidence: float) -> str:
        run_id = f"orch_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
        now = datetime.now(UTC).isoformat()
        with sqlite3.connect(str(self.db_path)) as con:
            con.execute(
                "INSERT INTO orchestrator_runs (run_id, agent_count, decision, confidence, timestamp) VALUES (?, ?, ?, ?, ?)",
                (run_id, agent_count, decision, confidence, now),
            )
        return run_id

    def get_stats(self, limit: int = 20) -> dict[str, Any]:
        with sqlite3.connect(str(self.db_path)) as con:
            total = con.execute("SELECT count(*) FROM runs").fetchone()[0]
            orch_total = con.execute("SELECT count(*) FROM orchestrator_runs").fetchone()[0]
            by_verdict = dict(con.execute("SELECT verdict, count(*) FROM runs GROUP BY verdict ORDER BY count(*) DESC").fetchall())
            by_agent = dict(con.execute("SELECT agent_name, count(*) FROM runs GROUP BY agent_name ORDER BY count(*) DESC LIMIT 20").fetchall())
            latest = con.execute(
                "SELECT agent_name, verdict, confidence, timestamp FROM runs ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            avg_conf = con.execute("SELECT avg(confidence) FROM runs").fetchone()[0]

        return {
            "total_runs": total,
            "orchestrator_runs": orch_total,
            "by_verdict": by_verdict,
            "by_agent": by_agent,
            "average_confidence": round(avg_conf or 0, 3),
            "latest_runs": [
                {"agent": r[0], "verdict": r[1], "confidence": r[2], "timestamp": r[3]}
                for r in latest[:10]
            ],
        }

    def close(self) -> None:
        pass


def print_stats(stats: dict[str, Any]) -> None:
    print(f"Agent Run Statistics")
    print(f"  Total runs:      {stats['total_runs']}")
    print(f"  Orchestrator runs: {stats['orchestrator_runs']}")
    print(f"  Avg confidence:  {stats['average_confidence']:.3f}")
    print(f"\n  By verdict:")
    for verdict, count in sorted(stats.get("by_verdict", {}).items(), key=lambda x: -x[1]):
        print(f"    {verdict:20} {count}")
    print(f"\n  By agent (top 10):")
    for agent, count in sorted(stats.get("by_agent", {}).items(), key=lambda x: -x[1])[:10]:
        print(f"    {agent:30} {count}")
    print(f"\n  Latest runs:")
    for run in stats.get("latest_runs", [])[:5]:
        conf = run.get("confidence", 0)
        print(f"    {run.get('agent','?'):30} {run.get('verdict','?'):15} conf={conf:.2f}")


__all__ = ["AgentStatsStore", "print_stats"]
