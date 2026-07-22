from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def check_all() -> dict[str, Any]:
    results: dict[str, Any] = {
        "checks": {},
        "all_pass": True,
        "summary": "",
    }

    # 1. DuckDB
    duckdb_path = Path("data/trading_data.duckdb")
    if duckdb_path.exists():
        try:
            import duckdb
            con = duckdb.connect(str(duckdb_path), read_only=True)
            tables = con.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='main'").fetchall()
            row_counts = {}
            for (t,) in tables:
                cnt = con.execute(f"SELECT count(*) FROM \"{t}\"").fetchone()[0]
                row_counts[t] = cnt
            con.close()
            results["checks"]["duckdb"] = {
                "status": "OK",
                "tables": len(tables),
                "rows": sum(row_counts.values()),
                "per_table": row_counts,
            }
        except Exception as e:
            results["checks"]["duckdb"] = {"status": "ERR", "error": str(e)}
            results["all_pass"] = False
    else:
        results["checks"]["duckdb"] = {"status": "MISS", "error": "File not found"}
        results["all_pass"] = False

    # 2. OutcomeTracker SQLite
    ot_path = Path("data/dean_os/outcome_tracker.sqlite")
    if ot_path.exists():
        try:
            con = sqlite3.connect(str(ot_path))
            events = con.execute("SELECT count(*) FROM tracked_events").fetchone()[0]
            outcomes = con.execute("SELECT count(*) FROM outcomes").fetchone()[0]
            con.close()
            results["checks"]["outcome_tracker"] = {"status": "OK", "events": events, "outcomes": outcomes}
        except Exception as e:
            results["checks"]["outcome_tracker"] = {"status": "ERR", "error": str(e)}
            results["all_pass"] = False
    else:
        results["checks"]["outcome_tracker"] = {"status": "MISS", "note": "No outcomes yet"}

    # 3. Agent Registry YAML
    reg_path = Path("dean_os/config/agent_registry.yaml")
    if reg_path.exists():
        try:
            with open(reg_path) as f:
                data = yaml.safe_load(f)
            agents = data.get("agents", {})
            enabled = sum(1 for a in agents.values() if a.get("enabled"))
            pipe_agents = [n for n, a in agents.items() if a.get("branch") == "pipeline" and a.get("enabled")]
            ana_agents = [n for n, a in agents.items() if a.get("branch") == "analytical" and a.get("enabled")]

            # Check for sharing execution_group
            groups: dict[str, list[str]] = {}
            for n, a in agents.items():
                eg = a.get("execution_group", "")
                if eg:
                    groups.setdefault(eg, []).append(n)
            conflicts = {g: names for g, names in groups.items() if len(names) > 1}

            results["checks"]["registry"] = {
                "status": "OK",
                "total": len(agents),
                "enabled": enabled,
                "pipeline": len(pipe_agents),
                "analytical": len(ana_agents),
                "execution_group_conflicts": conflicts,
            }
        except Exception as e:
            results["checks"]["registry"] = {"status": "ERR", "error": str(e)}
            results["all_pass"] = False
    else:
        results["checks"]["registry"] = {"status": "MISS", "error": "File not found"}
        results["all_pass"] = False

    # 4. Domain Profiles
    profiles_dir = Path("config/domain_profiles")
    if profiles_dir.is_dir():
        profiles = list(profiles_dir.glob("*.yaml"))
        valid = 0
        invalid = 0
        for pf in profiles:
            try:
                with open(pf) as f:
                    yaml.safe_load(f)
                valid += 1
            except Exception:
                invalid += 1
        results["checks"]["domain_profiles"] = {"status": "OK", "count": len(profiles), "valid": valid, "invalid": invalid}
        if invalid > 0:
            results["all_pass"] = False
    else:
        results["checks"]["domain_profiles"] = {"status": "MISS", "note": "No profiles dir"}

    # 5. DuckDB keyword index
    fts_path = Path("data/processed/features/news_keyword_index.duckdb")
    if fts_path.exists():
        try:
            import duckdb
            con = duckdb.connect(str(fts_path), read_only=True)
            kw_cnt = con.execute("SELECT count(*) FROM keyword_index").fetchone()[0]
            con.close()
            results["checks"]["keyword_index"] = {"status": "OK", "entries": kw_cnt}
        except Exception as e:
            results["checks"]["keyword_index"] = {"status": "ERR", "error": str(e)}
            results["all_pass"] = False
    else:
        results["checks"]["keyword_index"] = {"status": "MISS", "note": "No FTS index"}

    # 6. Key artifacts
    artifacts = {
        "features": "data/processed/features/features.parquet",
        "macro": "data/processed/features/macro_data.parquet",
        "predictions": "reports/dean_os/pipeline_prediction_source_review_current/latest.json",
        "runtime_semiconductor": "reports/dean_os/semiconductor_analyst_runtime_current/latest.json",
    }
    artifact_results = {}
    for name, path in artifacts.items():
        p = Path(path)
        if p.exists():
            size = p.stat().st_size
            artifact_results[name] = {"status": "OK", "size_mb": round(size / 1_000_000, 2)}
        else:
            artifact_results[name] = {"status": "MISS"}
    missing_any = any(v.get("status") == "MISS" for v in artifact_results.values())
    results["checks"]["artifacts"] = {
        "artifacts": artifact_results,
        "status": "OK" if not missing_any else "MISS",
    }

    results["summary"] = "All checks pass" if results["all_pass"] else "Some checks failed"
    return results


def print_report(results: dict[str, Any]) -> None:
    print(f"DEAN-OS System Health Check")
    print(f"  Summary: {results['summary']}")
    print()
    for name, check in sorted(results["checks"].items()):
        status = check.get("status", "?")
        icon = {"OK": "+", "MISS": "~", "ERR": "!"}.get(status, "?")
        details = ""
        if name == "duckdb" and status == "OK":
            details = f" | {check['tables']} tables, {check['rows']:,} rows"
        elif name == "outcome_tracker" and status == "OK":
            details = f" | {check['events']} events, {check['outcomes']} outcomes"
        elif name == "registry" and status == "OK":
            details = f" | {check['enabled']}/{check['total']} enabled ({check['pipeline']}p + {check['analytical']}a)"
            if check["execution_group_conflicts"]:
                details += f" | CONFLICTS: {check['execution_group_conflicts']}"
        elif name == "domain_profiles" and status == "OK":
            details = f" | {check['count']} profiles ({check['valid']} valid, {check['invalid']} invalid)"
        elif name == "keyword_index" and status == "OK":
            details = f" | {check['entries']} entries"
        elif name == "artifacts":
            arts = check.get("artifacts", {})
            ok_arts = [k for k, v in arts.items() if isinstance(v, dict) and v.get("status") == "OK"]
            miss_arts = [k for k, v in arts.items() if isinstance(v, dict) and v.get("status") == "MISS"]
            details = f" | {len(ok_arts)}/{len(arts)} present"
            if miss_arts:
                details += f" missing: {', '.join(miss_arts)}"
        print(f"  [{icon}] {name}{details}")
        if status == "ERR":
            print(f"       Error: {check.get('error', 'unknown')}")


__all__ = ["check_all", "print_report"]
