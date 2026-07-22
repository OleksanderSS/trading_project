import json
import sqlite3

import duckdb

from dean_os.industry_operational_source_coverage import IndustryOperationalSourceCoverageBuilder


def test_narrative_match_does_not_become_structured_metric(tmp_path):
    duck_path = tmp_path / "data.duckdb"
    d = duckdb.connect(str(duck_path))
    d.execute("create table news(title varchar, content varchar)")
    d.execute("create table keyword_index(source_table varchar, source_column varchar, keyword varchar, row_count bigint)")
    d.close()
    sqlite_path = tmp_path / "research.sqlite"
    s = sqlite3.connect(sqlite_path)
    s.execute("create table documents(document_id text, title text, source_type text, published_at text, text text)")
    s.execute("insert into documents values ('d1','Capacity narrative','article','2026-01-01','capacity utilization remains tight')")
    s.commit(); s.close()
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({"items": []}), encoding="utf-8")

    payload = IndustryOperationalSourceCoverageBuilder(tmp_path / "out").build(
        duckdb_path=duck_path, research_sqlite_path=sqlite_path,
        knowledge_pack_path=pack_path, save=False,
    )
    assert payload["summary"]["structured_operational_candidate_count"] == 0
    assert payload["summary"]["research_narrative_match_count"] == 1
    assert payload["summary"]["gate_status"] == "structured_adapter_ready_source_feed_missing"
    assert payload["semantic_boundary"]["narrative_match_is_structured_metric"] is False


def test_explicit_structured_column_is_candidate_not_auto_evidence(tmp_path):
    duck_path = tmp_path / "data.duckdb"
    d = duckdb.connect(str(duck_path))
    d.execute("create table industry(capacity_utilization double)")
    d.close()
    sqlite_path = tmp_path / "research.sqlite"
    s = sqlite3.connect(sqlite_path)
    s.execute("create table documents(document_id text, title text, source_type text, published_at text, text text)")
    s.commit(); s.close()
    pack_path = tmp_path / "pack.json"
    pack_path.write_text(json.dumps({"items": []}), encoding="utf-8")
    payload = IndustryOperationalSourceCoverageBuilder().build(
        duckdb_path=duck_path, research_sqlite_path=sqlite_path,
        knowledge_pack_path=pack_path, save=False,
    )
    assert payload["summary"]["structured_operational_candidate_count"] == 1
    assert payload["summary"]["metric_extraction_performed"] is False
