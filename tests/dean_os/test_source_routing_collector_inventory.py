"""SourceRoutingAgent must actually see collectors, from either input shape.

Two defects are pinned here:

1. A saved ``run_agent_collector_inventory.py`` payload nests the inventory under a
   ``collector_inventory`` key. Reading the envelope as if it were the inventory
   itself found no ``configured_collectors`` and routed zero collectors -- silently,
   with no warning, while reporting a valid-looking run.
2. With no inventory supplied at all the agent also routed zero collectors. Since
   the snapshot's producer was archived on 2026-07-24, "no path given" has to mean
   "scan the live config", not "assume there are no collectors".
"""
from __future__ import annotations

import json

from dean_os.agents.source_routing import inspect_source_routing

_INVENTORY = {
    "summary": {"status": "ok", "enabled_missing_classes": []},
    "configured_collectors": [
        {
            "name": "yahoo_finance",
            "type": "yahoo_finance",
            "enabled": True,
            "data_type": "market_data",
            "recommended_use": "pipeline_price_feed",
            "requires_api_key": False,
            "class_found": True,
        },
        {
            "name": "sec_filings",
            "type": "sec_filings",
            "enabled": False,
            "data_type": "fundamentals",
            "recommended_use": "research_specialist_feed",
            "requires_api_key": False,
            "class_found": True,
        },
    ],
}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


def test_nested_cli_payload_is_unwrapped(tmp_path):
    path = _write(tmp_path / "inventory.json", {"mode": "collector_inventory_agent", "collector_inventory": _INVENTORY})

    result = inspect_source_routing(collector_inventory_path=path)

    assert result["collectors"]["collector_count"] == 2
    assert result["summary"]["pipeline_feed_count"] == 1
    assert result["summary"]["research_specialist_feed_count"] == 1
    assert result["warnings"] == []


def test_bare_inventory_still_works(tmp_path):
    path = _write(tmp_path / "inventory.json", _INVENTORY)

    result = inspect_source_routing(collector_inventory_path=path)

    assert result["collectors"]["collector_count"] == 2


def test_inventory_dict_is_accepted_directly():
    result = inspect_source_routing(collector_inventory=_INVENTORY)

    assert result["collectors"]["collector_count"] == 2


def test_no_inventory_falls_back_to_a_live_scan():
    result = inspect_source_routing()

    # The repo ships a real src/config/collectors.yaml, so the live scan must find
    # collectors rather than reporting an empty routing set.
    assert result["collectors"]["collector_count"] > 0
    assert result["summary"]["pipeline_feed_count"] > 0


def test_empty_inventory_warns_instead_of_reporting_a_clean_run(tmp_path):
    path = _write(tmp_path / "inventory.json", {"summary": {"status": "ok"}, "configured_collectors": []})

    result = inspect_source_routing(collector_inventory_path=path)

    assert result["collectors"]["collector_count"] == 0
    assert any("configured_collectors" in warning for warning in result["warnings"])


def test_unreadable_inventory_warns(tmp_path):
    result = inspect_source_routing(collector_inventory_path=tmp_path / "missing.json")

    assert result["collectors"]["collector_count"] == 0
    assert any("could not be read" in warning for warning in result["warnings"])
