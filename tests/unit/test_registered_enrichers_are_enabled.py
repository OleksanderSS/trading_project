"""Registering an enricher is not what turns it on.

`enrichment.yaml` supplies an enricher's module, class and params.
`features.yaml` decides whether it runs, through `enabled_enrichers` (or the
older `enrichers.<id>.enabled`). `_is_enricher_enabled` reads only the
second.

corporate_filings was added to enrichment.yaml on 2026-08-21 and did not run.
The rebuild's log lists twenty-one enrichers and none of them is it, while
`sec_filings` sits in stage 3's inputs waiting for a reader that never loaded.
Nothing failed, nothing warned; the columns were simply absent, which is the
quietest way for a feature to not exist.

Two files have to agree, so a test makes them.
"""

from __future__ import annotations

import io

import pytest
import yaml


def _load(path: str) -> dict:
    return yaml.safe_load(io.open(path, encoding="utf-8")) or {}


def _find(node, key):
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find(value, key)
            if found is not None:
                return found
    return None


@pytest.fixture(scope="module")
def registered() -> dict:
    """Enrichers declared with a module and class in enrichment.yaml."""
    config = _load("src/config/enrichment.yaml")
    found = {}

    def walk(node):
        if isinstance(node, dict):
            for name, body in node.items():
                if isinstance(body, dict) and "module" in body and "class" in body:
                    found[name] = body
                else:
                    walk(body)

    walk(config)
    return found


@pytest.fixture(scope="module")
def enabled() -> dict:
    features = _load("src/config/features.yaml")
    switches = dict(_find(features, "enabled_enrichers") or {})
    for name, body in (_find(features, "enrichers") or {}).items():
        if isinstance(body, dict) and body.get("enabled"):
            switches.setdefault(name, True)
    return switches


def test_enrichment_yaml_declares_something(registered):
    assert registered, "no enricher carries module and class; the fixture is wrong"


def test_every_registered_enricher_has_a_switch(registered, enabled):
    """A module and class with no switch is a file nobody imports."""
    missing = sorted(name for name in registered if name not in enabled)
    assert not missing, (
        f"registered in enrichment.yaml but absent from features.yaml's "
        f"enabled_enrichers: {missing}. They will not run and nothing will say so."
    )


def test_corporate_filings_is_on(enabled):
    """Named, because this is the one that was silently off."""
    assert enabled.get("corporate_filings") is True


def test_the_orchestrator_actually_builds_it():
    """The config agreeing is not the same as the class instantiating."""
    from src.config.unified_config_manager import get_current_config
    from src.features.feature_orchestrator import FeatureOrchestrator

    orchestrator = FeatureOrchestrator.create_from_config(get_current_config())
    names = {enricher.name for enricher in orchestrator.enrichers}
    assert "corporate_filings" in names, sorted(names)
