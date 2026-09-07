"""
Cache entries whose shape depends on the producing code must be versioned.

A cached payload has the shape the collector gave it. Change the collector's
parsing and last week's payload no longer matches this week's code — but the
key still matches, so the stale payload is served for the rest of its TTL
(a week, for some collectors). Folding a fingerprint of the collector's source
into the key makes a collector invalidate its own entries, and only its own.

The opposite rule holds for per-record "already seen this content hash"
markers: their meaning is content identity, which no code change alters.
Versioning those would re-ingest the entire history on every edit.
"""
from __future__ import annotations

import ast
from pathlib import Path

COLLECTORS = Path("src/data/collectors")


def _cache_calls(tree: ast.Module):
    """Yield every ``<...>.cache_manager.get/set(...)`` call in the module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr not in {"get", "set"}:
            continue
        receiver = func.value
        if isinstance(receiver, ast.Attribute) and receiver.attr == "cache_manager":
            yield node


def _has_version(call: ast.Call) -> bool:
    return any(kw.arg == "version" for kw in call.keywords)


def _first_arg_is_bare_name(call: ast.Call, name: str) -> bool:
    """True for the dedup-marker form: cache_manager.get(h) / set(h, True)."""
    return bool(call.args) and isinstance(call.args[0], ast.Name) and call.args[0].id == name


def _iter_collector_modules():
    for path in sorted(COLLECTORS.rglob("*.py")):
        if "__pycache__" in path.parts or path.name == "__init__.py":
            continue
        yield path, ast.parse(path.read_text(encoding="utf-8", errors="ignore"))


def test_payload_cache_calls_pass_a_version():
    """
    Every cache call that is not a per-record hash marker must pass version=.

    This is the check a new collector will trip over if it caches a payload
    without versioning it.
    """
    offenders = []
    for path, tree in _iter_collector_modules():
        for call in _cache_calls(tree):
            if _first_arg_is_bare_name(call, "h"):
                continue  # dedup marker, deliberately unversioned
            if not _has_version(call):
                offenders.append(f"{path}:{call.lineno}")
    assert not offenders, (
        "These cache calls carry a payload whose shape depends on collector "
        "code, so they must pass version=self.cache_version: " + ", ".join(offenders)
    )


def test_dedup_hash_markers_stay_unversioned():
    """
    Per-record hash markers must NOT be versioned.

    'I have already ingested this content hash' means the same thing no matter
    which version of the collector wrote it. Versioning them would make every
    code change look like a fresh database and re-ingest everything.
    """
    offenders = []
    for path, tree in _iter_collector_modules():
        for call in _cache_calls(tree):
            if _first_arg_is_bare_name(call, "h") and _has_version(call):
                offenders.append(f"{path}:{call.lineno}")
    assert not offenders, (
        "Dedup hash markers must stay unversioned or every code change "
        "re-ingests the whole history: " + ", ".join(offenders)
    )


def test_at_least_one_of_each_kind_exists():
    """Guard against the two tests above passing because they matched nothing."""
    payload = markers = 0
    for _path, tree in _iter_collector_modules():
        for call in _cache_calls(tree):
            if _first_arg_is_bare_name(call, "h"):
                markers += 1
            else:
                payload += 1
    assert payload > 0, "no payload cache calls found — the scan is not matching"
    assert markers > 0, "no dedup marker calls found — the scan is not matching"
