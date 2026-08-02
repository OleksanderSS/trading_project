"""What position *i* of a context fingerprint actually means.

`ContextMapEnricher` builds the fingerprint as

    sorted(set(state_cols + temporal_cols))  ->  '1|-1|0|...'

so position *i* is a specific `state_<FEATURE>` column. That mapping was
never written down anywhere. Two things follow, and both are real:

- Any analysis of fingerprint components can only speak in positions.
  "Driver 37 loses 80% of the time" is not actionable and cannot be
  reviewed by a human.
- The mapping is not stable. Add one enricher, or let a column go all-NaN
  for a batch, and the sorted list shifts. Every fingerprint written before
  that point now decodes to different drivers -- silently, because a
  fingerprint carries no version.

This module records the ordered driver list under a content hash, so a
fingerprint can be decoded, and a decoding can be *invalidated* when the
schema moves. Anything derived from a fingerprint (rules, vulnerability
reports) should carry the schema id it was derived under.

Storage is a small JSON registry rather than a table: it is written once per
enrichment run, read rarely, and needs to survive independently of whichever
database happens to be open.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = Path("data/context_schema/context_drivers.json")

#: Guards against a corrupt or unbounded registry. One entry per distinct
#: driver ordering; a project that produces more than this is churning its
#: feature set faster than any rule derived from it could be validated.
MAX_SCHEMAS_RETAINED = 50


def registry_path() -> Path:
    """Where the registry lives; overridable for tests and alternate roots."""
    override = os.environ.get("CONTEXT_SCHEMA_REGISTRY")
    return Path(override) if override else DEFAULT_REGISTRY_PATH


def schema_id(drivers: list[str]) -> str:
    """Stable short id for an ordered driver list.

    Order matters and is part of the identity: the same columns in a
    different order produce different fingerprints, so they are a different
    schema.
    """
    # Newline separator: a column name cannot contain one, so two different
    # orderings can never hash to the same payload the way a space-joined
    # list could.
    payload = "\n".join(str(name) for name in drivers)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def record_schema(drivers: list[str], *, path: Path | None = None) -> str:
    """Register an ordered driver list, returning its id.

    Idempotent: re-registering the same ordering only refreshes `last_seen`,
    so repeated runs on an unchanged feature set do not grow the registry.
    """
    drivers = [str(name) for name in drivers]
    if not drivers:
        return ""

    target = path or registry_path()
    identifier = schema_id(drivers)
    now = datetime.now(timezone.utc).isoformat()

    registry = _load(target)
    schemas: dict[str, Any] = registry.setdefault("schemas", {})
    entry = schemas.get(identifier)
    if entry is None:
        schemas[identifier] = {
            "drivers": drivers,
            "first_seen": now,
            "last_seen": now,
        }
    else:
        entry["last_seen"] = now
        # Repair rather than trust: a hash collision is implausible, but a
        # hand-edited registry is not.
        entry["drivers"] = drivers

    registry["latest"] = identifier

    if len(schemas) > MAX_SCHEMAS_RETAINED:
        # Drop the least recently seen, never the latest.
        ordered = sorted(
            schemas.items(),
            key=lambda item: str(item[1].get("last_seen", "")),
        )
        for stale_id, _ in ordered:
            if len(schemas) <= MAX_SCHEMAS_RETAINED:
                break
            if stale_id != identifier:
                schemas.pop(stale_id, None)

    _save(target, registry)
    return identifier


def latest_schema(*, path: Path | None = None) -> tuple[str, list[str]]:
    """The most recently recorded ordering, or ('', []) when none exists."""
    registry = _load(path or registry_path())
    identifier = str(registry.get("latest") or "")
    entry = registry.get("schemas", {}).get(identifier) or {}
    return identifier, [str(name) for name in entry.get("drivers", [])]


def drivers_for(identifier: str, *, path: Path | None = None) -> list[str]:
    """The ordering recorded under `identifier`, or [] if it is unknown."""
    registry = _load(path or registry_path())
    entry = registry.get("schemas", {}).get(str(identifier)) or {}
    return [str(name) for name in entry.get("drivers", [])]


def driver_name(index: int, drivers: list[str]) -> str:
    """Human-readable name for a fingerprint position.

    Falls back to a position label rather than raising or inventing a name:
    an index past the end means the fingerprint was written under a different
    schema, and saying so is more useful than guessing which column moved.
    """
    if 0 <= index < len(drivers):
        return drivers[index]
    return f"driver_{index}"


def _load(path: Path) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError:
        # The ordinary first-run case, not a fault.
        logger.debug("No context driver registry at %s yet.", path)
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        # A registry that cannot be read must not take down feature
        # engineering -- but it must not be silent either: every rule derived
        # from a fingerprint will fall back to positional driver labels, and
        # that has to be traceable to this and not read as "the schema
        # changed".
        logger.warning(
            "Context driver registry at %s is unreadable (%s); fingerprint "
            "positions will not be resolvable to column names.", path, exc,
        )
        return {}
    if not isinstance(data, dict):
        logger.warning(
            "Context driver registry at %s holds %s, not an object; ignoring.",
            path, type(data).__name__,
        )
        return {}
    return data


def _save(path: Path, registry: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-replace so a crash mid-write cannot leave a truncated
        # registry that silently decodes every fingerprint to nothing.
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(registry, handle, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
    except OSError as exc:
        logger.warning(
            "Could not write the context driver registry to %s (%s); "
            "fingerprints from this run will decode to positions only.",
            path, exc,
        )
