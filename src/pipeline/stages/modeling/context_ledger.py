"""Which contexts have already been trained, and on exactly what data.

Stage 4 trains one model suite per (ticker, timeframe, target). A run that
dies in the eighth hour -- as the 2026-08-31 pooled run did, with a
MemoryError on the second daily target -- loses every context it had already
finished, because nothing on disk said they were finished. The 15m frame was
recomputed six times across runs 1-6 and produced byte-identical numbers
every time.

Two ways to make a restart cheap, and only one of them is safe:

- Look for the model file and assume a context with a model on disk is done.
  That is how a stale champion gets reused after the data underneath it
  changes, which is the single failure mode this project has spent a month
  removing from other places.
- Record what the context was trained ON, and reuse only when the data is
  bit-for-bit the same. That is this file.

The fingerprint deliberately covers more than the target: features change
here far more often than targets do. It is also deliberately cheap -- column
sums and a strided sample rather than a hash of the whole matrix -- because
it runs before every context and must not itself become the cost it is
avoiding, nor allocate the frame-wide copy that killed the run.

What it can and cannot promise, stated rather than assumed. Any changed
cell moves a column sum or a missing-count, so a re-run enricher, a widened
window or a corrected value all force a retrain. What could slip through is
a change that leaves every column total intact AND misses the sample --
swapping two rows, or two edits that cancel. Those are not shapes the
pipeline produces; they are shapes an adversary would produce. A fingerprint
that changes when the data did not merely costs a retrain, which is the safe
direction.

Reuse is OFF by default (`modeling.resume_completed_contexts`). A replayed
context is a result that was not computed in the run that reports it, and
that must be something an operator turned on, not something that happens
quietly after a crash.
"""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("ModelingContextLedger")

#: Rows sampled from the frame for the fingerprint. Enough that a changed
#: enricher moves it; few enough that hashing costs milliseconds.
FINGERPRINT_SAMPLE_ROWS = 2000

SCHEMA = "modeling_context_ledger_v1"


class ContextLedger:
    """A record of finished contexts, keyed by context and data fingerprint."""

    def __init__(self, path: Path | str = Path("data/trained_models/context_ledger.json")):
        self.path = Path(path)
        self._entries: dict[str, dict[str, Any]] = {}
        self._load()

    # -- reading -----------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning(
                "Context ledger at %s could not be read (%s); this run starts "
                "from nothing rather than from a half-parsed file.",
                self.path, e,
            )
            return
        if payload.get("schema") != SCHEMA:
            logger.warning(
                "Context ledger at %s was written by schema %r, not %r; "
                "ignoring it.", self.path, payload.get("schema"), SCHEMA,
            )
            return
        self._entries = payload.get("entries") or {}
        logger.info(
            "Context ledger loaded: %d finished context(s) on record.",
            len(self._entries),
        )

    def lookup(self, key: str, fingerprint: str) -> dict[str, Any] | None:
        """The stored outcome, but only if it was produced on THIS data."""
        entry = self._entries.get(key)
        if not entry:
            return None
        if entry.get("fingerprint") != fingerprint:
            logger.info(
                "Context %s is on record but its data has changed since "
                "%s; it will be trained again.",
                key, entry.get("recorded_at", "an earlier run"),
            )
            return None
        return entry

    # -- writing -----------------------------------------------------------

    def record(
        self,
        key: str,
        fingerprint: str,
        *,
        champion: dict[str, Any] | None = None,
        refusal: dict[str, Any] | None = None,
    ) -> None:
        """Remember one finished context and flush immediately.

        Flushed per context rather than at the end, because the run this
        exists for is the one that does not reach the end.
        """
        self._entries[key] = {
            "fingerprint": fingerprint,
            "recorded_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "outcome": "champion" if champion else "no_champion",
            "champion": _jsonable(champion),
            "refusal": _jsonable(refusal),
        }
        self._flush()

    def _flush(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"schema": SCHEMA, "entries": self._entries},
                    ensure_ascii=False, indent=1, default=str,
                ),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        except (OSError, TypeError, ValueError) as e:
            # A ledger that cannot be written must not take the run with it:
            # the worst case is a restart that recomputes, which is exactly
            # what happens today anyway.
            logger.warning("Could not write the context ledger (%s).", e)

    # -- fingerprinting ----------------------------------------------------

    @staticmethod
    def fingerprint(frame: pd.DataFrame, target_name: str) -> str:
        """What this context's data is, cheaply and specifically.

        Covers the shape, the column names, the time span, the whole target
        column, and a strided sample of the numeric feature matrix. A change
        to any enricher moves the last of these; a change to the collection
        window moves the first three.
        """
        digest = hashlib.sha256()
        digest.update(SCHEMA.encode())
        digest.update(f"rows={len(frame)};cols={len(frame.columns)}".encode())
        digest.update("|".join(sorted(str(c) for c in frame.columns)).encode())

        index = frame.index
        if len(index):
            digest.update(f"{index[0]}..{index[-1]}".encode())

        if target_name in frame.columns:
            target = pd.to_numeric(frame[target_name], errors="coerce")
            digest.update(
                np.ascontiguousarray(target.to_numpy(dtype=np.float64)).tobytes()
            )

        numeric = frame.select_dtypes(include=[np.number])
        if not numeric.empty:
            # Column-wise, so nothing the width of the frame is ever
            # materialised -- the whole reason the run this serves died. A
            # sum plus a missing-count per column moves for ANY changed
            # cell, anywhere, which the strided sample below cannot promise.
            digest.update(
                np.ascontiguousarray(
                    numeric.sum(numeric_only=True).to_numpy(dtype=np.float64)
                ).tobytes()
            )
            digest.update(
                np.ascontiguousarray(
                    numeric.isna().sum().to_numpy(dtype=np.int64)
                ).tobytes()
            )

            # And a strided sample of the values themselves, because two
            # different columns can share a sum: swapping two rows, or a pair
            # of offsetting edits, leaves the totals alone.
            step = max(1, len(numeric) // FINGERPRINT_SAMPLE_ROWS)
            sample = numeric.iloc[::step]
            if not sample.empty:
                digest.update(
                    np.ascontiguousarray(
                        sample.to_numpy(dtype=np.float64, na_value=np.nan)
                    ).tobytes()
                )
        return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    """Strip what json cannot hold, without losing the record entirely."""
    if value is None:
        return None
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError) as e:
        # Say it rather than coerce in silence. A champion record that had to
        # be stringified is one whose replay will not be byte-identical to
        # what was trained, and that is worth being able to find in a log.
        logger.debug(
            "Ledger entry needed coercion (%s: %s); values json cannot hold "
            "are stored as their string form.", type(e).__name__, e,
        )
        return json.loads(json.dumps(value, default=str))
