"""Open the project database for a test, or skip with the reason.

A contract test that needs the live database has three ways to be unable to
run, and only one of them is a defect:

    the file is missing        -- a fresh clone; skip
    the file is LOCKED         -- a measurement run holds it; skip
    the invariant is violated  -- fail, loudly

Until 2026-09-03 the first was a skip and the second was an unhandled
`IOException`, so eight contract tests ERRORED whenever anything was running.
In this project something is almost always running: a rebuild, a null, a
power curve. The suite was therefore red for reasons that had nothing to do
with the code, most of the time.

That is not a cosmetic problem. `.github/workflows/ci.yml` runs the whole
suite on every push and discards the result with `|| true`, and the note
beside it -- dated 2026-07-25, six weeks unactioned -- says the step "can
never fail the build, regardless of how many tests fail". A suite that goes
red for environmental reasons is exactly how a team arrives at `|| true`, and
once it is there a real ratchet can sit broken for a fortnight without anyone
seeing it. One did (REGISTER #240).

So the environment reports itself as an environment, and red is left to mean
one thing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "trading_data.duckdb"


def connect_or_skip(path: Path | None = None, *, read_only: bool = True):
    """A read-only connection, or `pytest.skip` naming which case applies.

    The lock message is matched on DuckDB's own text rather than on an
    exception subclass: it raises `IOException` both for a missing file and
    for a held one, so the type alone cannot tell "nothing to test" from
    "someone else is testing".
    """
    duckdb = pytest.importorskip("duckdb")
    target = path or DB_PATH
    if not target.exists():
        pytest.skip(f"no database at {target}")
    try:
        return duckdb.connect(str(target), read_only=read_only)
    except Exception as error:  # duckdb.IOException, but be liberal
        text = str(error)
        if "used by another process" in text or "Could not set lock" in text:
            pytest.skip(
                f"the database is held by another process, so this invariant "
                f"cannot be read right now. This is the environment, not a "
                f"failure: {text.splitlines()[0]}"
            )
        raise
