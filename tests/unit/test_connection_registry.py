"""Closing every connection is this registry's whole job.

Audited as part of the cross-cutting layer (reached by 4 pipeline stages).
Mostly clean: the broad `except Exception` in close_all is correct and
documented -- it runs as an atexit hook, where an escaping duckdb.Error on an
already-closed connection turns a successful run into a non-zero exit code --
and _logging_streams_open guards against logging after the streams are gone.

One structural gap: register() overwrote an existing entry without a word,
dropping the registry's last reference to whatever was there. Not a live leak
-- the sole caller, DataManager.get_connection(force_new=True), closes the old
connection first -- but silent is the wrong default for the one class whose
purpose is not leaking connections.
"""
from __future__ import annotations

import logging

import duckdb
import pytest

from src.core.system.connection_registry import ConnectionRegistry


@pytest.fixture(autouse=True)
def _isolate_registry():
    saved = dict(ConnectionRegistry._connections)
    ConnectionRegistry._connections.clear()
    yield
    ConnectionRegistry._connections.clear()
    ConnectionRegistry._connections.update(saved)


def _connection():
    return duckdb.connect(database=":memory:")


def test_a_registered_connection_can_be_retrieved():
    conn = _connection()
    ConnectionRegistry.register("probe", conn)

    assert ConnectionRegistry.get("probe") is conn


def test_an_unknown_name_returns_none():
    assert ConnectionRegistry.get("never_registered") is None


def test_close_all_closes_and_forgets():
    conn = _connection()
    ConnectionRegistry.register("probe", conn)

    ConnectionRegistry.close_all()

    assert ConnectionRegistry.get("probe") is None
    with pytest.raises(duckdb.Error):
        conn.execute("SELECT 1")


def test_close_all_survives_an_already_closed_connection():
    """The atexit case: DataManager closes the same connection first. An
    exception escaping here makes the process exit non-zero after a run that
    actually succeeded."""
    conn = _connection()
    ConnectionRegistry.register("probe", conn)
    conn.close()

    ConnectionRegistry.close_all()  # must not raise

    assert ConnectionRegistry.get("probe") is None


def test_close_all_keeps_going_after_one_failure():
    class Stubborn:
        def close(self):
            raise RuntimeError("cannot close")

    good = _connection()
    ConnectionRegistry.register("bad", Stubborn())
    ConnectionRegistry.register("good", good)

    ConnectionRegistry.close_all()

    assert ConnectionRegistry._connections == {}
    with pytest.raises(duckdb.Error):
        good.execute("SELECT 1")


def test_replacing_a_different_connection_is_reported(caplog):
    first, second = _connection(), _connection()
    ConnectionRegistry.register("probe", first)

    with caplog.at_level(logging.WARNING):
        ConnectionRegistry.register("probe", second)

    assert ConnectionRegistry.get("probe") is second
    assert any("replaced" in r.getMessage() for r in caplog.records)
    first.close()


def test_re_registering_the_same_object_is_quiet(caplog):
    """Idempotent registration is ordinary and must not cry wolf."""
    conn = _connection()
    ConnectionRegistry.register("probe", conn)

    with caplog.at_level(logging.WARNING):
        ConnectionRegistry.register("probe", conn)

    assert not [r for r in caplog.records if "replaced" in r.getMessage()]
