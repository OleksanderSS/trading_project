import atexit
import logging
import sqlite3
from typing import ClassVar

import duckdb

logger = logging.getLogger(__name__)


def _logging_streams_open() -> bool:
    """Return False during interpreter shutdown when log streams are closed."""
    handlers = list(logger.handlers) + list(logging.getLogger().handlers)
    for handler in handlers:
        stream = getattr(handler, "stream", None)
        if stream is not None and getattr(stream, "closed", False):
            return False
    return True

class ConnectionRegistry:
    """Centralized registry for all database connections to prevent resource leaks."""
    _connections: ClassVar[dict[str, duckdb.DuckDBPyConnection | sqlite3.Connection]] = {}

    @classmethod
    def register(cls, name: str, conn: duckdb.DuckDBPyConnection | sqlite3.Connection):
        """Register an active connection."""
        existing = cls._connections.get(name)
        if existing is not None and existing is not conn:
            # The only caller today (DataManager.get_connection with
            # force_new) closes the old connection before re-registering, so
            # this is not a live leak. Said out loud anyway: a silent
            # overwrite here drops the registry's last reference to a
            # connection, and closing them is this class's entire purpose.
            logger.warning(
                "Connection %r replaced while a different one was registered "
                "under that name; the previous connection will not be closed "
                "by this registry.", name,
            )
        cls._connections[name] = conn
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Connection registered: {name}")

    @classmethod
    def get(cls, name: str) -> duckdb.DuckDBPyConnection | sqlite3.Connection | None:
        """Retrieve a registered connection."""
        return cls._connections.get(name)

    @classmethod
    def close_all(cls):
        """Safely close all registered connections."""
        for name, conn in list(cls._connections.items()):
            try:
                conn.close()
                if _logging_streams_open():
                    logger.info(f"Closed connection: {name}")
            except Exception as e:
                # Broad catch is intentional: this runs as an atexit hook during
                # interpreter shutdown, where an escaping exception (e.g. duckdb.Error
                # on an already-closed connection) makes the OS-level exit code non-zero
                # even though the pipeline itself completed and logged success.
                if _logging_streams_open():
                    logger.error(f"Error closing connection {name}: {e}", exc_info=True)
        cls._connections.clear()

# Global cleanup hook
atexit.register(ConnectionRegistry.close_all)
