import logging
import duckdb
import sqlite3
from typing import Dict, Union, Any, Optional

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
    _connections: Dict[str, Union[duckdb.DuckDBPyConnection, sqlite3.Connection]] = {}

    @classmethod
    def register(cls, name: str, conn: Union[duckdb.DuckDBPyConnection, sqlite3.Connection]):
        """Register an active connection."""
        cls._connections[name] = conn
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Connection registered: {name}")

    @classmethod
    def get(cls, name: str) -> Optional[Union[duckdb.DuckDBPyConnection, sqlite3.Connection]]:
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
                if _logging_streams_open():
                    logger.error(f"Error closing connection {name}: {e}", exc_info=True)
        cls._connections.clear()

# Global cleanup hook
import atexit
atexit.register(ConnectionRegistry.close_all)
