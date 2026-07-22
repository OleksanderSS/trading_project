import atexit
import logging
import os
import time
from contextlib import contextmanager
from typing import ClassVar

import duckdb

logger = logging.getLogger(__name__)

MEMORY_DB = ':memory:'


class ConnectionHandler:
    """Handles DuckDB connection lifecycle and shared connections."""
    _connections: ClassVar[dict[str, duckdb.DuckDBPyConnection]] = {}
    _connection_lock: ClassVar[dict[str, bool]] = {}

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path
            ) if db_path != MEMORY_DB else db_path
        self.con = self.get_connection(self.db_path)

    @classmethod
    def close_all_connections(cls):
        """Close all open connections."""
        for db_path, conn in list(cls._connections.items()):
            try:
                conn.close()
                logger.info(f"Closed connection to '{db_path}'")
            except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                logger.warning(f"Error closing connection to '{db_path}': {e}")
        cls._connections.clear()
        cls._connection_lock.clear()

    # NOTE: atexit registration moved below the class body (was inside class
    # body which caused NameError — ConnectionHandler not yet defined there).

    @classmethod
    def get_connection(cls, db_path: str, force_new: bool=False,
        retry_count: int=3) ->duckdb.DuckDBPyConnection:
        """Get or create a DuckDB connection."""
        db_path = os.path.abspath(db_path
            ) if db_path != MEMORY_DB else db_path
        if force_new or db_path not in cls._connections:
            if force_new and db_path in cls._connections:
                try:
                    cls._connections[db_path].close()
                except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
                    logger.error(f'Виникла помилка: {e}', exc_info=True)
                    logger.warning(f'Error closing connection: {e}')
                    raise
                del cls._connections[db_path]
            if force_new and os.path.exists(db_path) and db_path != MEMORY_DB:
                try:
                    os.remove(db_path)
                except OSError as e:
                    logger.exception(
                        f"Error removing database file '{db_path}': {e}")
            last_error = None
            for attempt in range(retry_count):
                try:
                    cls._connections[db_path] = duckdb.connect(database=
                        db_path, read_only=False, config={'access_mode':
                        'READ_WRITE', 'threads': 4, 'max_memory': '2GB',
                        'temp_directory': 'data/temp',
                        'enable_object_cache': True, 'checkpoint_threshold':
                        '1GB'})
                    return cls._connections[db_path]
                except duckdb.Error as e:
                    last_error = e
                    if attempt < retry_count - 1:
                        import random
                        time.sleep((2 ** attempt) + random.uniform(0, 0.5))
                    else:
                        logger.error(
                            f"All {retry_count} connection attempts failed for '{db_path}'"
                            )
            try:
                cls._connections[db_path] = duckdb.connect(database=db_path,
                    read_only=False)
                return cls._connections[db_path]
            except Exception:
                raise RuntimeError(
                    f"Cannot connect to database '{db_path}': {last_error}"
                    ) from last_error
        return cls._connections[db_path]

    @contextmanager
    def transaction(self):
        """Context manager for DuckDB transactions."""
        try:
            self.con.begin()
            yield self.con
            self.con.commit()
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            logger.error(f"Error closing connection: {e}")
            raise e

# Register automatic cleanup on interpreter exit.
# Must be outside the class body so ConnectionHandler is already defined.
atexit.register(ConnectionHandler.close_all_connections)
