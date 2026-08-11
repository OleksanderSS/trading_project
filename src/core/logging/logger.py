import csv
import json
import logging
import queue
import re
import sys
import threading
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


class ContextAdapter(logging.LoggerAdapter):
    """
    Adapter to add ticker and timeframe context to log messages.
    """

    def process(self, msg, kwargs):
        ticker = self.extra.get('ticker', 'N/A')
        tf = self.extra.get('timeframe', 'N/A')
        return f'[{ticker}|{tf}] {msg}', kwargs


#: Query parameters and headers whose value is a credential. Matched
#: case-insensitively against `name=value` in any logged text.
_SECRET_PARAM_NAMES = (
    "api_key", "apikey", "access_token", "token", "auth", "key",
    "password", "secret", "client_secret", "signature",
)

_SECRET_PATTERN = re.compile(
    r"(?i)\b(" + "|".join(_SECRET_PARAM_NAMES) + r")=([^&\s\"'<>]+)"
)


class SecretRedactingFilter(logging.Filter):
    """Strip credential values out of every log record.

    httpx logs each request at INFO with the FULL url, so a collector that
    authenticates by query parameter writes its key into system.log and into
    every redirected run log on every call. Observed live: the FRED key
    appears in cleartext on each of the dozens of series requests per run.
    The keys are in `.env`, the logs are gitignored and none of the four
    tracked .log files contain one -- so this is exposure, not a breach --
    but a credential that is written thousands of times is one paste of a log
    away from being published.

    Filtering at the handler covers every logger in the process, including
    third-party ones this project does not call directly (httpx, urllib3),
    which is the only place that can be done once rather than per collector.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str) and "=" in record.msg:
                record.msg = _SECRET_PATTERN.sub(r"\1=***REDACTED***", record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {
                        key: (_SECRET_PATTERN.sub(r"\1=***REDACTED***", value)
                              if isinstance(value, str) else value)
                        for key, value in record.args.items()
                    }
                else:
                    record.args = tuple(
                        _SECRET_PATTERN.sub(r"\1=***REDACTED***", value)
                        if isinstance(value, str) else value
                        for value in record.args
                    )
        except (TypeError, ValueError, AttributeError):
            # A filter must never drop a record because redaction failed.
            pass
        return True


class ProjectLogger:
    """
    A centralized class for managing project-wide logging.
    Supports console, rotating file, and structured CSV logging for market context.
    """
    _is_configured = False
    _log_dir = Path('logs')
    _system_log_file = _log_dir / 'system.log'
    _market_movers_csv = _log_dir / 'market_movers_log.csv'
    _csv_lock = threading.Lock()
    _csv_queue = queue.Queue()
    _csv_listener = None

    @staticmethod
    def _get_config_path(key: str, default_path: str) ->Path:
        """Retrieves path from UnifiedConfigManager if available."""
        return Path(ProjectLogger._get_config_setting(key, default_path))

    @staticmethod
    def _get_config_setting(key: str, default: str) -> str:
        """Reads `system.<key>` from config, if config can safely be reached.

        Called during logging setup, so it must tolerate the config manager
        being absent or mid-initialisation -- hence the guards below rather
        than a plain import.
        """
        try:
            module = sys.modules.get("src.config.unified_config_manager")
            if module is not None:
                manager_cls = getattr(module, "UnifiedConfigManager", None)
                get_current_config = getattr(module, "get_current_config", None)
                if (
                    manager_cls is None
                    or get_current_config is None
                    or getattr(manager_cls, "_initializing", False)
                ):
                    return default
            else:
                from src.config.unified_config_manager import get_current_config

            value = get_current_config().get(f'system.{key}')
            if value:
                return str(value)
        except (ImportError, TypeError) as e:
            logging.getLogger('ProjectLogger').debug(
                f"Could not read logging setting '{key}' from config, using default '{default}': {e}"
            )
        except (ValueError, AttributeError, KeyError, ZeroDivisionError) as e:
            logging.getLogger('ProjectLogger').warning(
                f"Could not read logging setting '{key}' from config, using default '{default}': {e}"
            )
        return default

    @staticmethod
    def _start_csv_listener():
        """Starts a background thread to handle CSV writes from the queue."""

        def csv_writer():
            while True:
                record = ProjectLogger._csv_queue.get()
                if record is None:
                    break
                try:
                    with ProjectLogger._csv_lock:
                        with open(ProjectLogger._market_movers_csv, 'a',
                            newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            writer.writerow(record)
                except (OSError, TypeError, Exception) as e:
                    logging.error(f'Async CSV write failed: {e}', exc_info=True)
                    # We continue here to keep the listener thread alive, but we log the error
                ProjectLogger._csv_queue.task_done()
        threading.Thread(target=csv_writer, daemon=True).start()

    @staticmethod
    def setup_logging(level: str | None=None, format_string: str | None=None
        ) ->None:
        """
        Configures a clean logging setup with console and rotating file handlers.

        The level defaults to `system.log_level` in config (INFO), not to
        DEBUG. Every one of the seven callers invokes this with no arguments,
        so the old DEBUG default was the level the whole project actually ran
        at: 42.9% of the 36,146 lines in logs/system.log were DEBUG, while the
        file rotates at 10 MB with 5 backups. Half the retained history was
        noise, and the 83 CRITICAL and 380 ERROR lines were buried in it.
        Pass level='DEBUG' explicitly, or set system.log_level, when tracing
        something.
        """
        if ProjectLogger._is_configured:
            return
        if level is None:
            level = ProjectLogger._get_config_setting('log_level', 'INFO')
        ProjectLogger._log_dir = ProjectLogger._get_config_path('logs_path',
            'logs')
        ProjectLogger._system_log_file = ProjectLogger._log_dir / 'system.log'
        ProjectLogger._market_movers_csv = (ProjectLogger._log_dir /
            'market_movers_log.csv')
        ProjectLogger._log_dir.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        log_level = getattr(logging, level.upper(), None)
        unknown_level = None if isinstance(log_level, int) else level
        if unknown_level is not None:
            # A typo used to fall through to INFO without a word, so a run
            # configured with level='WARNIGN' looked deliberate. Reported
            # below, once handlers exist -- warning here would go to stderr
            # via logging's last resort and never reach system.log, since
            # every root handler was just removed.
            log_level = logging.INFO
        root_logger.setLevel(log_level)
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, 'reconfigure'):
                try:
                    stream.reconfigure(encoding='utf-8', errors='replace')
                except (OSError, TypeError, Exception) as e:
                    logging.getLogger('ProjectLogger').error(
                        f"Could not reconfigure stream encoding: {e}", exc_info=True)
                    # Continue without reconfiguration
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        formatter = logging.Formatter(format_string)
        secret_filter = SecretRedactingFilter()
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.addFilter(secret_filter)
        root_logger.addHandler(stream_handler)
        file_handler = RotatingFileHandler(ProjectLogger._system_log_file,
            maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.addFilter(secret_filter)
        root_logger.addHandler(file_handler)
        root_logger.propagate = False
        ProjectLogger._is_configured = True
        ProjectLogger._initialize_market_log()
        ProjectLogger._start_csv_listener()
        if unknown_level is not None:
            logging.getLogger('ProjectLogger').warning(
                "Unknown log level %r; falling back to INFO.", unknown_level
            )
        logging.getLogger('ProjectLogger').info(
            f'Logging configured. Level: {logging.getLevelName(log_level)}. '
            f'Path: {ProjectLogger._log_dir}'
            )

    @staticmethod
    def _initialize_market_log():
        """Creates the market movers CSV header if the file doesn't exist."""
        if not ProjectLogger._market_movers_csv.exists():
            header = ['timestamp', 'ticker', 'predicted_signal',
                'actual_outcome', 'volatility', 'top_news_sentiment',
                'context_json']
            with open(ProjectLogger._market_movers_csv, 'w', newline='',
                encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)

    @staticmethod
    def log_market_context_error(ticker: str, predicted: float, actual:
        float, volatility: float, sentiment: float, additional_context:
        dict[str, Any] | None=None) ->None:
        """
        Queues a specialized structured entry to market_movers_log.csv for Meta-Learning analysis.
        """
        if not ProjectLogger._is_configured:
            ProjectLogger.setup_logging()
        timestamp = datetime.now().isoformat()
        context_json = json.dumps(additional_context
            ) if additional_context else '{}'
        ProjectLogger._csv_queue.put([timestamp, ticker, predicted, actual,
            volatility, sentiment, context_json])

    @staticmethod
    def get_logger(name: str) ->logging.Logger:
        """
        Retrieves a logger instance for a specific module.
        """
        if not ProjectLogger._is_configured:
            ProjectLogger.setup_logging()
        return logging.getLogger(name)

    @staticmethod
    def get_context_logger(ticker: str, timeframe: str, name: str=
        'ContextLogger') ->logging.LoggerAdapter:
        """
        Returns an adapted logger that automatically injects ticker and timeframe context.
        """
        base_logger = ProjectLogger.get_logger(name)
        return ContextAdapter(base_logger, {'ticker': ticker, 'timeframe':
            timeframe})

    @staticmethod
    def log_structured(level: str, message: str, **kwargs) ->None:
        """
        Logs a JSON-formatted string for easier parsing of complex events.
        """
        logger = ProjectLogger.get_logger('StructuredLogger')
        structured_data = {'message': message, 'timestamp': datetime.now().
            isoformat(), **kwargs}
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(json.dumps(structured_data))
