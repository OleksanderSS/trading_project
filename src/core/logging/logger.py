import csv
import json
import logging
import queue
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
                    return Path(default_path)
            else:
                from src.config.unified_config_manager import get_current_config

            config = get_current_config()
            path_str = config.get(f'system.{key}')
            if path_str:
                return Path(path_str)
        except (ImportError, TypeError) as e:
            logging.getLogger('ProjectLogger').debug(
                f"Could not read logging path from config, using default '{default_path}': {e}"
            )
        except Exception as e:
            logging.getLogger('ProjectLogger').warning(
                f"Could not read logging path from config, using default '{default_path}': {e}"
            )
        return Path(default_path)

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
    def setup_logging(level: str='DEBUG', format_string: str | None=None
        ) ->None:
        """
        Configures a clean logging setup with console and rotating file handlers.
        """
        if ProjectLogger._is_configured:
            return
        ProjectLogger._log_dir = ProjectLogger._get_config_path('logs_path',
            'logs')
        ProjectLogger._system_log_file = ProjectLogger._log_dir / 'system.log'
        ProjectLogger._market_movers_csv = (ProjectLogger._log_dir /
            'market_movers_log.csv')
        ProjectLogger._log_dir.mkdir(parents=True, exist_ok=True)
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        log_level = getattr(logging, level.upper(), logging.INFO)
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
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        file_handler = RotatingFileHandler(ProjectLogger._system_log_file,
            maxBytes=10 * 1024 * 1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        root_logger.propagate = False
        ProjectLogger._is_configured = True
        ProjectLogger._initialize_market_log()
        ProjectLogger._start_csv_listener()
        logging.getLogger('ProjectLogger').info(
            f'Logging configured. Level: {level}. Path: {ProjectLogger._log_dir}'
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
