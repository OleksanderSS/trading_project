"""
Dashboard Data Bridge
Connects pipeline data to Streamlit dashboard
Provides real-time data access for dashboard components
"""

from datetime import datetime
from typing import Any

import pandas as pd

from src.core.logging.logger import ProjectLogger

try:
    from src.data.management.data_manager import DataManager
    DATAMANAGER_AVAILABLE = True
except ImportError:
    DATAMANAGER_AVAILABLE = False
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from src.data.management.data_manager import DataManager


class DashboardDataBridge:
    """
    Bridge between pipeline data and dashboard UI.
    Provides cached data access for dashboard components.
    """

    def __init__(self, config_manager=None, error_handler=None):
        """
        Initialize dashboard data bridge

        Args:
            config_manager: Configuration manager instance
            error_handler: Error handler instance
        """
        self.logger = ProjectLogger.get_logger(self.__class__.__name__)
        self.config_manager = config_manager
        self.error_handler = error_handler

        # Data cache
        self._data_cache = {}
        self._cache_timestamps = {}
        self._cache_ttl = 300  # 5 minutes

        # Initialize data sources
        self.data_manager = None
        self._initialize_data_sources()

    def _initialize_data_sources(self):
        """Initialize data sources if available"""
        if not DATAMANAGER_AVAILABLE:
            self.logger.warning("DataManager not available - using sample dashboard data")
            return

        try:
            if self.config_manager:
                self.data_manager = DataManager(self.config_manager)
                self.logger.info("✅ DataManager initialized for dashboard")
            else:
                self.logger.warning("No config manager provided - DataManager not initialized")
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize data sources: {e}")
            self.data_manager = None

    def get_dashboard_data(self, data_type: str, **kwargs) -> dict[str, Any]:
        """
        Get data for dashboard with caching

        Args:
            data_type: Type of data requested
            **kwargs: Additional parameters for data retrieval

        Returns:
            Dict with requested data
        """
        try:
            # Check cache first
            cache_key = f"{data_type}_{hash(str(sorted(kwargs.items())))}"
            if self._is_cache_valid(cache_key):
                cached_data = self._data_cache.get(cache_key, {})
                if isinstance(cached_data, dict):
                    return cached_data
                else:
                    return {}

            # Get fresh data based on type
            data = self._fetch_data_by_type(data_type, **kwargs)

            # Cache the result
            self._data_cache[cache_key] = data
            self._cache_timestamps[cache_key] = datetime.now()

            return data

        except Exception as e:
            self.logger.error(f"❌ Failed to get dashboard data for {data_type}: {e}")
            return {'error': str(e)}

    def _fetch_data_by_type(self, data_type: str, **kwargs) -> dict[str, Any]:
        """Fetch data based on type"""
        data_handlers = {
            "model_performance": self._get_model_performance_data,
            "trading_activity": self._get_trading_activity_data,
            "portfolio_metrics": self._get_portfolio_metrics_data,
            "market_data": self._get_market_data,
            "system_status": self._get_system_status_data,
            "ensemble_weights": self._get_ensemble_weights_data,
            "arena_results": self._get_arena_results_data,
        }

        handler = data_handlers.get(data_type)
        if handler:
            return handler(**kwargs)
        else:
            return {'error': f'Unknown data type: {data_type}'}

    def _query_df(self, query: str, params=None) -> pd.DataFrame:
        """Run a DataManager query using the active database API."""
        if not self.data_manager:
            return pd.DataFrame()
        if hasattr(self.data_manager, 'fetch_df'):
            return self.data_manager.fetch_df(query, params)
        if hasattr(self.data_manager, 'query_data'):
            return self.data_manager.query_data(query)
        return pd.DataFrame()

    def _mark_sample_data(self, data: dict[str, Any]) -> dict[str, Any]:
        data['data_source'] = 'sample'
        data['is_sample_data'] = True
        return data

    def _mark_real_data(self, data: dict[str, Any]) -> dict[str, Any]:
        data['data_source'] = 'database'
        data['is_sample_data'] = False
        return data

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid"""
        if cache_key not in self._cache_timestamps:
            return False

        cache_time = self._cache_timestamps[cache_key]
        age = (datetime.now() - cache_time).total_seconds()
        return bool(age < self._cache_ttl)

    def _get_model_performance_data(self, **kwargs) -> dict[str, Any]:
        """Get model performance data"""
        try:
            if self.data_manager:
                # Query model performance from database
                query = """
                SELECT model_name, model_type, avg_win_rate, avg_sharpe_ratio,
                       avg_precision, total_trades, last_updated
                FROM model_performance
                WHERE last_updated >= datetime('now', '-7 days')
                ORDER BY avg_sharpe_ratio DESC
                """
                data = self._query_df(query)
                if not data.empty:
                    return self._mark_real_data({
                        'models': data.to_dict('records'),
                        'total_models': len(data),
                        'last_updated': datetime.now().isoformat()
                    })

            return self._mark_sample_data({
                'models': [
                    {'model_name': 'LSTM_Ensemble', 'model_type': 'Neural', 'avg_win_rate': 0.65,
                     'avg_sharpe_ratio': 1.8, 'avg_precision': 0.72, 'total_trades': 1240},
                    {'model_name': 'RandomForest_Boost', 'model_type': 'Tree', 'avg_win_rate': 0.58,
                     'avg_sharpe_ratio': 1.4, 'avg_precision': 0.68, 'total_trades': 980},
                    {'model_name': 'XGBoost_Adaptive', 'model_type': 'Tree', 'avg_win_rate': 0.62,
                     'avg_sharpe_ratio': 1.6, 'avg_precision': 0.70, 'total_trades': 1100}
                ],
                'total_models': 3,
                'last_updated': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get model performance data: {e}")
            return {'error': str(e)}

    def _get_trading_activity_data(self, **kwargs) -> dict[str, Any]:
        """Get recent trading activity"""
        try:
            if self.data_manager:
                query = """
                SELECT ticker, signal_type, confidence, timestamp, pnl
                FROM trading_signals
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
                """
                data = self._query_df(query)
                if not data.empty:
                    return self._mark_real_data({
                        'signals': data.to_dict('records'),
                        'total_signals': len(data),
                        'last_updated': datetime.now().isoformat()
                    })

            return self._mark_sample_data({
                'signals': [
                    {'ticker': 'AAPL', 'signal_type': 'BUY', 'confidence': 0.85,
                     'timestamp': datetime.now().isoformat(), 'pnl': 0.023},
                    {'ticker': 'MSFT', 'signal_type': 'SELL', 'confidence': 0.78,
                     'timestamp': datetime.now().isoformat(), 'pnl': -0.015},
                    {'ticker': 'GOOGL', 'signal_type': 'HOLD', 'confidence': 0.92,
                     'timestamp': datetime.now().isoformat(), 'pnl': 0.008}
                ],
                'total_signals': 3,
                'last_updated': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get trading activity data: {e}")
            return {'error': str(e)}

    def _get_portfolio_metrics_data(self, **kwargs) -> dict[str, Any]:
        """Get portfolio performance metrics"""
        try:
            if self.data_manager:
                query = """
                SELECT total_value, returns, volatility, sharpe_ratio, max_drawdown
                FROM portfolio_performance
                WHERE date >= datetime('now', '-30 days')
                ORDER BY date DESC
                """
                data = self._query_df(query)
                if not data.empty:
                    latest = data.iloc[0]
                    return self._mark_real_data({
                        'total_value': float(latest['total_value']),
                        'returns': float(latest['returns']),
                        'volatility': float(latest['volatility']),
                        'sharpe_ratio': float(latest['sharpe_ratio']),
                        'max_drawdown': float(latest['max_drawdown']),
                        'last_updated': datetime.now().isoformat()
                    })

            return self._mark_sample_data({
                'total_value': 125000.0,
                'returns': 0.125,
                'volatility': 0.089,
                'sharpe_ratio': 1.4,
                'max_drawdown': -0.034,
                'last_updated': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get portfolio metrics data: {e}")
            return {'error': str(e)}

    def _get_market_data(self, **kwargs) -> dict[str, Any]:
        """Get market data for dashboard"""
        try:
            ticker = kwargs.get('ticker', 'SPY')

            if self.data_manager:
                query = """
                SELECT date, open, high, low, close, volume
                FROM market_data
                WHERE ticker = ? AND date >= datetime('now', '-30 days')
                ORDER BY date ASC
                """
                data = self._query_df(query, [ticker])
                if not data.empty:
                    return self._mark_real_data({
                        'ticker': ticker,
                        'data': data.to_dict('records'),
                        'last_updated': datetime.now().isoformat()
                    })

            # Mock data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            base_price = 4500.0
            return self._mark_sample_data({
                'ticker': ticker,
                'data': [
                    {
                        'date': dates[i].isoformat(),
                        'open': base_price + (i * 2.5),
                        'high': base_price + (i * 2.5) + 10,
                        'low': base_price + (i * 2.5) - 8,
                        'close': base_price + (i * 2.5) + 2,
                        'volume': 1000000 + (i * 50000)
                    }
                    for i in range(30)
                ],
                'last_updated': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get market data: {e}")
            return {'error': str(e)}

    def _get_system_status_data(self, **kwargs) -> dict[str, Any]:
        """Get system health status"""
        try:
            import psutil

            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory': {
                    'used_gb': memory.used / (1024**3),
                    'available_gb': memory.available / (1024**3),
                    'percent': memory.percent
                },
                'disk': {
                    'used_gb': disk.used / (1024**3),
                    'free_gb': disk.free / (1024**3),
                    'percent': (disk.used / disk.total) * 100
                },
                'status': 'healthy' if cpu_percent < 80 and memory.percent < 80 else 'warning',
                'last_updated': datetime.now().isoformat()
            }

        except ImportError:
            # Fallback if psutil not available
            return {
                'cpu_percent': 45.2,
                'memory': {'used_gb': 8.5, 'available_gb': 7.5, 'percent': 53.1},
                'disk': {'used_gb': 120.3, 'free_gb': 380.7, 'percent': 24.0},
                'status': 'healthy',
                'last_updated': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"❌ Failed to get system status: {e}")
            return {'error': str(e)}

    def _get_ensemble_weights_data(self, **kwargs) -> dict[str, Any]:
        """Get current ensemble model weights"""
        try:
            return self._mark_sample_data({
                'models': [
                    {'name': 'LSTM_Ensemble', 'weight': 0.35, 'performance': 0.72},
                    {'name': 'RandomForest_Boost', 'weight': 0.25, 'performance': 0.68},
                    {'name': 'XGBoost_Adaptive', 'weight': 0.30, 'performance': 0.70},
                    {'name': 'Linear_Regression', 'weight': 0.10, 'performance': 0.62}
                ],
                'last_rebalanced': datetime.now().isoformat(),
                'total_weight': 1.0
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get ensemble weights: {e}")
            return {'error': str(e)}

    def _get_arena_results_data(self, **kwargs) -> dict[str, Any]:
        """Get model arena competition results"""
        try:
            if self.data_manager:
                query = """
                SELECT model_name, battles_won, battles_lost, win_rate, avg_score
                FROM model_arena_results
                WHERE date >= datetime('now', '-7 days')
                ORDER BY win_rate DESC
                """
                data = self._query_df(query)
                if not data.empty:
                    return self._mark_real_data({
                        'results': data.to_dict('records'),
                        'total_models': len(data),
                        'last_updated': datetime.now().isoformat()
                    })

            return self._mark_sample_data({
                'results': [
                    {'model_name': 'LSTM_Ensemble', 'battles_won': 28, 'battles_lost': 12,
                     'win_rate': 0.70, 'avg_score': 0.85},
                    {'model_name': 'XGBoost_Adaptive', 'battles_won': 24, 'battles_lost': 16,
                     'win_rate': 0.60, 'avg_score': 0.78},
                    {'model_name': 'RandomForest_Boost', 'battles_won': 22, 'battles_lost': 18,
                     'win_rate': 0.55, 'avg_score': 0.72}
                ],
                'total_models': 3,
                'last_updated': datetime.now().isoformat()
            })

        except Exception as e:
            self.logger.error(f"❌ Failed to get arena results: {e}")
            return {'error': str(e)}

    def clear_cache(self):
        """Clear all cached data"""
        self._data_cache.clear()
        self._cache_timestamps.clear()
        self.logger.info("Dashboard data cache cleared")

    def get_cache_info(self) -> dict[str, Any]:
        """Get information about cache status"""
        return {
            'cached_items': len(self._data_cache),
            'cache_ttl_seconds': self._cache_ttl,
            'oldest_cache': min(self._cache_timestamps.values()) if self._cache_timestamps else None,
            'newest_cache': max(self._cache_timestamps.values()) if self._cache_timestamps else None
        }
