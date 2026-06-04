"""
Web UI Mode for Trading System
(ОНОВЛЕНО для використання UnifiedConfigManager)
"""
import http.server
import json
import socketserver
from datetime import datetime
from typing import Any

from src.main.modes.base import BaseMode

CONTENT_TYPE_HTML = 'text/html'


class WebUIMode(BaseMode):
    """Режим Web UI для Trading System"""

    def __init__(self):
        super().__init__()
        try:
            from src.monitoring import ModelPerformanceMonitor
            self.performance_monitor = ModelPerformanceMonitor()
        except ImportError:
            self.performance_monitor = None
            self.logger.warning(
                'ModelPerformanceMonitor not available, using None')
        self.data_collector = None

    def run(self, **kwargs) ->dict[str, Any]:
        """Запуск Web UI сервера"""
        host = kwargs.get('host', 'localhost')
        port = kwargs.get('port', 8080)
        try:
            self.logger.info('Starting Web UI on %s:%d', host, port)
            handler = self._create_handler()
            self._log_startup_info(host, port)
            with socketserver.TCPServer((host, port), handler) as httpd:
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    return self._handle_server_shutdown(host, port)
        except OSError as e:
            return self._handle_startup_error(e, host, port)
        except Exception as e:
            self.logger.error(f'Виникла помилка: {e}', exc_info=True)
            return self._handle_startup_error(e, host, port)

    def _log_startup_info(self, host: str, port: int) ->None:
        """Логує інформацію про запуск сервера."""
        self.logger.info('[START] Trading System Web UI')
        self.logger.info('[DATA] Dashboard: http://%s:%d', host, port)
        self.logger.info('💼 Trading Interface: http://%s:%d/trading', host,
            port)
        self.logger.info('[UP] System Overview: http://%s:%d/dashboard',
            host, port)
        self.logger.info('[RESTART] Auto-refresh enabled (30 seconds)')
        self.logger.info('⏹️  Press Ctrl+C to stop server')

    def _handle_server_shutdown(self, host: str, port: int) ->dict[str, Any]:
        """Обробляє коректне завершення роботи сервера."""
        self.logger.info('\n🛑 Server stopped')
        return {'status': 'stopped', 'mode': 'web-ui', 'host': host, 'port':
            port}

    def _handle_startup_error(self, error: Exception, host: str, port: int
        ) ->dict[str, Any]:
        """Обробляє помилки запуску сервера."""
        self.logger.error('Web UI failed to start: %s', str(error))
        return {'status': 'failed', 'mode': 'web-ui', 'host': host, 'port':
            port, 'error': str(error)}

    def _create_handler(self):
        """Stvorennya obrobnika zapytiv"""
        return self._create_trading_ui_handler()

    def _create_trading_ui_handler(self):
        """Create trading UI handler class"""


        class TradingUIHandler(http.server.SimpleHTTPRequestHandler):

            def __init__(self, *args, **kwargs):
                self.web_ui_mode = self.__class__.web_ui_mode
                super().__init__(*args, **kwargs)

            def do_GET(self):
                if self._is_static_file_request():
                    self._handle_static_file()
                elif self.path.startswith('/api/'):
                    self.handle_api_request()
                else:
                    self._handle_page_request()

            def _is_static_file_request(self):
                """Check if request is for static file"""
                return self.path not in ['/', '/dashboard', '/trading']

            def _handle_static_file(self):
                """Handle static file requests"""
                super().do_GET()

            def _handle_page_request(self):
                """Handle page requests"""
                page_handlers = {'/': ('index.html', CONTENT_TYPE_HTML),
                    '/dashboard': ('dashboard.html', CONTENT_TYPE_HTML),
                    '/trading': ('trading.html', CONTENT_TYPE_HTML)}
                if self.path in page_handlers:
                    filename, content_type = page_handlers[self.path]
                    self.serve_file(filename, content_type)
                else:
                    self.send_error(404)

            def serve_file(self, filename, content_type):
                try:
                    content = self.get_html_content(filename)
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                except Exception as e:
                    self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                    self.send_error(500, str(e))
                    raise

            def handle_api_request(self):
                try:
                    response = self._get_api_response()
                    if response is not None:
                        self.send_json_response(response)
                    else:
                        self.send_error(404)
                except Exception as e:
                    self.logger.error('API request failed: %s', str(e))
                    self.send_error(500, str(e))

            def _get_api_response(self):
                """Get API response based on path"""
                api_handlers = {'/api/system/overview': self.web_ui_mode.
                    get_system_overview, '/api/portfolio/status': self.
                    web_ui_mode.get_portfolio_status, '/api/market/data':
                    self.web_ui_mode.get_market_data,
                    '/api/performance/metrics': self.web_ui_mode.
                    get_performance_metrics}
                if self.path in api_handlers:
                    return api_handlers[self.path]()
                return None

            def send_json_response(self, data):
                content = json.dumps(data, default=str, ensure_ascii=False,
                    indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))

            def get_html_content(self, filename):
                html_handlers = {'index.html': self.web_ui_mode.
                    get_index_html, 'dashboard.html': self.web_ui_mode.
                    get_dashboard_html, 'trading.html': self.web_ui_mode.
                    get_trading_html}
                if filename in html_handlers:
                    return html_handlers[filename]()
                else:
                    raise FileNotFoundError(f'File {filename} not found')
        TradingUIHandler.web_ui_mode = self
        return TradingUIHandler

    def get_system_overview(self) ->dict[str, Any]:
        """Отримання огляду системи"""
        tickers = self.config_manager.get('data.tickers', [])
        return {'status': 'idle', 'last_update': datetime.now().isoformat(),
            'active_mode': None, 'running_tasks': [], 'config': {'tickers':
            tickers[:10], 'total_tickers': len(tickers), 'timeframes': self
            .config_manager.get('data.timeframes', []), 'initial_capital':
            self.config_manager.get('risk.initial_capital', 0)},
            'performance': self._get_performance_report()}

    def get_portfolio_status(self) ->dict[str, Any]:
        return {'total_value': 125000, 'cash_balance': 15000,
            'positions_count': 8, 'daily_pnl': 2500, 'daily_pnl_percent':
            2.04, 'positions': [{'ticker': 'TSLA', 'quantity': 50, 'value':
            12500, 'pnl': 500}, {'ticker': 'NVDA', 'quantity': 30, 'value':
            18000, 'pnl': 800}, {'ticker': 'AAPL', 'quantity': 100, 'value':
            17500, 'pnl': -200}]}

    def get_market_data(self) ->dict[str, Any]:
        market_data = {}
        tickers = self.config_manager.get('data.tickers', ['TSLA', 'NVDA',
            'AAPL', 'MSFT', 'GOOGL'])
        import random
        seed = self.config_manager.get('performance.random_seed', 42)
        random.seed(seed)
        for ticker in tickers[:5]:
            try:
                market_data[ticker] = {'ticker': ticker, 'price': round(
                    random.uniform(100, 500), 2), 'change': round(random.
                    uniform(-10, 10), 2), 'change_percent': round(random.
                    uniform(-5, 5), 2), 'volume': random.randint(1000000,
                    10000000), 'timestamp': datetime.now().isoformat(),
                    'source': 'simulation'}
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning('Failed to get market data for %s: %s',
                    ticker, str(e))
                market_data[ticker] = {'error': str(e)}
                raise
        return market_data

    def get_performance_metrics(self) ->dict[str, Any]:
        """Get performance metrics for API endpoint"""
        return {'recent_activity': [{'time': datetime.now().strftime(
            '%H:%M'), 'action': 'BUY', 'ticker': 'TSLA', 'quantity': 10,
            'price': 245.5}, {'time': datetime.now().strftime('%H:%M'),
            'action': 'SELL', 'ticker': 'AAPL', 'quantity': 5, 'price':
            178.25}, {'time': datetime.now().strftime('%H:%M'), 'action':
            'BUY', 'ticker': 'NVDA', 'quantity': 8, 'price': 485.75}],
            'total_trades': 156, 'win_rate': 0.65, 'avg_return': 0.023,
            'sharpe_ratio': 1.45, 'max_drawdown': -0.08}

    def _get_performance_report(self) ->dict[str, Any]:
        """Get performance report from monitor or fallback data"""
        if self.performance_monitor:
            try:
                return self.performance_monitor.get_performance_report()
            except Exception as e:
                self.logger.error(f'Виникла помилка: {e}', exc_info=True)
                self.logger.warning('Failed to get performance report: %s',
                    str(e))
                raise
        return {'status': 'active', 'cpu_usage': 45.2, 'memory_usage': 67.8,
            'disk_usage': 23.1, 'last_update': datetime.now().isoformat()}

    def get_index_html(self) ->str:
        return """<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .navbar { background: #2c3e50; color: white; padding: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1rem; }
        .status-idle { background: #95a5a6; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .price-up { color: #27ae60; }
        .price-down { color: #e74c3c; }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>[UP] Trading System</h1>
        <a href="/" style="color:white">Головна</a>
        <a href="/dashboard" style="color:white">Панель</a>
        <a href="/trading" style="color:white">Торговля</a>
    </nav>
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>🖥️ Статус системи</h3>
                <div id="system-status">Завантаження...</div>
            </div>
            <div class="card">
                <h3>💼 Портфоліо</h3>
                <div id="portfolio-status">Завантаження...</div>
            </div>
            <div class="card">
                <h3>[NOTIFY] Останні події</h3>
                <div id="recent-activity">Завантаження...</div>
            </div>
        </div>
        <div class="card">
            <h3>[DATA] Ринкові дані</h3>
            <div id="market-data">Завантаження...</div>
        </div>
    </div>
    <script>
        async function loadData() {
            try {
                const [system, portfolio, market, performance] = await Promise.all([
                    fetch('/api/system/overview').then(res => res.json()),
                    fetch('/api/portfolio/status').then(res => res.json()),
                    fetch('/api/market/data').then(res => res.json()),
                    fetch('/api/performance/metrics').then(res => res.json())
                ]);
                updateSystemStatus(system);
                updatePortfolioStatus(portfolio);
                updateMarketData(market);
                updateRecentActivity(performance);
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        function updateSystemStatus(data) {
            const statusClass = data.status === 'running' ? 'status-running' : 'status-idle';
            document.getElementById('system-status').innerHTML = `
                <div><span style="display:inline-block;width:12px;height:12px;border-radius:50%;margin-right:8px;" class="${statusClass}"></span> Статус: ${data.status}</div>
                <div>Активний режим: ${data.active_mode || 'Немає'}</div>
                <div>Тікери: ${data.config.total_tickers}</div>
                <div>Капітал: $${data.config.initial_capital.toLocaleString()}</div>`;
        }
        function updatePortfolioStatus(data) {
            document.getElementById('portfolio-status').innerHTML = `
                <div>Загальна вартість: $${data.total_value.toLocaleString()}</div>
                <div>Дохід за день: <span class="${data.daily_pnl >= 0 ? 'price-up' : 'price-down'}">$${data.daily_pnl.toLocaleString()}</span></div>
                <div>Позицій: ${data.positions_count}</div>
                <div>Готівка: $${data.cash_balance.toLocaleString()}</div>`;
        }
        function updateMarketData(data) {
            const html = Object.entries(data).map(([ticker, info]) => {
                if (info.error) return `<div><strong>${ticker}</strong>: Помилка</div>`;
                const changeClass = info.change >= 0 ? 'price-up' : 'price-down';
                const changeSymbol = info.change >= 0 ? '+' : '';
                return `<div><strong>${ticker}</strong>: $${info.price.toFixed(2)} <span class="${changeClass}">${changeSymbol}${info.change.toFixed(2)} (${changeSymbol}${info.change_percent.toFixed(2)}%)</span></div>`;
            }).join('');
            document.getElementById('market-data').innerHTML = html;
        }
        function updateRecentActivity(data) {
            const activities = data.recent_activity || [];
            const html = activities.map(act => `<div><small>${act.time}</small> ${act.action} ${act.quantity} ${act.ticker} @ $${act.price}</div>`).join('');
            document.getElementById('recent-activity').innerHTML = html || '<div>Немає</div>';
        }
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>"""

    def get_dashboard_html(self) ->str:
        return '<!DOCTYPE html> ... '

    def get_trading_html(self) ->str:
        return '<!DOCTYPE html> ... '
