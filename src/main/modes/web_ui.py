#!/usr/bin/env python3
"""
Web UI Mode for Trading System
(ОНОВЛЕНО для використання UnifiedConfigManager)
"""

import logging
import http.server
import socketserver
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# --- ІНТЕГРАЦІЯ: Використовуємо єдиний конфігураційний менеджер ---
from config.unified_config_manager import get_current_config
from utils.real_data_collector import RealDataCollector
from utils.common_utils import PerformanceMonitor


class WebUIMode:
    """Режим Web UI для Trading System"""
    
    # --- РЕФАКТОРИНГ: Оновлюємо конструктор для роботи з UnifiedConfigManager ---
    def __init__(self):
        self.config = get_current_config()
        self.logger = logging.getLogger(__name__)
        # --- РЕФАКТОРИНГ: RealDataCollector тепер не приймає конфігурацію ---
        self.data_collector = RealDataCollector()
        self.performance_monitor = PerformanceMonitor()
        
    def run(self, host: str = 'localhost', port: int = 8080) -> Dict[str, Any]:
        """Запуск Web UI сервера"""
        try:
            self.logger.info(f"Starting Web UI on {host}:{port}")
            
            handler = self._create_handler()
            
            print(f"[START] Trading System Web UI")
            print(f"[DATA] Dashboard: http://{host}:{port}")
            print(f"💼 Trading Interface: http://{host}:{port}/trading")
            print(f"[UP] System Overview: http://{host}:{port}/dashboard")
            print("[RESTART] Auto-refresh enabled (30 seconds)")
            print("⏹️  Press Ctrl+C to stop the server")
            
            with socketserver.TCPServer((host, port), handler) as httpd:
                try:
                    httpd.serve_forever()
                except KeyboardInterrupt:
                    print("\n🛑 Server stopped")
                    return {
                        'status': 'stopped',
                        'mode': 'web-ui',
                        'host': host,
                        'port': port
                    }
                    
        except Exception as e:
            self.logger.error(f"Web UI failed to start: {e}")
            return {
                'status': 'failed',
                'mode': 'web-ui',
                'error': str(e)
            }
    
    def _create_handler(self):
        """Створення обробника запитів"""
        class TradingUIHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.web_ui_mode = self.__class__.web_ui_mode
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                if self.path == '/':
                    self.serve_file('index.html', 'text/html')
                elif self.path == '/dashboard':
                    self.serve_file('dashboard.html', 'text/html')
                elif self.path == '/trading':
                    self.serve_file('trading.html', 'text/html')
                elif self.path.startswith('/api/'):
                    self.handle_api_request()
                else:
                    # Спроба віддати статичний файл, якщо він існує
                    # Це дозволить завантажувати CSS, JS і т.д.
                    # Важливо: встановлюємо директорію для SimpleHTTPRequestHandler
                    # оскільки ми не можемо змінити її під час виконання.
                    # Цей функціонал потребує більш просунутого сервера (напр. aiohttp, Flask)
                    super().do_GET()

            def serve_file(self, filename, content_type):
                try:
                    content = self.get_html_content(filename)
                    self.send_response(200)
                    self.send_header('Content-type', content_type)
                    self.send_header('Content-Length', str(len(content)))
                    self.end_headers()
                    self.wfile.write(content.encode('utf-8'))
                except Exception as e:
                    self.send_error(500, str(e))
            
            def handle_api_request(self):
                try:
                    if self.path == '/api/system/overview':
                        response = self.web_ui_mode.get_system_overview()
                    elif self.path == '/api/portfolio/status':
                        response = self.web_ui_mode.get_portfolio_status()
                    elif self.path == '/api/market/data':
                        response = self.web_ui_mode.get_market_data()
                    elif self.path == '/api/performance/metrics':
                        response = self.web_ui_mode.get_performance_metrics()
                    # ... інші API ...
                    else:
                        self.send_error(404)
                        return
                    
                    self.send_json_response(response)
                    
                except Exception as e:
                    self.send_error(500, str(e))

            def send_json_response(self, data):
                content = json.dumps(data, default=str, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))

            def get_html_content(self, filename):
                if filename == 'index.html':
                    return self.web_ui_mode.get_index_html()
                elif filename == 'dashboard.html':
                    return self.web_ui_mode.get_dashboard_html()
                elif filename == 'trading.html':
                    return self.web_ui_mode.get_trading_html()
                else:
                    raise FileNotFoundError(f'File {filename} not found')

        TradingUIHandler.web_ui_mode = self
        return TradingUIHandler

    def get_system_overview(self) -> Dict[str, Any]:
        """Отримання огляду системи"""
        # --- РЕФАКТОРИНГ: Використовуємо .get() для безпечного доступу ---
        tickers = self.config.get('data.tickers', [])
        return {
            'status': 'idle',
            'last_update': datetime.now().isoformat(),
            'active_mode': None,
            'running_tasks': [],
            'config': {
                'tickers': tickers[:10],
                'total_tickers': len(tickers),
                'timeframes': self.config.get('data.timeframes', []),
                'initial_capital': self.config.get('risk.initial_capital', 0)
            },
            'performance': self.performance_monitor.get_performance_report()
        }

    # ... (get_portfolio_status, get_market_data, etc. залишаються з симульованими даними) ...
    def get_portfolio_status(self) -> Dict[str, Any]:
        return {
            'total_value': 125000, 'cash_balance': 15000, 'positions_count': 8,
            'daily_pnl': 2500, 'daily_pnl_percent': 2.04,
            'positions': [
                {'ticker': 'TSLA', 'quantity': 50, 'value': 12500, 'pnl': 500},
                {'ticker': 'NVDA', 'quantity': 30, 'value': 18000, 'pnl': 800},
                {'ticker': 'AAPL', 'quantity': 100, 'value': 17500, 'pnl': -200}
            ]}
    
    def get_market_data(self) -> Dict[str, Any]:
        market_data = {}
        tickers = self.config.get('data.tickers', ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL'])
        for ticker in tickers[:5]:
            try:
                import random
                market_data[ticker] = {
                    'ticker': ticker,
                    'price': round(random.uniform(100, 500), 2),
                    'change': round(random.uniform(-10, 10), 2),
                    'change_percent': round(random.uniform(-5, 5), 2),
                    'volume': random.randint(1000000, 10000000),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'simulation'}
            except Exception as e:
                market_data[ticker] = {'error': str(e)}
        return market_data

    # ... (HTML-методи залишаються без змін) ...
    def get_index_html(self) -> str:
        return '''<!DOCTYPE html>
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
</html>'''
    def get_dashboard_html(self) -> str:
        return '''<!DOCTYPE html> ... ''' # Зміст скорочено
    def get_trading_html(self) -> str:
        return '''<!DOCTYPE html> ... ''' # Зміст скорочено
