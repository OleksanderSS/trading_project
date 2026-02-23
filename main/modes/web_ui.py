#!/usr/bin/env python3
"""
Web UI Mode for Trading System
Інтегрований режим для запуску веб-інтерфейсу
"""

import logging
import http.server
import socketserver
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from config.trading_config import TradingConfig
from utils.real_data_collector import RealDataCollector
from utils.common_utils import PerformanceMonitor


class WebUIMode:
    """Режим Web UI для Trading System"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_collector = RealDataCollector(config)
        self.performance_monitor = PerformanceMonitor()
        
    def run(self, host: str = 'localhost', port: int = 8080) -> Dict[str, Any]:
        """Запуск Web UI сервера"""
        try:
            self.logger.info(f"Starting Web UI on {host}:{port}")
            
            # Створення та запуск сервера
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
                    print("\\n🛑 Server stopped")
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
                # Зберігаємо посилання на WebUIMode
                self.web_ui_mode = self.__class__.web_ui_mode
                super().__init__(*args, **kwargs)
            
            def do_GET(self):
                """Обробка GET запитів"""
                if self.path == '/':
                    self.serve_file('index.html', 'text/html')
                elif self.path == '/dashboard':
                    self.serve_file('dashboard.html', 'text/html')
                elif self.path == '/trading':
                    self.serve_file('trading.html', 'text/html')
                elif self.path.startswith('/api/'):
                    self.handle_api_request()
                else:
                    self.send_error(404)
            
            def do_POST(self):
                """Обробка POST запитів"""
                if self.path.startswith('/api/'):
                    self.handle_api_request()
                else:
                    self.send_error(404)
            
            def serve_file(self, filename, content_type):
                """Віддача HTML файлу"""
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
                """Обробка API запитів"""
                try:
                    if self.path == '/api/system/overview':
                        response = self.web_ui_mode.get_system_overview()
                    elif self.path == '/api/portfolio/status':
                        response = self.web_ui_mode.get_portfolio_status()
                    elif self.path == '/api/market/data':
                        response = self.web_ui_mode.get_market_data()
                    elif self.path == '/api/performance/metrics':
                        response = self.web_ui_mode.get_performance_metrics()
                    elif self.path.startswith('/api/trading/start'):
                        response = self.web_ui_mode.handle_trading_start(self)
                    else:
                        self.send_error(404)
                        return
                    
                    self.send_json_response(response)
                    
                except Exception as e:
                    self.send_error(500, str(e))
            
            def send_json_response(self, data):
                """Відправка JSON відповіді"""
                content = json.dumps(data, ensure_ascii=False, indent=2)
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.send_header('Content-Length', str(len(content)))
                self.end_headers()
                self.wfile.write(content.encode('utf-8'))
            
            def get_html_content(self, filename):
                """Отримання HTML контенту"""
                if filename == 'index.html':
                    return self.web_ui_mode.get_index_html()
                elif filename == 'dashboard.html':
                    return self.web_ui_mode.get_dashboard_html()
                elif filename == 'trading.html':
                    return self.web_ui_mode.get_trading_html()
                else:
                    raise FileNotFoundError(f'File {filename} not found')
        
        # Встановлюємо посилання на WebUIMode
        TradingUIHandler.web_ui_mode = self
        return TradingUIHandler
    
    def get_system_overview(self) -> Dict[str, Any]:
        """Отримання огляду системи"""
        return {
            'status': 'idle',
            'last_update': datetime.now().isoformat(),
            'active_mode': None,
            'running_tasks': [],
            'config': {
                'tickers': self.config.data.tickers[:10],
                'total_tickers': len(self.config.data.tickers),
                'timeframes': [tf.value for tf in self.config.data.timeframes],
                'initial_capital': self.config.risk.initial_capital
            },
            'performance': self.performance_monitor.get_performance_report()
        }
    
    def get_portfolio_status(self) -> Dict[str, Any]:
        """Отримання статусу портфоліо"""
        return {
            'total_value': 125000,
            'cash_balance': 15000,
            'positions_count': 8,
            'daily_pnl': 2500,
            'daily_pnl_percent': 2.04,
            'positions': [
                {'ticker': 'TSLA', 'quantity': 50, 'value': 12500, 'pnl': 500},
                {'ticker': 'NVDA', 'quantity': 30, 'value': 18000, 'pnl': 800},
                {'ticker': 'AAPL', 'quantity': 100, 'value': 17500, 'pnl': -200}
            ]
        }
    
    def get_market_data(self) -> Dict[str, Any]:
        """Отримання ринкових data"""
        market_data = {}
        tickers = ['TSLA', 'NVDA', 'AAPL', 'MSFT', 'GOOGL']
        
        for ticker in tickers[:5]:
            try:
                # Симуляція data для стабільності
                import random
                market_data[ticker] = {
                    'ticker': ticker,
                    'price': round(random.uniform(100, 500), 2),
                    'change': round(random.uniform(-10, 10), 2),
                    'change_percent': round(random.uniform(-5, 5), 2),
                    'volume': random.randint(1000000, 10000000),
                    'timestamp': datetime.now().isoformat(),
                    'source': 'simulation'
                }
            except Exception as e:
                market_data[ticker] = {'error': str(e)}
        
        return market_data
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Отримання метрик продуктивності"""
        return {
            'system_performance': self.performance_monitor.get_performance_report(),
            'trading_metrics': {
                'total_trades': 156,
                'win_rate': 0.65,
                'avg_win': 250,
                'avg_loss': -120,
                'sharpe_ratio': 1.85,
                'max_drawdown': -0.08
            },
            'recent_activity': [
                {'time': '10:30', 'action': 'BUY', 'ticker': 'TSLA', 'quantity': 10, 'price': 250.5},
                {'time': '10:45', 'action': 'SELL', 'ticker': 'NVDA', 'quantity': 5, 'price': 600.2},
                {'time': '11:00', 'action': 'BUY', 'ticker': 'AAPL', 'quantity': 20, 'price': 175.3}
            ]
        }
    
    def handle_trading_start(self, handler) -> Dict[str, Any]:
        """Обробка запуску торгового режиму"""
        try:
            content_length = int(handler.headers['Content-Length'])
            post_data = handler.rfile.read(content_length)
            data = json.loads(post_data.decode('utf-8'))
            
            mode = data.get('mode')
            params = data.get('params', {})
            
            if not mode:
                return {'error': 'Mode is required'}
            
            # Симуляція запуску режиму
            task_id = f"task_{int(datetime.now().timestamp())}"
            
            self.logger.info(f"Starting trading mode: {mode} with params: {params}")
            
            return {
                'task_id': task_id,
                'status': 'started',
                'message': f'{mode} mode started successfully',
                'params': params
            }
            
        except Exception as e:
            self.logger.error(f"Failed to start trading mode: {e}")
            return {'error': str(e)}
    
    def get_index_html(self) -> str:
        """HTML для головної сторінки"""
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
        .navbar h1 { margin: 0; }
        .navbar a { color: white; text-decoration: none; margin-right: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1rem; }
        .status-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
        .status-running { background: #27ae60; }
        .status-idle { background: #95a5a6; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }
        .btn { background: #3498db; color: white; padding: 0.5rem 1rem; border: none; border-radius: 4px; cursor: pointer; }
        .btn:hover { background: #2980b9; }
        .market-ticker { border: 1px solid #ddd; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; }
        .price-up { color: #27ae60; }
        .price-down { color: #e74c3c; }
        .loading { text-align: center; padding: 2rem; }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>[UP] Trading System</h1>
        <a href="/">Головна</a>
        <a href="/dashboard">Панель</a>
        <a href="/trading">Торговля</a>
    </nav>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>🖥️ Статус системи</h3>
                <div id="system-status" class="loading">Завантаження...</div>
            </div>
            
            <div class="card">
                <h3>💼 Портфоліо</h3>
                <div id="portfolio-status" class="loading">Завантаження...</div>
            </div>
            
            <div class="card">
                <h3>[NOTIFY] Останні події</h3>
                <div id="recent-activity" class="loading">Завантаження...</div>
            </div>
        </div>
        
        <div class="card">
            <h3>[DATA] Ринкові дані</h3>
            <div id="market-data" class="loading">Завантаження...</div>
        </div>
    </div>
    
    <script>
        async function loadData() {
            try {
                const [systemResponse, portfolioResponse, marketResponse, performanceResponse] = await Promise.all([
                    fetch('/api/system/overview'),
                    fetch('/api/portfolio/status'),
                    fetch('/api/market/data'),
                    fetch('/api/performance/metrics')
                ]);
                
                const system = await systemResponse.json();
                const portfolio = await portfolioResponse.json();
                const market = await marketResponse.json();
                const performance = await performanceResponse.json();
                
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
                <div><span class="status-indicator ${statusClass}"></span> Статус: ${data.status}</div>
                <div>Активний режим: ${data.active_mode || 'Немає'}</div>
                <div>Тікери: ${data.config.total_tickers}</div>
                <div>Капітал: $${data.config.initial_capital.toLocaleString()}</div>
            `;
        }
        
        function updatePortfolioStatus(data) {
            document.getElementById('portfolio-status').innerHTML = `
                <div>Загальна вартість: $${data.total_value.toLocaleString()}</div>
                <div>Дохід за день: <span class="${data.daily_pnl >= 0 ? 'price-up' : 'price-down'}">$${data.daily_pnl.toLocaleString()}</span></div>
                <div>Позицій: ${data.positions_count}</div>
                <div>Готівка: $${data.cash_balance.toLocaleString()}</div>
            `;
        }
        
        function updateMarketData(data) {
            const html = Object.entries(data).map(([ticker, info]) => {
                if (info.error) {
                    return `<div class="market-ticker"><strong>${ticker}</strong>: Помилка завантаження</div>`;
                }
                
                const changeClass = info.change >= 0 ? 'price-up' : 'price-down';
                const changeSymbol = info.change >= 0 ? '+' : '';
                
                return `<div class="market-ticker">
                    <strong>${ticker}</strong>: $${info.price.toFixed(2)} 
                    <span class="${changeClass}">${changeSymbol}${info.change.toFixed(2)} (${changeSymbol}${info.change_percent.toFixed(2)}%)</span>
                </div>`;
            }).join('');
            
            document.getElementById('market-data').innerHTML = html;
        }
        
        function updateRecentActivity(data) {
            const activities = data.recent_activity || [];
            const html = activities.map(activity => 
                `<div><small>${activity.time}</small> ${activity.action} ${activity.quantity} ${activity.ticker} @ $${activity.price}</div>`
            ).join('');
            
            document.getElementById('recent-activity').innerHTML = html || '<div>Немає недавньої активності</div>';
        }
        
        loadData();
        setInterval(loadData, 30000);
    </script>
</body>
</html>'''
    
    def get_dashboard_html(self) -> str:
        """HTML для панелі приладів"""
        return '''<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dashboard - Trading System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .navbar { background: #2c3e50; color: white; padding: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; }
        .metric-card { background: #3498db; color: white; padding: 1.5rem; border-radius: 8px; text-align: center; }
        .metric-card h3 { margin-bottom: 0.5rem; }
        .metric-card .value { font-size: 2rem; font-weight: bold; }
        .position-item { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>[DATA] Trading Dashboard</h1>
        <a href="/" style="color: white; text-decoration: none; margin-right: 1rem;">Головна</a>
        <a href="/dashboard" style="color: white; text-decoration: none; margin-right: 1rem;">Панель</a>
        <a href="/trading" style="color: white; text-decoration: none;">Торговля</a>
    </nav>
    
    <div class="container">
        <div class="grid">
            <div class="metric-card">
                <h3>Загальна вартість</h3>
                <div class="value" id="total-value">$125,000</div>
                <small>+2.04% сьогодні</small>
            </div>
            
            <div class="metric-card" style="background: #27ae60;">
                <h3>Дохід за день</h3>
                <div class="value" id="daily-pnl">$2,500</div>
                <small>8 угод</small>
            </div>
            
            <div class="metric-card" style="background: #e74c3c;">
                <h3>Win Rate</h3>
                <div class="value" id="win-rate">65%</div>
                <small>156 угод всього</small>
            </div>
        </div>
        
        <div class="grid" style="grid-template-columns: 2fr 1fr;">
            <div class="card">
                <h3>💼 Позиції</h3>
                <div id="positions-list">
                    <div class="loading">Завантаження...</div>
                </div>
            </div>
            
            <div class="card">
                <h3>[UP] Метрики</h3>
                <div id="metrics-list">
                    <div class="loading">Завантаження...</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        async function loadDashboardData() {
            try {
                const response = await fetch('/api/portfolio/status');
                const data = await response.json();
                
                updateDashboardMetrics(data);
                updatePositionsList(data);
                
            } catch (error) {
                console.error('Error loading dashboard data:', error);
            }
        }
        
        function updateDashboardMetrics(data) {
            document.getElementById('total-value').textContent = `$${data.total_value.toLocaleString()}`;
            document.getElementById('daily-pnl').textContent = `$${data.daily_pnl.toLocaleString()}`;
            document.getElementById('win-rate').textContent = `${(data.win_rate * 100).toFixed(1)}%`;
        }
        
        function updatePositionsList(data) {
            const html = data.positions.map(pos => `
                <div class="position-item">
                    <div>
                        <strong>${pos.ticker}</strong><br>
                        <small>${pos.quantity} шт.</small>
                    </div>
                    <div style="text-align: right;">
                        <strong>$${pos.value.toLocaleString()}</strong><br>
                        <small class="${pos.pnl >= 0 ? 'price-up' : 'price-down'}">
                            ${pos.pnl >= 0 ? '+' : ''}$${pos.pnl}
                        </small>
                    </div>
                </div>
            `).join('');
            
            document.getElementById('positions-list').innerHTML = html;
        }
        
        loadDashboardData();
        setInterval(loadDashboardData, 30000);
    </script>
</body>
</html>'''
    
    def get_trading_html(self) -> str:
        """HTML для торгової сторінки"""
        return '''<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trading - Trading System</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #f5f5f5; }
        .navbar { background: #2c3e50; color: white; padding: 1rem; }
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem; }
        .card { background: white; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); padding: 1.5rem; margin-bottom: 1rem; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 1rem; }
        .form-group { margin-bottom: 1rem; }
        .form-group label { display: block; margin-bottom: 0.5rem; font-weight: bold; }
        .form-control { width: 100%; padding: 0.5rem; border: 1px solid #ddd; border-radius: 4px; }
        .btn { background: #3498db; color: white; padding: 0.75rem 1.5rem; border: none; border-radius: 4px; cursor: pointer; font-size: 1rem; }
        .btn:hover { background: #2980b9; }
        .btn:disabled { background: #95a5a6; cursor: not-allowed; }
        .task-item { border: 1px solid #ddd; padding: 1rem; margin: 0.5rem 0; border-radius: 4px; }
        .task-running { border-left: 4px solid #27ae60; }
        .alert { padding: 1rem; margin: 1rem 0; border-radius: 4px; }
        .alert-success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
        .alert-danger { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
        .loading { text-align: center; padding: 2rem; }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>💼 Trading Interface</h1>
        <a href="/" style="color: white; text-decoration: none; margin-right: 1rem;">Головна</a>
        <a href="/dashboard" style="color: white; text-decoration: none; margin-right: 1rem;">Панель</a>
        <a href="/trading" style="color: white; text-decoration: none;">Торговля</a>
    </nav>
    
    <div class="container">
        <div class="grid">
            <div class="card">
                <h3>[START] Запуск режиму</h3>
                <form id="trading-form">
                    <div class="form-group">
                        <label for="mode-select">Режим:</label>
                        <select id="mode-select" class="form-control">
                            <option value="backtest">Backtest</option>
                            <option value="comprehensive-backtest">Comprehensive Backtest</option>
                            <option value="optimized-backtest">Optimized Backtest</option>
                            <option value="real-data-backtest">Real Data Backtest</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label for="tickers-input">Тікери:</label>
                        <input type="text" id="tickers-input" class="form-control" 
                               placeholder="TSLA,NVDA,AAPL" value="TSLA,NVDA,AAPL">
                    </div>
                    
                    <div class="form-group">
                        <label for="capital-input">Початковий капітал:</label>
                        <input type="number" id="capital-input" class="form-control" 
                               value="100000" min="1000">
                    </div>
                    
                    <button type="submit" class="btn" id="submit-btn">
                        [START] Запустити
                    </button>
                </form>
            </div>
            
            <div class="card">
                <h3>[LIST] Активні задачі</h3>
                <div id="active-tasks" class="loading">
                    Завантаження...
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>[DATA] Результати</h3>
            <div id="results-container">
                <p style="color: #666;">Результати будуть відображені тут</p>
            </div>
        </div>
    </div>
    
    <script>
        document.addEventListener('DOMContentLoaded', function() {
            loadActiveTasks();
            setInterval(loadActiveTasks, 5000);
            
            const form = document.getElementById('trading-form');
            form.addEventListener('submit', handleTradingSubmit);
        });
        
        async function loadActiveTasks() {
            try {
                const response = await fetch('/api/system/overview');
                const data = await response.json();
                
                updateActiveTasks(data.running_tasks);
            } catch (error) {
                console.error('Error loading active tasks:', error);
            }
        }
        
        function updateActiveTasks(tasks) {
            const tasksDiv = document.getElementById('active-tasks');
            
            if (tasks.length === 0) {
                tasksDiv.innerHTML = '<p style="color: #666;">Немає активних задач</p>';
                return;
            }
            
            tasksDiv.innerHTML = tasks.map(taskId => `
                <div class="task-item task-running">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>${taskId}</strong><br>
                            <small style="color: #666;">Виконується...</small>
                        </div>
                    </div>
                </div>
            `).join('');
        }
        
        async function handleTradingSubmit(event) {
            event.preventDefault();
            
            const submitBtn = document.getElementById('submit-btn');
            submitBtn.disabled = true;
            submitBtn.textContent = '⏳ Запуск...';
            
            const mode = document.getElementById('mode-select').value;
            const tickers = document.getElementById('tickers-input').value;
            const capital = parseInt(document.getElementById('capital-input').value);
            
            const params = {
                tickers: tickers,
                initial_capital: capital
            };
            
            try {
                const response = await fetch('/api/trading/start', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        mode: mode,
                        params: params
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    showNotification('Режим запущено successfully!', 'success');
                    loadActiveTasks();
                    displayResults(data);
                } else {
                    showNotification(data.error || 'Помилка запуску режиму', 'error');
                }
                
            } catch (error) {
                showNotification('Помилка з\'єднання', 'error');
                console.error('Error:', error);
            } finally {
                submitBtn.disabled = false;
                submitBtn.textContent = '[START] Запустити';
            }
        }
        
        function displayResults(results) {
            const resultsDiv = document.getElementById('results-container');
            
            resultsDiv.innerHTML = `
                <div class="alert alert-success">
                    <h4>[OK] Результати виконання</h4>
                    <pre style="background: #f8f9fa; padding: 1rem; border-radius: 4px; overflow-x: auto;">${JSON.stringify(results, null, 2)}</pre>
                </div>
            `;
        }
        
        function showNotification(message, type) {
            const notification = document.createElement('div');
            notification.className = `alert alert-${type === 'error' ? 'danger' : 'success'}`;
            notification.style.position = 'fixed';
            notification.style.top = '20px';
            notification.style.right = '20px';
            notification.style.zIndex = '1000';
            notification.style.maxWidth = '400px';
            notification.innerHTML = `
                ${message}
                <button onclick="this.parentElement.remove()" style="background: none; border: none; float: right; font-size: 1.2rem; cursor: pointer;">×</button>
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentElement) {
                    notification.remove();
                }
            }, 5000);
        }
    </script>
</body>
</html>'''
