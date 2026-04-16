"""
Monitoring Dashboard - Дашборд моніторингу.

Генерує інтерактивні дашборди для візуалізації:
- Системних метрик (CPU, пам'ять, диск)
- Продуктивності моделей
- Якості даних
- Сповіщень та алертів

Використовує:
- Plotly для інтерактивних графіків
- Dash для веб-інтерфейсу
- Real-time оновлення даних
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import threading
import time

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import dash
    from dash import html, dcc
    from dash.dependencies import Input, Output
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly and Dash not available. Dashboard will use text-based output.")

from src.core.logging.logger import ProjectLogger
from .monitoring_system import MonitoringSystem, AlertSeverity

class MonitoringDashboardApp:
    """Веб-додаток дашборду моніторингу"""

    def __init__(self, monitoring_system: MonitoringSystem, config: Optional[Dict[str, Any]] = None):
        self.monitoring_system = monitoring_system
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("MonitoringDashboardApp")

        # Налаштування
        self.port = self.config.get('port', 8050)
        self.host = self.config.get('host', 'localhost')
        self.update_interval = self.config.get('update_interval', 5000)  # ms

        # Ініціалізація Dash додатку
        if PLOTLY_AVAILABLE:
            self.app = dash.Dash(__name__, title="Trading System Monitor")
            self._setup_layout()
            self._setup_callbacks()
        else:
            self.app = None

    def _setup_layout(self):
        """Налаштування макету дашборду"""
        self.app.layout = html.Div([
            html.H1("Trading System Monitoring Dashboard",
                   style={'textAlign': 'center', 'marginBottom': 30}),

            # Статус системи
            html.Div([
                html.H2("System Status"),
                html.Div(id='system-status', style={'fontSize': '24px', 'marginBottom': 20})
            ]),

            # Основні метрики
            html.Div([
                html.H2("Key Metrics"),
                html.Div([
                    dcc.Graph(id='cpu-memory-graph'),
                    dcc.Graph(id='disk-network-graph'),
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            ]),

            # Продуктивність моделей
            html.Div([
                html.H2("Model Performance"),
                dcc.Graph(id='model-performance-graph'),
            ]),

            # Якість даних
            html.Div([
                html.H2("Data Quality"),
                dcc.Graph(id='data-quality-graph'),
            ]),

            # Сповіщення
            html.Div([
                html.H2("Active Alerts"),
                html.Div(id='alerts-table'),
            ]),

            # Інтервал оновлення
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ], style={'padding': '20px'})

    def _setup_callbacks(self):
        """Налаштування callback функцій"""

        @self.app.callback(
            [Output('system-status', 'children'),
             Output('cpu-memory-graph', 'figure'),
             Output('disk-network-graph', 'figure'),
             Output('model-performance-graph', 'figure'),
             Output('data-quality-graph', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            """Оновлення всіх компонентів дашборду"""
            try:
                dashboard_data = self.monitoring_system.get_dashboard_data()

                # Статус системи
                system_status = dashboard_data.get('system_status', 'unknown')
                status_color = {
                    'healthy': 'green',
                    'degraded': 'orange',
                    'unhealthy': 'red',
                    'critical': 'darkred'
                }.get(system_status, 'gray')

                status_div = html.Div([
                    html.Span(f"Status: {system_status.upper()}",
                             style={'color': status_color, 'fontWeight': 'bold'}),
                    html.Br(),
                    html.Span(f"Active Monitors: {dashboard_data.get('summary', {}).get('active_monitors', 0)}"),
                    html.Br(),
                    html.Span(f"Active Alerts: {len(dashboard_data.get('alerts', {}).get('active', []))}")
                ])

                # Графіки
                cpu_memory_fig = self._create_cpu_memory_graph(dashboard_data)
                disk_network_fig = self._create_disk_network_graph(dashboard_data)
                model_perf_fig = self._create_model_performance_graph(dashboard_data)
                data_quality_fig = self._create_data_quality_graph(dashboard_data)

                # Таблиця сповіщень
                alerts_table = self._create_alerts_table(dashboard_data)

                return status_div, cpu_memory_fig, disk_network_fig, model_perf_fig, data_quality_fig, alerts_table

            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                return "Error loading dashboard", {}, {}, {}, {}, "Error loading alerts"

    def _create_cpu_memory_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Створення графіку CPU та пам'яті"""
        system_metrics = data.get('monitors', {}).get('system_health', {}).get('metrics', {})

        if not system_metrics:
            return go.Figure()

        fig = make_subplots(rows=1, cols=2, subplot_titles=('CPU Usage', 'Memory Usage'))

        # CPU
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_metrics.get('cpu_percent', 0),
                title={'text': "CPU %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkblue"}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )

        # Memory
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_metrics.get('memory_percent', 0),
                title={'text': "Memory %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkgreen"}},
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )

        fig.update_layout(height=300)
        return fig

    def _create_disk_network_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Створення графіку диска та мережі"""
        system_metrics = data.get('monitors', {}).get('system_health', {}).get('metrics', {})

        if not system_metrics:
            return go.Figure()

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Disk Usage', 'Network I/O'))

        # Disk
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=system_metrics.get('disk_percent', 0),
                title={'text': "Disk %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkred"}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )

        # Network (спрощено)
        network_sent = system_metrics.get('network_bytes_sent', 0) / (1024**2)  # MB
        network_recv = system_metrics.get('network_bytes_recv', 0) / (1024**2)  # MB

        fig.add_trace(
            go.Bar(
                x=['Sent', 'Received'],
                y=[network_sent, network_recv],
                marker_color=['lightblue', 'lightgreen'],
                name='Network I/O (MB)',
                showlegend=False
            ),
            row=1, col=2
        )

        fig.update_layout(height=300)
        return fig

    def _create_model_performance_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Створення графіку продуктивності моделей"""
        model_metrics = data.get('monitors', {}).get('model_performance', {}).get('metrics', {})

        if not model_metrics:
            return go.Figure()

        fig = go.Figure()

        # Метрики моделей
        metrics_data = [
            model_metrics.get('total_models', 0),
            model_metrics.get('active_models', 0),
            model_metrics.get('models_with_drift', 0),
            model_metrics.get('average_accuracy', 0) * 100
        ]

        fig.add_trace(go.Bar(
            x=['Total Models', 'Active Models', 'Models with Drift', 'Avg Accuracy %'],
            y=metrics_data,
            marker_color=['blue', 'green', 'orange', 'red']
        ))

        fig.update_layout(
            title="Model Performance Overview",
            height=400
        )

        return fig

    def _create_data_quality_graph(self, data: Dict[str, Any]) -> go.Figure:
        """Створення графіку якості даних"""
        data_metrics = data.get('monitors', {}).get('data_quality', {}).get('metrics', {})

        if not data_metrics:
            return go.Figure()

        fig = make_subplots(rows=1, cols=2, subplot_titles=('Data Completeness', 'Data Sources'))

        # Completeness
        completeness = data_metrics.get('average_completeness', 1.0) * 100
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=completeness,
                title={'text': "Avg Completeness %"},
                gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "darkcyan"}},
                domain={'row': 0, 'column': 0}
            ),
            row=1, col=1
        )

        # Sources
        sources_total = data_metrics.get('total_sources', 0)
        sources_issues = data_metrics.get('sources_with_issues', 0)

        fig.add_trace(
            go.Pie(
                labels=['Healthy Sources', 'Sources with Issues'],
                values=[sources_total - sources_issues, sources_issues],
                marker_colors=['lightgreen', 'salmon'],
                domain={'row': 0, 'column': 1}
            ),
            row=1, col=2
        )

        fig.update_layout(height=400)
        return fig

    def _create_alerts_table(self, data: Dict[str, Any]) -> html.Div:
        """Створення таблиці сповіщень"""
        alerts = data.get('alerts', {}).get('active', [])

        if not alerts:
            return html.Div("No active alerts")

        # Створення таблиці
        table_header = [
            html.Thead(html.Tr([
                html.Th("Severity"),
                html.Th("Monitor"),
                html.Th("Message"),
                html.Th("Time")
            ]))
        ]

        table_rows = []
        for alert in alerts[:10]:  # Показати максимум 10
            severity = alert.get('severity', 'unknown')
            color = {
                'info': 'blue',
                'warning': 'orange',
                'error': 'red',
                'critical': 'darkred'
            }.get(severity, 'black')

            row = html.Tr([
                html.Td(html.Span(severity.upper(), style={'color': color, 'fontWeight': 'bold'})),
                html.Td(alert.get('monitor', 'unknown')),
                html.Td(alert.get('message', 'No message')),
                html.Td(alert.get('timestamp', 'unknown')[:19])  # Формат часу
            ])
            table_rows.append(row)

        table_body = [html.Tbody(table_rows)]

        return html.Table(table_header + table_body, style={'width': '100%', 'borderCollapse': 'collapse'})

    def run_server(self, debug: bool = False):
        """Запуск веб-сервера дашборду"""
        if not self.app:
            self.logger.error("Dashboard app not available (missing Plotly/Dash)")
            return

        try:
            self.logger.info(f"Starting dashboard server on {self.host}:{self.port}")
            self.app.run_server(host=self.host, port=self.port, debug=debug)
        except Exception as e:
            self.logger.error(f"Error running dashboard server: {e}")

class TextBasedDashboard:
    """Текстовий дашборд для випадків, коли Plotly недоступний"""

    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring_system = monitoring_system
        self.logger = ProjectLogger.get_logger("TextBasedDashboard")

    def generate_report(self) -> str:
        """Генерація текстового звіту"""
        try:
            data = self.monitoring_system.get_dashboard_data()

            report_lines = []
            report_lines.append("=" * 60)
            report_lines.append("TRADING SYSTEM MONITORING DASHBOARD")
            report_lines.append("=" * 60)
            report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append("")

            # Статус системи
            system_status = data.get('system_status', 'unknown')
            summary = data.get('summary', {})
            report_lines.append(f"System Status: {system_status.upper()}")
            report_lines.append(f"Active Monitors: {summary.get('active_monitors', 0)}/{summary.get('total_monitors', 0)}")
            report_lines.append(f"Active Alerts: {len(data.get('alerts', {}).get('active', []))}")
            report_lines.append("")

            # Системні метрики
            system_metrics = data.get('monitors', {}).get('system_health', {}).get('metrics', {})
            if system_metrics:
                report_lines.append("SYSTEM METRICS:")
                report_lines.append(f"  CPU Usage: {system_metrics.get('cpu_percent', 0):.1f}%")
                report_lines.append(f"  Memory Usage: {system_metrics.get('memory_percent', 0):.1f}%")
                report_lines.append(f"  Disk Usage: {system_metrics.get('disk_percent', 0):.1f}%")
                report_lines.append(f"  Memory Used: {system_metrics.get('memory_used_gb', 0):.1f} GB")
                report_lines.append("")

            # Метрики моделей
            model_metrics = data.get('monitors', {}).get('model_performance', {}).get('metrics', {})
            if model_metrics:
                report_lines.append("MODEL PERFORMANCE:")
                report_lines.append(f"  Total Models: {model_metrics.get('total_models', 0)}")
                report_lines.append(f"  Active Models: {model_metrics.get('active_models', 0)}")
                report_lines.append(f"  Models with Drift: {model_metrics.get('models_with_drift', 0)}")
                report_lines.append(f"  Average Accuracy: {model_metrics.get('average_accuracy', 0):.3f}")
                report_lines.append("")

            # Якість даних
            data_metrics = data.get('monitors', {}).get('data_quality', {}).get('metrics', {})
            if data_metrics:
                report_lines.append("DATA QUALITY:")
                report_lines.append(f"  Total Sources: {data_metrics.get('total_sources', 0)}")
                report_lines.append(f"  Sources with Issues: {data_metrics.get('sources_with_issues', 0)}")
                report_lines.append(f"  Average Completeness: {data_metrics.get('average_completeness', 1.0):.1%}")
                report_lines.append("")

            # Активні сповіщення
            active_alerts = data.get('alerts', {}).get('active', [])
            if active_alerts:
                report_lines.append("ACTIVE ALERTS:")
                for i, alert in enumerate(active_alerts[:5], 1):  # Показати максимум 5
                    severity = alert.get('severity', 'unknown').upper()
                    monitor = alert.get('monitor', 'unknown')
                    message = alert.get('message', 'No message')
                    timestamp = alert.get('timestamp', 'unknown')[:19]
                    report_lines.append(f"  {i}. [{severity}] {monitor}: {message}")
                    report_lines.append(f"     Time: {timestamp}")
                report_lines.append("")

            # Недавні сповіщення
            recent_alerts = data.get('alerts', {}).get('recent', [])
            if recent_alerts:
                report_lines.append(f"RECENT ALERTS (last 24h): {len(recent_alerts)}")
                report_lines.append("")

            report_lines.append("=" * 60)

            return "\n".join(report_lines)

        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")
            return f"Error generating report: {e}"

    def print_report(self):
        """Виведення звіту в консоль"""
        report = self.generate_report()
        print(report)

    def save_report(self, filepath: str):
        """Збереження звіту у файл"""
        try:
            report = self.generate_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")

class MonitoringDashboardGenerator:
    """Генератор дашбордів моніторингу"""

    def __init__(self, monitoring_system: MonitoringSystem, config: Optional[Dict[str, Any]] = None):
        self.monitoring_system = monitoring_system
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("MonitoringDashboardGenerator")

        # Ініціалізація дашбордів
        if PLOTLY_AVAILABLE:
            self.web_dashboard = MonitoringDashboardApp(monitoring_system, config.get('web', {}))
        else:
            self.web_dashboard = None

        self.text_dashboard = TextBasedDashboard(monitoring_system)

        # Налаштування автоматичного збереження
        self.auto_save = self.config.get('auto_save', False)
        self.save_interval = self.config.get('save_interval', 3600)  # секунди
        self.save_path = self.config.get('save_path', 'monitoring_reports')

        # Створення директорії для звітів
        if self.auto_save:
            os.makedirs(self.save_path, exist_ok=True)

        # Потік автоматичного збереження
        self.save_thread = None
        self.is_running = False

    def start_auto_save(self):
        """Запуск автоматичного збереження звітів"""
        if not self.auto_save:
            return

        if self.is_running:
            return

        self.is_running = True
        self.save_thread = threading.Thread(target=self._auto_save_loop, daemon=True)
        self.save_thread.start()
        self.logger.info("Auto-save started")

    def stop_auto_save(self):
        """Зупинка автоматичного збереження"""
        self.is_running = False
        if self.save_thread:
            self.save_thread.join(timeout=5)
        self.logger.info("Auto-save stopped")

    def _auto_save_loop(self):
        """Цикл автоматичного збереження"""
        while self.is_running:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"monitoring_report_{timestamp}.txt"
                filepath = os.path.join(self.save_path, filename)

                self.text_dashboard.save_report(filepath)
                time.sleep(self.save_interval)

            except Exception as e:
                self.logger.error(f"Error in auto-save loop: {e}")
                time.sleep(60)  # Затримка при помилці

    def run_web_dashboard(self, debug: bool = False):
        """Запуск веб-дашборду"""
        if self.web_dashboard:
            self.web_dashboard.run_server(debug=debug)
        else:
            self.logger.warning("Web dashboard not available. Use text dashboard instead.")
            self.text_dashboard.print_report()

    def generate_text_report(self) -> str:
        """Генерація текстового звіту"""
        return self.text_dashboard.generate_report()

    def save_current_report(self, filepath: Optional[str] = None):
        """Збереження поточного звіту"""
        if not filepath:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = os.path.join(self.save_path, f"monitoring_report_{timestamp}.txt")

        self.text_dashboard.save_report(filepath)

    def get_dashboard_summary(self) -> Dict[str, Any]:
        """Отримання зведення дашборду"""
        data = self.monitoring_system.get_dashboard_data()

        return {
            'system_status': data.get('system_status'),
            'active_monitors': data.get('summary', {}).get('active_monitors', 0),
            'total_alerts': len(data.get('alerts', {}).get('active', [])),
            'last_update': datetime.now().isoformat(),
            'web_dashboard_available': self.web_dashboard is not None
        }