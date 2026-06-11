"""Monitoring Dashboard.

Generates interactive dashboards for visualizing system,
model, and data metrics.
"""

import os
import threading
import time
from datetime import datetime
from typing import Any, cast, Optional

from src.core.logging.logger import ProjectLogger

from .monitoring_system import MonitoringSystem

logger = ProjectLogger.get_logger("MonitoringDashboard")

# Constants to avoid duplication
GAUGE_NUMBER_MODE = "gauge+number"

try:
    import dash
    import plotly.graph_objects as go
    from dash import dcc, html
    from dash.dependencies import Input, Output
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    # Define dummy components for type safety
    html = cast(Any, type('html', (), {'Div': lambda *args, **kwargs: None, 'H1': lambda *args, **kwargs: None, 'H2': lambda *args, **kwargs: None, 'Th': lambda *args, **kwargs: None, 'Thead': lambda *args, **kwargs: None, 'Tr': lambda *args, **kwargs: None, 'Br': lambda *args, **kwargs: None, 'Span': lambda *args, **kwargs: None, 'Table': lambda *args, **kwargs: None, 'Td': lambda *args, **kwargs: None, 'Tbody': lambda *args, **kwargs: None}))
    dcc = cast(Any, type('dcc', (), {'Interval': lambda *args, **kwargs: None, 'Graph': lambda *args, **kwargs: None}))
    Input = cast(Any, lambda *args, **kwargs: None)
    Output = cast(Any, lambda *args, **kwargs: None)
    logger.warning("Plotly and Dash not available. Dashboard will use text-based output.")

class MonitoringDashboardApp:
    """Monitoring dashboard web app."""

    def __init__(
        self,
        monitoring_system: MonitoringSystem,
        config: dict[str, Any] | None = None,
    ):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly and Dash are required for MonitoringDashboardApp")
            
        self.monitoring_system = monitoring_system
        self.config = config or {}
        self.logger = ProjectLogger.get_logger("MonitoringDashboardApp")
        self.port = self.config.get('port', 8050)
        self.host = self.config.get('host', 'localhost')
        update_ms = self.config.get('update_interval', 5000)
        self.update_interval = update_ms  # ms

        self.app = dash.Dash(__name__, title="Trading System Monitor")
        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Dashboard layout configuration"""
        self.app.layout = html.Div([
            html.H1("Trading System Monitoring Dashboard",
                    style={'textAlign': 'center', 'marginBottom': 30}),

            html.Div([
                html.H2("System Status"),
                html.Div(id='system-status', style={'fontSize': '24px', 'marginBottom': 20})
            ]),

            html.Div([
                html.H2("Key Metrics"),
                html.Div([
                    dcc.Graph(id='cpu-memory-graph'),
                    dcc.Graph(id='disk-network-graph'),
                ], style={'display': 'flex', 'flexWrap': 'wrap'}),
            ]),

            html.Div([
                html.H2("Model Performance"),
                dcc.Graph(id='model-performance-graph'),
            ]),

            html.Div([
                html.H2("Data Quality"),
                dcc.Graph(id='data-quality-graph'),
            ]),

            html.Div([
                html.H2("Active alerts"),
                html.Div(id='alerts-table'),
            ]),

            dcc.Interval(
                id='interval-component',
                interval=self.update_interval,
                n_intervals=0
            )
        ], style={'padding': '20px'})

    def _setup_callbacks(self):
        """Callback functions configuration"""

        @self.app.callback(
            [Output('system-status', 'children'),
             Output('cpu-memory-graph', 'figure'),
             Output('disk-network-graph', 'figure'),
             Output('model-performance-graph', 'figure'),
             Output('data-quality-graph', 'figure'),
             Output('alerts-table', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n: int) -> tuple[Any, ...]:
            """Update all dashboard components"""
            try:
                dashboard_data = self.monitoring_system.get_dashboard_data()

                # System status
                system_status = dashboard_data.get('system_status', 'unknown')
                status_color = {
                    'healthy': 'green',
                    'degraded': 'orange',
                    'unhealthy': 'red',
                    'critical': 'darkred'
                }.get(system_status, 'gray')

                status_div = html.Div([
                    html.Span(
                        f"Status: {system_status.upper()}",
                        style={
                            'color': status_color,
                            'fontWeight': 'bold',
                        },
                    ),
                    html.Br(),
                    html.Span(
                        f"Active Monitors: "
                        f"{dashboard_data.get('summary', {}).get('active_monitors', 0)}"
                    ),
                    html.Br(),
                    html.Span(
                        f"Active alerts: "
                        f"{len(dashboard_data.get('alerts', {}).get('active', []))}"
                    ),
                ])

                # Charts
                cpu_memory_fig = self._create_cpu_memory_graph(dashboard_data)
                disk_network_fig = self._create_disk_network_graph(dashboard_data)
                model_perf_fig = self._create_model_performance_graph(dashboard_data)
                data_quality_fig = self._create_data_quality_graph(dashboard_data)

                # alerts table
                alerts_table = self._create_alerts_table(dashboard_data)

                return (
                    status_div,
                    cpu_memory_fig,
                    disk_network_fig,
                    model_perf_fig,
                    data_quality_fig,
                    alerts_table,
                )

            except Exception as e:
                self.logger.error(f"Error updating dashboard: {e}")
                return ("Error loading dashboard", {}, {}, {}, {}, "Error loading alerts")

    def _create_cpu_memory_graph(
        self, data: dict[str, Any]
    ) -> go.Figure:
        system_metrics = (
            data.get('monitors', {})
            .get('system_health', {})
            .get('metrics', {})
        )
        if not system_metrics:
            return go.Figure()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=('CPU Usage', 'Memory Usage'),
        )
        fig.add_trace(
            go.Indicator(
                mode=GAUGE_NUMBER_MODE,
                value=system_metrics.get('cpu_percent', 0),
                title={'text': "CPU %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                },
                domain={'row': 0, 'column': 0},
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Indicator(
                mode=GAUGE_NUMBER_MODE,
                value=system_metrics.get('memory_percent', 0),
                title={'text': "Memory %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                },
                domain={'row': 0, 'column': 1},
            ),
            row=1,
            col=2,
        )
        fig.update_layout(height=300)
        return fig

    def _create_disk_network_graph(
        self, data: dict[str, Any]
    ) -> go.Figure:
        system_metrics = (
            data.get('monitors', {})
            .get('system_health', {})
            .get('metrics', {})
        )
        if not system_metrics:
            return go.Figure()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=('Disk Usage', 'Network I/O'),
        )
        fig.add_trace(
            go.Indicator(
                mode=GAUGE_NUMBER_MODE,
                value=system_metrics.get('disk_percent', 0),
                title={'text': "Disk %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                },
                domain={'row': 0, 'column': 0},
            ),
            row=1,
            col=1,
        )

        network_sent = (
            system_metrics.get('network_bytes_sent', 0) / (1024**2)
        )
        network_recv = (
            system_metrics.get('network_bytes_recv', 0) / (1024**2)
        )

        fig.add_trace(
            go.Bar(
                x=['Sent', 'Received'],
                y=[network_sent, network_recv],
                marker_color=['lightblue', 'lightgreen'],
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.update_layout(height=300)
        return fig

    def _create_model_performance_graph(
        self, data: dict[str, Any]
    ) -> go.Figure:
        model_metrics = (
            data.get('monitors', {})
            .get('model_performance', {})
            .get('metrics', {})
        )
        if not model_metrics:
            return go.Figure()

        fig = go.Figure()
        metrics_data = [
            model_metrics.get('total_models', 0),
            model_metrics.get('active_models', 0),
            model_metrics.get('models_with_drift', 0),
            model_metrics.get('average_accuracy', 0) * 100,
        ]
        fig.add_trace(
            go.Bar(
                x=[
                    'Total Models',
                    'Active Models',
                    'Models with Drift',
                    'Avg Accuracy %',
                ],
                y=metrics_data,
                marker_color=['blue', 'green', 'orange', 'red'],
            )
        )
        fig.update_layout(
            title="Model Performance Overview", height=400
        )
        return fig

    def _create_data_quality_graph(
        self, data: dict[str, Any]
    ) -> go.Figure:
        data_metrics = (
            data.get('monitors', {})
            .get('data_quality', {})
            .get('metrics', {})
        )
        if not data_metrics:
            return go.Figure()

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=('Data Completeness', 'Data Sources'),
        )
        completeness = (
            data_metrics.get('average_completeness', 1.0) * 100
        )
        fig.add_trace(
            go.Indicator(
                mode=GAUGE_NUMBER_MODE,
                value=completeness,
                title={'text': "Avg Completeness %"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkcyan"},
                },
                domain={'row': 0, 'column': 0},
            ),
            row=1,
            col=1,
        )

        sources_total = data_metrics.get('total_sources', 0)
        sources_issues = data_metrics.get(
            'sources_with_issues', 0
        )
        fig.add_trace(
            go.Pie(
                labels=['Healthy Sources', 'Sources with Issues'],
                values=[
                    sources_total - sources_issues,
                    sources_issues,
                ],
                marker_colors=['lightgreen', 'salmon'],
                domain={'row': 0, 'column': 1},
            ),
            row=1,
            col=2,
        )
        fig.update_layout(height=400)
        return fig

    def _create_alerts_table(
        self, data: dict[str, Any]
    ) -> Any:
        alerts = data.get('alerts', {}).get('active', [])
        if not alerts:
            return html.Div("No active alerts")

        table_header = html.Thead(
            html.Tr([
                html.Th("Severity"),
                html.Th("Monitor"),
                html.Th("Message"),
                html.Th("Time"),
            ])
        )

        table_rows = []
        for alert in alerts[:10]:
            severity = alert.get('severity', 'unknown')
            color_map = {
                'info': 'blue',
                'warning': 'orange',
                'error': 'red',
                'critical': 'darkred',
            }
            color = color_map.get(severity, 'black')
            row = html.Tr([
                html.Td(
                    html.Span(
                        severity.upper(),
                        style={
                            'color': color,
                            'fontWeight': 'bold',
                        },
                    )
                ),
                html.Td(alert.get('monitor', 'unknown')),
                html.Td(alert.get('message', 'No message')),
                html.Td(alert.get('timestamp', 'unknown')[:19]),
            ])
            table_rows.append(row)

        table_body = html.Tbody(table_rows)
        return html.Table(
            [table_header, table_body],
            style={
                'width': '100%',
                'borderCollapse': 'collapse',
            },
        )

    def run_server(self, debug: bool = False):
        try:
            msg = (
                f"Starting dashboard server on {self.host}:{self.port}"
            )
            self.logger.info(msg)
            self.app.run_server(
                host=self.host,
                port=self.port,
                debug=debug,
            )
        except Exception as e:
            self.logger.error(
                f"Error running dashboard server: {e}"
            )

class DummyMonitoringDashboardApp:
    def __init__(self, *args: Any, **kwargs: Any):
        msg = (
            "Plotly and Dash are required for MonitoringDashboardApp"
        )
        raise ImportError(msg)

class TextBasedDashboard:
    """Text dashboard for console output"""

    def __init__(self, monitoring_system: MonitoringSystem):
        self.monitoring_system = monitoring_system
        self.logger = ProjectLogger.get_logger("TextBasedDashboard")

    def generate_report(self) -> str:
        try:
            data = self.monitoring_system.get_dashboard_data()
            timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            report_lines = [
                "=" * 60,
                "TRADING SYSTEM MONITORING DASHBOARD",
                "=" * 60,
                f"Generated at: {timestamp_str}",
                "",
            ]
            system_status = data.get('system_status', 'unknown')
            summary = data.get('summary', {})
            active_monitors = summary.get('active_monitors', 0)
            total_monitors = summary.get('total_monitors', 0)
            active_alerts = len(
                data.get('alerts', {}).get('active', [])
            )
            report_lines.extend([
                f"System Status: {system_status.upper()}",
                f"Active Monitors: {active_monitors}/{total_monitors}",
                f"Active alerts: {active_alerts}",
                "",
            ])

            sys_m = (
                data.get('monitors', {})
                .get('system_health', {})
                .get('metrics', {})
            )
            if sys_m:
                report_lines.extend([
                    "SYSTEM METRICS:",
                    f"  CPU Usage: {sys_m.get('cpu_percent', 0):.1f}%",
                    f"  Memory Usage: "
                    f"{sys_m.get('memory_percent', 0):.1f}%",
                    f"  Disk Usage: {sys_m.get('disk_percent', 0):.1f}%",
                    "",
                ])

            return "\n".join(report_lines)
        except Exception as e:
            self.logger.error(f"Error generating text report: {e}")
            return f"Error: {e}"

    def print_report(self):
        report = self.generate_report()
        self.logger.info("\n" + report)

    def save_report(self, filepath: str):
        try:
            report = self.generate_report()
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            self.logger.info(f"Report saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving report: {e}")

class MonitoringDashboardGenerator:
    """Monitoring dashboard generator."""

    def __init__(
        self,
        monitoring_system: MonitoringSystem,
        config: dict[str, Any] | None = None,
    ):
        self.monitoring_system = monitoring_system
        self.config = config or {}
        self.logger = ProjectLogger.get_logger(
            "MonitoringDashboardGenerator"
        )

        if PLOTLY_AVAILABLE:
            web_config = self.config.get('web', {})
            self.web_dashboard: Optional[MonitoringDashboardApp] = MonitoringDashboardApp(
                monitoring_system, web_config
            )
        else:
            self.web_dashboard = None

        self.text_dashboard = TextBasedDashboard(monitoring_system)
        self.auto_save = self.config.get('auto_save', False)
        self.save_interval = self.config.get('save_interval', 3600)
        self.save_path = self.config.get(
            'save_path', 'monitoring_reports'
        )

        if self.auto_save:
            os.makedirs(self.save_path, exist_ok=True)

        self.save_thread: Optional[threading.Thread] = None
        self.is_running = False

    def start_auto_save(self):
        if not self.auto_save or self.is_running:
            return
        self.is_running = True
        self.save_thread = threading.Thread(
            target=self._auto_save_loop, daemon=True
        )
        self.save_thread.start()
        self.logger.info("Auto-save started")

    def stop_auto_save(self):
        self.is_running = False
        if self.save_thread:
            self.save_thread.join(timeout=5)
        self.logger.info("Auto-save stopped")

    def _auto_save_loop(self):
        while self.is_running:
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = os.path.join(self.save_path, f"monitoring_report_{timestamp}.txt")
                self.text_dashboard.save_report(filepath)
                time.sleep(self.save_interval)
            except Exception as e:
                self.logger.error(f"Error in auto-save: {e}")
                time.sleep(60)

    def run_web_dashboard(self, debug: bool = False):
        if self.web_dashboard:
            self.web_dashboard.run_server(debug=debug)
        else:
            self.logger.warning("Web dashboard not available. Using text-based summary.")
            self.text_dashboard.print_report()
