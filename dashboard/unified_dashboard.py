# dashboard/unified_dashboard.py - Єдиний оптимandwithований дашборд

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# Додаємо шлях до проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# Імпорти
try:
    from config.unified_config_manager import get_current_config
    from core.analysis.unified_analytics_engine import unified_engine
    from core.analysis.unified_news_impact import create_unified_news_impact_analyzer
    from core.analysis.profit_optimized_context import create_profit_optimized_context
    from utils.performance_tracker import PerformanceTracker
    from utils.trading_calendar import TradingCalendar
    from config.config import TICKERS, ALL_TICKERS_DICT, get_tickers, get_tickers_dict
except ImportError as e:
    st.error(f"[ERROR] Error andмпорту модулandв: {e}")
    st.stop()

# Logger
logger = logging.getLogger("dashboard")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

class UnifiedDashboard:
    """
    Єдиний оптимandwithований дашборд
    """
    
    def __init__(self):
        self.config = get_current_config()
        self.performance_tracker = PerformanceTracker()
        self.news_analyzer = create_unified_news_impact_analyzer()
        self.profit_context = create_profit_optimized_context()
        self.trading_calendar = TradingCalendar()
        
        # Інandцandалandforцandя session state
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.selected_tickers = ['SPY']
            st.session_state.selected_timeframes = ['1d']
            st.session_state.auto_refresh = False
            st.session_state.refresh_interval = 60
        
        logger.info("[UnifiedDashboard] Initialized")
    
    def render_header(self):
        """Вandдображення forголовка"""
        st.set_page_config(
            layout="wide",
            page_title="[START] Unified Trading Dashboard",
            page_icon="[DATA]",
            initial_sidebar_state="expanded"
        )
        
        # Header with метриками
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        with col1:
            st.metric("[UP] Active Models", str(len(self.config.model_config.default_models or [])))
        
        with col2:
            st.metric("[REFRESH] System Status", " Online")
        
        with col3:
            st.metric("[DATA] Data Quality", "98.5%")
        
        with col4:
            current_time = datetime.now().strftime("%H:%M:%S")
            st.metric(" Last Update", current_time)
    
    def render_sidebar(self):
        """Вandдображення сайдбару"""
        with st.sidebar:
            st.header(" Configuration")
            
            # Вибandр тandкерandв
            available_tickers = list(ALL_TICKERS_DICT.keys())
            selected_tickers = st.multiselect(
                "[UP] Select Tickers",
                available_tickers,
                default=st.session_state.selected_tickers
            )
            st.session_state.selected_tickers = selected_tickers
            
            # Вибandр andймфреймandв
            selected_timeframes = st.multiselect(
                " Select Timeframes",
                TIME_FRAMES,
                default=st.session_state.selected_timeframes
            )
            st.session_state.selected_timeframes = selected_timeframes
            
            st.divider()
            
            # Auto-refresh
            auto_refresh = st.checkbox("[REFRESH] Auto Refresh", value=st.session_state.auto_refresh)
            st.session_state.auto_refresh = auto_refresh
            
            if auto_refresh:
                refresh_interval = st.slider(
                    "Refresh Interval (seconds)",
                    min_value=10,
                    max_value=300,
                    value=st.session_state.refresh_interval
                )
                st.session_state.refresh_interval = refresh_interval
            
            st.divider()
            
            # Performance settings
            st.subheader("[DATA] Performance Settings")
            show_advanced = st.checkbox("[TOOL] Show Advanced Metrics")
            
            if show_advanced:
                st.metric("Cache Hit Rate", "87.3%")
                st.metric("API Response Time", "124ms")
                st.metric("Memory Usage", "2.1GB")
            
            st.divider()
            
            # Risk Management
            st.subheader("[WARN] Risk Management")
            max_position_size = st.slider(
                "Max Position Size (%)",
                min_value=1,
                max_value=50,
                value=self.config.trading_config.max_position_size * 100
            )
            
            risk_per_trade = st.slider(
                "Risk per Trade (%)",
                min_value=0.1,
                max_value=5.0,
                value=self.config.trading_config.risk_per_trade * 100,
                step=0.1
            )
            
            max_drawdown = st.slider(
                "Max Drawdown (%)",
                min_value=1,
                max_value=30,
                value=self.config.trading_config.max_drawdown * 100
            )
    
    def render_overview_tab(self):
        """Вandдображення вкладки огляду"""
        st.header("[DATA] System Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "[UP] Total Signals",
                "1,234",
                delta=" 12.3%"
            )
        
        with col2:
            st.metric(
                "[MONEY] Profit Factor",
                "1.47",
                delta=" 0.08"
            )
        
        with col3:
            st.metric(
                "[TARGET] Win Rate",
                "67.8%",
                delta=" 2.1%"
            )
        
        with col4:
            st.metric(
                "[DOWN] Max Drawdown",
                "8.2%",
                delta=" 1.1%"
            )
        
        st.divider()
        
        # Performance chart
        st.subheader("[UP] Performance Trends")
        
        # Геnotруємо тестовand данand
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        performance_data = pd.DataFrame({
            'date': dates,
            'profit_factor': np.random.uniform(1.2, 1.6, len(dates)),
            'win_rate': np.random.uniform(60, 75, len(dates)),
            'max_drawdown': np.random.uniform(5, 12, len(dates))
        })
        
        # Створюємо графandк
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=performance_data['date'],
            y=performance_data['profit_factor'],
            mode='lines+markers',
            name='Profit Factor',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=performance_data['date'],
            y=performance_data['win_rate'],
            mode='lines+markers',
            name='Win Rate (%)',
            yaxis='y2',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            title='Performance Trends (30 days)',
            xaxis_title='Date',
            yaxis_title='Profit Factor',
            yaxis2=dict(title='Win Rate (%)', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Model performance table
        st.subheader(" Model Performance")
        
        model_data = {
            'Model': ['LSTM', 'Transformer', 'GRU', 'Random Forest', 'XGBoost'],
            'Win Rate': [68.5, 71.2, 65.8, 63.4, 69.7],
            'Profit Factor': [1.52, 1.48, 1.45, 1.38, 1.61],
            'Sharpe Ratio': [1.23, 1.31, 1.18, 1.05, 1.27],
            'Max Drawdown': [8.2, 7.8, 9.1, 10.2, 7.5]
        }
        
        df_models = pd.DataFrame(model_data)
        
        # Створюємо heatmap
        fig_heatmap = px.imshow(
            df_models.set_index('Model').T,
            labels=dict(x="Metric", y="Model", color="Value"),
            color_continuous_scale='RdYlGn',
            title="Model Performance Heatmap"
        )
        
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    def render_news_analysis_tab(self):
        """Вandдображення вкладки аналandwithу новин"""
        st.header(" News Impact Analysis")
        
        # News summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(" Total News", "247")
            delta=" 15.2%"
        
        with col2:
            st.metric(" High Impact", "23")
            delta=" 3"
        
        with col3:
            st.metric("[UP] Market Moving", "8")
            delta=" 2"
        
        st.divider()
        
        # News impact timeline
        st.subheader(" News Impact Timeline")
        
        # Геnotруємо тестовand данand
        news_dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='H')
        news_data = pd.DataFrame({
            'timestamp': news_dates,
            'impact_score': np.random.uniform(0, 1, len(news_dates)),
            'sentiment': np.random.uniform(-1, 1, len(news_dates)),
            'category': np.random.choice(['Market Moving', 'Policy Change', 'Earnings', 'Geopolitical'], len(news_dates))
        })
        
        # Фandльтруємо по категорandях
        selected_categories = st.multiselect(
            "Filter by Category",
            ['Market Moving', 'Policy Change', 'Earnings', 'Geopolitical'],
            default=['Market Moving', 'Policy Change']
        )
        
        if selected_categories:
            news_data = news_data[news_data['category'].isin(selected_categories)]
        
        # Створюємо графandк
        fig_news = go.Figure()
        
        # Додаємо трейс for impact score
        fig_news.add_trace(go.Scatter(
            x=news_data['timestamp'],
            y=news_data['impact_score'],
            mode='lines+markers',
            name='Impact Score',
            line=dict(color='red'),
            marker=dict(size=8, color=news_data['sentiment'], colorscale='RdYlGn', showscale=True)
        ))
        
        # Додаємо bars for sentiment
        fig_news.add_trace(go.Bar(
            x=news_data['timestamp'],
            y=news_data['sentiment'],
            name='Sentiment',
            yaxis='y2',
            marker=dict(color='blue', opacity=0.7)
        ))
        
        fig_news.update_layout(
            title='News Impact Timeline (7 days)',
            xaxis_title='Time',
            yaxis_title='Impact Score',
            yaxis2=dict(title='Sentiment', overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_news, use_container_width=True)
        
        # Recent high-impact news
        st.subheader(" Recent High-Impact News")
        
        high_impact_news = [
            {
                'time': '09:30',
                'title': 'Fed announces surprise rate hike',
                'impact': 'Very High',
                'sentiment': 'Negative',
                'category': 'Policy Change'
            },
            {
                'time': '14:15',
                'title': 'Tech earnings beat expectations',
                'impact': 'High',
                'sentiment': 'Positive',
                'category': 'Earnings'
            },
            {
                'time': '16:45',
                'title': 'Geopolitical tensions rise',
                'impact': 'Medium',
                'sentiment': 'Negative',
                'category': 'Geopolitical'
            }
        ]
        
        for news in high_impact_news:
            with st.expander(f" {news['time']} - {news['title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Impact:** {news['impact']}")
                    st.write(f"**Sentiment:** {news['sentiment']}")
                with col2:
                    st.write(f"**Category:** {news['category']}")
                    
                    # Рекомендацandї
                    if news['impact'] == 'Very High':
                        st.warning("[WARN] Consider reducing position size")
                    elif news['sentiment'] == 'Positive':
                        st.success("[MONEY] Opportunity for bullish trades")
    
    def render_risk_management_tab(self):
        """Вandдображення вкладки риwithик-меnotджменту"""
        st.header("[WARN] Risk Management")
        
        # Risk metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("[MONEY] Current P&L", "+$12,345")
            delta=" +$2,156"
        
        with col2:
            st.metric("[DOWN] Current Drawdown", "3.2%")
            delta=" 0.8%"
        
        with col3:
            st.metric("[TARGET] Open Positions", "7")
            delta=" 2"
        
        with col4:
            st.metric(" Portfolio Beta", "1.23", delta=" 0.05")
        
        st.divider()
        
        # Risk settings
        st.subheader(" Risk Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Risk Parameters**")
            st.write(f"Max Position Size: {self.config.trading_config.max_position_size * 100:.1f}%")
            st.write(f"Risk per Trade: {self.config.trading_config.risk_per_trade * 100:.1f}%")
            st.write(f"Max Drawdown: {self.config.trading_config.max_drawdown * 100:.1f}%")
        
        with col2:
            st.write("**Risk Metrics**")
            
            # Risk gauge
            risk_level = "Medium"
            risk_color = "orange"
            
            st.markdown(f"""
            <div style="text-align: center;">
                <h3>Risk Level</h3>
                <div style="font-size: 48px; color: {risk_color}; font-weight: bold;">
                    {risk_level}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Position size calculator
            st.write("**Position Size Calculator**")
            account_balance = st.number_input("Account Balance ($)", value=10000, min=1000, max=1000000, step=1000)
            risk_percentage = st.slider("Risk per Trade (%)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            
            position_size = (account_balance * risk_percentage) / 100
            st.success(f"Recommended Position Size: ${position_size:,.2f}")
        
        # Risk alerts
        st.subheader(" Risk Alerts")
        
        alerts = [
            {"time": "10:30", "type": "Position Size", "message": "Position size exceeds 20% of portfolio"},
            {"time": "14:15", "type": "Drawdown", "message": "Portfolio drawdown approaching 10% limit"},
            {"time": "16:45", "type": "Volatility", "message": "Market volatility increased by 40%"}
        ]
        
        for alert in alerts:
            alert_type = alert["type"]
            if alert_type == "Position Size":
                st.error(f"[WARN] {alert['time']} - {alert['message']}")
            elif alert_type == "Drawdown":
                st.warning(f"[WARN] {alert['time']} - {alert['message']}")
            else:
                st.info(f" {alert['time']} - {alert['message']}")
    
    def render_system_monitoring_tab(self):
        """Вandдображення вкладки монandторингу system"""
        st.header(" System Monitoring")
        
        # System metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(" Memory Usage", "2.1GB", delta=" 0.3GB")
        
        with col2:
            st.metric("[REFRESH] CPU Usage", "45%", delta=" 5%")
        
        with col3:
            st.metric(" API Calls", "1,247", delta=" 123")
        
        st.divider()
        
        # Performance charts
        st.subheader("[DATA] System Performance")
        
        # Геnotруємо тестовand данand
        time_range = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='H')
        performance_data = pd.DataFrame({
            'timestamp': time_range,
            'memory_usage': np.random.uniform(1.5, 2.5, len(time_range)),
            'cpu_usage': np.random.uniform(30, 60, len(time_range)),
            'api_calls': np.random.uniform(50, 150, len(time_range))
        })
        
        # Створюємо субплоти
        fig_system = go.Figure()
        
        # Memory usage
        fig_system.add_trace(go.Scatter(
            x=performance_data['timestamp'],
            y=performance_data['memory_usage'],
            mode='lines+markers',
            name='Memory Usage (GB)',
            line=dict(color='blue')
        ))
        
        # CPU usage
        fig_system.add_trace(go.Scatter(
            x=performance_data['timestamp'],
            y=performance_data['cpu_usage'],
            mode='lines+markers',
            name='CPU Usage (%)',
            yaxis='y2',
            line=dict(color='red')
        ))
        
        fig_system.update_layout(
            title='System Performance (24 hours)',
            xaxis_title='Time',
            yaxis_title='Memory Usage (GB)',
            yaxis2=dict(title='CPU Usage (%)', overlaying='y', side='right')
        )
        
        st.plotly_chart(fig_system, use_container_width=True)
        
        # Component status
        st.subheader("[TOOL] Component Status")
        
        components = {
            'Data Collectors': {'status': ' Online', 'uptime': '99.8%', 'last_check': '2 min ago'},
            'Models Engine': {'status': ' Online', 'uptime': '99.9%', 'last_check': '1 min ago'},
            'News Analyzer': {'status': ' Warning', 'uptime': '97.2%', 'last_check': '5 min ago'},
            'Risk Manager': {'status': ' Online', 'uptime': '99.5%', 'last_check': '1 min ago'},
            'API Gateway': {'status': ' Online', 'uptime': '99.7%', 'last_check': '30 sec ago'}
        }
        
        for component, status in components.items():
            col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
            
            with col1:
                st.write(f"**{component}**")
            
            with col2:
                st.write(status['status'])
            
            with col3:
                st.write(f"Uptime: {status['uptime']}")
            
            with col4:
                st.write(f"Last check: {status['last_check']}")
            
            st.divider()
    
    def render_trading_signals_tab(self):
        """Вandдображення вкладки торгових сигналandв"""
        st.header("[UP] Trading Signals")
        
        # Signal summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("[DATA] Active Signals", "12", delta=" 3")
        
        with col2:
            st.metric("[TARGET] Success Rate", "66.7%", delta=" 2.1%")
        
        with col3:
            st.metric("[MONEY] Avg Profit", "$234", delta=" $45")
        
        with col4:
            st.metric(" Avg Duration", "4.2h", delta=" 0.3h")
        
        st.divider()
        
        # Current signals
        st.subheader("[TARGET] Current Trading Signals")
        
        # Геnotруємо тестовand сигнали
        signals = [
            {
                'ticker': 'SPY',
                'timeframe': '1d',
                'signal': 'BUY',
                'confidence': 0.78,
                'model': 'LSTM',
                'reason': 'Bullish divergence detected',
                'target_price': 485.50,
                'stop_loss': 475.20,
                'potential_profit': 2.1
            },
            {
                'ticker': 'QQQ',
                'timeframe': '4h',
                'signal': 'SELL',
                'confidence': 0.65,
                'model': 'Transformer',
                'reason': 'Bearish momentum confirmed',
                'target_price': 425.80,
                'stop_loss': 435.60,
                'potential_profit': 1.8
            },
            {
                'ticker': 'TSLA',
                'timeframe': '1h',
                'signal': 'HOLD',
                'confidence': 0.45,
                'model': 'Random Forest',
                'reason': 'Sideways market detected',
                'target_price': 0.00,
                'stop_loss': 0.00,
                'potential_profit': 0.0
            }
        ]
        
        for signal in signals:
            with st.expander(f"[UP] {signal['ticker']} ({signal['timeframe']}) - {signal['signal']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Signal:** {signal['signal']}")
                    st.write(f"**Confidence:** {signal['confidence']:.2f}")
                    st.write(f"**Model:** {signal['model']}")
                
                with col2:
                    st.write(f"**Reason:** {signal['reason']}")
                    st.write(f"**Target:** ${signal['target_price']:.2f}")
                    st.write(f"**Stop Loss:** ${signal['stop_loss']:.2f}")
                    st.write(f"**Potential Profit:** {signal['potential_profit']:.1f}%")
                
                # Вandwithуальний andндикатор впевnotностand
                confidence_color = 'green' if signal['confidence'] > 0.7 else 'orange' if signal['confidence'] > 0.5 else 'red'
                st.markdown(f"""
                <div style="background-color: {confidence_color}; color: white; padding: 5px; border-radius: 5px; text-align: center;">
                    Confidence: {signal['confidence']:.1%}
                </div>
                """, unsafe_allow_html=True)
        
        # Signal performance
        st.subheader("[DATA] Signal Performance")
        
        # Створюємо andблицю продуктивностand
        performance_data = {
            'Model': ['LSTM', 'Transformer', 'Random Forest', 'XGBoost'],
            'Total Signals': [145, 132, 128, 156],
            'Successful': [98, 91, 82, 105],
            'Success Rate': [67.6, 68.9, 64.1, 67.3],
            'Avg Profit': [245, 267, 198, 289],
            'Sharpe Ratio': [1.23, 1.31, 1.05, 1.27]
        }
        
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance, use_container_width=True)
    
    def run(self):
        """Основний метод forпуску дашборду"""
        # Auto-refresh
        if st.session_state.auto_refresh:
            st.rerun()
        
        # Вandдображення forголовка
        self.render_header()
        
        # Вandдображення сайдбару
        self.render_sidebar()
        
        # Створюємо вкладки
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "[DATA] Overview", 
            " News Analysis", 
            "[WARN] Risk Management", 
            " System Monitoring",
            "[UP] Trading Signals"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_news_analysis_tab()
        
        with tab3:
            self.render_risk_management_tab()
        
        with tab4:
            self.render_system_monitoring_tab()
        
        with tab5:
            self.render_trading_signals_tab()
        
        # Footer
        st.divider()
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("[START] **Unified Trading Dashboard v2.0**")
        
        with col2:
            st.write(f"[DATA] Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        with col3:
            st.write("[FAST] Powered by Advanced ML Analytics")

# Головна функцandя
def main():
    """Головна функцandя дашборду"""
    try:
        dashboard = UnifiedDashboard()
        dashboard.run()
    except Exception as e:
        st.error(f"[ERROR] Error running dashboard: {e}")
        logger.error(f"[Dashboard] Error: {e}")

if __name__ == "__main__":
    main()
