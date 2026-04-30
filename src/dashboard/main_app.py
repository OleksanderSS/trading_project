# dashboard/main_app.py - Unified optimized dashboard

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
import psutil

# Add project path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Imports
try:
    from src.config.unified_config_manager import get_current_config
    from src.data.management.data_manager import DataManager
    from src.core.error_handling.error_handler import ErrorHandler
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
    from src.analytics.calculators.fama_french_factors import FamaFrenchFactors
    from src.analytics.analyzers.hedge_fund_analyzer import HedgeFundAnalyzer
except ImportError as e:
    st.error(f"[ERROR] Module import error: {e}")
    st.stop()

# Logger
logger = logging.getLogger("dashboard")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)

@st.cache_data(ttl=60)
def get_data_from_db(_db_manager, query):
    """ Cached function for getting data from DB """
    return _db_manager.load_data(query)

@st.cache_resource
def get_all_configured_tickers() -> List[str]:
    """Gets list of all unique tickers from configuration."""
    config_manager = get_current_config()
    assets = config_manager.get("assets", {})
    tickers = list(assets.keys()) if isinstance(assets, dict) else []
    if 'SPY' not in tickers:
        tickers.append('SPY')
    return sorted(tickers)

class UnifiedDashboard:
    """
    Unified optimized dashboard
    """
    
    def __init__(self):
        self.config_manager = get_current_config()
        self.error_handler = ErrorHandler()
        self.db_manager = DataManager(self.config_manager, self.error_handler)
        self.available_tickers = get_all_configured_tickers()
        
        # Initialize session state
        if 'dashboard_initialized' not in st.session_state:
            st.session_state.dashboard_initialized = True
            st.session_state.selected_tickers = ['SPY']
            st.session_state.selected_timeframes = ['1d']
            st.session_state.auto_refresh = False
            st.session_state.refresh_interval = 60
        
        logger.info("[UnifiedDashboard] Initialized")
    
    def render_header(self):
        """Render header with dynamic metrics."""
        st.set_page_config(layout="wide", page_title="DEAN Unified Trading Dashboard", page_icon="📈")
        
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        
        # --- Real metrics ---
        model_perf = get_data_from_db(self.db_manager, "SELECT COUNT(DISTINCT model_name) as count FROM model_performance")
        active_models = model_perf.iloc[0]['count'] if not model_perf.empty else 0
        
        with col1:
            st.metric("[UP] Active Models", str(active_models))
        with col2:
            st.metric("[REFRESH] System Status", "Online")
        with col3:
            st.metric("[DATA] Storage", "DuckDB Active")
        with col4:
            st.metric("Last Update", datetime.now().strftime("%H:%M:%S"))
    
    def render_sidebar(self):
        """Render sidebar"""
        with st.sidebar:
            st.header("Configuration")
            st.session_state.selected_tickers = st.multiselect("[UP] Tickers", self.available_tickers, default=st.session_state.selected_tickers)
            st.session_state.auto_refresh = st.checkbox("[REFRESH] Auto Refresh", value=st.session_state.auto_refresh)
            if st.session_state.auto_refresh:
                st.session_state.refresh_interval = st.slider("Interval (s)", 10, 300, st.session_state.refresh_interval)

    def render_overview_tab(self):
        """Render overview tab"""
        st.header("[DATA] System Overview")
        
        col1, col2, col3, _ = st.columns(4)
        perf_data = get_data_from_db(self.db_manager, "SELECT AVG(profit_factor) as avg_pf, AVG(win_rate) as avg_wr, MAX(max_drawdown) as max_dd FROM model_performance")
        
        avg_pf = perf_data.iloc[0]['avg_pf'] if not perf_data.empty else 0
        avg_wr = perf_data.iloc[0]['avg_wr'] if not perf_data.empty else 0
        max_dd = perf_data.iloc[0]['max_dd'] if not perf_data.empty else 0

        with col1: st.metric("[MONEY] Avg Profit Factor", f"{avg_pf:.2f}")
        with col2: st.metric("[TARGET] Avg Win Rate", f"{avg_wr * 100:.1f}%")
        with col3: st.metric("[DOWN] Max Drawdown", f"{max_dd * 100:.1f}%")
        
        st.divider()
        st.subheader("Alpha & Factor Exposure (Hedge Fund Analysis)")
        # Assume presence of factor_exposures table from HedgeFundAnalyzer
        exposures = get_data_from_db(self.db_manager, "SELECT * FROM factor_exposures ORDER BY timestamp DESC LIMIT 10")
        if not exposures.empty:
            st.dataframe(exposures, use_container_width=True)
            fig = px.bar(exposures, x='factor', y='exposure', color='model_name', barmode='group', title="Factor Exposures")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No factor exposure data available yet.")

    def render_trading_signals_tab(self):
        """Render trading signals tab"""
        st.header("[UP] Trading Signals")
        
        # Adaptive Thresholds Visualization
        st.subheader("Adaptive Thresholds & Confidence")
        thresholds_data = get_data_from_db(self.db_manager, "SELECT * FROM adaptive_thresholds ORDER BY timestamp DESC LIMIT 20")
        if not thresholds_data.empty:
            fig = go.Figure()
            fig.add_trace({
                'x': thresholds_data['timestamp'],
                'y': thresholds_data['sentiment_pos'],
                'name': 'Pos Threshold',
                'line': {'color': 'green'},
                'type': 'scatter'
            })
            fig.add_trace({
                'x': thresholds_data['timestamp'],
                'y': thresholds_data['sentiment_neg'],
                'name': 'Neg Threshold',
                'line': {'color': 'red'},
                'type': 'scatter'
            })
            fig.add_trace({
                'x': thresholds_data['timestamp'],
                'y': thresholds_data['min_prediction_prob'],
                'name': 'Min Conf',
                'line': {'dash': 'dash'},
                'type': 'scatter'
            })
            st.plotly_chart(fig, use_container_width=True)

        signals = get_data_from_db(self.db_manager, "SELECT * FROM trading_signals ORDER BY timestamp DESC LIMIT 20")
        if not signals.empty:
            st.dataframe(signals, use_container_width=True)
        else:
            st.info("No active signals.")

    def render_news_analysis_tab(self):
        """Render news analysis tab"""
        st.header("News Sentiment Analysis")
        news_sentiment = get_data_from_db(self.db_manager, "SELECT timestamp, AVG(sentiment_score) as avg_sentiment FROM news_data GROUP BY timestamp ORDER BY timestamp ASC")
        if not news_sentiment.empty:
            fig = px.line(news_sentiment, x='timestamp', y='avg_sentiment', title="Market Sentiment Trend")
            st.plotly_chart(fig, use_container_width=True)
        
        latest_news = get_data_from_db(self.db_manager, "SELECT timestamp, ticker, title, sentiment_score FROM news_data ORDER BY timestamp DESC LIMIT 50")
        st.dataframe(latest_news, use_container_width=True)

    def render_risk_management_tab(self):
        st.header("[WARN] Risk & Health")
        df = get_data_from_db(self.db_manager, "SELECT model_name, sharpe_ratio, max_drawdown, win_rate FROM model_performance")
        if not df.empty:
            st.dataframe(df.style.background_gradient(cmap='RdYlGn', subset=['sharpe_ratio', 'win_rate']), use_container_width=True)

    def render_system_monitoring_tab(self):
        st.header("System Monitoring")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CPU Usage", f"{psutil.cpu_percent()}%")
            st.metric("Memory Usage", f"{psutil.virtual_memory().percent}%")
        with col2:
            db_size = os.path.getsize(self.db_manager.db_path) / (1024*1024) if os.path.exists(self.db_manager.db_path) else 0
            st.metric("DB Size", f"{db_size:.2f} MB")

    def run(self):
        if st.session_state.auto_refresh:
            st.empty() # Placeholder for refresh logic
        self.render_header()
        self.render_sidebar()
        
        tabs = st.tabs(["Overview", "Signals", "News", "Risk", "System"])
        with tabs[0]: self.render_overview_tab()
        with tabs[1]: self.render_trading_signals_tab()
        with tabs[2]: self.render_news_analysis_tab()
        with tabs[3]: self.render_risk_management_tab()
        with tabs[4]: self.render_system_monitoring_tab()

if __name__ == "__main__":
    UnifiedDashboard().run()