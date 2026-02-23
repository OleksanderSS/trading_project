# dashboard/main_app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime

# Logger
logger = logging.getLogger("dashboard")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)
logger.propagate = False

# Шляхи
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_dir, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Імпорти
try:
    from config.config import PATHS, TICKERS, TIME_FRAMES
    from core.trading_advisor import TradingAdvisor
except ImportError as e:
    st.error(f"Не вдалося andмпортувати модулand: {e}")
    st.stop()

# Конфandг
MAIN_TICKER_SYMBOL = list(TICKERS.keys())[0] if TICKERS else "SPY"
DEFAULT_INTERVAL = TIME_FRAMES[0] if TIME_FRAMES else '1d'
DATA_PATH = PATHS.get("data", "data")

st.set_page_config(layout="wide", page_title=f"[UP] {MAIN_TICKER_SYMBOL} Trading Advisor", page_icon="[DATA]")
st.title(f"[UP] {MAIN_TICKER_SYMBOL} Trading Advisor Dashboard")

selected_interval = st.selectbox(
    "Виберandть часовий andнтервал", TIME_FRAMES,
    index=TIME_FRAMES.index(DEFAULT_INTERVAL) if DEFAULT_INTERVAL in TIME_FRAMES else 0
)
st.markdown(f"Інтервал: **{selected_interval}**")

# Інandцandалandwithуємо TradingAdvisor
try:
    advisor = TradingAdvisor(ticker=MAIN_TICKER_SYMBOL, interval=selected_interval)
    layer_summary = advisor.get_layer_summary()
except Exception as e:
    st.error(f"Error andнandцandалandforцandї TradingAdvisor: {e}")
    st.stop()

# Покаwithуємо andнформацandю про систему
st.header(" Багатошарова архandтектура")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Всього шарandв", len(layer_summary))
with col2:
    active_layers = len([l for l in layer_summary.values() if l['weight'] != 1.0])
    st.metric("Активних шарandв", active_layers)
with col3:
    total_features = sum(l['features_count'] for l in layer_summary.values())
    st.metric("Всього фandчей", total_features)

# Топ шарandв
st.subheader("Топ-5 шарandв for кandлькandстю фandчей:")
top_layers = sorted(layer_summary.items(), key=lambda x: x[1]['features_count'], reverse=True)[:5]
for i, (name, info) in enumerate(top_layers, 1):
    st.write(f"{i}. **{name}**: {info['features_count']} фandчей (вага: {info['weight']})")

# Сandтус system
st.header("[DATA] Сandтус system")
st.success("[OK] TradingAdvisor andнandцandалandwithовано")
st.info(" Всand шари мають notйтральну вагу 1.0 - готово for тренування моwhereлей")

# Інструкцandї
st.header("[START] Наступнand кроки")
st.write("""
1. Запустandть `py run.py` for тренування моwhereлей
2. Пandсля тренування налаштуйте ваги шарandв
3. Всandновandть streamlit and plotly for повного dashboard
""")