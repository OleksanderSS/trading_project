# dashboard/simple_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Додаємо шлях до проекту
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from config.config import TICKERS, TIME_FRAMES, PATHS
    from core.trading_advisor import TradingAdvisor
except ImportError as e:
    st.error(f"Error andмпорту: {e}")
    st.stop()

# Конфandгурацandя
st.set_page_config(page_title="Trading Dashboard", layout="wide")
st.title("[UP] Trading System Dashboard")

# Sidebar
st.sidebar.header("Налаштування")
ticker = st.sidebar.selectbox("Тandкер", list(TICKERS.keys()) if TICKERS else ["SPY"])
interval = st.sidebar.selectbox("Інтервал", TIME_FRAMES if TIME_FRAMES else ["1d"])

# Основна паnotль
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Поточний тandкер", ticker)
    st.metric("Інтервал", interval)

with col2:
    # Інandцandалandwithуємо TradingAdvisor
    try:
        advisor = TradingAdvisor(ticker=ticker, interval=interval)
        layer_summary = advisor.get_layer_summary()
        
        total_layers = len(layer_summary)
        active_layers = len([l for l in layer_summary.values() if l['weight'] != 1.0])
        
        st.metric("Всього шарandв", total_layers)
        st.metric("Активних шарandв", active_layers)
        
        # Отримуємо сигнали
        try:
            signals = advisor.get_current_signals()
            if signals:
                signal_strength = signals.get('signal_strength', 0)
                signal_direction = signals.get('direction', 'HOLD')
                st.metric("Поточний сигнал", f"{signal_direction} ({signal_strength:.2f})")
            else:
                st.metric("Поточний сигнал", "Немає data")
        except Exception:
            st.metric("Поточний сигнал", "Error")
        
    except Exception as e:
        st.error(f"Error TradingAdvisor: {e}")

with col3:
    # Перевandряємо наявнandсть data
    data_files = [
        "data/trading_data.db",
        "data/news.db", 
        "data/stage3_features.parquet"
    ]
    
    data_status = sum(1 for f in data_files if os.path.exists(f))
    st.metric("Файлandв data", f"{data_status}/{len(data_files)}")
    
    if data_status == len(data_files):
        st.metric("Сandтус system", "[OK] Готово")
    else:
        st.metric("Сandтус system", "[WARN] Потрandбнand данand")

# Інформацandя про шари
st.header(" Багатошарова архandтектура")

if 'advisor' in locals():
    # Покаwithуємо топ-5 шарandв
    top_layers = sorted(layer_summary.items(), 
                       key=lambda x: x[1]['features_count'], 
                       reverse=True)[:5]
    
    st.subheader("Топ-5 шарandв for кandлькandстю фandчей:")
    for i, (name, info) in enumerate(top_layers, 1):
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.write(f"{i}. **{name}**")
        with col2:
            st.write(f"{info['features_count']} фandчей")
        with col3:
            st.write(f"Вага: {info['weight']}")

# Сandтус data and реwithульandти
st.header("[DATA] Сandтус data and реwithульandти")

data_path = PATHS.get("data", "data") if 'PATHS' in locals() else "data"

col1, col2 = st.columns(2)

with col1:
    st.subheader(" Файли data")
    if os.path.exists(data_path):
        files = os.listdir(data_path)
        st.write(f"Всього fileandв: {len(files)}")
        
        # Перевandряємо ключовand fileи
        key_files = {
            "trading_data.db": " Цandновand данand",
            "news.db": " Новини", 
            "stage3_features.parquet": "[TOOL] Фandчand"
        }
        
        for filename, description in key_files.items():
            if filename in files:
                file_path = os.path.join(data_path, filename)
                size = os.path.getsize(file_path) / (1024*1024)  # MB
                st.write(f"[OK] {description}: {size:.1f} MB")
            else:
                st.write(f"[ERROR] {description}: вandдсутнandй")
    else:
        st.warning("Папка data/ not withнайwhereна")

with col2:
    st.subheader(" Реwithульandти моwhereлей")
    
    # Перевandряємо чи є withбереженand реwithульandти моwhereлей
    try:
        if 'advisor' in locals():
            # Спробуємо отримати осandннand реwithульandти
            model_results = advisor.get_model_performance_summary()
            if model_results:
                for model, metrics in model_results.items():
                    accuracy = metrics.get('accuracy', 0)
                    st.write(f"[UP] {model.upper()}: {accuracy:.2%}")
            else:
                st.write("[WARN] Моwhereлand ще not натренованand")
                st.write("Запустandть: `py run.py`")
        else:
            st.write("[ERROR] TradingAdvisor unavailable")
    except Exception as e:
        st.write("[WARN] Реwithульandти моwhereлей notдоступнand")
        st.write("Запустandть тренування: `py run.py`")

# Рекомендацandї and наступнand кроки
st.header("[START] Рекомендацandї")

# Динамandчнand рекомендацandї на основand сandну system
if 'data_status' in locals() and data_status < len(data_files):
    st.warning("[WARN] **Потрandбно withandбрати данand**")
    st.code("py collect_data.py")
    st.write("Пandсля withбору data поверandйтесь до dashboard")
elif 'advisor' in locals():
    try:
        model_results = advisor.get_model_performance_summary()
        if not model_results:
            st.info("[DATA] **Готово до тренування моwhereлей**")
            st.code("py run.py")
            st.write("This forйме 10-30 хвилин forлежно вandд обсягу data")
        else:
            st.success("[OK] **Система готова до роботи!**")
            st.write("Можете:")
            st.write("- Переглядати сигнали в реальному часand")
            st.write("- Налаштовувати ваги шарandв")
            st.write("- Запускати backtesting")
            
            # Покаwithуємо code for settings
            st.code("""
# Налаштування вагandв шарandв
from core.trading_advisor import TradingAdvisor
advisor = TradingAdvisor()
advisor.update_layer_weight("news", 0.8)  # Зменшити вплив новин
advisor.update_layer_weight("technical", 1.2)  # Збandльшити технandчний аналandwith
""")
    except Exception:
        st.info("[REFRESH] **Переforпустandть dashboard пandсля тренування**")
else:
    st.error("[ERROR] **Error andнandцandалandforцandї system**")
    st.write("Перевandрте конфandгурацandю and API ключand")

# Команди
st.header(" Кориснand команди")
st.code("""
# Швидка перевandрка system
py quick_layer_check.py

# Тест TradingAdvisor
py test_trading_advisor.py

# Збandр data
py collect_data.py

# Тренування моwhereлей
py run.py
""")

# Сandтус dashboard
if 'advisor' in locals() and 'data_status' in locals():
    if data_status == len(data_files):
        st.success("[OK] Dashboard активний! Система працює.")
    else:
        st.warning("[WARN] Dashboard активний, але потрandбнand данand for повної функцandональностand.")
else:
    st.error("[ERROR] Dashboard має problemsи with andнandцandалandforцandєю.")

# Автооновлення
st.write("---")
if st.button("[REFRESH] Оновити данand"):
    st.rerun()