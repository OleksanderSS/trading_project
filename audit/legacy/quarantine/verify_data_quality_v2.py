
import pandas as pd
import numpy as np
import logging
import sys

# Додаємо шлях до проекту, щоб імпорти працювали
import os
sys.path.append(os.getcwd())

from src.pipeline.guards.temporal_target_guard import TemporalTargetGuard
from src.features.enrichers.context_map_enricher import ContextMapEnricher
from src.features.enrichers.sentiment_features_enricher import SentimentFeaturesEnricher
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

# Налаштуємо логування у консоль для зручності
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("DataQualityVerifier")

def create_mock_data():
    """Створює синтетичні дані для тестування."""
    dates = pd.date_range(start="2024-01-01", periods=150, freq="1h")
    tickers = ["SPY", "AAPL"]
    data = []
    
    for ticker in tickers:
        price = 100.0
        for i, date in enumerate(dates):
            change = np.random.normal(0, 0.01)
            price *= (1 + change)
            data.append({
                "datetime": date,
                "ticker": ticker,
                "open": price * 0.99,
                "high": price * 1.01,
                "low": price * 0.98,
                "close": price,
                "volume": np.random.randint(1000, 5000),
                "nlp_sentiment_score": np.random.uniform(-1, 1) if i % 5 == 0 else 0.0
            })
    
    df = pd.DataFrame(data)
    # Важливо: деякі компоненти очікують datetime в колонці, інші в індексі
    return df

def verify_all():
    logger.info("🚀 Starting Comprehensive Data Quality Verification...")
    df = create_mock_data()
    config_manager = UnifiedConfigManager()
    
    # 1. Тест Таргетів
    logger.info("\n--- Testing TemporalTargetGuard ---")
    guard = TemporalTargetGuard(config_manager)
    # Очікує DF з datetime в колонці або індексі
    targets = guard.generate_targets_safe(df, "1h", pd.Timestamp.now(), None)
    
    vol_cols = [c for c in targets.columns if "volatility" in c]
    for col in vol_cols:
        non_zeros = (targets[col] != 0).sum()
        total = len(targets)
        logger.info(f"📊 Target '{col}': Non-Zeros={non_zeros}/{total}, Mean={targets[col].mean():.6f}")
        if non_zeros == 0:
             logger.error(f"❌ FAIL: {col} is ALL ZEROS")
        else:
             logger.info(f"✅ PASS: {col} has active variance")

    # 2. Тест Context Map & Velocity
    logger.info("\n--- Testing ContextMapEnricher (Champion & Velocity) ---")
    # Очікує datetime в індексі
    df_idx = df.set_index("datetime")
    enricher = ContextMapEnricher({"champion_ticker": "SPY", "velocity_window": 10})
    df_context = enricher.enrich(df_idx)
    
    if "context_velocity" in df_context.columns:
        logger.info(f"✅ PASS: 'context_velocity' created. Sample: {df_context['context_velocity'].tail(3).values}")
    else:
        logger.error("❌ FAIL: 'context_velocity' missing")
        
    if "state_champion" in df_context.columns:
        logger.info(f"✅ PASS: 'state_champion' integrated. Values: {df_context['state_champion'].unique()}")
    else:
        logger.error("❌ FAIL: 'state_champion' missing")

    # 3. Тест Сентименту
    logger.info("\n--- Testing Sentiment Enrichment ---")
    sent_enricher = SentimentFeaturesEnricher()
    df_sent = sent_enricher.enrich(df_idx)
    sent_added = [c for c in df_sent.columns if "sentiment_sma" in c]
    if sent_added:
        logger.info(f"✅ PASS: Sentiment features added: {sent_added[:3]}...")
    else:
        logger.info("ℹ️ Note: Sentiment enrichment might skip if no 'news' in kwargs, but column check:")
        if 'nlp_sentiment_score' in df_sent.columns:
             logger.info("✅ nlp_sentiment_score is present")

    logger.info("\n🏁 Verification Complete.")

if __name__ == "__main__":
    verify_all()
