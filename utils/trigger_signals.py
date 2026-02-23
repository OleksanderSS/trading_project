# utils/trigger_signals.py

from utils.logger import ProjectLogger
from config.trigger_signals_config import SIGNAL_RULES


logger = ProjectLogger.get_logger("TradingProjectLogger")

def generate_trigger_signals(trigger_data: dict) -> dict:
    """
    Геnotрує трейдинговand сигнали на основand тригерних оwithнак.
    Правила беруться with SIGNAL_RULES у конфandгу.
    """
    if not trigger_data:
        logger.warning("[trigger_signals] [WARN] trigger_data порожнandй, поверandю пустий словник")
        return {}

    signals = {}

    for feature, rule in SIGNAL_RULES.items():
        if feature in trigger_data and trigger_data[feature].sum() > 0:
            signals[feature] = rule
            logger.info(f"[trigger_signals] [OK] Згеnotровано сигнал {rule} for {feature}")

    logger.info(f"[trigger_signals] [DATA] Всього сигналandв: {len(signals)}")
    return signals