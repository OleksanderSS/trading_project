from src.features.builders.news_event.enricher import NewsGlobalEnricher
from src.meta_learning.awareness.context.manager import ContextAwarenessEngine
from src.meta_learning.awareness.context.scanner import EventScanner


def test_news_context_map_uses_macro_threshold_states():
    enricher = NewsGlobalEnricher()

    result = enricher.calculate_context_map({
        "macro_vixcls": 30.0,
        "macro_dgs10": 1.5,
        "macro_unrate": 5.0,
    })

    assert result["state_vixcls"] == 1
    assert result["state_dgs10"] == -1
    assert result["state_unrate"] == 0
    assert result["context_fingerprint"] == "1|-1|0"


def test_context_awareness_estimates_derived_market_levels():
    engine = ContextAwarenessEngine.__new__(ContextAwarenessEngine)

    assert engine._estimate_fear_greed_index(-0.5) == 25.0
    assert engine._estimate_fear_greed_index(0.75) == 75.0
    assert engine._estimate_vix_level("elevated") == 25.0


def test_event_scanner_does_not_emit_synthetic_market_events():
    scanner = EventScanner(config={"test": "https://example.invalid"})

    assert scanner.scan_all_sources() == []
