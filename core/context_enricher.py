# core/context_enricher.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from utils.logger_fixed import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

class ContextEnricher:
    """Enriches trading signals with layer context and historical performance"""
    
    def __init__(self):
        self.logger = logger
        self.layer_performance_history = {}  # Track layer performance over time
        self.signal_history = []  # Store all signals with context
        
    def enrich_signal_with_context(self, 
                                 signal: Dict[str, Any], 
                                 layer_info: Dict[str, Any],
                                 current_time: datetime = None) -> Dict[str, Any]:
        """Add layer context to signal"""
        if current_time is None:
            current_time = datetime.now()
            
        enriched_signal = signal.copy()
        
        # Add layer context
        enriched_signal['context'] = {
            'timestamp': current_time,
            'active_layers': layer_info.get('active_layers', []),
            'total_layers': layer_info.get('total_layers', 0),
            'neutral_layers': layer_info.get('neutral_layers', 0),
            'layer_weights': layer_info.get('layer_weights', {}),
            'layer_confidence': self._calculate_layer_confidence(layer_info),
            'historical_performance': self._get_layer_historical_performance(layer_info),
            'market_regime': self._detect_market_regime(),
            'volatility_context': self._get_volatility_context()
        }
        
        # Store for future learning
        self.signal_history.append(enriched_signal)
        
        return enriched_signal
    
    def update_layer_performance(self, 
                               layer_name: str, 
                               performance_metrics: Dict[str, float]):
        """Update layer performance history"""
        if layer_name not in self.layer_performance_history:
            self.layer_performance_history[layer_name] = []
            
        self.layer_performance_history[layer_name].append({
            'timestamp': datetime.now(),
            'metrics': performance_metrics
        })
        
        # Keep only last 100 records per layer
        if len(self.layer_performance_history[layer_name]) > 100:
            self.layer_performance_history[layer_name] = self.layer_performance_history[layer_name][-100:]
    
    def _calculate_layer_confidence(self, layer_info: Dict[str, Any]) -> float:
        """Calculate confidence based on layer weights and activity"""
        active_layers = layer_info.get('active_layers', [])
        total_layers = layer_info.get('total_layers', 1)
        
        if total_layers == 0:
            return 0.0
            
        # Confidence based on proportion of active layers
        base_confidence = len(active_layers) / total_layers
        
        # Adjust for neutral layers (they reduce confidence)
        neutral_ratio = layer_info.get('neutral_layers', 0) / total_layers
        adjusted_confidence = base_confidence * (1 - neutral_ratio * 0.3)
        
        return min(adjusted_confidence, 1.0)
    
    def _get_layer_historical_performance(self, layer_info: Dict[str, Any]) -> Dict[str, float]:
        """Get historical performance for active layers"""
        active_layers = layer_info.get('active_layers', [])
        performance_summary = {}
        
        for layer_name in active_layers:
            if layer_name in self.layer_performance_history:
                history = self.layer_performance_history[layer_name]
                if history:
                    # Calculate average performance metrics
                    recent_performance = history[-10:]  # Last 10 records
                    avg_accuracy = np.mean([h['metrics'].get('accuracy', 0) for h in recent_performance])
                    avg_profit = np.mean([h['metrics'].get('profit', 0) for h in recent_performance])
                    
                    performance_summary[layer_name] = {
                        'avg_accuracy': avg_accuracy,
                        'avg_profit': avg_profit,
                        'sample_size': len(recent_performance)
                    }
                else:
                    performance_summary[layer_name] = {
                        'avg_accuracy': 0.0,
                        'avg_profit': 0.0,
                        'sample_size': 0
                    }
            else:
                performance_summary[layer_name] = {
                    'avg_accuracy': 0.0,
                    'avg_profit': 0.0,
                    'sample_size': 0
                }
        
        return performance_summary
    
    def _detect_market_regime(self) -> str:
        """Detect current market regime (trending, ranging, volatile)"""
        # This is a simplified implementation
        # In practice, you'd analyze recent price action, volatility, etc.
        
        if not self.signal_history:
            return "unknown"
            
        # Analyze last 20 signals
        recent_signals = self.signal_history[-20:]
        
        # Count signal directions
        buy_signals = len([s for s in recent_signals if s.get('final_signal') == 'BUY'])
        sell_signals = len([s for s in recent_signals if s.get('final_signal') == 'SELL'])
        
        total_signals = len(recent_signals)
        if total_signals == 0:
            return "unknown"
            
        # Determine regime based on signal distribution
        buy_ratio = buy_signals / total_signals
        sell_ratio = sell_signals / total_signals
        
        if buy_ratio > 0.6:
            return "bullish"
        elif sell_ratio > 0.6:
            return "bearish"
        else:
            return "ranging"
    
    def _get_volatility_context(self) -> Dict[str, float]:
        """Get volatility context from recent signals"""
        if not self.signal_history:
            return {"volatility_level": 0.5, "trend_strength": 0.5}
            
        # Calculate volatility based on signal changes
        recent_signals = self.signal_history[-20:]
        
        # Convert signals to numeric for volatility calculation
        signal_values = []
        for signal in recent_signals:
            final_signal = signal.get('final_signal', 'HOLD')
            if final_signal == 'BUY':
                signal_values.append(1)
            elif final_signal == 'SELL':
                signal_values.append(-1)
            else:
                signal_values.append(0)
        
        if len(signal_values) < 2:
            return {"volatility_level": 0.5, "trend_strength": 0.5}
        
        # Calculate volatility (standard deviation of signal changes)
        signal_changes = np.diff(signal_values)
        volatility = np.std(signal_changes) if len(signal_changes) > 0 else 0
        
        # Calculate trend strength (absolute mean of signals)
        trend_strength = np.mean(np.abs(signal_values))
        
        return {
            "volatility_level": float(volatility),
            "trend_strength": float(trend_strength)
        }
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of all context data"""
        return {
            'total_signals_processed': len(self.signal_history),
            'layer_performance_history': {
                layer: len(history) for layer, history in self.layer_performance_history.items()
            },
            'current_market_regime': self._detect_market_regime(),
            'current_volatility': self._get_volatility_context(),
            'last_updated': datetime.now()
        }
    
    def export_context_data(self, filepath: str):
        """Export all context data for analysis"""
        import json
        
        context_data = {
            'signal_history': self.signal_history,
            'layer_performance_history': self.layer_performance_history,
            'summary': self.get_context_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(context_data, f, indent=2, default=str)
        
        self.logger.info(f"Context data exported to {filepath}")
