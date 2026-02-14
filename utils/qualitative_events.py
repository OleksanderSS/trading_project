# utils/qualitative_events.py 

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List

class QualitativeEventsAnalyzer:
    def __init__(self, data_path: str = "data/signals/historical_qualitative_events.json"):
        self.data_path = data_path
        self.events = self._load_events()
    
    def _load_events(self) -> Dict:
        try:
            with open(self.data_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"financial_crises": [], "geopolitical_events": []}

    def find_analog_by_magnitude(self, indicator_name: str, current_value: float) -> List[Dict]:
        """
        Шукає andсторичнand подandї not for наwithвою, а for силою шоку.
        Наприклад: Chicago PMI 36.3 -> withнайти all PMI < 40 в andсторandї.
        """
        analogs = []
        for category, events in self.events.items():
            for event in events:
                # Перевandряємо, чи є в andсторandї forпис про силу аналогandчного покаwithника
                hist_val = event.get("magnitude", {}).get(indicator_name)
                if hist_val and abs(hist_val - current_value) / hist_val < 0.15: # 15% похибка
                    analogs.append(event)
        return analogs

    def get_event_features(self, date: datetime) -> Dict[str, float]:
        """Геnotрує фandчand forтухання for векторandв (Exponential Decay)"""
        features = {"systemic_fragility": 0.0, "historical_context_weight": 0.0}
        
        for category, events in self.events.items():
            for event in events:
                event_date = datetime.strptime(event["date"], "%Y-%m-%d")
                days_diff = (date - event_date).days
                
                if 0 <= days_diff < 730: # Дивимось на подandї осandннandх 2 рокandв
                    # Експоnotнцandальnot forтухання впливу (Half-life = 180 днandв)
                    influence = 2 ** (-days_diff / 180)
                    features["historical_context_weight"] = max(features["historical_context_weight"], influence)
                    
                    if event.get("severity") == "high":
                        features["systemic_fragility"] = max(features["systemic_fragility"], influence)
        
        return features