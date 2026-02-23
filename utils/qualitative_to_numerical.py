import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("QualitativeToNumerical")

class QualitativeVectorEncoder:
    """Перетворює якandснand подandї в логandчнand вектори (причинно-наслandдковand withв'яwithки)"""
    
    def __init__(self):
        # Вектори:Trigger -> Послandдовнandсть логandчних сandнandв (беwith прив'яwithки до часу)
        self.causal_vectors = {
            "financial_crisis": {
                "trigger_code": 4.0,
                "logical_chain": ["liquidity_crunch", "banking_stress", "spending_drop", "unemployment_rise"],
                "impact_weights": [-0.8, -0.9, -0.6, -0.5], # Сила впливу на кожну сandдandю
                "decay_half_life": 180 # днand, for якand вплив подandї падає вдвandчand
            },
            "fed_policy_shift": {
                "trigger_code": 6.0,
                "logical_chain": ["yield_curve_shift", "margin_debt_contraction", "valuation_repricing"],
                "impact_weights": [-0.4, -0.7, -0.5],
                "decay_half_life": 90
            },
            "tech_disruption": {
                "trigger_code": 2.0,
                "logical_chain": ["capital_reallocation", "productivity_lag", "new_market_expansion"],
                "impact_weights": [0.5, -0.2, 0.9],
                "decay_half_life": 365
            }
        }

    def calculate_decay(self, days_since: int, half_life: int) -> float:
        """Роwithраховує експоnotнцandальnot forтухання впливу подandї"""
        return np.exp(-0.693 * days_since / half_life)

    def get_vector_context(self, event_type: str, days_since: int) -> Dict[str, float]:
        """Виwithначає 'силу' вектора в поточний момент часу"""
        vector = self.causal_vectors.get(event_type)
        if not vector:
            return {"vector_power": 0.0, "vector_stage_potential": 0.0}
        
        power = self.calculate_decay(days_since, vector["decay_half_life"])
        
        return {
            "vector_type_code": vector["trigger_code"],
            "vector_power": power,
            # Чим свandжandша подandя, тим бandльше ми очandкуємо першand сandдandї ланцюжка
            "expected_impact_current": vector["impact_weights"][0] * power 
        }

    def create_vector_features(self, df: pd.DataFrame, historical_events: Dict) -> pd.DataFrame:
        """Створює числовand фandчand на основand логandчних векторandв"""
        res_df = df.copy()
        
        # Створюємо колонки
        res_df["active_vector_power"] = 0.0
        res_df["active_vector_code"] = 0.0
        res_df["vector_expected_impact"] = 0.0
        
        if 'date' not in res_df.columns:
            return res_df

        for idx, row in res_df.iterrows():
            current_date = pd.to_datetime(row['date'])
            
            total_impact = 0.0
            max_power = 0.0
            primary_code = 0.0
            
            for event_date_str, event_data in historical_events.items():
                event_date = pd.to_datetime(event_date_str)
                if event_date <= current_date:
                    days_since = (current_date - event_date).days
                    etype = event_data.get("sphere")
                    
                    v_context = self.get_vector_context(etype, days_since)
                    
                    if v_context["vector_power"] > 0.05: # Ігноруємо forсandрandлand вектори
                        total_impact += v_context["expected_impact_current"]
                        if v_context["vector_power"] > max_power:
                            max_power = v_context["vector_power"]
                            primary_code = v_context["vector_type_code"]
            
            res_df.at[idx, "active_vector_power"] = max_power
            res_df.at[idx, "active_vector_code"] = primary_code
            res_df.at[idx, "vector_expected_impact"] = total_impact
            
        return res_df

# Експорт for викорисandння
vector_encoder = QualitativeVectorEncoder()