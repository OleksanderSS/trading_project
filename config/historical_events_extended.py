# config/historical_events_extended.py - Роwithширенand andсторичнand подandї

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class HistoricalEventsExtended:
    """
    Роwithширенand andсторичнand подandї for аналandwithу ринку
    """
    
    def __init__(self):
        self.events = self._define_extended_events()
        
    def _define_extended_events(self) -> Dict[str, Dict]:
        """Виwithначити роwithширенand andсторичнand подandї"""
        
        events = {
            # Криwithовand подandї
            'crisis_similarity_2008': {
                'name': 'Фandнансова криfor 2008',
                'start_date': '2008-09-15',
                'end_date': '2009-03-09',
                'type': 'crisis',
                'severity': 10,
                'key_indicators': ['bank_failures', 'credit_freeze', 'housing_collapse'],
                'market_impact': {'sp500_drop': -57, 'vix_spike': 80},
                'duration_days': 176,
                'recovery_time': '4+ years'
            },
            
            'covid_crash_2020': {
                'name': 'COVID-19 крах 2020',
                'start_date': '2020-02-20',
                'end_date': '2020-03-23',
                'type': 'crisis',
                'severity': 9,
                'key_indicators': ['pandemic', 'lockdown', 'supply_chain_disruption'],
                'market_impact': {'sp500_drop': -34, 'vix_spike': 82},
                'duration_days': 32,
                'recovery_time': '5 months'
            },
            
            'dot_com_bubble': {
                'name': 'Dot-com бульбашка',
                'start_date': '2000-03-10',
                'end_date': '2002-10-09',
                'type': 'bubble_burst',
                'severity': 8,
                'key_indicators': ['tech_overvaluation', 'nasdaq_crash', 'speculation'],
                'market_impact': {'nasdaq_drop': -78, 'sp500_drop': -49},
                'duration_days': 943,
                'recovery_time': '7+ years'
            },
            
            'black_monday': {
                'name': 'Чорний поnotдandлок 1987',
                'start_date': '1987-10-19',
                'end_date': '1987-10-20',
                'type': 'crash',
                'severity': 9,
                'key_indicators': ['program_trading', 'market_mechanism_failure'],
                'market_impact': {'sp500_drop': -20.5, 'vix_spike': 150},
                'duration_days': 1,
                'recovery_time': '2 years'
            },
            
            'great_recession': {
                'name': 'Велика рецесandя',
                'start_date': '2007-12-01',
                'end_date': '2009-06-01',
                'type': 'recession',
                'severity': 8,
                'key_indicators': ['housing_crisis', 'unemployment_spike', 'gdp_contraction'],
                'market_impact': {'sp500_drop': -57, 'unemployment_rate': 10},
                'duration_days': 548,
                'recovery_time': '4+ years'
            },
            
            # Інфляцandйнand подandї
            'inflation_spike_2022': {
                'name': 'Інфляцandйний спайк 2022',
                'start_date': '2021-12-01',
                'end_date': '2022-11-01',
                'type': 'inflation',
                'severity': 7,
                'key_indicators': ['cpi_spike', 'supply_chain', 'wage_pressure'],
                'market_impact': {'inflation_rate': 9.1, 'rate_hikes': 4},
                'duration_days': 335,
                'recovery_time': '2+ years'
            },
            
            'rate_hike_cycle_2022': {
                'name': 'Цикл пandдвищення сandвок 2022',
                'start_date': '2022-03-01',
                'end_date': '2023-07-01',
                'type': 'monetary_policy',
                'severity': 6,
                'key_indicators': ['fed_rate_hikes', 'tightening_cycle', 'yield_curve_inversion'],
                'market_impact': {'rate_hikes': 11, 'terminal_rate': 5.5},
                'duration_days': 488,
                'recovery_time': '1+ years'
            },
            
            # Банкandвськand криwithи
            'banking_crisis_2023': {
                'name': 'Банкandвська криfor 2023',
                'start_date': '2023-03-08',
                'end_date': '2023-05-01',
                'type': 'banking_crisis',
                'severity': 7,
                'key_indicators': ['regional_bank_failures', 'credit_contraction', 'depositor_runs'],
                'market_impact': {'bank_failures': 3, 'credit_contraction': 15},
                'duration_days': 54,
                'recovery_time': '6 months'
            },
            
            # Технологandчнand бульбашки
            'tech_bubble_2021': {
                'name': 'Технологandчна бульбашка 2021',
                'start_date': '2020-11-01',
                'end_date': '2022-12-31',
                'type': 'bubble_burst',
                'severity': 6,
                'key_indicators': ['tech_valuation', 'growth_stocks', 'speculation'],
                'market_impact': {'nasdaq_drop': -35, 'tech_etf_decline': -40},
                'duration_days': 425,
                'recovery_time': '2+ years'
            },
            
            'crypto_crash_2022': {
                'name': 'Крипто крах 2022',
                'start_date': '2022-04-01',
                'end_date': '2022-12-31',
                'type': 'crash',
                'severity': 5,
                'key_indicators': ['crypto_liquidations', 'defi_crisis', 'stablecoin_failure'],
                'market_impact': {'bitcoin_drop': -65, 'crypto_market_cap_loss': -70},
                'duration_days': 274,
                'recovery_time': '1+ years'
            },
            
            # Геополandтичнand криwithи
            'geopolitical_crisis': {
                'name': 'Геополandтична криfor',
                'start_date': '2022-02-24',
                'end_date': '2022-12-31',
                'type': 'geopolitical',
                'severity': 6,
                'key_indicators': ['war_impact', 'sanctions', 'energy_crisis'],
                'market_impact': {'commodity_spike': 50, 'energy_prices': 200},
                'duration_days': 310,
                'recovery_time': '2+ years'
            },
            
            # Еnotргетичнand криwithи
            'energy_crisis': {
                'name': 'Еnotргетична криfor',
                'start_date': '2021-09-01',
                'end_date': '2022-12-31',
                'type': 'energy',
                'severity': 7,
                'key_indicators': ['oil_prices', 'gas_prices', 'energy_shortage'],
                'market_impact': {'oil_price_spike': 120, 'energy_inflation': 40},
                'duration_days': 486,
                'recovery_time': '2+ years'
            },
            
            # Криwithи ланцюгandв посandчання
            'supply_chain_crisis': {
                'name': 'Криfor ланцюгandв посandчання',
                'start_date': '2020-03-01',
                'end_date': '2022-06-01',
                'type': 'supply_chain',
                'severity': 6,
                'key_indicators': ['shipping_costs', 'inventory_shortages', 'production_delays'],
                'market_impact': {'shipping_cost_spike': 300, 'inventory_shortages': 25},
                'duration_days': 823,
                'recovery_time': '2+ years'
            },
            
            # Житловand криwithи
            'housing_crisis': {
                'name': 'Житлова криfor',
                'start_date': '2006-07-01',
                'end_date': '2009-03-01',
                'type': 'housing',
                'severity': 8,
                'key_indicators': ['mortgage_crisis', 'foreclosure_wave', 'housing_price_decline'],
                'market_impact': {'housing_price_drop': -30, 'foreclosure_rate': 5},
                'duration_days': 974,
                'recovery_time': '5+ years'
            },
            
            # Ринковand механandчнand withбої
            'flash_crash_2010': {
                'name': 'Flash crash 2010',
                'start_date': '2010-05-06',
                'end_date': '2010-05-06',
                'type': 'flash_crash',
                'severity': 7,
                'key_indicators': ['high_frequency_trading', 'market_mechanism', 'liquidity_evaporation'],
                'market_impact': {'sp500_drop': -9, 'recovery_time': '20 minutes'},
                'duration_days': 0.01,
                'recovery_time': 'hours'
            },
            
            # Європейська боргова криfor
            'european_debt_crisis': {
                'name': 'Європейська боргова криfor',
                'start_date': '2010-01-01',
                'end_date': '2012-07-01',
                'type': 'debt_crisis',
                'severity': 7,
                'key_indicators': ['sovereign_debt', 'euro_crisis', 'austerity'],
                'market_impact': {'euro_decline': -20, 'bond_spreads': 400},
                'duration_days': 912,
                'recovery_time': '3+ years'
            },
            
            # Аwithandйська фandнансова криfor
            'asian_financial_crisis': {
                'name': 'Аwithandйська фandнансова криfor',
                'start_date': '1997-07-01',
                'end_date': '1998-12-31',
                'type': 'currency_crisis',
                'severity': 8,
                'key_indicators': ['currency_devaluation', 'contagion', 'imf_intervention'],
                'market_impact': {'currency_decline': -50, 'regional_market_drop': -60},
                'duration_days': 548,
                'recovery_time': '2+ years'
            }
        }
        
        return events
    
    def get_event_similarity(self, current_date: datetime, event_name: str) -> float:
        """
        Роwithрахувати схожandсть поточної дати with andсторичною подandєю
        
        Args:
            current_date: Поточна даand
            event_name: Наwithва подandї
            
        Returns:
            float: Схожandсть (0-1)
        """
        if event_name not in self.events:
            return 0.0
        
        event = self.events[event_name]
        event_start = pd.to_datetime(event['start_date'])
        event_end = pd.to_datetime(event['end_date'])
        
        # Calculating вandдсandнь в днях
        days_since_start = (current_date - event_start).days
        days_since_end = (current_date - event_end).days
        
        # Якщо ми в periodand подandї
        if event_start <= current_date <= event_end:
            return 1.0
        
        # Якщо подandя notщодавня
        if days_since_end <= 30:
            return 0.8
        elif days_since_end <= 90:
            return 0.6
        elif days_since_end <= 365:
            return 0.4
        elif days_since_end <= 365 * 2:
            return 0.2
        else:
            return 0.1
    
    def get_event_features(self, current_date: datetime) -> Dict[str, float]:
        """
        Отримати фandчand andсторичних подandй for поточної дати
        
        Args:
            current_date: Поточна даand
            
        Returns:
            Dict with фandчами подandй
        """
        features = {}
        
        for event_name, event_data in self.events.items():
            similarity = self.get_event_similarity(current_date, event_name)
            
            if similarity > 0.1:
                # Баwithовand фandчand
                features[f'{event_name}_similarity'] = similarity
                features[f'{event_name}_severity'] = event_data['severity'] * similarity
                
                # Фandчand типу подandї
                event_type = event_data['type']
                features[f'{event_type}_event_active'] = max(features.get(f'{event_type}_event_active', 0), similarity)
                
                # Фandчand впливу на ринок
                if 'market_impact' in event_data:
                    impact = event_data['market_impact']
                    for impact_key, impact_value in impact.items():
                        features[f'{event_name}_{impact_key}'] = impact_value * similarity
        
        # Агрегованand фandчand
        features['total_crisis_exposure'] = sum(features.get(f'{name}_similarity', 0) 
                                              for name, data in self.events.items() 
                                              if data['type'] in ['crisis', 'crash', 'bubble_burst'])
        
        features['total_inflation_exposure'] = sum(features.get(f'{name}_similarity', 0) 
                                              for name, data in self.events.items() 
                                              if data['type'] in ['inflation'])
        
        features['total_policy_exposure'] = sum(features.get(f'{name}_similarity', 0) 
                                           for name, data in self.events.items() 
                                           if data['type'] in ['monetary_policy'])
        
        return features
    
    def get_event_periods(self) -> List[Tuple[str, datetime, datetime]]:
        """
        Отримати periodи allх подandй
        
        Returns:
            List with (event_name, start_date, end_date)
        """
        periods = []
        
        for event_name, event_data in self.events.items():
            start_date = pd.to_datetime(event_data['start_date'])
            end_date = pd.to_datetime(event_data['end_date'])
            periods.append((event_name, start_date, end_date))
        
        return periods
    
    def get_events_by_type(self, event_type: str) -> Dict[str, Dict]:
        """
        Отримати подandї for типом
        
        Args:
            event_type: Тип подandї
            
        Returns:
            Dict with подandями вкаforного типу
        """
        return {name: data for name, data in self.events.items() 
                if data['type'] == event_type}
    
    def get_events_by_severity(self, min_severity: int) -> Dict[str, Dict]:
        """
        Отримати подandї for мandнandмальною серйоwithнandстю
        
        Args:
            min_severity: Мandнandмальна серйоwithнandсть
            
        Returns:
            Dict with подandями вкаforної серйоwithностand
        """
        return {name: data for name, data in self.events.items() 
                if data['severity'] >= min_severity}

# Глобальний екwithемпляр
historical_events = HistoricalEventsExtended()

# Функцandї for сумandсностand
def get_historical_event_features(current_date: datetime) -> Dict[str, float]:
    """Отримати фandчand andсторичних подandй (сумandснandсть)"""
    return historical_events.get_event_features(current_date)

def get_event_similarity(current_date: datetime, event_name: str) -> float:
    """Отримати схожandсть подandї (сумandснandсть)"""
    return historical_events.get_event_similarity(current_date, event_name)

if __name__ == "__main__":
    # Тестування
    test_date = datetime(2023, 3, 15)
    features = get_historical_event_features(test_date)
    
    print(f"Historical event features for {test_date}:")
    for feature, value in features.items():
        if value > 0:
            print(f"  {feature}: {value:.3f}")
