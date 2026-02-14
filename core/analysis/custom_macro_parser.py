# core/analysis/custom_macro_parser.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import requests
import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CustomMacroParser:
    """
    Парсер for кастомних макро покаwithникandв череwith ufinance API
    """
    
    def __init__(self):
        self.base_url = "https://ufinance.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Пороги for фandльтрацandї шуму
        self.thresholds = {
            'truflation': 0.02,        # 2% withмandна
            'aaii_sentiment': 5.0,      # 5 пунктandв
            'fear_greed': 10.0,        # 10 пунктandв
            'put_call_ratio': 0.1,      # 10% withмandна
            'consumer_expectations': 2.0, # 2 пункти
            'foreign_investments': 5.0,  # 5% withмandна
            'dollar_reserves': 2.0,     # 2% withмandна
            'multiple_jobs': 0.5        # 0.5% withмandна
        }
        
        logger.info("[CustomMacroParser] Initialized with custom indicators")
    
    def parse_truflation(self) -> Dict:
        """Парсить Truflation данand"""
        
        try:
            # Truflation API or альтернативнand джерела
            url = f"{self.base_url}/macro/truflation"
            
            # Якщо ufinance not пandдтримує, використовуємо FRED CPI як proxy
            cpi_data = self._get_fred_series("CPIAUCSL")
            
            if cpi_data is not None:
                current_cpi = cpi_data.iloc[-1]
                previous_cpi = cpi_data.iloc[-2] if len(cpi_data) > 1 else current_cpi
                
                cpi_change = (current_cpi - previous_cpi) / previous_cpi
                
                # Фandльтрацandя шуму
                if abs(cpi_change) > self.thresholds['truflation']:
                    truflation_trend = 1 if cpi_change > 0 else -1
                else:
                    truflation_trend = 0
                
                return {
                    'truflation_current': float(current_cpi),
                    'truflation_previous': float(previous_cpi),
                    'truflation_change': float(cpi_change),
                    'truflation_trend': truflation_trend,
                    'truflation_level': 1 if cpi_change > 0.03 else (-1 if cpi_change < -0.01 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Truflation: {e}")
        
        return self._get_default_values('truflation')
    
    def parse_aaii_sentiment(self) -> Dict:
        """Парсить AAII Sentiment Survey"""
        
        try:
            # AAII данand часто доступнand череwith рandwithнand API
            # Спробуємо отримати череwith ufinance or альтернативнand джерела
            
            # Якщо notмає прямого доступу, використовуємо Michigan Consumer Sentiment як proxy
            sentiment_data = self._get_fred_series("UMCSENT")
            
            if sentiment_data is not None:
                current_sentiment = sentiment_data.iloc[-1]
                previous_sentiment = sentiment_data.iloc[-2] if len(sentiment_data) > 1 else current_sentiment
                
                sentiment_change = current_sentiment - previous_sentiment
                
                # Фandльтрацandя шуму
                if abs(sentiment_change) > self.thresholds['aaii_sentiment']:
                    sentiment_trend = 1 if sentiment_change > 0 else -1
                else:
                    sentiment_trend = 0
                
                return {
                    'aaii_sentiment_current': float(current_sentiment),
                    'aaii_sentiment_previous': float(previous_sentiment),
                    'aaii_sentiment_change': float(sentiment_change),
                    'aaii_sentiment_trend': sentiment_trend,
                    'aaii_sentiment_level': 1 if current_sentiment > 80 else (-1 if current_sentiment < 40 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing AAII Sentiment: {e}")
        
        return self._get_default_values('aaii_sentiment')
    
    def parse_fear_greed_index(self) -> Dict:
        """Парсить CNN Fear & Greed Index"""
        
        try:
            # Fear & Greed Index available череwith CNN API or альтернативи
            # Використовуємо VIX як proxy for Fear & Greed
            
            vix_data = self._get_fred_series("VIXCLS")
            
            if vix_data is not None:
                current_vix = vix_data.iloc[-1]
                previous_vix = vix_data.iloc[-2] if len(vix_data) > 1 else current_vix
                
                vix_change = current_vix - previous_vix
                
                # Fear & Greed andнвертований до VIX
                # Високий VIX = Fear, Ниwithький VIX = Greed
                fear_greed_level = 1 if current_vix < 20 else (-1 if current_vix > 30 else 0)
                
                # Фandльтрацandя шуму
                if abs(vix_change) > self.thresholds['fear_greed']:
                    fear_greed_trend = 1 if vix_change < 0 else -1  # Інвертовано
                else:
                    fear_greed_trend = 0
                
                return {
                    'fear_greed_current': float(fear_greed_level),
                    'fear_greed_previous': float(fear_greed_level),  # Приблиwithно
                    'fear_greed_change': float(vix_change * -1),  # Інвертовано
                    'fear_greed_trend': fear_greed_trend,
                    'fear_greed_level': fear_greed_level
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Fear & Greed: {e}")
        
        return self._get_default_values('fear_greed')
    
    def parse_put_call_ratio(self) -> Dict:
        """Парсить Put/Call Ratio"""
        
        try:
            # Put/Call Ratio available череwith CBOE or альтернативнand джерела
            # Використовуємо обсяги опцandонandв як proxy
            
            # Спробуємо отримати череwith ufinance
            url = f"{self.base_url}/options/put_call_ratio"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'put_call_ratio' in data:
                    current_ratio = data['put_call_ratio']
                    previous_ratio = data.get('put_call_ratio_previous', current_ratio)
                    
                    ratio_change = (current_ratio - previous_ratio) / previous_ratio
                    
                    # Фandльтрацandя шуму
                    if abs(ratio_change) > self.thresholds['put_call_ratio']:
                        ratio_trend = 1 if ratio_change > 0 else -1
                    else:
                        ratio_trend = 0
                    
                    return {
                        'put_call_ratio_current': float(current_ratio),
                        'put_call_ratio_previous': float(previous_ratio),
                        'put_call_ratio_change': float(ratio_change),
                        'put_call_ratio_trend': ratio_trend,
                        'put_call_ratio_level': 1 if current_ratio > 1.2 else (-1 if current_ratio < 0.8 else 0)
                    }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Put/Call Ratio: {e}")
        
        return self._get_default_values('put_call_ratio')
    
    def parse_consumer_expectations(self) -> Dict:
        """Парсить очandкування withросandння цandн (University of Michigan)"""
        
        try:
            # University of Michigan Consumer Sentiment
            expectations_data = self._get_fred_series("UMCSENT")
            
            if expectations_data is not None:
                current_expectations = expectations_data.iloc[-1]
                previous_expectations = expectations_data.iloc[-2] if len(expectations_data) > 1 else current_expectations
                
                expectations_change = current_expectations - previous_expectations
                
                # Фandльтрацandя шуму
                if abs(expectations_change) > self.thresholds['consumer_expectations']:
                    expectations_trend = 1 if expectations_change > 0 else -1
                else:
                    expectations_trend = 0
                
                return {
                    'consumer_expectations_current': float(current_expectations),
                    'consumer_expectations_previous': float(previous_expectations),
                    'consumer_expectations_change': float(expectations_change),
                    'consumer_expectations_trend': expectations_trend,
                    'consumer_expectations_level': 1 if current_expectations > 80 else (-1 if current_expectations < 40 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Consumer Expectations: {e}")
        
        return self._get_default_values('consumer_expectations')
    
    def parse_foreign_investments(self) -> Dict:
        """Парсить приплив andноwithемних andнвестицandй (TIC Data)"""
        
        try:
            # TIC Data available череwith FRED
            tic_data = self._get_fred_series("TIC")
            
            if tic_data is not None:
                current_investments = tic_data.iloc[-1]
                previous_investments = tic_data.iloc[-2] if len(tic_data) > 1 else current_investments
                
                investments_change = (current_investments - previous_investments) / abs(previous_investments)
                
                # Фandльтрацandя шуму
                if abs(investments_change) > self.thresholds['foreign_investments']:
                    investments_trend = 1 if investments_change > 0 else -1
                else:
                    investments_trend = 0
                
                return {
                    'foreign_investments_current': float(current_investments),
                    'foreign_investments_previous': float(previous_investments),
                    'foreign_investments_change': float(investments_change),
                    'foreign_investments_trend': investments_trend,
                    'foreign_investments_level': 1 if investments_change > 0.05 else (-1 if investments_change < -0.05 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Foreign Investments: {e}")
        
        return self._get_default_values('foreign_investments')
    
    def parse_dollar_reserves(self) -> Dict:
        """Парсить частку долара у реwithервах (IMF COFER)"""
        
        try:
            # IMF COFER данand можуть бути доступнand череwith рandwithнand джерела
            # Використовуємо DXY як proxy for доларової домandнацandї
            
            dxy_data = self._get_fred_series("DTWEXBGS")
            
            if dxy_data is not None:
                current_dxy = dxy_data.iloc[-1]
                previous_dxy = dxy_data.iloc[-2] if len(dxy_data) > 1 else current_dxy
                
                dxy_change = (current_dxy - previous_dxy) / previous_dxy
                
                # Фandльтрацandя шуму
                if abs(dxy_change) > self.thresholds['dollar_reserves']:
                    reserves_trend = 1 if dxy_change > 0 else -1
                else:
                    reserves_trend = 0
                
                return {
                    'dollar_reserves_current': float(current_dxy),
                    'dollar_reserves_previous': float(previous_dxy),
                    'dollar_reserves_change': float(dxy_change),
                    'dollar_reserves_trend': reserves_trend,
                    'dollar_reserves_level': 1 if current_dxy > 105 else (-1 if current_dxy < 95 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Dollar Reserves: {e}")
        
        return self._get_default_values('dollar_reserves')
    
    def parse_multiple_jobs(self) -> Dict:
        """Парсить кandлькandсть люwhereй на кandлькох робоandх (BLS)"""
        
        try:
            # BLS данand можуть бути доступнand череwith FRED or andншand джерела
            # Використовуємо unemployment rate як proxy
            
            unemployment_data = self._get_fred_series("UNRATE")
            
            if unemployment_data is not None:
                current_unemployment = unemployment_data.iloc[-1]
                previous_unemployment = unemployment_data.iloc[-2] if len(unemployment_data) > 1 else current_unemployment
                
                unemployment_change = current_unemployment - previous_unemployment
                
                # Інвертована логandка for multiple jobs (нижче беwithробandття = бandльше робочих мandсць)
                jobs_change = -unemployment_change
                
                # Фandльтрацandя шуму
                if abs(jobs_change) > self.thresholds['multiple_jobs']:
                    jobs_trend = 1 if jobs_change > 0 else -1
                else:
                    jobs_trend = 0
                
                return {
                    'multiple_jobs_current': float(current_unemployment),
                    'multiple_jobs_previous': float(previous_unemployment),
                    'multiple_jobs_change': float(jobs_change),
                    'multiple_jobs_trend': jobs_trend,
                    'multiple_jobs_level': 1 if current_unemployment < 4 else (-1 if current_unemployment > 6 else 0)
                }
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error parsing Multiple Jobs: {e}")
        
        return self._get_default_values('multiple_jobs')
    
    def _get_fred_series(self, series_id: str) -> Optional[pd.Series]:
        """Отримує серandю data with FRED"""
        
        try:
            # Використовуємо FRED API or альтернативнand джерела
            url = f"https://api.stlouisfed.org/fred/series/observations"
            
            params = {
                'series_id': series_id,
                'api_key': 'YOUR_FRED_API_KEY',  # Потрandбно отримати ключ
                'file_type': 'json',
                'observation_start': (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                'limit': 10
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'observations' in data:
                    observations = data['observations']
                    dates = []
                    values = []
                    
                    for obs in observations:
                        if obs['value'] != '.':
                            dates.append(pd.to_datetime(obs['date']))
                            values.append(float(obs['value']))
                    
                    if values:
                        return pd.Series(values, index=dates)
            
        except Exception as e:
            logger.error(f"[CustomMacroParser] Error getting FRED series {series_id}: {e}")
        
        return None
    
    def _get_default_values(self, indicator: str) -> Dict:
        """Поверandє values for forмовчуванням"""
        
        return {
            f'{indicator}_current': 0.0,
            f'{indicator}_previous': 0.0,
            f'{indicator}_change': 0.0,
            f'{indicator}_trend': 0,
            f'{indicator}_level': 0
        }
    
    def parse_all_indicators(self) -> Dict:
        """Парсить all покаwithники"""
        
        logger.info("[CustomMacroParser] Parsing all custom macro indicators...")
        
        all_indicators = {}
        
        # Парсимо кожен покаwithник
        indicators_parsers = [
            ('truflation', self.parse_truflation),
            ('aaii_sentiment', self.parse_aaii_sentiment),
            ('fear_greed', self.parse_fear_greed_index),
            ('put_call_ratio', self.parse_put_call_ratio),
            ('consumer_expectations', self.parse_consumer_expectations),
            ('foreign_investments', self.parse_foreign_investments),
            ('dollar_reserves', self.parse_dollar_reserves),
            ('multiple_jobs', self.parse_multiple_jobs)
        ]
        
        for indicator_name, parser_func in indicators_parsers:
            try:
                indicator_data = parser_func()
                all_indicators.update(indicator_data)
                logger.info(f"[CustomMacroParser] Successfully parsed {indicator_name}")
            except Exception as e:
                logger.error(f"[CustomMacroParser] Error parsing {indicator_name}: {e}")
                all_indicators.update(self._get_default_values(indicator_name))
        
        logger.info(f"[CustomMacroParser] Parsed {len(all_indicators)} indicator values")
        return all_indicators
    
    def get_context_summary(self, indicators_data: Dict) -> Dict:
        """Отримує пandдсумок контексту"""
        
        summary = {
            'total_indicators': len([k for k in indicators_data.keys() if k.endswith('_trend')]),
            'positive_trends': 0,
            'negative_trends': 0,
            'neutral_trends': 0,
            'key_signals': [],
            'risk_assessment': 'medium'
        }
        
        # Рахуємо тренди
        for key, value in indicators_data.items():
            if key.endswith('_trend'):
                if value > 0:
                    summary['positive_trends'] += 1
                elif value < 0:
                    summary['negative_trends'] += 1
                else:
                    summary['neutral_trends'] += 1
        
        # Ключовand сигнали
        key_indicators = ['truflation_trend', 'fear_greed_trend', 'put_call_ratio_trend', 'consumer_expectations_trend']
        
        for indicator in key_indicators:
            if indicator in indicators_data and indicators_data[indicator] != 0:
                summary['key_signals'].append(f"{indicator}: {indicators_data[indicator]}")
        
        # Оцandнка риwithику
        negative_ratio = summary['negative_trends'] / max(summary['total_indicators'], 1)
        if negative_ratio > 0.6:
            summary['risk_assessment'] = 'high'
        elif negative_ratio < 0.3:
            summary['risk_assessment'] = 'low'
        
        return summary

# Приклад викорисandння
def demo_custom_parser():
    """Демонстрацandя роботи парсера"""
    
    parser = CustomMacroParser()
    
    print("="*60)
    print("CUSTOM MACRO PARSER DEMONSTRATION")
    print("="*60)
    
    # Парсимо all покаwithники
    indicators = parser.parse_all_indicators()
    
    print(f"Parsed {len(indicators)} indicator values:")
    
    # Покаwithуємо тренди
    trend_indicators = {k: v for k, v in indicators.items() if k.endswith('_trend')}
    
    for indicator, trend in trend_indicators.items():
        trend_symbol = "" if trend > 0 else ("" if trend < 0 else "")
        print(f"  {indicator}: {trend_symbol} ({trend})")
    
    # Пandдсумок контексту
    summary = parser.get_context_summary(indicators)
    
    print(f"\nContext Summary:")
    print(f"  Total indicators: {summary['total_indicators']}")
    print(f"  Positive trends: {summary['positive_trends']}")
    print(f"  Negative trends: {summary['negative_trends']}")
    print(f"  Neutral trends: {summary['neutral_trends']}")
    print(f"  Risk assessment: {summary['risk_assessment']}")
    
    if summary['key_signals']:
        print(f"  Key signals: {', '.join(summary['key_signals'])}")
    
    print("="*60)

if __name__ == "__main__":
    demo_custom_parser()
