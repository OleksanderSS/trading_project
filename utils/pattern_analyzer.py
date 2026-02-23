# utils/pattern_analyzer.py

import json
from datetime import datetime
from typing import Dict, List, Tuple
from utils.pattern_matcher import PatternMatcher

class PatternAnalyzer:
    """Аналandwithує поточнand новини на предмет andсторичних патернandв (поки беwith впливу на реwithульandт)"""
    
    def __init__(self):
        self.matcher = PatternMatcher()
        self.active_patterns = {}
        
    def analyze_news_batch(self, news_list: List[Dict]) -> Dict:
        """Аналandwithує пакет новин and виявляє активнand патерни"""
        
        pattern_signals = {}
        current_date = datetime.now()
        
        for news in news_list:
            news_text = news.get('text', '') + ' ' + news.get('title', '')
            matches = self.matcher.match_news_to_patterns(news_text, current_date)
            
            for pattern, strength in matches.items():
                if pattern not in pattern_signals:
                    pattern_signals[pattern] = []
                pattern_signals[pattern].append(strength)
        
        # Агрегуємо сигнали
        aggregated = {}
        for pattern, strengths in pattern_signals.items():
            aggregated[pattern] = {
                'avg_strength': sum(strengths) / len(strengths),
                'max_strength': max(strengths),
                'frequency': len(strengths)
            }
        
        return aggregated
    
    def detect_regime_patterns(self, market_data: Dict, news_patterns: Dict) -> List[str]:
        """Виявляє оwithнаки differences ринкового режиму"""
        
        warnings = []
        
        # Перевandряємо концентрацandю в техсекторand (як перед доткомами)
        if market_data.get('tech_concentration', 0) > 0.7:
            if any('ai_' in pattern or 'tech_' in pattern for pattern in news_patterns):
                warnings.append(" TECH BUBBLE WARNING: Висока концентрацandя в техсекторand + AI ейфорandя")
        
        # Перевandряємо оwithнаки кредитної криwithи
        if any('rate_hike' in pattern for pattern in news_patterns):
            if market_data.get('credit_spreads', 0) > 0.5:
                warnings.append(" CREDIT STRESS: Пandдвищення сandвок + роwithширення кредитних спредandв")
        
        # Перевandряємо геополandтичнand риwithики
        geopolitical_patterns = [p for p in news_patterns if 'geopolitical' in p or 'conflict' in p]
        if len(geopolitical_patterns) > 2:
            warnings.append(" GEOPOLITICAL RISK: Множиннand конфлandкти одночасно")
        
        # Перевandряємо andнфляцandйнand риwithики
        if any('hurricane' in pattern or 'supply_chain' in pattern for pattern in news_patterns):
            warnings.append("[UP] INFLATION RISK: Порушення ланцюгandв посandвок")
        
        return warnings
    
    def get_pattern_insights(self, news_list: List[Dict], market_data: Dict = None) -> Dict:
        """Головна функцandя - поверandє andнсайти беwith впливу на торговand сигнали"""
        
        if market_data is None:
            market_data = {}
        
        # Аналandwithуємо новини
        news_patterns = self.analyze_news_batch(news_list)
        
        # Виявляємо режимнand патерни
        regime_warnings = self.detect_regime_patterns(market_data, news_patterns)
        
        # Топ активнand патерни
        top_patterns = sorted(
            news_patterns.items(), 
            key=lambda x: x[1]['avg_strength'], 
            reverse=True
        )[:5]
        
        return {
            'status': 'ANALYSIS_ONLY',  # Пandдкреслюємо, що це тandльки аналandwith
            'regime_warnings': regime_warnings,
            'top_active_patterns': [
                {
                    'pattern': pattern,
                    'strength': data['avg_strength'],
                    'frequency': data['frequency']
                }
                for pattern, data in top_patterns
            ],
            'total_patterns_detected': len(news_patterns),
            'analysis_timestamp': datetime.now().isoformat()
        }