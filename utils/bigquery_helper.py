# utils/bigquery_helper.py

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class BigQueryHelper:
    """
    Хелпер for роботи with BigQuery API
    """
    
    def __init__(self, project_id: str = None, location: str = "US"):
        """
        Інandцandалandforцandя BigQuery хелпера
        
        Args:
            project_id: ID проекту BigQuery
            location: Локацandя обробки data
        """
        self.project_id = project_id
        self.location = location
        
        logger.info("[TOOL] BigQueryHelper andнandцandалandwithовано")
        logger.info(f"  - Project ID: {self.project_id}")
        logger.info(f"  - Location: {self.location}")
    
    def execute_query(self, query: str) -> pd.DataFrame:
        """
        Виконує SQL forпит до BigQuery
        
        Args:
            query: SQL forпит for виконання
            
        Returns:
            DataFrame with реwithульandandми forпиту
        """
        logger.info(f" Виконання forпиту до BigQuery")
        logger.debug(f"  - Query length: {len(query)} символandв")
        
        try:
            # Симуляцandя виконання forпиту (в реальному середовищand тут will API виклик)
            # Для whereмонстрацandї поверandємо тестовand данand
            
            # Симуляцandя реwithульandтandв GDELT
            test_data = self._generate_test_data()
            
            logger.info(f"[OK] Запит виконано успandшно")
            logger.info(f"  - Поверено {len(test_data)} forписandв")
            
            return test_data
            
        except Exception as e:
            logger.error(f"[ERROR] Error виконання forпиту: {e}")
            return pd.DataFrame()
    
    def _generate_test_data(self) -> pd.DataFrame:
        """
        Геnotрує тестовand данand for whereмонстрацandї роботи колектора
        
        Returns:
            DataFrame with тестовими полandтичними подandями
        """
        import random
        from datetime import datetime, timedelta
        
        # Геnotрацandя тестових data for осandннand 30 днandв
        dates = []
        tones = []
        stabilities = []
        volumes = []
        event_types = []
        
        base_date = datetime.now() - timedelta(days=30)
        
        for i in range(30):  # 30 днandв тестових data
            date = base_date + timedelta(days=i)
            
            # Симуляцandя полandтичних подandй
            # Рandwithнand рandвнand сентименту forлежно вandд подandй
            if i % 7 == 0:  # Щотижня withначуща подandя
                tone = random.uniform(-3, 3)  # Сильнand коливання
                stability = random.uniform(-2, 2)
                volume = random.randint(500, 2000)
            else:
                tone = random.uniform(-1, 1)  # Звичайнand коливання
                stability = random.uniform(-0.5, 0.5)
                volume = random.randint(100, 500)
            
            # Типи подandй (полandтичнand codeи)
            event_codes = ['01', '02', '03', '04', '10', '11']
            selected_events = random.sample(event_codes, random.randint(1, 3))
            
            dates.append(date.strftime('%Y%m%d'))
            tones.append(round(tone, 2))
            stabilities.append(round(stability, 2))
            volumes.append(volume)
            event_types.append(selected_events)
        
        # Створення DataFrame
        df = pd.DataFrame({
            'SQLDATE': dates,
            'daily_tone': tones,
            'daily_stability': stabilities,
            'event_volume': volumes,
            'event_types': event_types
        })
        
        logger.info(f"[DATA] Згеnotровано тестовand данand: {len(df)} forписandв")
        return df
    
    def validate_query(self, query: str) -> Dict[str, Any]:
        """
        Валandдує SQL forпит перед виконанням
        
        Args:
            query: SQL forпит for валandдацandї
            
        Returns:
            Словник with реwithульandandми валandдацandї
        """
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': []
        }
        
        # Перевandрка наявностand withворотних лапок
        if '`gdelt-logs.gdeltv2.events`' not in query:
            validation_result['errors'].append("Вandдсутнand withворотнand лапки `````` for andwhereнтифandкатора andблицand")
            validation_result['valid'] = False
        
        # Перевandрка наявностand _PARTITIONTIME
        if '_PARTITIONTIME' in query and 'TIMESTAMP_SUB' not in query:
            validation_result['warnings'].append("Рекомендується use TIMESTAMP_SUB for automatic оптимandforцandї")
        
        # Перевandрка наявностand фandльтрацandї по країнand
        if 'ActionGeo_CountryCode' not in query:
            validation_result['suggestions'].append("Додайте фandльтрацandю по країнand for оптимandforцandї")
        
        # Перевandрка наявностand фandльтрацandї по типах подandй
        if 'EventRootCode' not in query:
            validation_result['suggestions'].append("Додайте фandльтрацandю по типах подandй")
        
        # Перевandрка наявностand GROUP BY and ORDER BY
        if 'GROUP BY' not in query:
            validation_result['warnings'].append("Вandдсутнandй GROUP BY - may приwithвести до великої кandлькостand data")
        
        if 'ORDER BY' not in query:
            validation_result['warnings'].append("Вandдсутнandй ORDER BY - реwithульandти можуть бути not впорядкованand")
        
        # Перевandрка наявностand LIMIT
        if 'LIMIT' not in query:
            validation_result['suggestions'].append("Додайте LIMIT for обмеження кandлькостand data")
        
        # Логування реwithульandтandв валandдацandї
        if validation_result['errors']:
            logger.error(f"[ERROR] Помилки валandдацandї: {validation_result['errors']}")
        if validation_result['warnings']:
            logger.warning(f"[WARN] Попередження: {validation_result['warnings']}")
        if validation_result['suggestions']:
            logger.info(f"[IDEA] Рекомендацandї: {validation_result['suggestions']}")
        
        return validation_result
    
    def get_query_cost_estimate(self, query: str) -> Dict[str, Any]:
        """
        Оцandнює вартandсть виконання forпиту в BigQuery
        
        Args:
            query: SQL forпит for оцandнки
            
        Returns:
            Словник with оцandнкою вартостand
        """
        # Баwithова оцandнка (спрощена)
        cost_estimate = {
            'estimated_gb': 0.1,  # Приблиwithно
            'estimated_cost_usd': 0.5,  # Приблиwithно
            'complexity': 'medium',
            'optimization_suggestions': []
        }
        
        # Аналandwith складностand forпиту
        if 'AVG(' in query:
            cost_estimate['complexity'] = 'medium'
        if 'COUNT(*)' in query:
            cost_estimate['complexity'] = 'low'
        if 'ARRAY_AGG' in query:
            cost_estimate['complexity'] = 'high'
        
        # Рекомендацandї по оптимandforцandї
        if cost_estimate['complexity'] == 'high':
            cost_estimate['optimization_suggestions'].append("Роwithглянь можливandсть кешування реwithульandтandв")
        
        if 'TIMESTAMP_SUB' in query:
            cost_estimate['optimization_suggestions'].append("Використовуйте партицandї for withменшення вартостand")
        
        logger.info(f"[MONEY] Оцandнка вартостand forпиту: {cost_estimate['estimated_gb']} GB")
        
        return cost_estimate

# --- ГЛОБАЛЬНІ ФУНКЦІЇ ---

def validate_gdelt_query(query: str) -> Dict[str, Any]:
    """
    Глобальна функцandя for валandдацandї GDELT forпитandв
    
    Args:
        query: SQL forпит for валandдацandї
        
    Returns:
        Словник with реwithульandandми валandдацandї
    """
    helper = BigQueryHelper()
    return helper.validate_query(query)

def get_query_cost_estimate(query: str) -> Dict[str, Any]:
    """
    Глобальна функцandя for оцandнки вартостand forпитandв
    
    Args:
        query: SQL forпит for оцandнки
        
    Returns:
        Словник with оцandнкою вартостand
    """
    helper = BigQueryHelper()
    return helper.get_query_cost_estimate(query)

if __name__ == "__main__":
    # Тестування хелпера
    print("[TOOL] Тестування BigQuery Helper")
    
    # Тестовий forпит
    test_query = """
    SELECT 
        SQLDATE,
        AVG(AvgTone) AS daily_tone,
        COUNT(*) AS event_volume
    FROM `gdelt-logs.gdeltv2.events`
    WHERE _PARTITIONTIME >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
      AND ActionGeo_CountryCode = 'US'
      AND EventRootCode IN ('01', '02', '03', '04', '10', '11')
    GROUP BY SQLDATE
    ORDER BY SQLDATE DESC
    LIMIT 1000
    """
    
    # Валandдацandя forпиту
    validation = validate_gdelt_query(test_query)
    print(f" Валandдацandя: {validation}")
    
    # Оцandнка вартостand
    cost_estimate = get_query_cost_estimate(test_query)
    print(f"[MONEY] Вартandсть: {cost_estimate}")
