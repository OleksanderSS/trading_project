# utils/simple_temporal_indicators.py

import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger("SimpleTemporalIndicators")


class SimpleTemporalIndicators:
    """
    Прості часові показники - просто числа без складної логіки
    """
    
    def __init__(self):
        self.logger = logging.getLogger("SimpleTemporalIndicators")
        
        # Прості числові показники - просто числа без ваг
        self.temporal_values = {
            'day_of_week': {
                0: 0,  # Monday
                1: 1,  # Tuesday
                2: 2,  # Wednesday
                3: 3,  # Thursday
                4: 4,  # Friday
                5: 5,  # Saturday
                6: 6   # Sunday
            },
            'day_of_month': {
                # Просто число дня місяця 1-31
                day: day for day in range(1, 32)
            },
            'week_of_year': {
                # Просто номер тижня 1-52
                week: week for week in range(1, 53)
            },
            'month_of_year': {
                # Просто номер місяця 1-12
                month: month for month in range(1, 13)
            },
            'quarter': {
                # Просто номер кварталу 1-4
                quarter: quarter for quarter in range(1, 5)
            },
            'hour_of_day': {
                # Просто година 0-23
                hour: hour for hour in range(24)
            }
        }
        
        self.logger.info("SimpleTemporalIndicators initialized")
    
    def get_temporal_indicators(self, date: datetime = None) -> Dict[str, Any]:
        """
        Отримати прості часові показники для дати
        
        Args:
            date: Дата (якщо None - поточна)
            
        Returns:
            Dict: Часові показники
        """
        if date is None:
            date = datetime.now()
        
        indicators = {
            'date': date.isoformat(),
            'day_of_week': date.weekday(),  # 0=Monday, 6=Sunday
            'day_of_month': date.day,
            'week_of_year': date.isocalendar()[1],
            'month_of_year': date.month,
            'quarter': (date.month - 1) // 3 + 1,
            'hour_of_day': date.hour,
            'day_of_year': date.timetuple().tm_yday
        }
        
        # Просто повертаємо числа без ваг
        indicators['values'] = {
            'day_of_week': indicators['day_of_week'],
            'day_of_month': indicators['day_of_month'],
            'week_of_year': indicators['week_of_year'],
            'month_of_year': indicators['month_of_year'],
            'quarter': indicators['quarter'],
            'hour_of_day': indicators['hour_of_day']
        }
        
        # Прості описи
        indicators['descriptions'] = {
            'day_of_week': self._get_day_of_week_description(date.weekday()),
            'day_of_month': f"Day {date.day} of month",
            'week_of_year': f"Week {date.isocalendar()[1]} of year",
            'month_of_year': f"Month {date.month} of year",
            'quarter': f"Quarter {(date.month - 1) // 3 + 1}",
            'hour_of_day': f"Hour {date.hour} of day"
        }
        
        return indicators
    
    def get_simple_values(self, df: pd.DataFrame, date_col: str = 'date') -> Dict[str, Any]:
        """
        Отримати прості числові значення
        
        Args:
            df: DataFrame з датами
            date_col: Назва колонки з датами
            
        Returns:
            Dict: Числові значення
        """
        indicators = self.get_temporal_indicators(date)
        return indicators['values']
    
    def get_day_of_week_value(self, date: datetime = None) -> int:
        """Отримати день тижня як число (0-6)"""
        if date is None:
            date = datetime.now()
        return date.weekday()
    
    def get_advanced_temporal_features(self, df: pd.DataFrame, date_col: str = 'date') -> Dict[str, Any]:
        """
        Розширені часові фічі для кращого ML
        
        Args:
            df: DataFrame з датами
            date_col: Назва колонки з датами
            
        Returns:
            Dict: Розширені часові фічі
        """
        if date_col not in df.columns:
            self.logger.warning(f"Date column '{date_col}' not found in DataFrame")
            return {}
        
        # Перетворюємо дати
        dates = pd.to_datetime(df[date_col])
        
        features = {}
        
        # Базові часові фічі
        features.update(self.get_simple_values(df, date_col))
        
        # Розширені фічі
        features.update({
            # Квартали
            'quarter': dates.dt.quarter,
            'quarter_of_year': dates.dt.quarter,
            
            # Тиждень року
            'week_of_year': dates.dt.isocalendar().week,
            
            # День року
            'day_of_year': dates.dt.dayofyear,
            
            # Частина дня
            'time_of_day': dates.dt.hour + dates.dt.minute / 60.0,
            
            # Частина тижня (робочі/вихідні)
            'is_weekend': (dates.dt.dayofweek >= 5).astype(int),
            'is_monday': (dates.dt.dayofweek == 0).astype(int),
            'is_friday': (dates.dt.dayofweek == 4).astype(int),
            
            # Частина місяця (початок/кінець)
            'is_month_start': (dates.dt.day <= 5).astype(int),
            'is_month_end': (dates.dt.day >= 25).astype(int),
            
            # Частина року (сезонність)
            'is_winter': dates.dt.month.isin([12, 1, 2]).astype(int),
            'is_spring': dates.dt.month.isin([3, 4, 5]).astype(int),
            'is_summer': dates.dt.month.isin([6, 7, 8]).astype(int),
            'is_fall': dates.dt.month.isin([9, 10, 11]).astype(int),
            
            # Циклічні фічі (для моделей)
            'day_of_year_sin': np.sin(2 * np.pi * dates.dt.dayofyear / 365.0),
            'day_of_year_cos': np.cos(2 * np.pi * dates.dt.dayofyear / 365.0),
            'hour_sin': np.sin(2 * np.pi * dates.dt.hour / 24.0),
            'hour_cos': np.cos(2 * np.pi * dates.dt.hour / 24.0),
            'day_of_week_sin': np.sin(2 * np.pi * dates.dt.dayofweek / 7.0),
            'day_of_week_cos': np.cos(2 * np.pi * dates.dt.dayofweek / 7.0),
        })
        
        return features
    
    def get_day_of_month_value(self, date: datetime = None) -> int:
        """Отримати день місяця як число (1-31)"""
        if date is None:
            date = datetime.now()
        return date.day
    
    def get_month_of_year_value(self, date: datetime = None) -> int:
        """Отримати місяць як число (1-12)"""
        if date is None:
            date = datetime.now()
        return date.month
    
    def get_hour_of_day_value(self, date: datetime = None) -> int:
        """Отримати годину як число (0-23)"""
        if date is None:
            date = datetime.now()
        return date.hour
    
    def analyze_temporal_pattern(self, dates: List[datetime]) -> Dict[str, Any]:
        """
        Аналізує часовий патерн для списку дат
        
        Args:
            dates: Список дат
            
        Returns:
            Dict: Аналіз патерну
        """
        analysis = {
            'total_dates': len(dates),
            'date_range': {
                'start': min(dates).isoformat(),
                'end': max(dates).isoformat()
            },
            'patterns': {},
            'statistics': {}
        }
        
        # Аналізуємо кожен показник
        for indicator_name in ['day_of_week', 'day_of_month', 'month_of_year', 'quarter']:
            values = []
            weights = []
            
            for date in dates:
                indicators = self.get_temporal_indicators(date)
                values.append(indicators[indicator_name])
                weights.append(indicators['weights'].get(indicator_name, 0.0))
            
            if values:
                analysis['patterns'][indicator_name] = {
                    'values': values,
                    'weights': weights,
                    'average_weight': np.mean(weights),
                    'best_value': values[np.argmax(weights)] if weights else None,
                    'worst_value': values[np.argmin(weights)] if weights else None
                }
        
        # Загальна статистика
        all_weights = []
        for date in dates:
            indicators = self.get_temporal_indicators(date)
            all_weights.append(indicators['total_weight'])
        
        if all_weights:
            analysis['statistics'] = {
                'average_bias': np.mean(all_weights),
                'positive_days': len([w for w in all_weights if w > 0]),
                'negative_days': len([w for w in all_weights if w < 0]),
                'neutral_days': len([w for w in all_weights if abs(w) < 0.05]),
                'best_day': dates[np.argmax(all_weights)].isoformat(),
                'worst_day': dates[np.argmin(all_weights)].isoformat()
            }
        
        return analysis
    
    def _get_day_of_week_description(self, day_of_week: int) -> str:
        """Опис дня тижня"""
        descriptions = {
            0: "Monday - negative sentiment",
            1: "Tuesday - recovery",
            2: "Wednesday - neutral",
            3: "Thursday - momentum",
            4: "Friday - optimism",
            5: "Saturday - weekend",
            6: "Sunday - weekend"
        }
        return descriptions.get(day_of_week, "Unknown")
    
    def _get_day_of_month_description(self, day_of_month: int) -> str:
        """Опис дня місяця"""
        if 1 <= day_of_month <= 5:
            return "Payday effect - positive"
        elif 6 <= day_of_month <= 20:
            return "Normal period - neutral"
        elif 21 <= day_of_month <= 25:
            return "End of month - slightly negative"
        else:
            return "Bill pressure - negative"
    
    def _get_week_of_year_description(self, week_of_year: int) -> str:
        """Опис тижня року"""
        if 1 <= week_of_year <= 5:
            return "New Year optimism"
        elif 6 <= week_of_year <= 45:
            return "Normal period"
        else:
            return "Tax loss harvesting"
    
    def _get_month_of_year_description(self, month: int) -> str:
        """Опис місяця року"""
        descriptions = {
            1: "January - New Year optimism",
            2: "February - Valentine's effect",
            3: "March - Spring optimism",
            4: "April - Tax season",
            5: "May - Summer approach",
            6: "June - Summer start",
            7: "July - Summer lull",
            8: "August - Summer end",
            9: "September - Back to school",
            10: "October - Halloween effect",
            11: "November - Pre-holiday",
            12: "December - Holiday season"
        }
        return descriptions.get(month, "Unknown")
    
    def _get_quarter_description(self, quarter: int) -> str:
        """Опис кварталу"""
        descriptions = {
            1: "Q1 - New Year optimism",
            2: "Q2 - Mid-year neutral",
            3: "Q3 - Summer lull",
            4: "Q4 - Holiday season"
        }
        return descriptions.get(quarter, "Unknown")
    
    def _get_hour_of_day_description(self, hour: int) -> str:
        """Опис години дня"""
        if 9 <= hour <= 11:
            return "Morning - high activity"
        elif 12 <= hour <= 13:
            return "Lunch break"
        elif 14 <= hour <= 16:
            return "Afternoon - trading hours"
        elif 17 <= hour <= 18:
            return "Evening - closing hours"
        else:
            return "Off hours"


# Приклад використання:
if __name__ == "__main__":
    # Створюємо калькулятор
    temporal_calc = SimpleTemporalIndicators()
    
    # Отримуємо показники для поточної дати
    indicators = temporal_calc.get_temporal_indicators()
    print("Часові показники:", indicators)
    
    # Отримуємо простий bias score
    bias_score = temporal_calc.get_simple_bias_score()
    print(f"Bias score: {bias_score}")
    
    # Отримуємо рекомендації
    recommendations = temporal_calc.get_trading_recommendations()
    print("Рекомендації:", recommendations)
    
    # Отримуємо коефіцієнт корекції
    adjustment = temporal_calc.get_temporal_adjustment_factor()
    print(f"Коефіцієнт корекції: {adjustment}")
    
    # Аналізуємо патерн для останнього місяця
    from datetime import datetime, timedelta
    dates = [datetime.now() - timedelta(days=i) for i in range(30)]
    pattern_analysis = temporal_calc.analyze_temporal_pattern(dates)
    print("Аналіз патерну:", pattern_analysis)
