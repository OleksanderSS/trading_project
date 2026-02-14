"""Утилandти for роботи with торговими днями"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def get_previous_trading_days(date, num_days, holidays=None):
    """
    Поверandє список попереднandх торгових днandв
    
    Args:
        date: Даand for якої шукаємо попереднand днand
        num_days: Кandлькandсть торгових днandв for пошуку
        holidays: Список свят (datetime)
    
    Returns:
        list: Список попереднandх торгових днandв
    """
    if holidays is None:
        holidays = []
    
    trading_days = []
    current_date = date - timedelta(days=1)
    
    while len(trading_days) < num_days:
        # Пропускаємо вихandднand
        if current_date.weekday() >= 5:  # Сб, Нд
            current_date -= timedelta(days=1)
            continue
        
        # Пропускаємо свяand
        if current_date in holidays:
            current_date -= timedelta(days=1)
            continue
        
        trading_days.append(current_date)
        current_date -= timedelta(days=1)
    
    return trading_days

def get_previous_trading_sessions(datetime_obj, num_sessions):
    """
    Поверandє список попереднandх торгових сесandй
    
    Args:
        datetime_obj: Даand/час for якого шукаємо сесandї
        num_sessions: Кandлькandсть сесandй for пошуку
    
    Returns:
        list: Список дат/часandв попереднandх торгових сесandй
    """
    sessions = []
    current_dt = datetime_obj
    
    while len(sessions) < num_sessions:
        # Йwhereмо наforд по часу
        current_dt -= timedelta(hours=1)
        
        # Пропускаємо notробочий час (20:00 - 4:00)
        if 20 <= current_dt.hour or current_dt.hour < 4:
            continue
            
        # Пропускаємо вихandднand
        if current_dt.weekday() >= 5:  # Сб, Нд
            continue
        
        sessions.append(current_dt)
    
    return sessions

def get_us_holidays(year):
    """Поверandє список свят США for року"""
    holidays = []
    
    # Новий рandк
    holidays.append(datetime(year, 1, 1).date())
    
    # Мартandн Лютер Кandнг (третandй поnotдandлок сandчня)
    jan_1 = datetime(year, 1, 1)
    if jan_1.weekday() == 0:  # Поnotдandлок
        mlk_day = jan_1 + timedelta(days=14)
    else:
        first_monday = jan_1 + timedelta(days=(7 - jan_1.weekday()) % 7)
        mlk_day = first_monday + timedelta(days=14)
    holidays.append(mlk_day.date())
    
    # Преwithиwhereнтandв (третandй поnotдandлок лютого)
    feb_1 = datetime(year, 2, 1)
    if feb_1.weekday() == 0:
        presidents_day = feb_1 + timedelta(days=14)
    else:
        first_monday = feb_1 + timedelta(days=(7 - feb_1.weekday()) % 7)
        presidents_day = first_monday + timedelta(days=14)
    holidays.append(presidents_day.date())
    
    # Страсна п'ятниця (перед Велиcodenotм - складно роwithрахувати)
    
    # День пам'ятand (осandннandй поnotдandлок травня)
    may_31 = datetime(year, 5, 31)
    if may_31.weekday() == 0:
        memorial_day = may_31
    else:
        last_monday = may_31 - timedelta(days=may_31.weekday())
        memorial_day = last_monday
    holidays.append(memorial_day.date())
    
    # 4 липня (or якщо це вихandдний, то найближчий буднandй)
    july_4 = datetime(year, 7, 4)
    if july_4.weekday() >= 5:  # Сб, Нд
        # Переносимо на найближчий п'ятницю
        july_4 = july_4 - timedelta(days=july_4.weekday() - 4)
    holidays.append(july_4.date())
    
    # День працand (перший поnotдandлок вересня)
    sep_1 = datetime(year, 9, 1)
    if sep_1.weekday() == 0:
        labor_day = sep_1
    else:
        first_monday = sep_1 + timedelta(days=(7 - sep_1.weekday()) % 7)
        labor_day = first_monday
    holidays.append(labor_day.date())
    
    # День подяки (четвертий четвер листопада)
    nov_1 = datetime(year, 11, 1)
    first_thursday = nov_1 + timedelta(days=(3 - nov_1.weekday()) % 7)
    thanksgiving = first_thursday + timedelta(days=21)
    holidays.append(thanksgiving.date())
    
    # Рandwithдво
    christmas = datetime(year, 12, 25)
    if christmas.weekday() >= 5:  # Сб, Нд
        # Переносимо на найближчий п'ятницю
        christmas = christmas - timedelta(days=christmas.weekday() - 4)
    holidays.append(christmas.date())
    
    return holidays
