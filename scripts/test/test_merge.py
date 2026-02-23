#!/usr/bin/env python3
"""
Тестуємо merge_asof логandку
"""

import pandas as pd
import numpy as np

# Створюємо тестовand данand
print("=== TESTING MERGE_ASOF LOGIC ===")

# Тестовand цandни (як у реальних data)
price_data = pd.DataFrame({
    'date': pd.date_range('2025-10-27 13:30', periods=5, freq='15min'),
    'spy_price': [100, 101, 102, 103, 104],
    'qqq_price': [200, 201, 202, 203, 204]
})

print("Price data:")
print(price_data)

# Тестовand новини
news_data = pd.DataFrame({
    'trade_date': [pd.Timestamp('2025-10-28 07:00:00')],
    'news_id': ['news1']
})

print("\nNews data:")
print(news_data)

# Тестуємо merge_asof with backward
merged_backward = pd.merge_asof(
    news_data.sort_values('trade_date'),
    price_data.sort_values('date'),
    left_on='trade_date',
    right_on='date',
    direction='backward',
    allow_exact_matches=True
)

print("\nMerge with direction='backward':")
print(merged_backward)

# Тестуємо merge_asof with forward
merged_forward = pd.merge_asof(
    news_data.sort_values('trade_date'),
    price_data.sort_values('date'),
    left_on='trade_date',
    right_on='date',
    direction='forward',
    allow_exact_matches=True
)

print("\nMerge with direction='forward':")
print(merged_forward)

# Тестуємо merge_asof with nearest
merged_nearest = pd.merge_asof(
    news_data.sort_values('trade_date'),
    price_data.sort_values('date'),
    left_on='trade_date',
    right_on='date',
    direction='nearest',
    allow_exact_matches=True
)

print("\nMerge with direction='nearest':")
print(merged_nearest)

# Тестуємо with ffill()
price_data_filled = price_data.copy()
price_data_filled['date'] = pd.date_range('2025-10-27 13:30', periods=10, freq='15min')
price_data_filled['spy_price'] = price_data_filled['spy_price'].ffill()
price_data_filled['qqq_price'] = price_data_filled['qqq_price'].ffill()

print("\nPrice data after ffill:")
print(price_data_filled)

merged_with_ffill = pd.merge_asof(
    news_data.sort_values('trade_date'),
    price_data_filled.sort_values('date'),
    left_on='trade_date',
    right_on='date',
    direction='backward',
    allow_exact_matches=True
)

print("\nMerge with ffill data:")
print(merged_with_ffill)
