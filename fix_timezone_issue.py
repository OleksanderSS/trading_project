import pandas as pd
import numpy as np

batch_dir = 'data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5'

print("🔧 ВИПРАВЛЕННЯ TIMEZONE ПРОБЛЕМИ")
print("="*80)

# Завантажуємо дані
features_df = pd.read_parquet(f'{batch_dir}/features.parquet')
targets_df = pd.read_parquet(f'{batch_dir}/targets.parquet')

print(f"BEFORE:")
print(f"  Features: {features_df.shape}")
print(f"  published_at timezone: {features_df['published_at'].dt.tz}")
print(f"  datetime timezone: {features_df['datetime'].dt.tz}")
print(f"  Targets datetime timezone: {targets_df['datetime'].dt.tz}")

# ВИПРАВЛЕННЯ 1: Видаляємо timezone з published_at
features_df['published_at'] = features_df['published_at'].dt.tz_localize(None)

# ВИПРАВЛЕННЯ 2: Перевіряємо, чи datetime в targets має timezone
if targets_df['datetime'].dt.tz is not None:
    targets_df['datetime'] = targets_df['datetime'].dt.tz_localize(None)

print(f"\nAFTER:")
print(f"  published_at timezone: {features_df['published_at'].dt.tz}")
print(f"  datetime timezone: {features_df['datetime'].dt.tz}")
print(f"  Targets datetime timezone: {targets_df['datetime'].dt.tz}")

# ВИПРАВЛЕННЯ 3: Тепер перевіримо різницю
diff = features_df['published_at'].iloc[0] - features_df['datetime'].iloc[0]
print(f"\n🔍 РІЗНИЦЯ МІЖ published_at та datetime:")
print(f"  Різниця: {diff}")

# ВИПРАВЛЕННЯ 4: Якщо різниця 0, то використовуємо published_at як datetime
if diff == pd.Timedelta(0):
    print("✅ published_at та datetime однакові - все ОК")
else:
    print(f"⚠️ Різниця {diff} - треба синхронізувати")
    # Використовуємо published_at як основу для datetime
    features_df['datetime'] = features_df['published_at'].copy()
    print("✅ datetime синхронізовано з published_at")

# ТЕСТ MERGE
print(f"\n🔗 ТЕСТ MERGE:")
merged = features_df.merge(targets_df, on=['datetime', 'ticker'], how='inner')
print(f"  Merged shape: {merged.shape}")

if merged.shape[0] > 100:
    print("✅ УСПІХ! Merge працює правильно")
    
    # Зберігаємо виправлені дані
    features_df.to_parquet(f'{batch_dir}/features.parquet')
    targets_df.to_parquet(f'{batch_dir}/targets.parquet')
    print(f"💾 Виправлені дані збережено")
else:
    print("❌ Merge все ще не працює")
    
    # Детальна діагностика
    print(f"\n🔍 ДЕТАЛЬНА ДІАГНОСТИКА:")
    features_dt = set(features_df['datetime'].dt.floor('s'))
    targets_dt = set(targets_df['datetime'].dt.floor('s'))
    common_dt = features_dt & targets_dt
    print(f"  Features datetime: {len(features_dt)} унікальних")
    print(f"  Targets datetime: {len(targets_dt)} унікальних")
    print(f"  Співпадінь: {len(common_dt)}")
    
    if len(common_dt) < 50:
        print(f"  Перші 5 features datetime: {sorted(features_dt)[:5]}")
        print(f"  Перші 5 targets datetime: {sorted(targets_dt)[:5]}")