import pandas as pd

batch_dir = 'data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5'

print("🔧 ПРИМУСОВЕ ВИПРАВЛЕННЯ TARGETS")
print("="*80)

# Завантажуємо targets
targets_df = pd.read_parquet(f'{batch_dir}/targets.parquet')

print(f"BEFORE:")
print(f"  Targets shape: {targets_df.shape}")
print(f"  Sample datetime: {targets_df['datetime'].iloc[0]}")

# Додаємо 2 години
targets_df['datetime'] = targets_df['datetime'] + pd.Timedelta(hours=2)

print(f"AFTER (+2 години):")
print(f"  Sample datetime: {targets_df['datetime'].iloc[0]}")

# Зберігаємо
targets_df.to_parquet(f'{batch_dir}/targets.parquet')
print(f"💾 Targets збережено")

# Перевіряємо merge
features_df = pd.read_parquet(f'{batch_dir}/features.parquet')
merged = features_df.merge(targets_df, on=['datetime', 'ticker'], how='inner')
print(f"\n🔗 ФІНАЛЬНИЙ ТЕСТ:")
print(f"  Features: {features_df.shape}")
print(f"  Targets: {targets_df.shape}")
print(f"  Merged: {merged.shape}")

if merged.shape[0] > 100:
    print("✅ УСПІХ! Timezone проблему виправлено")
else:
    print("❌ Все ще проблема")