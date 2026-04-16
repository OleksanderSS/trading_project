import pandas as pd
import numpy as np

batch_dir = 'data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5'

print("🧠 РОЗУМНЕ ВИПРАВЛЕННЯ TIMEZONE")
print("="*80)

# Завантажуємо дані
features_df = pd.read_parquet(f'{batch_dir}/features.parquet')
targets_df = pd.read_parquet(f'{batch_dir}/targets.parquet')

print(f"Features: {features_df.shape}")
print(f"Targets: {targets_df.shape}")

# Створюємо DataFrame для аналізу
features_keys = features_df[['datetime', 'ticker']].copy()
features_keys['source'] = 'features'

targets_keys = targets_df[['datetime', 'ticker']].copy()
targets_keys['source'] = 'targets'

# Об'єднуємо для аналізу
all_keys = pd.concat([features_keys, targets_keys], ignore_index=True)

# Групуємо по ticker та сортуємо по datetime
all_keys = all_keys.sort_values(['ticker', 'datetime'])

print(f"\nВсього ключів: {len(all_keys)}")

# Знаходимо найближчі пари features-targets
matched_pairs = []
unmatched_features = []
unmatched_targets = []

features_times = features_df['datetime'].values
targets_times = targets_df['datetime'].values

print(f"Пошук найближчих пар...")

for i, feat_time in enumerate(features_times):
    # Знаходимо найближчий час в targets
    time_diffs = np.abs(targets_times - feat_time)
    min_diff_idx = np.argmin(time_diffs)
    min_diff = time_diffs[min_diff_idx]
    
    # Якщо різниця менше 3 годин, вважаємо це парою
    if min_diff <= pd.Timedelta(hours=3):
        target_time = targets_times[min_diff_idx]
        matched_pairs.append({
            'features_idx': i,
            'targets_idx': min_diff_idx,
            'features_time': feat_time,
            'targets_time': target_time,
            'diff': min_diff
        })
    else:
        unmatched_features.append(i)

print(f"Знайдено пар: {len(matched_pairs)}")
print(f"Неспарених features: {len(unmatched_features)}")

if matched_pairs:
    # Аналізуємо різниці
    diffs = [pair['diff'] for pair in matched_pairs]
    unique_diffs = list(set(diffs))
    print(f"Унікальні різниці: {unique_diffs}")
    
    # Виправляємо targets datetime, щоб збігався з features
    new_targets_df = targets_df.copy()
    
    for pair in matched_pairs:
        targets_idx = pair['targets_idx']
        features_time = pair['features_time']
        new_targets_df.iloc[targets_idx, new_targets_df.columns.get_loc('datetime')] = features_time
    
    print(f"Виправлено {len(matched_pairs)} datetime в targets")
    
    # Зберігаємо
    new_targets_df.to_parquet(f'{batch_dir}/targets.parquet')
    print(f"💾 Виправлені targets збережено")
    
    # Тестуємо merge
    merged = features_df.merge(new_targets_df, on=['datetime', 'ticker'], how='inner')
    print(f"\n🔗 ФІНАЛЬНИЙ ТЕСТ:")
    print(f"  Features: {features_df.shape}")
    print(f"  Targets: {new_targets_df.shape}")
    print(f"  Merged: {merged.shape}")
    
    if merged.shape[0] >= len(matched_pairs):
        print("✅ УСПІХ! Timezone проблему повністю виправлено")
    else:
        print(f"⚠️ Частковий успіх: {merged.shape[0]} з {len(matched_pairs)} очікуваних")
else:
    print("❌ Не вдалося знайти жодної пари")