#!/usr/bin/env python3
"""
Скрипт для виправлення проблем з якістю даних

Виправляє:
1. Видаляє колонки з 100% пропусків (news_id, news_title)
2. Заповнює пропуски в макро-даних (forward fill)
3. Перевіряє якість таргета
4. Зберігає очищені дані
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_nulls(df: pd.DataFrame, name: str) -> dict:
    """Аналіз пропусків в DataFrame"""
    nulls = df.isnull().sum()
    total_cells = df.shape[0] * df.shape[1]
    total_nulls = nulls.sum()
    
    null_cols = {}
    for col in df.columns:
        null_count = nulls[col]
        if null_count > 0:
            null_pct = (null_count / len(df)) * 100
            null_cols[col] = {
                'count': int(null_count),
                'percentage': round(null_pct, 2)
            }
    
    return {
        'name': name,
        'shape': df.shape,
        'total_nulls': int(total_nulls),
        'total_cells': int(total_cells),
        'null_percentage': round((total_nulls / total_cells) * 100, 2),
        'null_columns': null_cols
    }


def fix_features(features_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Виправлення проблем в features
    
    Returns:
        (fixed_df, report)
    """
    logger.info("🔧 Виправлення features...")
    
    report = {
        'before': analyze_nulls(features_df, 'features_before'),
        'actions': [],
        'after': None
    }
    
    # 1. Видалити колонки з 100% пропусків
    null_pct = (features_df.isnull().sum() / len(features_df)) * 100
    cols_to_drop = null_pct[null_pct == 100].index.tolist()
    
    if cols_to_drop:
        logger.info(f"❌ Видалення колонок з 100% пропусків: {cols_to_drop}")
        features_df = features_df.drop(columns=cols_to_drop)
        report['actions'].append({
            'action': 'drop_100_percent_null_columns',
            'columns': cols_to_drop,
            'count': len(cols_to_drop)
        })
    
    # 2. Заповнити пропуски в макро-даних (forward fill)
    macro_cols = [col for col in features_df.columns if col.startswith('macro_')]
    
    for col in macro_cols:
        null_count_before = features_df[col].isnull().sum()
        if null_count_before > 0:
            logger.info(f"🔧 Заповнення пропусків в {col}: {null_count_before} nulls")
            
            # Forward fill
            features_df[col] = features_df[col].fillna(method='ffill')
            
            # Якщо залишилися пропуски на початку - backward fill
            features_df[col] = features_df[col].fillna(method='bfill')
            
            # Якщо все ще є пропуски - заповнити нулями
            features_df[col] = features_df[col].fillna(0)
            
            null_count_after = features_df[col].isnull().sum()
            
            report['actions'].append({
                'action': 'fill_macro_nulls',
                'column': col,
                'nulls_before': int(null_count_before),
                'nulls_after': int(null_count_after)
            })
    
    # 3. Заповнити інші пропуски нулями (якщо залишилися)
    remaining_nulls = features_df.isnull().sum().sum()
    if remaining_nulls > 0:
        logger.warning(f"⚠️ Залишилося {remaining_nulls} пропусків, заповнюємо нулями")
        features_df = features_df.fillna(0)
        report['actions'].append({
            'action': 'fill_remaining_nulls_with_zero',
            'count': int(remaining_nulls)
        })
    
    report['after'] = analyze_nulls(features_df, 'features_after')
    
    logger.info(f"✅ Features виправлено: {report['before']['shape']} -> {report['after']['shape']}")
    
    return features_df, report


def analyze_target(targets_df: pd.DataFrame) -> dict:
    """Аналіз якості таргета"""
    target_col = [col for col in targets_df.columns if col != 'ticker'][0]
    target_values = targets_df[target_col]
    
    return {
        'column': target_col,
        'shape': targets_df.shape,
        'statistics': {
            'count': int(target_values.count()),
            'mean': float(target_values.mean()),
            'std': float(target_values.std()),
            'min': float(target_values.min()),
            'max': float(target_values.max()),
            'median': float(target_values.median()),
            'q25': float(target_values.quantile(0.25)),
            'q75': float(target_values.quantile(0.75))
        },
        'quality': {
            'unique_values': int(target_values.nunique()),
            'zero_count': int((target_values == 0).sum()),
            'zero_percentage': round(((target_values == 0).sum() / len(target_values)) * 100, 2),
            'null_count': int(target_values.isnull().sum())
        }
    }


def main():
    """Головна функція"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Виправлення проблем з якістю даних')
    parser.add_argument(
        '--batch-dir',
        type=str,
        default='data/colab/accumulated/test_ticker_amd_target_return_1d_ep5_iter5',
        help='Шлях до директорії з batch даними'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Шлях для збереження виправлених даних (за замовчуванням - той самий)'
    )
    parser.add_argument(
        '--backup',
        action='store_true',
        help='Створити backup перед виправленням'
    )
    
    args = parser.parse_args()
    
    batch_dir = Path(args.batch_dir)
    output_dir = Path(args.output_dir) if args.output_dir else batch_dir
    
    logger.info("=" * 80)
    logger.info("🔧 ВИПРАВЛЕННЯ ЯКОСТІ ДАНИХ")
    logger.info("=" * 80)
    logger.info(f"📁 Batch: {batch_dir}")
    logger.info(f"📁 Output: {output_dir}")
    
    # Завантажити дані
    features_path = batch_dir / 'features.parquet'
    targets_path = batch_dir / 'targets.parquet'
    
    if not features_path.exists():
        logger.error(f"❌ Файл не знайдено: {features_path}")
        return
    
    if not targets_path.exists():
        logger.error(f"❌ Файл не знайдено: {targets_path}")
        return
    
    logger.info(f"📥 Завантаження features з {features_path}...")
    features_df = pd.read_parquet(features_path)
    logger.info(f"✅ Features завантажено: {features_df.shape}")
    
    logger.info(f"📥 Завантаження targets з {targets_path}...")
    targets_df = pd.read_parquet(targets_path)
    logger.info(f"✅ Targets завантажено: {targets_df.shape}")
    
    # Backup
    if args.backup:
        backup_dir = batch_dir / 'backup_before_fix'
        backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"💾 Створення backup в {backup_dir}...")
        features_df.to_parquet(backup_dir / 'features.parquet')
        targets_df.to_parquet(backup_dir / 'targets.parquet')
        logger.info("✅ Backup створено")
    
    # Виправити features
    fixed_features, features_report = fix_features(features_df)
    
    # Аналізувати targets
    logger.info("📊 Аналіз targets...")
    targets_report = analyze_target(targets_df)
    
    # Зберегти виправлені дані
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"💾 Збереження виправлених features в {output_dir}...")
    fixed_features.to_parquet(output_dir / 'features.parquet')
    logger.info("✅ Features збережено")
    
    # Targets не змінюємо, просто копіюємо якщо output_dir інший
    if output_dir != batch_dir:
        logger.info(f"💾 Копіювання targets в {output_dir}...")
        targets_df.to_parquet(output_dir / 'targets.parquet')
        logger.info("✅ Targets скопійовано")
    
    # Зберегти звіт
    report = {
        'timestamp': datetime.now().isoformat(),
        'batch_dir': str(batch_dir),
        'output_dir': str(output_dir),
        'features': features_report,
        'targets': targets_report
    }
    
    report_path = output_dir / 'data_quality_fix_report.json'
    logger.info(f"📝 Збереження звіту в {report_path}...")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("✅ Звіт збережено")
    
    # Підсумок
    logger.info("")
    logger.info("=" * 80)
    logger.info("📊 ПІДСУМОК")
    logger.info("=" * 80)
    logger.info(f"Features:")
    logger.info(f"  До:    {features_report['before']['shape']}, nulls: {features_report['before']['null_percentage']}%")
    logger.info(f"  Після: {features_report['after']['shape']}, nulls: {features_report['after']['null_percentage']}%")
    logger.info(f"  Дії:   {len(features_report['actions'])}")
    logger.info("")
    logger.info(f"Targets:")
    logger.info(f"  Shape: {targets_report['shape']}")
    logger.info(f"  Нулів: {targets_report['quality']['zero_count']} ({targets_report['quality']['zero_percentage']}%)")
    logger.info(f"  Унікальних: {targets_report['quality']['unique_values']}")
    logger.info("")
    logger.info("✅ Виправлення завершено!")
    logger.info(f"📋 Детальний звіт: {report_path}")


if __name__ == '__main__':
    main()
