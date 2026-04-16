#!/usr/bin/env python3
"""
Скрипт для запуску гібридного пайплайну.

Використання:
    python run_hybrid_pipeline.py --mode local      # Тільки локальна частина
    python run_hybrid_pipeline.py --mode full       # Повний пайплайн
    python run_hybrid_pipeline.py --mode prepare    # Підготовка для Colab
"""

import asyncio
import argparse
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Налаштування кодування для Windows консолі
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# Додаємо кореневу директорію проєкту до sys.path.
# Всі внутрішні імпорти проєкту повинні використовувати префікс 'src.',
# наприклад: from src.config.tickers import ...
# Це запобігає конфліктам імен при динамічному завантаженні модулів.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.config.unified_config_manager import UnifiedConfigManager
from src.pipeline.hybrid_orchestrator import HybridOrchestrator
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)


def validate_arguments(args, config_manager):
    """
    ✅ Валідація аргументів командного рядка.
    
    Перевіряє:
    - Тікери існують у конфігу
    - Таргети існують у конфігу
    - Моделі існують у конфігу
    - Режим та інші параметри коректні
    """
    errors = []
    warnings = []
    
    # Отримуємо доступні тікери
    assets_config = config_manager.get_config('assets') or {}
    active_preset = assets_config.get('active_preset', 'default_volatile')
    available_tickers = (
        assets_config
        .get('presets', {})
        .get(active_preset, {})
        .get('tickers', [])
    )
    
    # Отримуємо доступні таргети
    targets_config = config_manager.get_config('targets') or {}
    available_targets = list(targets_config.keys())
    
    # Отримуємо доступні моделі
    models_config = config_manager.get_config('models') or {}
    available_models = list(models_config.get('model_definitions', {}).keys())
    if not available_models:
        available_models = list(models_config.get('available', {}).keys())
    
    # Валідація --test-ticker
    if args.test_ticker:
        if args.test_ticker not in available_tickers:
            errors.append(
                f"❌ Тікер '{args.test_ticker}' не знайдено. "
                f"Доступні: {', '.join(available_tickers)}"
            )
    
    # Валідація --test-target
    if args.test_target:
        # Нормалізуємо назву таргету
        target_name = args.test_target
        if not target_name.startswith('target_'):
            target_name = f"target_{target_name}"
        
        if target_name not in available_targets:
            errors.append(
                f"❌ Таргет '{args.test_target}' не знайдено. "
                f"Доступні: {', '.join(available_targets[:5])}... (всього {len(available_targets)})"
            )
    
    # Валідація --test-model
    if args.test_model:
        if args.test_model not in available_models:
            errors.append(
                f"❌ Модель '{args.test_model}' не знайдено. "
                f"Доступні: {', '.join(available_models)}"
            )
    
    # Валідація --mode
    valid_modes = ['local', 'full', 'prepare', 'light', 'continue']
    if args.mode not in valid_modes:
        errors.append(
            f"❌ Режим '{args.mode}' невалідний. "
            f"Доступні: {', '.join(valid_modes)}"
        )
    
    # Валідація --max-iterations
    if args.max_iterations < 1:
        errors.append(f"❌ --max-iterations повинен бути >= 1, отримано {args.max_iterations}")
    
    # Валідація --epochs
    if args.epochs < 1:
        errors.append(f"❌ --epochs повинен бути >= 1, отримано {args.epochs}")
    
    # Валідація --stages (тільки для continue mode)
    if args.stages:
        if args.mode != 'continue':
            warnings.append(
                f"⚠️ --stages використовується тільки в режимі 'continue', "
                f"але вказано режим '{args.mode}'. Параметр буде ігнорований."
            )
        else:
            # Перевіряємо що етапи в діапазоні 4-7
            for stage in args.stages:
                if stage < 4 or stage > 7:
                    errors.append(
                        f"❌ Етап {stage} невалідний. "
                        f"Доступні етапи: 4 (light models), 5 (prediction), 6 (trading), 7 (evaluation)"
                    )
    
    # Виводимо помилки
    if errors:
        logger.error("❌ ПОМИЛКИ ВАЛІДАЦІЇ:")
        for error in errors:
            logger.error(f"  {error}")
        sys.exit(1)
    
    # Виводимо попередження
    if warnings:
        logger.warning("⚠️ ПОПЕРЕДЖЕННЯ:")
        for warning in warnings:
            logger.warning(f"  {warning}")
    
    logger.info("✅ Аргументи валідовані успішно")


async def main():
    parser = argparse.ArgumentParser(description='Гібридний пайплайн для трейдингу')
    parser.add_argument(
        '--mode',
        choices=['local', 'full', 'prepare', 'light', 'continue'],
        default='full',
        help='Режим виконання'
    )
    parser.add_argument(
        '--tickers',
        nargs='+',
        help='Список тікерів (опціонально, використовується з конфігу)'
    )
    parser.add_argument(
        '--timeframes',
        nargs='+',
        help='Список таймфреймів (опціонально, використовується з конфігу)'
    )
    parser.add_argument(
        '--batch-name',
        help='Назва пакету для Colab'
    )
    parser.add_argument(
        '--no-accumulate',
        action='store_true',
        help='Не акумулювати дані, створити новий batch'
    )
    parser.add_argument(
        '--force-training',
        action='store_true',
        help='Форсувати тренування навіть якщо немає нових даних'
    )
    parser.add_argument(
        '--force-feature-selection',
        action='store_true',
        help='Форсувати повторний відбір ознак навіть якщо вони вже існують'
    )
    parser.add_argument(
        '--skip-colab',
        action='store_true',
        help='Пропустити виконання в Colab, виконати тільки локальну частину'
    )
    parser.add_argument(
        '--test-ticker',
        help='Тільки один тікер для швидкого тесту (наприклад, AMD)'
    )
    parser.add_argument(
        '--test-target',
        help='Тільки один таргет для швидкого тесту (наприклад, target_return_15m)'
    )
    parser.add_argument(
        '--test-model',
        help='Тільки один тип моделі для швидкого тесту (наприклад, mlp, xgb, cnn)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help='Максимальна кількість ітерацій для tree моделей (default: 100, для швидкого тесту: 10-20)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Кількість епох для важких моделей (default: 50, для швидкого тесту: 5-10)'
    )
    # ✅ NEW: STAGE SELECTION
    parser.add_argument(
        '--stages',
        nargs='+',
        type=int,
        help='Конкретні етапи для виконання (наприклад, --stages 4 5 6 7 або --stages 6). Працює тільки в continue mode.'
    )
    
    args = parser.parse_args()
    
    # ✅ ІНІЦІАЛІЗАЦІЯ CONFIG MANAGER ДЛЯ ВАЛІДАЦІЇ
    config_manager = UnifiedConfigManager()
    
    # ✅ ПЕРЕВІРКА ВЕРСІЙНОСТІ
    try:
        from src.core.version_checker import VersionChecker
        version_checker = VersionChecker(config_manager)
        version_ok, version_result = version_checker.check_all()
        
        if not version_ok:
            logger.error("❌ Версійні вимоги не задоволені:")
            for issue in version_result.get('package_issues', []):
                logger.error(f"  {issue}")
            logger.warning("⚠️ Продовжуємо виконання, але можливі проблеми...")
    except Exception as e:
        logger.warning(f"⚠️ Не вдалося перевірити версійність: {e}")
    
    # ✅ ВАЛІДАЦІЯ АРГУМЕНТІВ
    validate_arguments(args, config_manager)
    
    # ✅ ГНУЧКА ЛОГІКА BATCH NAME
    # Якщо вказані детальні параметри → генеруємо специфічний batch_name
    # Якщо нічого не вказано → використовуємо main_database (всі показники)
    if not args.batch_name:
        # Перевіряємо чи є хоч один тестовий параметр
        has_test_params = args.test_ticker or args.test_target or args.test_model
        
        if has_test_params:
            # ТЕСТОВИЙ РЕЖИМ: генеруємо специфічний batch_name
            parts = []
            if args.test_ticker:
                ticker_clean = args.test_ticker.lower().replace('ticker_', '')
                parts.append(f"ticker_{ticker_clean}")
            if args.test_target:
                target_clean = args.test_target.lower().replace('target_', '')
                parts.append(f"target_{target_clean}")
            if args.test_model:
                parts.append(f"model_{args.test_model.lower()}")
            
            # ✅ FIX: В режимі continue шукаємо існуючий batch замість створення нового
            if args.mode == 'continue':
                # Шукаємо існуючий batch з такими параметрами (без ep/iter)
                base_pattern = "test_" + "_".join(parts) if parts else "manual_run"
                base_pattern = base_pattern.replace('target_target_', 'target_').replace('ticker_ticker_', 'ticker_')
                
                # Шукаємо всі батчі що відповідають патерну
                accumulated_dir = Path("data/colab/accumulated")
                if accumulated_dir.exists():
                    # Шукаємо тільки директорії (не файли)
                    matching_batches = [p for p in accumulated_dir.glob(f"{base_pattern}_ep*_iter*") if p.is_dir()]
                    if matching_batches:
                        # Використовуємо найновіший batch (за датою модифікації)
                        latest_batch = max(matching_batches, key=lambda p: p.stat().st_mtime)
                        args.batch_name = latest_batch.name
                        logger.info(f"🔄 CONTINUE MODE: Використовуємо існуючий batch: {args.batch_name}")
                    else:
                        logger.warning(f"⚠️ Не знайдено існуючого batch для {base_pattern}_ep*_iter*")
                        logger.warning(f"⚠️ Створюємо новий batch з дефолтними параметрами")
                        parts.append(f"ep{args.epochs}")
                        parts.append(f"iter{args.max_iterations}")
                        generated_name = "test_" + "_".join(parts) if parts else "manual_run"
                        args.batch_name = generated_name.replace('target_target_', 'target_').replace('ticker_ticker_', 'ticker_')
                else:
                    logger.error(f"❌ Директорія {accumulated_dir} не існує")
                    parts.append(f"ep{args.epochs}")
                    parts.append(f"iter{args.max_iterations}")
                    generated_name = "test_" + "_".join(parts) if parts else "manual_run"
                    args.batch_name = generated_name.replace('target_target_', 'target_').replace('ticker_ticker_', 'ticker_')
            else:
                # Режим prepare: створюємо новий batch з вказаними параметрами
                parts.append(f"ep{args.epochs}")
                parts.append(f"iter{args.max_iterations}")
                generated_name = "test_" + "_".join(parts) if parts else "manual_run"
                args.batch_name = generated_name.replace('target_target_', 'target_').replace('ticker_ticker_', 'ticker_')
            
            logger.info(f"🧪 ТЕСТОВИЙ РЕЖИМ: batch_name = {args.batch_name}")
        else:
            # ПОВНИЙ РЕЖИМ: використовуємо main_database (всі тікери, всі таргети)
            args.batch_name = "main_database"
            logger.info(f"📦 ПОВНИЙ РЕЖИМ: batch_name = {args.batch_name} (всі показники)")

    # ✅ ЗБЕРЕЖЕННЯ RUNTIME ПАРАМЕТРІВ для Colab
    # Визначаємо папку для збереження runtime_params
    # ✅ FIX: Завжди зберігаємо в папку, що відповідає batch_name
    batch_dir = Path(f"data/colab/accumulated/{args.batch_name}")
    
    # Створюємо папку якщо не існує
    batch_dir.mkdir(parents=True, exist_ok=True)
    runtime_params_path = batch_dir / "runtime_params.json"
    
    runtime_params = {
        "description": "Runtime parameters for both local and Colab training",
        "last_updated": datetime.now().isoformat(),
        "mode": args.mode,
        "test_mode": {
            "enabled": bool(args.test_ticker or args.test_target or args.test_model),
            "test_ticker": args.test_ticker,
            "test_target": args.test_target,
            "test_model": args.test_model,
            "reduced_epochs": args.epochs if args.epochs != 50 else None,
            "max_iterations": args.max_iterations
        },
        "training": {
            "tickers": args.tickers or [],
            "timeframes": args.timeframes or [],
            "force_training": args.force_training,
            "force_feature_selection": args.force_feature_selection,
            "skip_colab": args.skip_colab,
            "accumulate": not args.no_accumulate
        },
        "models": {
            "max_iterations": args.max_iterations,
            "epochs": args.epochs
        },
        "batch": {
            "batch_name": args.batch_name,
            "no_accumulate": args.no_accumulate
        }
    }
    
    # Зберігаємо параметри в правильну папку
    with open(runtime_params_path, 'w') as f:
        json.dump(runtime_params, f, indent=2)
    logger.info(f"✅ Runtime параметри збережено: {runtime_params_path}")

    runtime_params_central = Path("data/runtime/runtime_params.json")
    runtime_params_central.parent.mkdir(parents=True, exist_ok=True)
    with open(runtime_params_central, 'w') as f:
        json.dump(runtime_params, f, indent=2)
    logger.info(f"✅ Runtime параметри також збережено: {runtime_params_central}")

    if runtime_params["test_mode"]["enabled"]:
        logger.info("🧪 ТЕСТОВИЙ РЕЖИМ АКТИВОВАНО:")
        if args.test_ticker:
            logger.info(f"   Тікер: {args.test_ticker}")
        if args.test_target:
            logger.info(f"   Таргет: {args.test_target}")
        if args.test_model:
            logger.info(f"   Модель: {args.test_model}")
        if args.epochs != 50:
            logger.info(f"   Епохи: {args.epochs}")
        if args.max_iterations != 100:
            logger.info(f"   Ітерації: {args.max_iterations}")
    
    # ✅ Встановлюємо максимальну кількість ітерацій
    if args.max_iterations != 100:
        os.environ['MAX_ITERATIONS'] = str(args.max_iterations)
        logger.info(f"⚡ MAX_ITERATIONS: {args.max_iterations} (default: 100)")
    
    # Ініціалізація (використовуємо згенерований batch_name для ізоляції)
    logger.info(f"🚀 Запуск гібридного пайплайну (batch: {args.batch_name})...")
    orchestrator = HybridOrchestrator(config_manager, batch_name=args.batch_name)
    
    # Отримуємо тікери та таймфрейми
    tickers = args.tickers
    if args.test_ticker:
        tickers = [args.test_ticker]
        logger.info(f"🧪 FAST MODE: Використовуємо тільки тікер {args.test_ticker}")
    elif not tickers:
        # Використовуємо з конфігу
        assets_config = config_manager.get_config('assets') or {}
        active_preset = assets_config.get('active_preset')
        tickers = (
            assets_config
            .get('presets', {})
            .get(active_preset, {})
            .get('tickers', [])
        )
    
    timeframes = args.timeframes
    if not timeframes:
        system_config = config_manager.get_config('system') or {}
        timeframes = system_config.get('timeframes', ['15m', '1h', '1d'])
    
    logger.info(f"📊 Тікери: {tickers}")
    logger.info(f"⏱️ Таймфрейми: {timeframes}")
    
    # Виконання відповідно до режиму
    if args.mode == 'local':
        logger.info("💻 Режим: Локальний пайплайн (етапи 0-3)")
        results = await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )
        
    elif args.mode == 'light':
        logger.info("💡 Режим: Тренування легких моделей")
        # Спочатку запускаємо локальний пайплайн
        local_results = await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )
        
        # Потім тренуємо легкі моделі
        features_df = local_results['results'].get('features_df')
        targets_df = local_results['results'].get('targets_df')
        
        if features_df is not None and targets_df is not None:
            results = await orchestrator.run_light_models(
                features_df=features_df,
                targets_df=targets_df,
                tickers=tickers
            )
        else:
            logger.error("❌ Немає даних для тренування")
            return
    
    elif args.mode == 'prepare':
        logger.info("📦 Режим: Підготовка пакету для Colab")
        # Спочатку запускаємо локальний пайплайн
        local_results = await orchestrator.run_local_pipeline(
            tickers=tickers,
            timeframes=timeframes
        )
        
        logger.info(f"local_results type: {type(local_results)}")
        logger.info(f"local_results keys: {local_results.keys() if isinstance(local_results, dict) else 'N/A'}")
        
        # Підготовлюємо пакет
        results_dict = local_results.get('results', {})
        logger.info(f"results_dict keys: {results_dict.keys() if isinstance(results_dict, dict) else 'N/A'}")
        
        # ✅ FIX: Використовуємо вже розділені features_df та targets_df з run_local_pipeline
        features_df = results_dict.get('features_df')
        targets_df = results_dict.get('targets_df')
        
        logger.info(f"features_df type: {type(features_df)}")
        logger.info(f"features_df is None: {features_df is None}")
        if features_df is not None:
            logger.info(f"features_df shape: {features_df.shape}")
            logger.info(f"features_df empty: {features_df.empty}")
            # Перевіряємо чи є target колонки в features
            target_cols_in_features = [c for c in features_df.columns if c.startswith('target_')]
            if target_cols_in_features:
                logger.warning(f"⚠️ Features містить {len(target_cols_in_features)} target колонок: {target_cols_in_features}")
            else:
                logger.info(f"✅ Features НЕ містить target колонок")
        
        if features_df is not None and not features_df.empty:
            results = orchestrator.prepare_colab_batch(
                features_df=features_df,
                targets_df=targets_df,
                tickers=tickers,
                timeframes=timeframes,
                batch_name=args.batch_name,
                accumulate=not args.no_accumulate
            )
            
            # Додаємо статус до результатів
            results['status'] = 'prepare_complete'
            
            # Виводимо інструкції
            instructions = orchestrator._generate_colab_instructions(results)
            print("\n" + "="*80)
            print(instructions)
            print("="*80 + "\n")
            
            print(f"📂 Дані для Colab підготовлено в: {results['batch_dir']}/")
            print(f"📝 Конфіг: {results['batch_dir']}/config.json")
        else:
            logger.error("❌ Немає даних для підготовки пакету")
            return
    
    elif args.mode == 'full':
        logger.info("🌐 Режим: Повний гібридний пайплайн")
        results = await orchestrator.run_full_hybrid_pipeline(
            tickers=tickers,
            timeframes=timeframes,
            accumulate=not args.no_accumulate,
            force_training=args.force_training,
            skip_colab=args.skip_colab,
            force_feature_selection=args.force_feature_selection
        )
        
        # Обробка різних статусів
        if results.get('status') == 'no_new_data_no_results':
            logger.warning("\n" + "⚠️ "*40)
            logger.warning("⚠️ НЕМАЄ НОВИХ РЯДКІВ ДАНИХ І НЕМАЄ РЕЗУЛЬТАТІВ COLAB")
            logger.warning("⚠️ "*40)
            logger.info(f"📋 Повідомлення: {results.get('message')}")
            return
        
        elif results.get('status') == 'completed_without_colab':
            logger.info("\n" + "✅ "*40)
            logger.info("✅ ПАЙПЛАЙН ЗАВЕРШЕНО БЕЗ COLAB")
            logger.info("✅ Використано тільки локальні моделі")
            logger.info("✅ "*40)
            logger.info(f"📋 Повідомлення: {results.get('message')}")
            return
        
        elif results.get('status') == 'paused_for_colab':
            logger.info("\n" + "* "*40)
            logger.info("PAUSE: Feature selection + Heavy models training in Google Colab")
            logger.info("* "*40)
            
            # Виводимо інструкції для Colab
            if 'colab_instructions' in results:
                print("\n" + "="*80)
                print(results['colab_instructions'])
                print("="*80 + "\n")
            
            print("  7. Результати автоматично збережуться:")
            print("      - selected_features_{model}.json (для легких моделей)")
            print("      - colab_results_*.json (результати важких моделей)")
            print("\n⏳ Очікуваний час: 2-6 годин (залежно від GPU)")
            print("\n📁 Дані для Colab збережено в:")
            batch_info = results.get('batch_info') or results.get('colab_batch', {})
            if batch_info:
                print(f"   {batch_info.get('batch_dir', 'data/colab/accumulated/main_database')}")
            print("\n🔴 "*40)
            
            # ⏸️ ПАУЗА: Очікування користувача
            print("\n" + "⏸️ "*40)
            print("⏸️ ПАУЗА: Очікування завершення Colab (вибір фіч + тренування)...")
            print("⏸️ "*40)
            user_input = input("\n✅ Натисніть ENTER після завершення Colab для продовження... ")
            print("\n" + "▶️ "*40)
            print("▶️ Продовжуємо пайплайн...")
            print("▶️ "*40)
            
            print("\n🔄 Продовжуємо пайплайн...")
            
            # Завантажуємо результати з Colab
            batch_name = batch_info.get('batch_name', 'main_database')
            batch_dir = orchestrator.output_dir / batch_name
            
            # Гнучкий пошук selected_features файлів (в підпапці або в кореневій папці)
            selected_features_files = list(batch_dir.glob("selected_features_*.json"))
            if not selected_features_files:
                # Якщо не знайдено в підпапці, шукаємо в кореневій папці accumulated/
                root_dir = orchestrator.output_dir
                selected_features_files = list(root_dir.glob("selected_features_*.json"))
                if selected_features_files:
                    logger.info(f"✅ Знайдено {len(selected_features_files)} файлів selected_features в кореневій папці")
                else:
                    logger.warning("⚠️ Не знайдено файлів selected_features_*.json")
                    logger.warning("⚠️ Можливо Colab не завершив вибір фіч")
                    logger.info("💡 Перевірте чи всі клітинки в Colab виконались успішно")
            else:
                logger.info(f"✅ Знайдено {len(selected_features_files)} файлів selected_features в підпапці")
            
            # Автоматично запускаємо скрипт аккумулятора, якщо є colab_results_*.json файли
            colab_results_files = list(batch_dir.glob("colab_results_*.json"))
            if colab_results_files and len(colab_results_files) > 0:
                logger.info(f"📦 Знайдено {len(colab_results_files)} файлів colab_results")

            logger.info(f"📥 Завантаження фінальних результатів з Colab: {batch_name}")
            colab_results = orchestrator.load_colab_results(batch_name)
            
            if colab_results.get('status') == 'not_found':
                logger.error(f"❌ Результати не знайдено для пакету: {batch_name}")
                logger.info("💡 Переконайтесь, що файл colab_results_summary.json або colab_results.json існує в папці пакету")
                return
            
            if colab_results.get('status') == 'not_found':
                logger.error(f"❌ Результати не знайдено для пакету: {batch_name}")
                logger.info("💡 Переконайтесь, що файл colab_results_summary.json або colab_results.json існує в папці пакету")
                return
            
            # Завантажуємо features та targets з кешу
            features_path = batch_dir / "features.parquet"
            targets_path = batch_dir / "targets.parquet"
            
            if not features_path.exists() or not targets_path.exists():
                logger.error(f"❌ Не знайдено файли даних в пакеті: {batch_name}")
                return
            
            import pandas as pd
            features_df = pd.read_parquet(features_path)
            targets_df = pd.read_parquet(targets_path)
            
            logger.info(f"✅ Завантажено features з кешу: {features_df.shape}")
            logger.info(f"✅ Завантажено targets з кешу: {targets_df.shape}")
            
            # 🎯 ЕТАП 5: Тренування легких моделей на вибраних фічах
            logger.info("\n" + "💡 "*40)
            logger.info("💡 ЕТАП 5: Тренування легких моделей")
            logger.info("💡 (використовуємо фічі, вибрані в Colab)")
            logger.info("💡 "*40)
            
            light_results = await orchestrator.run_light_models_with_selected_features(
                features_df=features_df,
                targets_df=targets_df,
                batch_name=batch_name,
                tickers=tickers,
                force=args.force_training  # ✅ NEW: Передаємо force
            )
            
            if light_results.get('status') == 'error':
                logger.error(f"❌ Помилка тренування легких моделей: {light_results.get('message')}")
                logger.info("💡 Продовжуємо з результатами важких моделей...")
            else:
                logger.info(f"✅ Легкі моделі протреновані: {light_results.get('models_trained', 0)} моделей")
            
            # Запускаємо етапи 6-7 (Prediction, Trading, Evaluation)
            logger.info("\n🎯 ЕТАПИ 6-7: Запуск аналізу результатів (Prediction, Trading, Evaluation)...")
            final_results = await orchestrator.run_final_stages(
                features_df=features_df,
                targets_df=targets_df,
                colab_results=colab_results,
                tickers=tickers,
                timeframes=timeframes,
                batch_name=batch_name  # ✅ ADD batch_name
            )
            
            # Оновлюємо results
            results.update(final_results)
            results['light_models'] = light_results
            results['status'] = 'full_pipeline_complete'
    
    elif args.mode == 'continue':
        logger.info("🔄 Режим: Продовження після Colab")
        
        # ✅ FIX: Читаємо batch_name з централізованого runtime_params.json, потім з legacy src/config для зворотної сумісності
        if not args.batch_name:
            runtime_params_path = Path("data/runtime/runtime_params.json")
            if not runtime_params_path.exists():
                runtime_params_path = Path("src/config/runtime_params.json")

            if runtime_params_path.exists():
                with open(runtime_params_path, 'r') as f:
                    loaded_params = json.load(f)
                    args.batch_name = loaded_params.get('batch', {}).get('batch_name', 'main_database')
                    logger.info(f"📝 Завантажено batch_name з {runtime_params_path}: {args.batch_name}")
            else:
                logger.error("❌ Для режиму 'continue' потрібно вказати --batch-name або мати runtime_params.json")
                return
        
        batch_dir = orchestrator.output_dir / args.batch_name
        logger.info(f"📂 Використовуємо batch: {args.batch_name}")
        
        # [FIX] СУПЕР-ВАЖЛИВО: СПОЧАТКУ АКУМУЛЮЄМО, ПОТІМ ЗАВАНТАЖУЄМО
        # Автоматично запускаємо скрипт аккумулятора, якщо є colab_results_*.json файли
        # Гнучкий пошук: спочатку в підпапці, потім в кореневій папці
        colab_results_files = list(batch_dir.glob("colab_results_*.json"))
        if not colab_results_files:
            # Шукаємо в кореневій папці accumulated/
            root_dir = orchestrator.output_dir
            colab_results_files = list(root_dir.glob("colab_results_*.json"))
            if colab_results_files:
                logger.info(f"📦 Знайдено {len(colab_results_files)} файлів colab_results в кореневій папці, запуск аккумулятора...")
                batch_dir = root_dir  # Оновлюємо batch_dir для подальшого використання
        
        if colab_results_files and len(colab_results_files) > 0:
            if batch_dir == orchestrator.output_dir:
                logger.info(f"📦 Знайдено {len(colab_results_files)} файлів colab_results в кореневій папці")
            else:
                logger.info(f"📦 Знайдено {len(colab_results_files)} файлів colab_results в підпапці")
        
        # Тепер завантажуємо вже акумульовані результати з Colab
        logger.info(f"📥 Завантаження фінальних результатів з Colab: {args.batch_name}")
        colab_results = orchestrator.load_colab_results(args.batch_name)
        
        if colab_results.get('status') == 'not_found':
            logger.error(f"❌ Результати не знайдено для пакету: {args.batch_name}")
            logger.info("💡 Переконайтесь, що файл colab_results_summary.json або colab_results.json існує в папці пакету")
            return
        
        # Завантажуємо features та targets з пакету
        # Гнучкий пошук: спочатку в підпапці, потім в кореневій папці
        features_path = batch_dir / "features.parquet"
        targets_path = batch_dir / "targets.parquet"
        
        if not features_path.exists() or not targets_path.exists():
            # Шукаємо в кореневій папці accumulated/
            root_dir = orchestrator.output_dir
            features_path = root_dir / "features.parquet"
            targets_path = root_dir / "targets.parquet"
            if features_path.exists() and targets_path.exists():
                logger.info(f"✅ Знайдено файли даних в кореневій папці")
                batch_dir = root_dir  # Оновлюємо batch_dir
            else:
                logger.error(f"❌ Не знайдено файли даних ні в підпапці, ні в кореневій папці")
                return
        
        import pandas as pd
        features_df = pd.read_parquet(features_path)
        targets_df = pd.read_parquet(targets_path)
        
        logger.info(f"✅ Завантажено features: {features_df.shape}")
        logger.info(f"✅ Завантажено targets: {targets_df.shape}")
        
        # Отримуємо список таргетів
        target_cols = [c for c in targets_df.columns if c.startswith('target_')]
        if args.test_target:
            if args.test_target in target_cols:
                target_cols = [args.test_target]
                logger.info(f"🧪 FAST MODE: Використовуємо тільки таргет {args.test_target}")
            else:
                logger.warning(f"⚠️ Таргет {args.test_target} не знайдено. Використовуємо всі {len(target_cols)} таргетів.")
        
        # 🎯 ЕТАП 4-7: Тренування легких моделей та запуск аналізу
        # ✅ NEW: Підтримка вибору конкретних етапів
        if args.stages:
            logger.info(f"🎯 Вибрано етапи: {args.stages}")
            stages_to_run = args.stages
        else:
            # За замовчуванням: всі етапи 4-7
            stages_to_run = None  # run_final_stages використає [5, 6, 7] за замовчуванням
            logger.info("🎯 Запуск всіх етапів (4-7)")
        
        # Якщо етап 4 вибрано, запускаємо тренування легких моделей окремо
        if args.stages and 4 in args.stages:
            logger.info("\n" + "💡 " * 40)
            logger.info("💡 ЕТАП 4: Тренування легких моделей")
            logger.info("💡 (використовуємо фічі, вибрані в Colab)")
            logger.info(f"💡 Тікери: {tickers}, Таргети: {target_cols}")
            logger.info("💡 " * 40)
            
            light_results = await orchestrator.run_light_models_with_selected_features(
                features_df=features_df,
                targets_df=targets_df,
                batch_name=args.batch_name,
                tickers=tickers
            )
            
            if light_results.get('status') == 'error':
                logger.error(f"❌ Помилка тренування легких моделей: {light_results.get('message')}")
            else:
                logger.info(f"✅ Легкі моделі протреновані: {light_results.get('models_trained', 0)} моделей")
        elif not args.stages:
            # Якщо етапи не вказані, запускаємо етап 4 за замовчуванням
            logger.info("\n" + "💡 " * 40)
            logger.info("💡 ЕТАП 4: Тренування легких моделей")
            logger.info("💡 (використовуємо фічі, вибрані в Colab)")
            logger.info(f"💡 Тікери: {tickers}, Таргети: {target_cols}")
            logger.info("💡 " * 40)
            
            light_results = await orchestrator.run_light_models_with_selected_features(
                features_df=features_df,
                targets_df=targets_df,
                batch_name=args.batch_name,
                tickers=tickers
            )
            
            if light_results.get('status') == 'error':
                logger.error(f"❌ Помилка тренування легких моделей: {light_results.get('message')}")
            else:
                logger.info(f"✅ Легкі моделі протреновані: {light_results.get('models_trained', 0)} моделей")

        # Запускаємо етапи 5-7 (або вибрані етапи)
        results = await orchestrator.run_final_stages(
            features_df=features_df,
            targets_df=targets_df,
            colab_results=colab_results,
            tickers=tickers,
            timeframes=timeframes,
            batch_name=args.batch_name,
            stages_to_run=stages_to_run  # ✅ NEW: Передаємо вибрані етапи
        )
    
    # Виводимо результати
    logger.info("✅ Виконання завершено")
    logger.info(f"📋 Статус: {results.get('status')}")
    
    if 'saved_files' in results:
        logger.info("📁 Збережені файли:")
        for key, path in results['saved_files'].items():
            logger.info(f"  - {key}: {path}")
    
    if 'metadata_path' in results:
        logger.info(f"📋 Метадані: {results['metadata_path']}")


if __name__ == '__main__':
    asyncio.run(main())
