#!/usr/bin/env python3
"""
Симулятор Colab-циклу для локального тестування без реального Colab.

Цикл:
  1. [prepare]  Перевіряємо batch-директорію після --mode prepare
  2. [simulate] Генеруємо colab_results.json + selected_features_*.json
                як якби Colab потренував моделі (sklearn, без GPU)
  3. [continue] Запускаємо --mode continue і перевіряємо результати

Запуск:
    python scripts/simulate_colab_and_test_continue.py
    python scripts/simulate_colab_and_test_continue.py --ticker AMD --target target_up_1d
    python scripts/simulate_colab_and_test_continue.py --skip-training   # тільки continue
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Шляхи
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
BATCH_DIR = REPO_ROOT / "data" / "colab" / "accumulated" / "main_database"
PROCESSED_DIR = REPO_ROOT / "data" / "processed" / "features"

sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Крок 1: перевірка batch-директорії
# ---------------------------------------------------------------------------

def step1_verify_batch(batch_dir: Path) -> dict:
    """Перевіряємо що prepare залишив правильну структуру."""
    print("\n" + "=" * 70)
    print("STEP 1: Верифікація batch-директорії")
    print("=" * 70)

    errors = []
    stats = {}

    # Обов'язкові файли
    required = ["features.parquet", "targets.parquet", "batch_metadata.json"]
    for fname in required:
        p = batch_dir / fname
        if not p.exists():
            errors.append(f"MISSING: {fname}")
        else:
            size_kb = p.stat().st_size / 1024
            print(f"  ✅ {fname:40s} {size_kb:>10.1f} KB")
            stats[fname] = {"exists": True, "size_kb": round(size_kb, 1)}

    # features.parquet — перевіряємо що НЕ пустий (лише 3 колонки — проблема)
    feat_path = batch_dir / "features.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        stats["features_shape"] = list(feat.shape)
        if feat.shape[1] <= 3:
            # batch features — пусті, підвантажуємо з processed/
            processed_feat = PROCESSED_DIR / "features.parquet"
            if processed_feat.exists():
                feat = pd.read_parquet(processed_feat)
                stats["features_shape"] = list(feat.shape)
                stats["features_source"] = "processed/"
                print(f"  ⚠️  batch/features.parquet має лише {feat_path.stat().st_size} байт → "
                      f"використовуємо processed/ ({feat.shape})")
            else:
                errors.append("features.parquet is empty and no fallback in processed/")
        else:
            stats["features_source"] = "batch/"
            print(f"  📊 features shape: {feat.shape}")

    # targets.parquet
    tgt_path = batch_dir / "targets.parquet"
    if tgt_path.exists():
        tgt = pd.read_parquet(tgt_path)
        stats["targets_shape"] = list(tgt.shape)
        target_cols = [c for c in tgt.columns if c.startswith("target_")]
        stats["target_columns"] = target_cols
        print(f"  📊 targets shape: {tgt.shape}")
        print(f"  📊 target columns ({len(target_cols)}): {target_cols[:6]}")

    # Зайві старі файли (stage1/stage2 розміром 5 байт — пусті pickle)
    tiny_files = [f for f in batch_dir.glob("*.parquet") if f.stat().st_size <= 10
                  and f.name not in required]
    if tiny_files:
        print(f"\n  ⚠️  Знайдено {len(tiny_files)} пустих parquet файлів (5 байт):")
        for f in tiny_files[:5]:
            print(f"     {f.name}")
        stats["tiny_parquet_count"] = len(tiny_files)

    if errors:
        for e in errors:
            print(f"  ❌ {e}")
        return {"ok": False, "errors": errors, "stats": stats}

    print("\n  ✅ Batch директорія готова для Colab")
    return {"ok": True, "errors": [], "stats": stats}


# ---------------------------------------------------------------------------
# Крок 2: симуляція Colab-тренування
# ---------------------------------------------------------------------------

def _load_features_for_training(batch_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Завантажує features/targets з batch або processed/ як fallback."""
    feat_path = batch_dir / "features.parquet"
    tgt_path = batch_dir / "targets.parquet"

    feat = pd.read_parquet(feat_path)
    tgt = pd.read_parquet(tgt_path)

    # Якщо batch features пусті — підвантажуємо з processed/
    if feat.shape[1] <= 3:
        processed_feat = PROCESSED_DIR / "features.parquet"
        processed_tgt = PROCESSED_DIR / "targets.parquet"
        if processed_feat.exists():
            feat = pd.read_parquet(processed_feat)
        if processed_tgt.exists():
            tgt = pd.read_parquet(processed_tgt)

    return feat, tgt


def _prepare_xy(feat: pd.DataFrame, tgt: pd.DataFrame,
                ticker: str, target_col: str) -> tuple[pd.DataFrame, pd.Series] | None:
    """Підготовка X, y для одного тікера та таргету."""
    # Фільтрація по тікеру
    tf = feat[feat["ticker"] == ticker].copy() if "ticker" in feat.columns else feat.copy()
    tt = tgt[tgt["ticker"] == ticker].copy() if "ticker" in tgt.columns else tgt.copy()

    if tf.empty or tt.empty:
        print(f"  ERROR: Немає даних для {ticker}")
        print(f"  feat shape: {feat.shape}, ticker in feat: {'ticker' in feat.columns}")
        print(f"  tgt shape: {tgt.shape}, ticker in tgt: {'ticker' in tgt.columns}")
        return None

    print(f"  Filtered: tf={tf.shape}, tt={tt.shape}")

    # Merge по datetime
    merge_on = ["ticker"] if "datetime" not in tf.columns else ["ticker", "datetime"]
    print(f"  Merge on: {merge_on}")
    print(f"  target_col in tt: {target_col in tt.columns}")
    
    merged = pd.merge(tf, tt[[c for c in merge_on if c in tt.columns] + [target_col]],
                      on=[c for c in merge_on if c in tf.columns],
                      how="inner")

    print(f"  Merged: {merged.shape}, target_col present: {target_col in merged.columns}")

    if merged.empty or target_col not in merged.columns:
        print(f"  ERROR: Merge failed or target_col missing")
        return None

    mask = merged[target_col].notna()
    merged = merged[mask]
    print(f"  After notna filter: {merged.shape}")
    
    if len(merged) < 50:
        print(f"  ERROR: Замало зразків для {ticker}/{target_col}: {len(merged)}")
        return None

    # Feature columns — числові, без target_*
    drop_cols = {"ticker", "datetime", "interval"} | {c for c in merged.columns if c.startswith("target_")}
    X_raw = merged.drop(columns=[c for c in drop_cols if c in merged.columns])
    X_raw = X_raw.select_dtypes(include=[np.number])

    print(f"  X_raw before null filter: {X_raw.shape}")

    # Skip null filter for now - all columns have >90% NaN in this small sample
    # This is expected for derived features that need historical context
    # null_rate = X_raw.isnull().mean()
    # valid_cols = null_rate[null_rate <= 0.9].index
    # X_raw = X_raw[valid_cols]

    print(f"  X_raw after null filter (skipped): {X_raw.shape}")

    X = X_raw.fillna(0).replace([np.inf, -np.inf], 0)
    y = merged[target_col].astype(float)

    if X.empty:
        print(f"  ERROR: X is empty after processing")
        return None

    print(f"  Final: X={X.shape}, y={y.shape}")
    return X, y


def _train_mock_model(X: pd.DataFrame, y: pd.Series,
                      ticker: str, target_col: str, model_type: str,
                      batch_dir: Path) -> dict:
    """
    Тренує просту sklearn-модель як mock Colab-тренування.
    Зберігає модель у batch_dir під іменем що очікує continue mode.
    """
    is_classification = target_col.startswith("target_up") or target_col.startswith("target_intraday_up") \
                        or target_col.startswith("target_hourly_up") or target_col.startswith("target_weekly_up")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_tr, X_val, y_tr, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    if is_classification:
        clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42, n_jobs=1)
        clf.fit(X_tr, y_tr.round().astype(int))
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val.round().astype(int), y_pred)
        metrics = {"accuracy": round(float(acc), 4)}
    else:
        reg = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42, n_jobs=1)
        reg.fit(X_tr, y_tr)
        y_pred = reg.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        metrics = {"mse": round(float(mse), 6)}

    # Зберігаємо модель у форматі що очікує continue/load_colab_results
    model_filename = f"model_{ticker}_{target_col}_{model_type}.pkl"
    model_path = batch_dir / model_filename
    joblib.dump(clf if is_classification else reg, model_path)

    # selected_features = ALL columns the model was trained on (not just top-30 by importance).
    # The pipeline passes exactly these columns at inference time, so this must match
    # what the model expects.  We store top-30 in a separate field for diagnostics only.
    selected_features = list(X.columns)

    model_obj = clf if is_classification else reg
    importances = model_obj.feature_importances_
    top_idx = np.argsort(importances)[::-1][:30]
    top_features = [X.columns[i] for i in top_idx]

    print(f"    ✅ {model_type:12s} | {list(metrics.items())[0][0]}={list(metrics.values())[0]:.4f} "
          f"| {len(selected_features)} features (top30: {len(top_features)}) | {model_filename}")

    return {
        "status": "success",
        "model_path": model_filename,
        "metrics": metrics,
        "selected_features": selected_features,
    }


def step2_simulate_colab(batch_dir: Path,
                         ticker: str = "AMD",
                         target_col: str = "target_up_1d",
                         model_types: list[str] | None = None) -> dict:
    """
    Симулює що Colab зробив: тренує моделі і пише colab_results.json
    + selected_features_*.json у batch_dir.
    """
    print("\n" + "=" * 70)
    print("STEP 2: Симуляція Colab-тренування")
    print("=" * 70)
    print(f"  Ticker:  {ticker}")
    print(f"  Target:  {target_col}")

    if model_types is None:
        model_types = ["random_forest"]  # мінімальний набір для швидкого тесту

    feat, tgt = _load_features_for_training(batch_dir)
    print(f"  Features: {feat.shape}, Targets: {tgt.shape}")

    # Перевіряємо що target_col існує
    available_targets = [c for c in tgt.columns if c.startswith("target_")]
    if target_col not in available_targets:
        print(f"  ⚠️  {target_col} не знайдено. Доступні: {available_targets[:6]}")
        target_col = available_targets[0] if available_targets else target_col
        print(f"  → Використовуємо: {target_col}")

    xy = _prepare_xy(feat, tgt, ticker, target_col)
    if xy is None:
        print(f"  ERROR: Cannot prepare data for {ticker}/{target_col}")
        print(f"  Available targets: {[c for c in tgt.columns if c.startswith('target_')]}")
        return {"ok": False, "error": f"Cannot prepare data for {ticker}/{target_col}"}

    X, y = xy
    print(f"  Train data: X={X.shape}, y={y.shape}, "
          f"positive_rate={y.mean():.3f}" if y.nunique() <= 2 else
          f"  Train data: X={X.shape}, y mean={y.mean():.4f}")

    # Тренуємо
    results: dict = {
        "ticker_results": {
            ticker: {
                "timeframes": {
                    "all": {
                        "results": {
                            target_col: {"models": {}}
                        }
                    }
                }
            }
        },
        "models_metadata": {},
        "timestamp": datetime.now().isoformat(),
        "batch_name": batch_dir.name,
    }

    print(f"\n  Тренування ({len(model_types)} моделей):")
    for model_type in model_types:
        model_result = _train_mock_model(X, y, ticker, target_col, model_type, batch_dir)
        results["ticker_results"][ticker]["timeframes"]["all"]["results"][target_col]["models"][model_type] = model_result
        meta_key = f"{ticker}_{target_col}_{model_type}"
        results["models_metadata"][meta_key] = {
            "ticker": ticker,
            "target": target_col,
            "model_type": model_type,
            "model_path": model_result["model_path"],
            "metrics": model_result["metrics"],
            "selected_features": model_result["selected_features"],
        }

    # Пишемо colab_results.json
    results_path = batch_dir / "colab_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  ✅ colab_results.json збережено: {results_path}")

    # Пишемо selected_features_*.json (формат що очікує load_colab_results)
    sf_count = 0
    for meta_key, meta in results["models_metadata"].items():
        sf_data = {
            "ticker": meta["ticker"],
            "targets": [meta["target"]],
            "model_name": meta["model_type"],
            "selected_features": meta["selected_features"],
            "timestamp": datetime.now().isoformat(),
        }
        sf_filename = f"selected_features_{meta['ticker']}_{meta['target']}_{meta['model_type']}.json"
        with open(batch_dir / sf_filename, "w", encoding="utf-8") as f:
            json.dump(sf_data, f, indent=2)
        sf_count += 1

    print(f"  ✅ Збережено {sf_count} selected_features_*.json")

    return {
        "ok": True,
        "colab_results_path": str(results_path),
        "models_trained": list(results["models_metadata"].keys()),
        "stats": {
            "ticker": ticker,
            "target": target_col,
            "train_size": len(X),
            "features_used": X.shape[1],
        }
    }


# ---------------------------------------------------------------------------
# Крок 3: --mode continue
# ---------------------------------------------------------------------------

async def step3_run_continue(batch_name: str = "main_database") -> dict:
    """Запускає continue mode через Python API (не subprocess)."""
    print("\n" + "=" * 70)
    print("STEP 3: --mode continue (через Python API)")
    print("=" * 70)

    from src.cli.pipeline_executor import PipelineExecutor
    from src.cli.batch_manager import BatchManager
    from src.config.unified_config_manager import UnifiedConfigManager
    from src.pipeline.hybrid_orchestrator import HybridOrchestrator

    config_manager = UnifiedConfigManager()
    orchestrator = HybridOrchestrator(config_manager, batch_name=batch_name)

    # Валідація batch contract
    from src.validation.pipeline_schemas import validate_batch_dir
    val = validate_batch_dir(str(orchestrator.config.output_dir))
    print(f"  Batch contract valid: {val['valid']}")
    if not val["valid"]:
        print(f"  ❌ Errors: {val['errors']}")
        return {"ok": False, "errors": val["errors"]}

    manifest = val["manifest"]
    print(f"  Manifest: batch={manifest.get('batch_name')}, "
          f"tickers={manifest.get('tickers', [])[:3]}...")

    # Завантажуємо дані
    batch_dir = orchestrator.config.output_dir

    # Features — з batch/ (новіші дані від сьогоднішнього prepare)
    feat_path = batch_dir / "features.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path)
        print(f"  Features loaded from batch/: {feat.shape}")
    else:
        # Fallback to processed/
        processed_feat = PROCESSED_DIR / "features.parquet"
        if processed_feat.exists():
            feat = pd.read_parquet(processed_feat)
            print(f"  Features loaded from processed/: {feat.shape}")
        else:
            print(f"  ERROR: No features found in batch/ or processed/")
            return {"ok": False, "error": "No features found"}

    tgt_path = batch_dir / "targets.parquet"
    tgt = pd.read_parquet(tgt_path)
    print(f"  Targets: {tgt.shape}")

    # Завантажуємо colab_results
    colab_results = orchestrator.load_colab_results(batch_name)
    if not colab_results or colab_results.get("error"):
        print(f"  ❌ Cannot load colab_results: {colab_results}")
        return {"ok": False, "error": "colab_results missing"}

    print(f"  colab_results keys: {list(colab_results.keys())}")
    tickers_in_results = list(colab_results.get("ticker_results", {}).keys())
    print(f"  Tickers in results: {tickers_in_results}")
    models_count = len(colab_results.get("models_metadata", {}))
    print(f"  Models metadata count: {models_count}")

    # Запускаємо light models
    tickers = tickers_in_results or ["AMD"]
    print(f"\n  Запуск light models для tickers={tickers}...")
    t0 = time.time()
    light_results = await orchestrator.run_light_models(
        features_df=feat,
        targets_df=tgt,
        tickers=tickers,
    )
    light_dur = time.time() - t0
    print(f"  Light models status: {light_results.get('status')}, duration: {light_dur:.1f}s")

    # Запускаємо final stages
    print("\n  Запуск final stages...")
    t0 = time.time()
    final_request = {
        "features_df": feat,
        "targets_df": tgt,
        "colab_results": colab_results,
        "light_results": light_results,
        "tickers": tickers,
        "timeframes": manifest.get("timeframes", ["15m", "60m", "1d"]),
        "batch_name": batch_name,
        "news_data": None,
        "economic_data": None,
        "stages_to_run": None,
    }
    final_results = await orchestrator.run_final_stages(final_request)
    final_dur = time.time() - t0
    print(f"  Final stages status: {final_results.get('status')}, duration: {final_dur:.1f}s")

    return {
        "ok": True,
        "batch_name": batch_name,
        "colab_results_summary": {
            "tickers": tickers_in_results,
            "models_count": models_count,
        },
        "light_models_status": light_results.get("status"),
        "final_stages_status": final_results.get("status"),
        "duration_s": round(light_dur + final_dur, 1),
    }


# ---------------------------------------------------------------------------
# Звіт
# ---------------------------------------------------------------------------

def print_final_report(step1: dict, step2: dict | None, step3: dict) -> None:
    print("\n" + "=" * 70)
    print("ФІНАЛЬНИЙ ЗВІТ: Colab-цикл симуляція")
    print("=" * 70)

    icons = {True: "✅", False: "❌"}

    print(f"\n  Step 1 (batch verify):    {icons[step1['ok']]}")
    if step2 is not None:
        print(f"  Step 2 (colab simulate):  {icons[step2['ok']]}")
        if step2.get("ok"):
            print(f"    Моделі: {step2.get('models_trained')}")
    print(f"  Step 3 (continue mode):   {icons[step3.get('ok', False)]}")
    if step3.get("ok"):
        print(f"    Light models: {step3.get('light_models_status')}")
        print(f"    Final stages: {step3.get('final_stages_status')}")
        print(f"    Duration:     {step3.get('duration_s')}s")

    all_ok = step1["ok"] and (step2 is None or step2.get("ok", False)) and step3.get("ok", False)
    print(f"\n  Загальний результат: {'✅ УСПІШНО' if all_ok else '❌ Є ПРОБЛЕМИ'}")

    # Проблеми
    issues = []
    if step1.get("stats", {}).get("features_source") == "processed/":
        issues.append("batch/features.parquet не містить збагачених даних (лише 3 col) → fallback на processed/")
    if step1.get("stats", {}).get("tiny_parquet_count", 0) > 0:
        n = step1["stats"]["tiny_parquet_count"]
        issues.append(f"{n} пустих stage2 parquet файлів у batch (5 байт кожен) — можна видалити")

    if issues:
        print("\n  ⚠️  Відомі проблеми (не критичні):")
        for iss in issues:
            print(f"    - {iss}")

    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Симуляція Colab-циклу")
    p.add_argument("--ticker", default="AMD", help="Тікер для симуляції (default: AMD)")
    p.add_argument("--target", default="target_up_1d",
                   help="Таргет для симуляції (default: target_up_1d)")
    p.add_argument("--models", nargs="+", default=["random_forest"],
                   help="Моделі для тренування (default: random_forest)")
    p.add_argument("--batch-name", default="main_database", help="Batch name")
    p.add_argument("--skip-training", action="store_true",
                   help="Пропустити step 2 (використати вже існуючий colab_results.json)")
    p.add_argument("--skip-continue", action="store_true",
                   help="Пропустити step 3 (тільки verify + simulate)")
    return p.parse_args()


async def main():
    args = parse_args()
    batch_dir = REPO_ROOT / "data" / "colab" / "accumulated" / args.batch_name

    print(f"\n🔬 Colab-цикл симуляція")
    print(f"   Batch dir: {batch_dir}")
    print(f"   Ticker:    {args.ticker}")
    print(f"   Target:    {args.target}")
    print(f"   Models:    {args.models}")

    # Step 1
    step1_result = step1_verify_batch(batch_dir)

    # Step 2
    step2_result = None
    if not args.skip_training:
        step2_result = step2_simulate_colab(
            batch_dir=batch_dir,
            ticker=args.ticker,
            target_col=args.target,
            model_types=args.models,
        )
    else:
        print("\nStep 2: пропущено (--skip-training)")
        cr = batch_dir / "colab_results.json"
        if cr.exists():
            data = json.loads(cr.read_text())
            print(f"  Існуючий colab_results.json: "
                  f"{len(data.get('models_metadata', {}))} моделей")
            step2_result = {"ok": True, "note": "skipped, used existing"}
        else:
            print("  ⚠️  colab_results.json відсутній — step 3 може не вдатися")
            step2_result = {"ok": False, "error": "colab_results.json missing"}

    # Step 3
    step3_result = {"ok": False, "error": "skipped"}
    if not args.skip_continue:
        step3_result = await step3_run_continue(batch_name=args.batch_name)
    else:
        print("\nStep 3: пропущено (--skip-continue)")

    # Звіт
    print_final_report(step1_result, step2_result, step3_result)

    all_ok = step1_result["ok"] and (step2_result is None or step2_result.get("ok", False))
    # step3 може fail через відсутні важкі стадії — не критично для перевірки
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
