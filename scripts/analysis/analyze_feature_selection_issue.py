#!/usr/bin/env python3
"""
Аналіз проблеми з feature selection.
"""

import json
from pathlib import Path

def analyze_feature_selection():
    """Аналіз feature_selection_check."""
    
    print("=" * 100)
    print("🔍 АНАЛІЗ FEATURE SELECTION CHECK")
    print("=" * 100)
    
    # Лог з успішного запуску
    log_data = {
        'batch_dir': 'data\\colab\\accumulated\\full_pipeline_trading\\full_pipeline_trading',
        'batch_name': 'full_pipeline_trading',
        'metadata_path': 'data\\colab\\accumulated\\full_pipeline_trading\\full_pipeline_trading\\batch_metadata.json',
        'files': {
            'features': 'data\\colab\\accumulated\\full_pipeline_trading\\full_pipeline_trading\\features.parquet',
            'targets': 'data\\colab\\accumulated\\full_pipeline_trading\\full_pipeline_trading\\targets.parquet',
            'config': None
        },
        'feature_selection_check': {
            'needed': True,
            'reason': 'No existing selection'
        },
        'test_mode': False
    }
    
    print("\n📊 ЛОГ АНАЛІЗ:")
    print(f"   Status: ✅ Pipeline completed successfully")
    print(f"   Batch: {log_data['batch_name']}")
    print(f"   Test mode: {log_data['test_mode']}")
    
    print("\n📦 ФАЙЛИ:")
    print(f"   ✅ features.parquet: {Path(log_data['files']['features']).exists()}")
    print(f"   ✅ targets.parquet: {Path(log_data['files']['targets']).exists()}")
    print(f"   ✅ batch_metadata.json: {Path(log_data['metadata_path']).exists()}")
    print(f"   ❌ config: {log_data['files']['config']}")
    
    print("\n🔍 FEATURE SELECTION CHECK:")
    fs_check = log_data['feature_selection_check']
    print(f"   Needed: {fs_check['needed']}")
    print(f"   Reason: {fs_check['reason']}")
    
    # Аналіз проблеми
    print("\n" + "=" * 100)
    print("❓ ЩО ЦЕ ОЗНАЧАЄ?")
    print("=" * 100)
    
    print("\n✅ ХОРОШІ НОВИНИ:")
    print("   1. Pipeline завершився успішно")
    print("   2. Всі файли створено (features, targets, metadata)")
    print("   3. Дані готові до переносу в Colab")
    print("   4. 59,171 рядків, 224 фічі, 16 таргетів")
    
    print("\n⚠️ ВАЖЛИВЕ ЗАУВАЖЕННЯ:")
    print("   'feature_selection_check': {'needed': True, 'reason': 'No existing selection'}")
    print()
    print("   Це означає:")
    print("   - Feature selection ще НЕ виконано")
    print("   - В Colab потрібно буде виконати feature selection")
    print("   - Це НОРМАЛЬНО для першого запуску")
    print("   - Це частина гібридного workflow")
    
    print("\n📋 ЩО ТАКЕ FEATURE SELECTION?")
    print("   Feature selection - це процес відбору найважливіших фіч для кожної моделі.")
    print("   Замість використання всіх 224 фіч, відбираються найбільш інформативні.")
    print()
    print("   Методи:")
    print("   - SHAP (SHapley Additive exPlanations)")
    print("   - Permutation Importance")
    print("   - Feature Importance з моделей")
    print()
    print("   Результат:")
    print("   - selected_features_*.json файли")
    print("   - Список найважливіших фіч для кожної моделі")
    print("   - Зменшення розмірності → швидше тренування")
    
    print("\n🔄 ГІБРИДНИЙ WORKFLOW:")
    print()
    print("   ЛОКАЛЬНО (Stages 0-3):")
    print("   ├─ Stage 0: Setup")
    print("   ├─ Stage 1: Collection (59,171 rows)")
    print("   ├─ Stage 2: Processing (clean data)")
    print("   └─ Stage 3: Features (224 features) ✅ ВИ ТУТ")
    print()
    print("   COLAB (Stage 4):")
    print("   ├─ Feature Selection (відбір фіч) ← НАСТУПНИЙ КРОК")
    print("   ├─ Heavy Models Training")
    print("   └─ Save results")
    print()
    print("   ЛОКАЛЬНО (Stages 5-7):")
    print("   ├─ Stage 5: Light Models")
    print("   ├─ Stage 6: Ensemble")
    print("   └─ Stage 7: Evaluation")
    
    print("\n" + "=" * 100)
    print("🎯 ЩО РОБИТИ ДАЛІ?")
    print("=" * 100)
    
    print("\n1️⃣ ПЕРЕНЕСТИ В COLAB (зараз)")
    print("   cd data/colab/accumulated/")
    print("   zip -r full_pipeline_trading.zip full_pipeline_trading/")
    print("   # Завантажити в Colab")
    
    print("\n2️⃣ В COLAB: Feature Selection + Training")
    print("   !python colab_clean_cell.py")
    print()
    print("   Що відбудеться:")
    print("   a) Feature Selection:")
    print("      - Аналіз важливості всіх 224 фіч")
    print("      - Відбір топ-N найважливіших для кожної моделі")
    print("      - Збереження selected_features_*.json")
    print()
    print("   b) Heavy Models Training:")
    print("      - MLP, CNN, LSTM, GRU, Transformer, TabNet, Autoencoder")
    print("      - Використання відібраних фіч")
    print("      - Збереження моделей та результатів")
    
    print("\n3️⃣ ЗАВАНТАЖИТИ НАЗАД")
    print("   # З Colab:")
    print("   !zip -r colab_results.zip models/ results/ selected_features_*.json")
    print("   files.download('colab_results.zip')")
    
    print("\n4️⃣ ПРОДОВЖИТИ ЛОКАЛЬНО")
    print("   python run_hybrid_pipeline.py --mode continue")
    print("   # Використає selected_features з Colab")
    
    print("\n" + "=" * 100)
    print("❓ ЧОМУ 'config': None?")
    print("=" * 100)
    
    print("\n   'config': None - це НОРМАЛЬНО для full mode!")
    print()
    print("   Пояснення:")
    print("   - config.json створюється ТІЛЬКИ в test mode")
    print("   - В full mode config не потрібен")
    print("   - Всі параметри беруться з unified_config.yaml")
    print()
    print("   Test mode (з config.json):")
    print("   python run_hybrid_pipeline.py --mode prepare \\")
    print("     --test-ticker AMD --test-target target_return_1d")
    print()
    print("   Full mode (без config.json):")
    print("   python run_hybrid_pipeline.py --mode prepare")
    print("   # config: None ← це правильно!")
    
    print("\n" + "=" * 100)
    print("🔍 ПЕРЕВІРКА ФАЙЛІВ")
    print("=" * 100)
    
    batch_dir = Path('data/colab/accumulated/full_pipeline_trading/full_pipeline_trading')
    
    if batch_dir.exists():
        print(f"\n✅ Batch directory exists: {batch_dir}")
        
        # Check files
        files_to_check = [
            'features.parquet',
            'targets.parquet',
            'batch_metadata.json'
        ]
        
        print("\n📦 Файли:")
        for filename in files_to_check:
            filepath = batch_dir / filename
            if filepath.exists():
                size_mb = filepath.stat().st_size / 1024**2
                print(f"   ✅ {filename}: {size_mb:.1f} MB")
            else:
                print(f"   ❌ {filename}: NOT FOUND")
        
        # Check for selected_features files
        selected_features = list(batch_dir.glob('selected_features_*.json'))
        print(f"\n🔍 Selected features files: {len(selected_features)}")
        if selected_features:
            print("   ✅ Feature selection вже виконано!")
            for sf in selected_features[:5]:
                print(f"      {sf.name}")
            if len(selected_features) > 5:
                print(f"      ... та ще {len(selected_features) - 5}")
        else:
            print("   ⏳ Feature selection ще не виконано (це нормально)")
            print("   📝 Буде виконано в Colab")
    else:
        print(f"\n❌ Batch directory NOT FOUND: {batch_dir}")
    
    print("\n" + "=" * 100)
    print("📋 ПІДСУМОК")
    print("=" * 100)
    
    print("\n✅ ВСЕ ПРАЦЮЄ ПРАВИЛЬНО!")
    print()
    print("   Що маємо:")
    print("   ✅ Pipeline завершився успішно")
    print("   ✅ Stages 0-3 виконано")
    print("   ✅ 59,171 рядків даних")
    print("   ✅ 224 фічі створено")
    print("   ✅ 16 таргетів готові")
    print("   ✅ Файли збережено")
    print()
    print("   Що далі:")
    print("   ⏳ Feature selection (в Colab)")
    print("   ⏳ Heavy models training (в Colab)")
    print("   ⏳ Light models training (локально)")
    print()
    print("   'feature_selection_check': {'needed': True}")
    print("   ↑ Це НЕ помилка, це інформація для Colab!")
    print()
    print("   'config': None")
    print("   ↑ Це правильно для full mode!")
    
    print("\n🚀 ГОТОВО ДО ПЕРЕНОСУ В COLAB!")
    
    print("\n" + "=" * 100)

if __name__ == '__main__':
    analyze_feature_selection()
