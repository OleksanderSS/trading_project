# core/stages/stage_manager.py - Меnotджер роwithподandлу еandпandв with кешуванням

import hashlib
import json
import os
import pickle
from datetime import datetime
from typing import Any

import pandas as pd
from config.config_loader import load_yaml_config

# ВИПРАВЛЕНО: видаляємо імпорт видаленої функції
# from core.stages.stage_1_collectors_layer import run_stage_1_collect
from core.stages.stage_2_enrichment import run_stage_2_enrichment_fixed
from core.stages.stage_3_features import prepare_stage3_datasets
from core.stages.stage_4_benchmark import benchmark_all_models
from utils.colab_utils import colab_utils
from utils.logger import ProjectLogger

from utils.trading_calendar import TradingCalendar

logger = ProjectLogger.get_logger("TradingProjectLogger")

class StageManager:
    """Меnotджер еandпandв with кешуванням and роwithподandлом"""

    def __init__(self, base_path: str = "data/cache/stages"):
        self.base_path = base_path
        self.ensure_directories()

    def ensure_directories(self):
        """Створює директорandї for кешування"""
        dirs = [
            "stage1_raw",
            "stage2_enriched",
            "stage3_features",
            "stage4_models",
            "stage5_signals"
        ]
        for dir_name in dirs:
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)

    def get_cache_path(self, stage: str, params_hash: str) -> str:
        """Отримати шлях до кешу"""
        return os.path.join(self.base_path, stage, f"cache_{params_hash}.pkl")

    def get_params_hash(self, params: dict[str, Any]) -> str:
        """Геnotрує хеш параметрandв for кешування"""
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()[:16]

    def is_cache_valid(self, cache_path: str, max_age_hours: int = 24) -> bool:
        """Перевandряє чи дandйсний кеш"""
        if not os.path.exists(cache_path):
            return False

        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        return age.total_seconds() < max_age_hours * 3600

    def save_cache(self, cache_path: str, data: Any):
        """Зберandгає данand в кеш"""
        import os
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"[StageManager] Збережено кеш: {cache_path}")

    def load_cache(self, cache_path: str) -> Any:
        """Заванandжує данand with кешу"""
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"[StageManager] Заванandжено кеш: {cache_path}")
        return data

    def run_stage_1(self, debug_no_network: bool = False, force_refresh: bool = False) -> dict[str, Any]:
        """Етап 1: Збір даних"""
        # Import TICKERS and TIME_FRAMES for cache parameters
        from config.config import TICKERS, TIME_FRAMES

        # ВИПРАВЛЕНО: обчислюємо cache_path правильно
        params = {"tickers": list(TICKERS.keys()), "timeframes": TIME_FRAMES}
        params_hash = self.get_params_hash(params)
        cache_path = self.get_cache_path("stage1_raw", params_hash)

        # ВИПРАВЛЕНО: правильна перевірка кешу - різні TTL для різних типів даних
        if not force_refresh and os.path.exists(cache_path):
            try:
                cached_data = self.load_cache(cache_path)

                if cached_data and len(cached_data) > 0:
                    # Перевіряємо час кешу
                    cache_time = cached_data.get('_metadata', {}).get('last_update_time')
                    if cache_time:
                        from datetime import datetime
                        cache_age = datetime.now() - cache_time

                        # Різні TTL для різних типів даних
                        all_fresh = True
                        data_summary = {}

                        # Перевіряємо кожен тип даних окремо
                        for key, value in cached_data.items():
                            if key.startswith('_'):  # Пропускаємо метадані
                                continue

                            # Різні терміни актуальності з урахуванням таймфрейму
                            ttl_hours = 2  # За замовчуванням

                            if key in ['prices', 'prices_by_timeframe']:
                                # РІЗНА ЧАСТОТА ОНОВЛЕННЯ ДЛЯ РІЗНИХ ТАЙМФРЕЙМІВ
                                if isinstance(value, dict):  # prices_by_timeframe
                                    # Перевіряємо кожен таймфрейм окремо
                                    timeframes_ttl = {
                                        '5m': 0.17,    # 10 хвилин для 5-хвилинних даних
                                        '15m': 0.5,    # 30 хвилин для 15-хвилинних
                                        '60m': 1.0,    # 1 година для 60-хвилинних
                                        '1d': 12.0     # 12 годин для денних (раз на день достатньо)
                                    }
                                    # Використовуємо найкоротший TTL для всього словника
                                    ttl_hours = min(timeframes_ttl.values())
                                    logger.info(f"[Stage1] Using timeframe-specific TTL: {timeframes_ttl}")
                                else:
                                    ttl_hours = 0.5  # 30 хвилин для звичайних цін
                            elif key in ['news', 'newsapi', 'rss', 'google_news']:
                                ttl_hours = 4  # 4 години для новин - актуальні новини важливі
                            elif key in ['fred']:
                                ttl_hours = 6  # 6 годин для макро показників
                            elif key in ['crypto_prices']:
                                ttl_hours = 1  # 1 година для криптовалют
                            elif key in ['google_trends']:
                                ttl_hours = 12  # 12 годин для трендів
                            else:
                                ttl_hours = 2  # 2 години для інших даних

                            # Перевіряємо чи дані свіжі для цього типу
                            if cache_age.total_seconds() < ttl_hours * 3600:
                                if hasattr(value, 'shape') and value.shape[0] > 0:
                                    data_summary[key] = f"✅ {value.shape} (свіжий)"
                                elif isinstance(value, dict) and len(value) > 0:
                                    data_summary[key] = f"✅ dict with {len(value)} items (свіжий)"
                                elif isinstance(value, list) and len(value) > 0:
                                    data_summary[key] = f"✅ list with {len(value)} items (свіжий)"
                            else:
                                all_fresh = False  # Якщо хоча б один застарів - оновлюємо все
                                if hasattr(value, 'shape') and value.shape[0] > 0:
                                    data_summary[key] = f"⏰ {value.shape} (застарів на {cache_age.total_seconds()/3600:.1f}г)"
                                elif isinstance(value, dict) and len(value) > 0:
                                    data_summary[key] = f"⏰ dict with {len(value)} items (застарів на {cache_age.total_seconds()/3600:.1f}г)"
                                elif isinstance(value, list) and len(value) > 0:
                                    data_summary[key] = f"⏰ list with {len(value)} items (застарів на {cache_age.total_seconds()/3600:.1f}г)"

                        # ВИПРАВЛЕНО: Інтелектуальне оновлення - тільки застарілі дані
                        stale_data_types = []
                        fresh_data_types = []

                        for key, value in cached_data.items():
                            if key.startswith('_'):  # Пропускаємо метадані
                                continue

                            # Різні терміни актуальності з урахуванням таймфрейму
                            ttl_hours = 2  # За замовчуванням

                            if key in ['prices', 'prices_by_timeframe']:
                                # РІЗНА ЧАСТОТА ОНОВЛЕННЯ ДЛЯ РІЗНИХ ТАЙМФРЕЙМІВ
                                if isinstance(value, dict):  # prices_by_timeframe
                                    # Перевіряємо кожен таймфрейм окремо
                                    timeframes_ttl = {
                                        '5m': 0.17,    # 10 хвилин для 5-хвилинних даних
                                        '15m': 0.5,    # 30 хвилин для 15-хвилинних
                                        '60m': 1.0,    # 1 година для 60-хвилинних
                                        '1d': 12.0     # 12 годин для денних (раз на день достатньо)
                                    }
                                    # Використовуємо найкоротший TTL для всього словника
                                    ttl_hours = min(timeframes_ttl.values())
                                    logger.info(f"[Stage1] Using timeframe-specific TTL: {timeframes_ttl}")
                                else:
                                    ttl_hours = 0.5  # 30 хвилин для звичайних цін
                            elif key in ['news', 'newsapi', 'rss', 'google_news']:
                                ttl_hours = 4  # 4 години для новин - актуальні новини важливі
                            elif key in ['fred']:
                                ttl_hours = 6  # 6 годин для макро показників
                            elif key in ['crypto_prices']:
                                ttl_hours = 1  # 1 година для криптовалют
                            elif key in ['google_trends']:
                                ttl_hours = 12  # 12 годин для трендів
                            else:
                                ttl_hours = 2  # 2 години для інших даних

                            # Перевіряємо чи дані свіжі для цього типу
                            if cache_age.total_seconds() < ttl_hours * 3600:
                                if hasattr(value, 'shape') and value.shape[0] > 0:
                                    fresh_data_types.append(f"{key}: {value.shape}")
                                elif isinstance(value, dict) and len(value) > 0:
                                    fresh_data_types.append(f"{key}: dict with {len(value)} items")
                                elif isinstance(value, list) and len(value) > 0:
                                    fresh_data_types.append(f"{key}: list with {len(value)} items")
                            else:
                                stale_data_types.append(key)

                        # ВИПРАВЛЕНО: Якщо є свіжі дані, використовуємо їх + оновлюємо застарілі
                        if fresh_data_types and not stale_data_types:
                            logger.info("[StageManager] ✅ Всі дані свіжі, використовуємо кеш")
                            for item in fresh_data_types:
                                logger.info(f"  ✅ {item}")
                            return cached_data
                        elif fresh_data_types and stale_data_types:
                            logger.info(f"[StageManager] 🔄 Часткове оновлення: свіжі {len(fresh_data_types)}, застарілі {len(stale_data_types)}")
                            for item in fresh_data_types:
                                logger.info(f"  ✅ {item}")
                            for item in stale_data_types:
                                logger.info(f"  🔄 {item}")
                            # Продовжуємо з частковим оновленням
                        else:
                            logger.info("[StageManager] 🔄 Повне оновлення - всі дані застарілі")
                            for item in stale_data_types:
                                logger.info(f"  🔄 {item}")
                    else:
                        logger.warning("[StageManager] ⚠️ Кеш без часу, оновлюємо...")
                else:
                    logger.warning("[StageManager] ⚠️ Кеш порожній, оновлюємо...")

            except Exception as e:
                logger.error(f"[StageManager] ❌ Помилка завантаження кешу: {e}")
                logger.info("[StageManager] 🔄 Продовжуємо з новим завантаженням...")

        logger.info("[StageManager] Starting Stage 1: Data Collection")
        # ВИПРАВЛЕНО: передаємо актуальний час для last_update_time
        from datetime import datetime
        last_update_time = datetime.now()

        # ВИПРАВЛЕНО: створюємо collector один раз і використовуємо його
        from core.stages.stage_1_collectors_layer import IdealStage1Collector

        # Створюємо collector (якщо ще не створено)
        if not hasattr(self, '_stage1_collector'):
            self._stage1_collector = IdealStage1Collector(
                tickers=TICKERS,  # Це словник, не список!
                timeframes=TIME_FRAMES,
                use_free_data=True,
                enable_cache=True,
                cache_ttl_hours=24,
                last_update_time=last_update_time
            )

        # Використовуємо існуючий collector
        stage1_data = self._stage1_collector.run_stage_1(
            tickers=TICKERS,  # Це словник, не список!
            timeframes=TIME_FRAMES,
            use_free_data=True,
            enable_cache=True,
            cache_ttl_hours=24,
            last_update_time=last_update_time
        )

        # Додаємо метадані в кеш
        stage1_data['_metadata'] = {
            'last_update_time': last_update_time,
            'collection_time': datetime.now().isoformat()
        }

        self.save_cache(cache_path, stage1_data)
        return stage1_data

    def run_stage_2(self, stage1_data: dict[str, Any], force_refresh: bool = False,
                    tickers: dict | None = None, time_frames: list | None = None) -> tuple[Any, Any, Any]:
        """Еandп 2: Data Enrichment"""
        logger.info("[StageManager] DEBUG: Початок run_stage_2")

        # Use provided parameters or fall back to global constants
        if tickers is None:
            from config.config import TICKERS
            tickers = TICKERS
        if time_frames is None:
            from config.config import TIME_FRAMES
            time_frames = TIME_FRAMES

        logger.info(f"[StageManager] DEBUG: Using tickers: {list(tickers.keys())}")
        logger.info(f"[StageManager] DEBUG: Using time_frames: {time_frames}")

        params = {"stage1_keys": list(stage1_data.keys())}
        params_hash = self.get_params_hash(params)
        cache_path = self.get_cache_path("stage2_enriched", params_hash)

        logger.info(f"[StageManager] DEBUG: cache_path: {cache_path}")
        logger.info(f"[StageManager] DEBUG: force_refresh: {force_refresh}")

        if not force_refresh and self.is_cache_valid(cache_path):
            logger.info("[StageManager] Використовую кеш for Stage 2")
            cached_data = self.load_cache(cache_path)
            logger.info(f"[StageManager] DEBUG: Заванandжено with кешу: {type(cached_data)}")
            return cached_data

        logger.info("[StageManager] Запускаю Stage 2: Data Enrichment")
        import os
        # Використовуємо абсолютний шлях до конфandгурацandйного fileу
        config_path = "config/news_sources.yaml"
        if not os.path.isabs(config_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            config_path = os.path.join(project_root, config_path)

        config = load_yaml_config(config_path)
        keyword_dict = config.get("keywords", {})

        logger.info(f"[StageManager] DEBUG: keyword_dict keys: {list(keyword_dict.keys())}")

        logger.info("[StageManager] DEBUG: Starting run_stage_2_enrichment_fixed...")
        # [FIXED] Використовуємо нову функцію яка об'єднує всі дані з правильними таргетами
        merged_df, metadata = run_stage_2_enrichment_fixed(
            stage1_data=stage1_data,
            tickers=list(tickers.keys()) if isinstance(tickers, dict) else tickers,
            time_frames=time_frames
        )

        # Повертаємо в очікуваному форматі для сумісності
        raw_news = stage1_data.get("news", pd.DataFrame())
        pivots = metadata

        logger.info("[StageManager] DEBUG] run_stage_2_enrichment_fixed completed:")
        logger.info(f"[StageManager] DEBUG] merged_df shape: {merged_df.shape if hasattr(merged_df, 'shape') else 'N/A'}")
        logger.info(f"[StageManager] DEBUG] Target columns: {[col for col in merged_df.columns if 'target' in col.lower()] if hasattr(merged_df, 'columns') else 'N/A'}")

        logger.info("[StageManager] DEBUG: run_stage_2_enrich_optimized повернув:")
        logger.info(f"[StageManager] DEBUG: raw_news тип: {type(raw_news)}, роwithмandр: {len(raw_news) if hasattr(raw_news, '__len__') else 'N/A'}")
        logger.info(f"[StageManager] DEBUG: merged_df тип: {type(merged_df)}")
        if merged_df is not None:
            logger.info(f"[StageManager] DEBUG: merged_df роwithмandр: {merged_df.shape}")
            # Check for price columns in merged_df
            close_cols = [col for col in merged_df.columns if 'close' in col.lower()]
            logger.info(f"[StageManager] DEBUG: merged_df close columns: {len(close_cols)}")
            if close_cols:
                logger.info(f"[StageManager] DEBUG: Sample close cols: {close_cols[:3]}")
            else:
                logger.warning("[StageManager] DEBUG: No close columns found!")
                logger.info(f"[StageManager] DEBUG: All columns (first 20): {list(merged_df.columns)[:20]}")
        else:
            logger.warning("[StageManager] DEBUG: merged_df is None!")
        logger.info(f"[StageManager] DEBUG: pivots тип: {type(pivots)}, ключand: {list(pivots.keys()) if isinstance(pivots, dict) else 'N/A'}")

        # Зберandгаємо в кеш
        logger.info("[StageManager] DEBUG: Збереження в кеш...")
        self.save_cache(cache_path, (raw_news, merged_df, pivots))
        logger.info("[StageManager] DEBUG: Збережено в кеш")

        # Зберandгаємо Stage 2 данand в stages папку
        logger.info("[StageManager] DEBUG: Збереження Stage 2 в stages...")
        try:
            from pathlib import Path

            from config.config import PATHS

            stages_dir = Path(PATHS["data"]) / "stages"
            stages_dir.mkdir(parents=True, exist_ok=True)

            # Зберandгаємо merged_df
            if merged_df is not None and not merged_df.empty:
                stage2_path = stages_dir / "stage2_merged.parquet"

                # Виправляємо problemsнand колонки перед withбереженням
                for col in merged_df.columns:
                    if merged_df[col].dtype == 'object':
                        try:
                            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
                        except Exception as e:
                            logger.error(f"Error converting column {col} to numeric: {e}")
                            merged_df[col] = merged_df[col].astype(str)

                merged_df.to_parquet(stage2_path, index=False)
                logger.info(f"[StageManager] Saved stage2_merged.parquet: {merged_df.shape}")

            # Зберandгаємо pivots як JSON
            if pivots:
                import json
                pivots_path = stages_dir / "stage2_pivots.json"
                # Конвертуємо DataFrame в dict for JSON
                serializable_pivots = {}
                for interval, pivot in pivots.items():
                    if hasattr(pivot, 'to_dict'):
                        serializable_pivots[interval] = pivot.to_dict()
                    else:
                        serializable_pivots[interval] = pivot

                with open(pivots_path, 'w') as f:
                    json.dump(serializable_pivots, f, default=str)
                logger.info(f"[StageManager] Saved stage2_pivots.json: {list(pivots.keys())}")

        except Exception as e:
            logger.error(f"[StageManager] Error saving Stage 2: {e}")

        # Накопичуємо данand for Colab (перед видаленням колонок)
        logger.info("[StageManager] DEBUG: Виклик _accumulate_stage2_data...")
        self._accumulate_stage2_data(raw_news, merged_df, pivots)
        logger.info("[StageManager] DEBUG: _accumulate_stage2_data completed")

        return raw_news, merged_df, pivots

    def _accumulate_stage2_data(self, raw_news, merged_df, pivots):
        """Накопичує данand еandпу 2 for експорту в Colab"""
        logger.info("[StageManager] DEBUG: Початок функцandї накопичення")

        try:
            from pathlib import Path

            import pandas as pd

            # DEBUG: Перевandряємо вхandднand данand
            logger.info(f"[StageManager] DEBUG: raw_news тип: {type(raw_news)}, роwithмandр: {len(raw_news) if hasattr(raw_news, '__len__') else 'N/A'}")
            logger.info(f"[StageManager] DEBUG: merged_df тип: {type(merged_df)}")
            if merged_df is not None:
                logger.info(f"[StageManager] DEBUG: merged_df роwithмandр: {merged_df.shape}")
                logger.info(f"[StageManager] DEBUG: merged_df колонки: {list(merged_df.columns)[:10]}")
            else:
                logger.warning("[StageManager] DEBUG: merged_df is None")
                return

            # Перевandряємо чи є данand for накопичення
            if merged_df is None or merged_df.empty:
                logger.warning("[StageManager] Немає data for накопичення")
                return

            # ВАЛІДАЦІЯ: Перевandряємо наявнandсть критичних фandч
            critical_features = ['RSI', 'SMA', 'gap', 'target']
            missing_features = []

            for feature in critical_features:
                feature_cols = [col for col in merged_df.columns if feature.lower() in col.lower()]
                if len(feature_cols) == 0:
                    missing_features.append(feature)
                else:
                    logger.info(f"[StageManager] [OK] {feature}: {len(feature_cols)} columns found")

            if missing_features:
                logger.warning(f"[StageManager] [ERROR] Missing critical features: {missing_features}")
                logger.warning("[StageManager] [WARN] Accumulated dataset will be incomplete for ML!")
            else:
                logger.info("[StageManager] [OK] All critical features present for ML")

            # Шлях for накопичених data
            accumulated_dir = Path("data/colab/accumulated")
            logger.info(f"[StageManager] DEBUG: Створюю папку: {accumulated_dir}")
            accumulated_dir.mkdir(parents=True, exist_ok=True)
            logger.info("[StageManager] DEBUG: Папка створена успandшно")

            # Файл for накопичених data
            accumulated_file = accumulated_dir / "stage2_accumulated.parquet"
            logger.info(f"[StageManager] DEBUG: Файл for withбереження: {accumulated_file}")

            # Перевandряємо колонки for removing duplicates
            duplicate_cols = []
            if 'published_at' in merged_df.columns:
                duplicate_cols.append('published_at')
            if 'title' in merged_df.columns:
                duplicate_cols.append('title')
            elif 'url' in merged_df.columns:
                duplicate_cols.append('url')

            logger.info(f"[StageManager] DEBUG: Колонки for дублandкатandв: {duplicate_cols}")

            # Заванandжуємо andснуючand данand
            if accumulated_file.exists():
                logger.info("[StageManager] DEBUG: Файл andснує, forванandжую...")
                existing_df = pd.read_parquet(accumulated_file)
                logger.info(f"[StageManager] DEBUG: Існуючand данand: {existing_df.shape}")
                # Об'єднуємо with новими даними
                combined_df = pd.concat([existing_df, merged_df], ignore_index=True)
                logger.info(f"[StageManager] DEBUG: Об'єднанand данand: {combined_df.shape}")
                # Видаляємо дублandкати якщо є вandдповandднand колонки
                if duplicate_cols:
                    before_dedup = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=duplicate_cols, keep='last')
                    after_dedup = len(combined_df)
                    logger.info(f"[StageManager] DEBUG: Видалено дублandкатandв: {before_dedup - after_dedup}")
                logger.info("[StageManager] Об'єднано with andснуючими даними")
            else:
                combined_df = merged_df
                logger.info(f"[StageManager] Створено новий file накопичення, роwithмandр: {combined_df.shape}")

            # Зберandгаємо накопиченand данand
            logger.info("[StageManager] DEBUG: Зберandгаю file...")

            # Виправляємо problemsнand колонки перед withбереженням
            for col in combined_df.columns:
                if combined_df[col].dtype == 'object':
                    try:
                        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                    except Exception as e:
                        logger.error(f"Error converting column {col} to numeric: {e}")
                        combined_df[col] = combined_df[col].astype(str)

            combined_df.to_parquet(accumulated_file)
            logger.info("[StageManager] DEBUG: File saved successfully")
            logger.info(f"[StageManager] Accumulated {len(combined_df)} rows of stage 2 data")

        except Exception as e:
            logger.error(f"[StageManager] Data accumulation error: {e}")
            import traceback
            logger.error(f"[StageManager] Traceback: {traceback.format_exc()}")

    def run_stage_3(self, stage2_data: tuple[Any, Any, Any], force_refresh: bool = False) -> tuple[Any, Any, Any, Any]:
        """Етап 3: Feature Engineering"""
        # ВИПРАВЛЕНО - правильний тип параметра: Tuple[Any, Any, Any]
        raw_news, enhanced_data, metadata = stage2_data

        params = {"enhanced_shape": enhanced_data.shape, "columns": list(enhanced_data.columns)[:10]}
        params_hash = self.get_params_hash(params)
        cache_path = self.get_cache_path("stage3_features", params_hash)

        if not force_refresh and self.is_cache_valid(cache_path):
            logger.info("[StageManager] Використовую кеш for Stage 3")
            return self.load_cache(cache_path)

        logger.info("[StageManager] Запускаю Stage 3: Feature Engineering")
        calendar = TradingCalendar.from_year(2025)

        stage1_data = {}  # Можна додати дані з етапу 1 якщо needed
        stage2_data_formatted = {'merged_data': enhanced_data}  # Правильний формат
        config = {'calendar': calendar}

        # ВИПРАВЛЕНО: правильно розпаковуємо результат з Dict
        stage3_result = prepare_stage3_datasets(stage1_data, stage2_data_formatted, config)

        # Перевіряємо тип результату
        if isinstance(stage3_result, dict):
            merged_stage3 = stage3_result.get('features', {}).get('technical', pd.DataFrame())
            context_df = stage3_result.get('context', pd.DataFrame())
            features_df = stage3_result.get('features', {}).get('technical', pd.DataFrame())
            trigger_data = stage3_result.get('triggers', pd.DataFrame())
        else:
            # Якщо повернуто кортеж (старий формат)
            merged_stage3, context_df, features_df, trigger_data = stage3_result

        self.save_cache(cache_path, (merged_stage3, context_df, features_df, trigger_data))
        return merged_stage3, context_df, features_df, trigger_data

    def run_stage_4(self, features_df: pd.DataFrame, models: list | None = None, force_refresh: bool = False) -> pd.DataFrame:
        params = {"features_shape": features_df.shape, "models": models}
        params_hash = self.get_params_hash(params)
        cache_path = self.get_cache_path("stage4_models", params_hash)

        if not force_refresh and self.is_cache_valid(cache_path):
            logger.info("[StageManager] Використовую кеш for Stage 4")
            return self.load_cache(cache_path)

        logger.info("[StageManager] Запускаю Stage 4: Model Training")
        results_df = benchmark_all_models(features_df, models=models)

        self.save_cache(cache_path, results_df)
        return results_df

    def run_pipeline_incremental(self,
                                stage_to_run: str = None,
                                debug_no_network: bool = False,
                                models: list | None = None,
                                force_refresh: bool = False) -> dict[str, Any]:
        """"
        Запускає pipeline andнкременandльно
        
        Args:
            stage_to_run: '1', '2', '3', '4', 'all' - which еandп forпускати
            debug_no_network: чи use реальнand forпити
            models: список моwhereлей for еandпу 4
            force_refresh: чи примусово оновити кеш
            use_cache: чи use кеш
            
        Returns:
            Dict with реwithульandandми allх еandпandв
        """
        logger.info(f"[StageManager] Pipeline: Початок, stage_to_run={stage_to_run}")
        results = {}

        # Еandп 1
        if stage_to_run in ['1', 'all', None]:
            results['stage1'] = self.run_stage_1(debug_no_network, force_refresh)

        # Еandп 2
        if stage_to_run in ['2', 'all', None]:
            # Якщо forпускаємо тandльки еandп 2, потрandбен еandп 1
            if stage_to_run == '2' and 'stage1' not in results:
                logger.info("[StageManager] Pipeline: Еandп 2 потребує еandпу 1, forпускаю...")
                results['stage1'] = self.run_stage_1(debug_no_network, force_refresh)

            if 'stage1' in results:
                logger.info(f"[StageManager] Pipeline: Запускаю еandп 2 with force_refresh={force_refresh}")
                raw_news, merged_df, pivots = self.run_stage_2(results['stage1'], force_refresh)
            results['stage2'] = {
                'raw_news': raw_news,
                'merged_df': merged_df,
                'pivots': pivots
            }

        # Етап 3
        if stage_to_run in ['3', 'all', None] and 'stage2' in results:
            stage2_data_tuple = (results['stage2']['raw_news'], results['stage2']['merged_df'], results['stage2']['pivots'])
            merged_stage3, context_df, features_df, trigger_data = self.run_stage_3(stage2_data_tuple, force_refresh)
            results['stage3'] = {
                'merged_stage3': merged_stage3,
                'context_df': context_df,
                'features_df': features_df,
                'trigger_data': trigger_data
            }

        # Еandп 4
        if stage_to_run in ['4', 'all', None] and 'stage3' in results:
            results_df = self.run_stage_4(results['stage3']['features_df'], models, force_refresh)
            results['stage4'] = results_df

        logger.info(f"[StageManager] Pipeline forвершено. Еandпи: {list(results.keys())}")
        return results

    def export_for_colab(self, stage: str = '2', results: dict[str, Any] = None) -> str:
        """
        Експортує данand for Colab
        
        Args:
            stage: '2' or '3' - which еandп експортувати
            results: реwithульandти pipeline
            
        Returns:
            Шлях до експортованого fileу
        """
        if stage == '2' and results and 'stage2' in results:
            merged_df = results['stage2']['merged_df']
            return colab_utils.export_stage2_data(merged_df)
        elif stage == '3' and results and 'stage3' in results:
            stage3_data = results['stage3']
            return colab_utils.export_stage3_data(
                stage3_data['features_df'],
                stage3_data['context_df'],
                stage3_data['trigger_data']
            )
        elif stage == '4' and results and 'stage4' in results:
            results_df = results['stage4']
            return colab_utils.export_stage4_data(results_df)
        else:
            raise ValueError(f"Неможливо експортувати еandп {stage}")

    def import_from_colab(self, results_file: str) -> pd.DataFrame:
        """
        Імпортує реwithульandти with Colab
        
        Args:
            results_file: Шлях до fileу with реwithульandandми
            
        Returns:
            DataFrame with реwithульandandми моwhereлей
        """
        try:
            return colab_utils.import_colab_results(results_file)
        except Exception as e:
            logger.error(f"[StageManager] Error andмпорту реwithульandтandв with Colab: {e}")
            import traceback
            logger.error(f"[StageManager] Traceback: {traceback.format_exc()}")
            return None

    def create_colab_template(self, output_path: str = "colab_template.ipynb") -> str:
        """
        Створює шаблон Colab notebook
        
        Args:
            output_path: Шлях for withбереження шаблону
            
        Returns:
            Шлях до createdго шаблону
        """
        try:
            return colab_utils.create_colab_notebook_template(output_path)
        except Exception as e:
            logger.error(f"[StageManager] Error створення шаблону Colab: {e}")
            import traceback
            logger.error(f"[StageManager] Traceback: {traceback.format_exc()}")
            return None

    def clear_cache(self, stage: str = None):
        """Очищує кеш"""
        try:
            if stage:
                cache_dir = os.path.join(self.base_path, stage)
                if os.path.exists(cache_dir):
                    for file in os.listdir(cache_dir):
                        os.remove(os.path.join(cache_dir, file))
                    logger.info(f"[StageManager] Очищено кеш for {stage}")
            else:
                for dir_name in os.listdir(self.base_path):
                    dir_path = os.path.join(self.base_path, dir_name)
                    if os.path.isdir(dir_path):
                        for file in os.listdir(dir_path):
                            os.remove(os.path.join(dir_path, file))
                logger.info("[StageManager] Очищено весь кеш")
        except Exception as e:
            logger.error(f"[StageManager] Error очищення кешу: {e}")
            import traceback
            logger.error(f"[StageManager] Traceback: {traceback.format_exc()}")
            cache_dir = os.path.join(self.base_path, stage)
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    os.remove(os.path.join(cache_dir, file))
                logger.info(f"[StageManager] Очищено кеш for {stage}")
        else:
            for dir_name in os.listdir(self.base_path):
                dir_path = os.path.join(self.base_path, dir_name)
                if os.path.isdir(dir_path):
                    for file in os.listdir(dir_path):
                        os.remove(os.path.join(dir_path, file))
            logger.info("[StageManager] Очищено весь кеш")

# Глобальний екwithемпляр
stage_manager = StageManager()
