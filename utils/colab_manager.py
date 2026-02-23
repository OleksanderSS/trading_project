# utils/colab_manager.py - Унandфandкований меnotджер Colab

import pandas as pd
from pathlib import Path
from datetime import datetime
from utils.logger import ProjectLogger

logger = ProjectLogger.get_logger("ColabManager")

class ColabManager:
    """Унandфandкований меnotджер for роботи with Colab"""
    
    def __init__(self):
        self.colab_dir = Path("data/colab")
        self.accumulated_file = Path("data/cache/stages/stage4_models/accumulated_results.parquet")
        self.colab_dir.mkdir(exist_ok=True)
        self.accumulated_file.parent.mkdir(parents=True, exist_ok=True)
    
    def sync_to_colab(self):
        """Синхронandforцandя фandч в Colab"""
        try:
            # Експортуємо найновandшand фandчand еandпу 3
            from core.stages.stage_manager import stage_manager
            
            # Заванandжуємо реwithульandти еandпу 3
            stage3_results = stage_manager.get_cached_results('stage3')
            if stage3_results is None:
                logger.error("Реwithульandти еandпу 3 not withнайwhereно")
                return False
            
            # Експортуємо в Colab
            from utils.colab_utils import ColabUtils
            colab_utils = ColabUtils()
            
            # Експортуємо фandчand
            features_path = colab_utils.export_stage3_data(
                stage3_results['features_df'],
                stage3_results.get('context_df'),
                stage3_results.get('trigger_data')
            )
            
            logger.info(f"Фandчand експортовано: {features_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error синхронandforцandї: {e}")
            return False
    
    def create_advanced_colab_notebook(self, output_path: str = "advanced_colab_template.ipynb") -> str:
        """Створює роwithширену клandтинку with багатоцandльовим прогноwithуванням"""
        
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# [START] Advanced Trading Models - Multi-Target Training\n",
                        "\n",
                        "Роwithширена версandя with багатоцandльовим прогноwithуванням:\n",
                        "- [DATA] Price targets (close, high, low)\n",
                        "- [UP] Volatility targets\n",
                        "- [REFRESH] Direction targets\n",
                        "-  Time-based targets"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Всandновлення (якщо потрandбно)\n",
                        "!pip install pandas numpy scikit-learn lightgbm xgboost tensorflow pytorch-tabnet\n",
                        "\n",
                        "from google.colab import files\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import os\n",
                        "import json\n",
                        "from datetime import datetime\n",
                        "import xgboost as xgb\n",
                        "import lightgbm as lgb\n",
                        "import tensorflow as tf\n",
                        "from sklearn.ensemble import RandomForestClassifier\n",
                        "from sklearn.model_selection import train_test_split\n",
                        "from sklearn.metrics import accuracy_score, mean_squared_error\n",
                        "\n",
                        "print(\"[START] Advanced Multi-Target Training Ready!\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "def auto_load_latest_files():\n",
                        "    \"\"\"Автоматично forванandжує найновandшand fileи\"\"\"\n",
                        "    colab_dir = '/content/drive/MyDrive/trading_project/'\n",
                        "    accumulated_dir = '/content/drive/MyDrive/trading_project/colab/accumulated'\n",
                        "    \n",
                        "    if os.path.exists(accumulated_dir):\n",
                        "        accumulated_files = [f for f in os.listdir(accumulated_dir)\n",
                        "                           if 'stage2_accumulated' in f and f.endswith('.parquet')]\n",
                        "        if accumulated_files:\n",
                        "            latest_file = max(accumulated_files, key=lambda x: os.path.getmtime(os.path.join(accumulated_dir, x)))\n",
                        "            features_df = pd.read_parquet(os.path.join(accumulated_dir, latest_file))\n",
                        "            \n",
                        "            # Очищення object колонок\n",
                        "            object_cols = features_df.select_dtypes(include=['object']).columns\n",
                        "            if len(object_cols) > 0:\n",
                        "                print(f\" Видаляю object колонки: {list(object_cols)}\")\n",
                        "                features_df = features_df.drop(columns=object_cols)\n",
                        "            \n",
                        "            print(f\"[OK] Заванandжено: {latest_file}\")\n",
                        "            print(f\"[DATA] Роwithмandр: {features_df.shape}\")\n",
                        "            return features_df\n",
                        "    \n",
                        "    print(\"[WARN] Файли not withнайwhereно, forванandжте вручну\")\n",
                        "    uploaded = files.upload()\n",
                        "    file_name = list(uploaded.keys())[0]\n",
                        "    return pd.read_parquet(file_name)"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "def create_multi_targets(features_df, tickers, timeframes):\n",
                        "    \"\"\"Створює багатоцandльовand withмandннand\"\"\"\n",
                        "    \n",
                        "    # Price targets\n",
                        "    for ticker in tickers:\n",
                        "        for timeframe in timeframes:\n",
                        "            # Дandагностика колонок - ВИПРАВЛЕНО\n",
                        "            print(\"[SEARCH] Перевandряємо доступнand колонки:\")\n",
                        "            for ticker in tickers:\n",
                        "                for timeframe in timeframes:\n",
                        "                    # Шукаємо правильну наwithву close колонки\n",
                        "                    target_col = None\n",
                        "                    possible_names = [\n",
                        "                        f'{timeframe}_{ticker}_close',    # 1d_nvda_close (ПРАВИЛЬНИЙ)\n",
                        "                        f'{ticker}_{timeframe}_close',    # nvda_1d_close\n",
                        "                        f'{timeframe}_close_{ticker}',    # 15m_close_nvda (сandрий)\n",
                        "                        f'{ticker}_{timeframe}_open',     # fallback\n",
                        "                        f'{timeframe}_{ticker}_open'      # fallback\n",
                        "                    ]\n",
                        "                    \n",
                        "                    found_names = []\n",
                        "                    for name in possible_names:\n",
                        "                        if name in features_df.columns:\n",
                        "                            target_col = name\n",
                        "                            found_names.append(name)\n",
                        "                    \n",
                        "                    if target_col:\n",
                        "                        non_null = features_df[target_col].notna().sum()\n",
                        "                        print(f\"  [OK] {target_col}: {non_null} withначень\")\n",
                        "                    else:\n",
                        "                        print(f\"  [ERROR] Не withнайwhereно close for {ticker} {timeframe}\")\n",
                        "                        print(f\"     Але є: {found_names[:3]}\")\n",
                        "                    # Шукаємо цandновand колонки\n",
                        "                    close_col = None\n",
                        "                    high_col = None\n",
                        "                    low_col = None\n",
                        "                    \n",
                        "                    for col in features_df.columns:\n",
                        "                        if ticker in col and timeframe in col:\n",
                        "                            if 'close' in col:\n",
                        "                                close_col = col\n",
                        "                            elif 'high' in col:\n",
                        "                                high_col = col\n",
                        "                            elif 'low' in col:\n",
                        "                                low_col = col\n",
                        "                    \n",
                        "                    if close_col:\n",
                        "                        # Price change targets\n",
                        "                        features_df[f'target_price_change_{ticker}_{timeframe}'] = (\n",
                        "                            features_df[close_col].pct_change() * 100\n",
                        "                        ).fillna(0)\n",
                        "                        \n",
                        "                        # Direction targets (binary)\n",
                        "                        features_df[f'target_direction_{ticker}_{timeframe}'] = (\n",
                        "                            features_df[close_col].pct_change() > 0\n",
                        "                        ).astype(int)\n",
                        "                        \n",
                        "                        # Volatility targets\n",
                        "                        if high_col and low_col:\n",
                        "                            features_df[f'target_volatility_{ticker}_{timeframe}'] = (\n",
                        "                                (features_df[high_col] - features_df[low_col]) / features_df[close_col] * 100\n",
                        "                            ).fillna(0)\n",
                        "                        \n",
                        "                        # Multi-horizon targets\n",
                        "                        for horizon in [1, 3, 5]:\n",
                        "                            features_df[f'target_price_{ticker}_{timeframe}_{horizon}h'] = (\n",
                        "                                features_df[close_col].shift(-horizon).pct_change() * 100\n",
                        "                            ).fillna(0)\n",
                        "                    \n",
                        "                        low_col = col\n",
                        "            \n",
                        "            if close_col:\n",
                        "                # Price change targets\n",
                        "                features_df[f'target_price_change_{ticker}_{timeframe}'] = (\n",
                        "                    features_df[close_col].pct_change() * 100\n",
                        "                ).fillna(0)\n",
                        "                \n",
                        "                # Direction targets (binary)\n",
                        "                features_df[f'target_direction_{ticker}_{timeframe}'] = (\n",
                        "                    features_df[close_col].pct_change() > 0\n",
                        "                ).astype(int)\n",
                        "                \n",
                        "                # Volatility targets\n",
                        "                if high_col and low_col:\n",
                        "                    features_df[f'target_volatility_{ticker}_{timeframe}'] = (\n",
                        "                        (features_df[high_col] - features_df[low_col]) / features_df[close_col] * 100\n",
                        "                    ).fillna(0)\n",
                        "                \n",
                        "                # Multi-horizon targets\n",
                        "                for horizon in [1, 3, 5]:\n",
                        "                    features_df[f'target_price_{ticker}_{timeframe}_{horizon}h'] = (\n",
                        "                        features_df[close_col].shift(-horizon).pct_change() * 100\n",
                        "                    ).fillna(0)\n",
                        "    \n",
                        "    return features_df\n",
                        "\n",
                        "def train_advanced_models(features_df, tickers, timeframes):\n",
                        "    \"\"\"Тренує роwithширенand моwhereлand\"\"\"\n",
                        "    \n",
                        "    # Створюємо багатоцandльовand withмandннand\n",
                        "    features_df = create_multi_targets(features_df, tickers, timeframes)\n",
                        "    \n",
                        "    models = {\n",
                        "        'ensemble': ['lgb', 'xgb', 'rf'],\n",
                        "        'neural': ['gru', 'lstm', 'transformer'],\n",
                        "        'hybrid': ['cnn', 'tabnet', 'autoencoder']\n",
                        "    }\n",
                        "    \n",
                        "    results = []\n",
                        "    total_combinations = len(tickers) * len(timeframes) * sum(len(v) for v in models.values())\n",
                        "    current = 0\n",
                        "    \n",
                        "    print(f\"[DATA] Данand for тренування: {features_df.shape}\")\n",
                        "    print(f\"[TARGET] Таргети created: {[col for col in features_df.columns if 'target_' in col][:10]}...\")\n",
                        "    \n",
                        "    for ticker in tickers:\n",
                        "        for timeframe in timeframes:\n",
                        "            # Перевandряємо наявнandсть data\n",
                        "            target_cols = [col for col in features_df.columns \n",
                        "                        if f'target_{ticker}_{timeframe}' in col]\n",
                        "            \n",
                        "            if not target_cols:\n",
                        "                print(f\"[WARN] Немає andргетandв for {ticker} {timeframe}\")\n",
                        "                continue\n",
                        "            \n",
                        "            # Пandдготовка data\n",
                        "            feature_cols = [col for col in features_df.select_dtypes(include=[np.number]).columns \n",
                        "                          if not col.startswith('target_')]\n",
                        "            \n",
                        "            X = features_df[feature_cols].fillna(0)\n",
                        "            \n",
                        "            # Тренуємо for кожного andргету\n",
                        "            for target_col in target_cols[:3]:  # Обмежуємо for quicklyстand\n",
                        "                y = features_df[target_col].fillna(0)\n",
                        "                \n",
                        "                # Видаляємо NaN рядки\n",
                        "                mask = ~np.isnan(y)\n",
                        "                X_clean = X[mask]\n",
                        "                y_clean = y[mask]\n",
                        "                \n",
                        "                if len(X_clean) < 50:\n",
                        "                    continue\n",
                        "                \n",
                        "                X_train, X_test, y_train, y_test = train_test_split(\n",
                        "                    X_clean, y_clean, test_size=0.2, random_state=42\n",
                        "                )\n",
                        "                \n",
                        "                # Вибandр моwhereлand forлежно вandд типу andргету\n",
                        "                if 'direction' in target_col:\n",
                        "                    # Classification\n",
                        "                    model_type = 'classification'\n",
                        "                    model = lgb.LGBMClassifier(random_state=42)\n",
                        "                    metric = 'accuracy'\n",
                        "                else:\n",
                        "                    # Regression\n",
                        "                    model_type = 'regression'\n",
                        "                    model = lgb.LGBMRegressor(random_state=42)\n",
                        "                    metric = 'mse'\n",
                        "                \n",
                        "                # Тренування\n",
                        "                try:\n",
                        "                    model.fit(X_train, y_train)\n",
                        "                    predictions = model.predict(X_test)\n",
                        "                    \n",
                        "                    # Метрики\n",
                        "                    if model_type == 'classification':\n",
                        "                        score = accuracy_score(y_test, predictions)\n",
                        "                        metric_name = 'accuracy'\n",
                        "                    else:\n",
                        "                        score = mean_squared_error(y_test, predictions)\n",
                        "                        metric_name = 'mse'\n",
                        "                    \n",
                        "                    results.append({\n",
                        "                        'model': 'lgb_advanced',\n",
                        "                        'ticker': ticker,\n",
                        "                        'timeframe': timeframe,\n",
                        "                        'target_type': target_col,\n",
                        "                        'model_type': model_type,\n",
                        "                        'metric': metric_name,\n",
                        "                        'score': score,\n",
                        "                        'timestamp': datetime.now().isoformat()\n",
                        "                    })\n",
                        "                    \n",
                        "                    current += 1\n",
                        "                    print(f\"[OK] [{current}/{total_combinations}] {target_col}: {score:.4f}\")\n",
                        "                    \n",
                        "                except Exception as e:\n",
                        "                    print(f\"[ERROR] Error {target_col}: {e}\")\n",
                        "    \n",
                        "    return pd.DataFrame(results)\n",
                        "\n",
                        "# Головnot виконання\n",
                        "print(\"[START] ПОЧИНАЄМО РОЗШИРЕНЕ ТРЕНУВАННЯ...\")\n",
                        "\n",
                        "# Параметри\n",
                        "tickers = ['nvda', 'qqq', 'spy', 'tsla']\n",
                        "timeframes = ['15m', '60m', '1d']\n",
                        "\n",
                        "# Заванandження data\n",
                        "features_df = auto_load_latest_files()\n",
                        "\n",
                        "# Тренування моwhereлей\n",
                        "advanced_results = train_advanced_models(features_df, tickers, timeframes)\n",
                        "\n",
                        "print(f\"\\n[TARGET] Реwithульandти: {len(advanced_results)} моwhereлей\")\n",
                        "print(advanced_results.head(10))"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.8.5"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(notebook_content, f, indent=2, ensure_ascii=False)

        logger.info(f"[ColabManager] ???????????????? ?????????????????? ????????????????: {output_path}")
        return output_path

    def accumulate_results(self):
        """???????????????????? ?????????????????????? ?? Colab"""
        try:
            # Шукаємо осandннandй parquet file
            parquet_files = list(self.colab_dir.glob("stage4_models_*.parquet"))
            
            if not parquet_files:
                logger.info("Файлandв реwithульandтandв not withнайwhereно")
                return False
            
            # Беремо осandннandй file
            latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
            new_results = pd.read_parquet(latest_file)
            
            # Акумулюємо
            accumulated = self._merge_results(new_results)
            
            # Видаляємо сandрand fileи
            self._cleanup_old_files(latest_file)
            
            logger.info(f"Акумульовано {len(accumulated)} реwithульandтandв")
            return True
            
        except Exception as e:
            logger.error(f"Error акумуляцandї: {e}")
            return False
    
    def _merge_results(self, new_results):
        """Merging реwithульandтandв беwith дублandкатandв"""
        # Заванandжуємо andснуючand
        if self.accumulated_file.exists():
            accumulated = pd.read_parquet(self.accumulated_file)
        else:
            accumulated = pd.DataFrame()
        
        # Об'єднуємо
        combined = pd.concat([accumulated, new_results], ignore_index=True)
        
        # Видаляємо дублandкати
        combined = combined.drop_duplicates(
            subset=['model', 'ticker', 'timeframe', 'timestamp'], 
            keep='last'
        )
        
        # Зберandгаємо
        combined.to_parquet(self.accumulated_file)
        return combined
    
    def _cleanup_old_files(self, keep_file):
        """Видалення сandрих fileandв"""
        parquet_files = list(self.colab_dir.glob("stage4_models_*.parquet"))
        metadata_files = list(self.colab_dir.glob("stage4_models_*_metadata.json"))
        
        # Видаляємо сandрand parquet fileи
        for file_path in parquet_files:
            if file_path != keep_file:
                file_path.unlink()
                logger.info(f"Видалено: {file_path.name}")
        
        # Видаляємо сandрand metadata fileи
        for file_path in metadata_files:
            if not file_path.exists():
                continue  # Пропускаємо якщо вже deleted
            
            # Видаляємо якщо notмає вandдповandдного parquet
            corresponding_parquet = file_path.name.replace('_metadata.json', '.parquet')
            if not (self.colab_dir / corresponding_parquet).exists():
                file_path.unlink()
                logger.info(f"Видалено metadata: {file_path.name}")
    
    def get_accumulated_results(self):
        """Поверandє акумульованand реwithульandти"""
        if self.accumulated_file.exists():
            return pd.read_parquet(self.accumulated_file)
        return pd.DataFrame()
    
    def get_stats(self):
        """Сandтистика реwithульandтandв"""
        accumulated = self.get_accumulated_results()
        
        if accumulated.empty:
            return "Реwithульandтandв ще notмає"
        
        stats = {
            "total_results": len(accumulated),
            "models": accumulated['model'].nunique(),
            "tickers": accumulated['ticker'].nunique(),
            "timeframes": accumulated['timeframe'].nunique(),
            "latest_update": accumulated['timestamp'].max()
        }
        
        return stats

# Глобальний екwithемпляр
colab_manager = ColabManager()
