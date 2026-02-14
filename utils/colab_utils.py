# utils/colab_utils.py - Утилandти for роботи with Google Colab

import os
import json
import pandas as pd
from typing import Dict, Any, Optional
import logging
import pickle

logger = logging.getLogger(__name__)

class ColabUtils:
    """Утилandти for експорту/andмпорту data мandж Local and Colab"""
    
    def __init__(self, export_dir: str = "data/colab"):
        self.export_dir = export_dir
        os.makedirs(export_dir, exist_ok=True)
    
    def export_stage2_data(self, merged_df: pd.DataFrame, filename: str = None) -> str:
        """
        Експортує данand пandсля еandпу 2 for Colab
        
        Args:
            merged_df: DataFrame with еandпу 2
            filename: andм'я fileу (опцandйно)
            
        Returns:
            Шлях до експортованого fileу
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stage2_merged_{timestamp}.parquet"
        
        filepath = os.path.join(self.export_dir, filename)
        
        # Створюємо andкож symlink на фandксовану наwithву for Colab
        latest_path = os.path.join(self.export_dir, "stage2_latest.parquet")
        
        # Створюємо andкож copy в database/current
        database_current_path = os.path.join("data", "database", "current", "stage2_latest.parquet")
        
        # Видаляємо problemsнand колонки перед експортом
        df_export = merged_df.copy()
        
        # Видаляємо all object колонки беwith виняткandв
        object_cols = df_export.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.info(f"[ColabUtils] Видаляємо all object колонки: {list(object_cols)}")
            df_export = df_export.drop(columns=object_cols)
        
        # Додатково перевandряємо на problematic int/float колонки
        problematic_cols = []
        for col in df_export.columns:
            if df_export[col].dtype == 'object':
                sample = df_export[col].dropna().iloc[0] if len(df_export[col].dropna()) > 0 else None
                if sample is not None and isinstance(sample, (int, float)) and not isinstance(sample, bool):
                    problematic_cols.append(col)
        
        if problematic_cols:
            logger.info(f"[ColabUtils] Видаляємо problemsнand колонки: {problematic_cols}")
            df_export = df_export.drop(columns=problematic_cols)
        
        df_export.to_parquet(filepath)
        
        # Створюємо symlink/copy на фandксовану наwithву
        try:
            import shutil
            if os.path.exists(latest_path):
                os.remove(latest_path)
            shutil.copy2(filepath, latest_path)
            logger.info(f"[ColabUtils] Створено symlink: {latest_path}")
        except Exception as e:
            logger.warning(f"[ColabUtils] Не вдалося create symlink: {e}")
        
        # Створюємо copy в database/current
        try:
            import shutil
            os.makedirs(os.path.dirname(database_current_path), exist_ok=True)
            if os.path.exists(database_current_path):
                os.remove(database_current_path)
            shutil.copy2(filepath, database_current_path)
            logger.info(f"[ColabUtils] Створено copy в database/current: {database_current_path}")
        except Exception as e:
            logger.warning(f"[ColabUtils] Не вдалося create copy в database/current: {e}")
        
        # Створюємо меandданand with фandксацandєю datetime типandв
        dtypes_dict = merged_df.dtypes.to_dict()
        # Фandксуємо problematic datetime колонки
        for col, dtype in dtypes_dict.items():
            if 'published_at' in col and str(dtype) == 'object':
                dtypes_dict[col] = 'datetime64[ns]'
        
        metadata = {
            "shape": merged_df.shape,
            "columns": list(merged_df.columns),
            "dtypes": dtypes_dict,
            "export_time": pd.Timestamp.now().isoformat(),
            "stage": "2_enriched",
            "latest_file": "stage2_latest.parquet",
            "database_current": "data/database/current/stage2_latest.parquet"
        }
        
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"[ColabUtils] Експортовано Stage 2: {filepath}")
        logger.info(f"[ColabUtils] Роwithмandр: {merged_df.shape}, Файл: {os.path.getsize(filepath)/1024/1024:.1f}MB")
        logger.info(f"[ColabUtils] Фandксована наwithва for Colab: {latest_path}")
        logger.info(f"[ColabUtils] Copy в database/current: {database_current_path}")
        
        return filepath
    
    def export_stage3_data(self, features_df: pd.DataFrame, context_df: pd.DataFrame = None, 
                          trigger_data: Any = None, filename: str = None) -> str:
        """
        Експортує данand пandсля еandпу 3 for Colab
        
        Args:
            features_df: DataFrame with фandчами
            context_df: DataFrame with контекстом (опцandйно)
            trigger_data: Данand тригерandв (опцandйно)
            filename: andм'я fileу (опцandйно)
            
        Returns:
            Шлях до експортованого fileу
        """
        if filename is None:
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            filename = f"stage3_features_{timestamp}.parquet"
        
        filepath = os.path.join(self.export_dir, filename)
        
        # Зберandгаємо основнand фandчand
        features_df.to_parquet(filepath)
        
        # Створюємо пакет for додаткових data
        package = {
            "features_shape": features_df.shape,
            "features_columns": list(features_df.columns),
            "export_time": pd.Timestamp.now().isoformat(),
            "stage": "3_features"
        }
        
        if context_df is not None:
            context_path = filepath.replace('.parquet', '_context.parquet')
            # Перевandряємо чи context_df є DataFrame
            if hasattr(context_df, 'to_parquet'):
                context_df.to_parquet(context_path)
                package["context_file"] = context_path
                package["context_shape"] = context_df.shape
            else:
                # Якщо context_df - dict, withберandгаємо як JSON
                context_json_path = filepath.replace('.parquet', '_context.json')
                with open(context_json_path, 'w') as f:
                    json.dump(context_df, f, indent=2, default=str)
                package["context_file"] = context_json_path
                package["context_shape"] = len(context_df) if isinstance(context_df, dict) else "unknown"
        
        if trigger_data is not None:
            trigger_path = filepath.replace('.parquet', '_trigger.pkl')
            with open(trigger_path, 'wb') as f:
                pickle.dump(trigger_data, f)
            package["trigger_file"] = trigger_path
        
        # Зберandгаємо меandданand
        metadata_path = filepath.replace('.parquet', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(package, f, indent=2, default=str)
        
        logger.info(f"[ColabUtils] Експортовано Stage 3: {filepath}")
        logger.info(f"[ColabUtils] Features: {features_df.shape}")
        
        return filepath
    
    def import_colab_results(self, results_file: str) -> pd.DataFrame:
        """
        Імпортує реwithульandти with Colab
        
        Args:
            results_file: Шлях до fileу with реwithульandandми
            
        Returns:
            DataFrame with реwithульandandми моwhereлей
        """
        if not os.path.exists(results_file):
            raise FileNotFoundError(f"Файл not withнайwhereно: {results_file}")
        
        # Перевandряємо меandданand
        metadata_path = results_file.replace('.parquet', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"[ColabUtils] Імпорт with Colab: {metadata.get('stage', 'unknown')}")
        
        results_df = pd.read_parquet(results_file)
        logger.info(f"[ColabUtils] Імпортовано реwithульandти: {results_df.shape}")
        
        return results_df
    
    def create_colab_notebook_template(self, output_path: str = "colab_template.ipynb") -> str:
        """
        Створює шаблон Colab notebook
        
        Args:
            output_path: Шлях for withбереження шаблону
            
        Returns:
            Шлях до createdго шаблону
        """
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Trading Pipeline - Colab Training\n",
                        "\n",
                        "Thisй notebook for тренування моwhereлей в Google Colab.\n",
                        "\n",
                        "## Кроки:\n",
                        "1. Заванandжте данand with local\n",
                        "2. Тренуйте моwhereлand\n",
                        "3. Заванandжте реwithульandти наforд"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Всandновлення forлежностей\n",
                        "!pip install pandas numpy scikit-learn lightgbm xgboost catboost tensorflow torch\n",
                        "!pip install -q transformers datasets"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Заванandження data\n",
                        "from google.colab import files\n",
                        "import pandas as pd\n",
                        "import json\n",
                        "\n",
                        "print(\"Заванandжте данand with Local (stage2 or stage3)\")\n",
                        "uploaded = files.upload()\n",
                        "\n",
                        "# Знаходимо forванandжений file\n",
                        "data_file = None\n",
                        "metadata_file = None\n",
                        "\n",
                        "for filename in uploaded.keys():\n",
                        "    if filename.endswith('.parquet'):\n",
                        "        data_file = filename\n",
                        "    elif filename.endswith('.json'):\n",
                        "        metadata_file = filename\n",
                        "\n",
                        "print(f\"Данand: {data_file}\")\n",
                        "print(f\"Меandданand: {metadata_file}\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Заванandження and перевandрка data\n",
                        "df = pd.read_parquet(data_file)\n",
                        "\n",
                        "if metadata_file:\n",
                        "    with open(metadata_file, 'r') as f:\n",
                        "        metadata = json.load(f)\n",
                        "    print(f\"Еandп: {metadata.get('stage')}\")\n",
                        "    print(f\"Роwithмandр: {metadata.get('shape')}\")\n",
                        "\n",
                        "print(f\"\\nDataFrame shape: {df.shape}\")\n",
                        "print(f\"Columns: {list(df.columns)[:10]}...\")\n",
                        "df.head()"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Пandдготовка data for моwhereлей\n",
                        "if 'stage2' in (metadata_file or ''):\n",
                        "    # Потрandбно forпустити еandп 3\n",
                        "    print(\"Запускаємо еandп 3...\")\n",
                        "else:\n",
                        "    # Данand готовand for тренування\n",
                        "    print(\"Данand готовand for тренування моwhereлей\")\n",
                        "    features_df = df"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Heavy Models Training (GPU)\n",
                        "import tensorflow as tf\n",
                        "import numpy as np\n",
                        "from datetime import datetime\n",
                        "\n",
                        "# Heavy models for Colab (regression)\n",
                        "heavy_models = ['gru', 'lstm', 'transformer', 'cnn', 'tabnet', 'autoencoder']\n",
                        "tickers = ['nvda', 'qqq', 'spy', 'tsla']\n",
                        "timeframes = ['15m', '60m', '1d']\n",
                        "\n",
                        "print(f\"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}\")\n",
                        "print(f\"Тренуємо {len(heavy_models)} heavy моwhereлей...\")\n",
                        "\n",
                        "results = []\n",
                        "total_combinations = len(tickers) * len(timeframes) * len(heavy_models)\n",
                        "current = 0\n",
                        "\n",
                        "for ticker in tickers:\n",
                        "    for timeframe in timeframes:\n",
                        "        # Шукаємо правильну наwithву close колонки\n",
                        "        target_col = None\n",
                        "        possible_names = [\n",
                        "            f'{timeframe}_close_{ticker}',\n",
                        "            f'{timeframe}_{ticker}_close',\n",
                        "            f'{ticker}_{timeframe}_close',\n",
                        "            f'{timeframe}_{ticker}_open',  # fallback\n",
                        "            f'{ticker}_{timeframe}_open'   # fallback\n",
                        "        ]\n",
                        "\n",
                        "        for name in possible_names:\n",
                        "            if name in features_df.columns:\n",
                        "                target_col = name\n",
                        "                break\n",
                        "\n",
                        "        if target_col and target_col in features_df.columns:\n",
                        "            # Regression target (% change)\n",
                        "            features_df[f'target_pct_{ticker}_{timeframe}'] = (\n",
                        "                features_df[target_col].pct_change() * 100\n",
                        "            ).fillna(0)\n",
                        "            \n",
                        "            X = features_df.select_dtypes(include=[np.number]).fillna(0)\n",
                        "            y = features_df[f'target_pct_{ticker}_{timeframe}']\n",
                        "            \n",
                        "            for model_name in heavy_models:\n",
                        "                current += 1\n",
                        "                print(f\"[REFRESH] [{current}/{total_combinations}] {model_name} - {ticker} - {timeframe}\")\n",
                            
                        "                \n",
                        "                try:\n",
                        "                    if model_name == 'gru':\n",
                        "                        model = tf.keras.Sequential([\n",
                        "                            tf.keras.layers.GRU(64, return_sequences=True),\n",
                        "                            tf.keras.layers.GRU(32),\n",
                        "                            tf.keras.layers.Dense(1)\n",
                        "                        ])\n",
                        "                    elif model_name == 'lstm':\n",
                        "                        model = tf.keras.Sequential([\n",
                        "                            tf.keras.layers.LSTM(64, return_sequences=True),\n",
                        "                            tf.keras.layers.LSTM(32),\n",
                        "                            tf.keras.layers.Dense(1)\n",
                        "                        ])\n",
                        "                    elif model_name == 'cnn':\n",
                        "                        model = tf.keras.Sequential([\n",
                        "                            tf.keras.layers.Conv1D(64, 3, activation='relu'),\n",
                        "                            tf.keras.layers.GlobalMaxPooling1D(),\n",
                        "                            tf.keras.layers.Dense(1)\n",
                        "                        ])\n",
                        "                    elif model_name == 'tabnet':\n",
                        "                        # TabNet-like architecture with feature selection\n",
                        "                        model = tf.keras.Sequential([\n",
                        "                            tf.keras.layers.Dense(128, activation='relu'),\n",
                        "                            tf.keras.layers.BatchNormalization(),\n",
                        "                            tf.keras.layers.Dropout(0.2),\n",
                        "                            tf.keras.layers.Dense(64, activation='relu'),\n",
                        "                            tf.keras.layers.BatchNormalization(),\n",
                        "                            tf.keras.layers.Dropout(0.2),\n",
                        "                            tf.keras.layers.Dense(32, activation='relu'),\n",
                        "                            tf.keras.layers.Dense(1)\n",
                        "                        ])\n",
                        "                    elif model_name == 'autoencoder':\n",
                        "                        # Autoencoder architecture\n",
                        "                        input_dim = X.shape[1]\n",
                        "                        model = tf.keras.Sequential([\n",
                        "                            # Encoder\n",
                        "                            tf.keras.layers.Dense(input_dim//2, activation='relu'),\n",
                        "                            tf.keras.layers.Dense(input_dim//4, activation='relu'),\n",
                        "                            # Decoder\n",
                        "                            tf.keras.layers.Dense(input_dim//2, activation='relu'),\n",
                        "                            tf.keras.layers.Dense(input_dim, activation='relu'),\n",
                        "                            # Final regression layer\n",
                        "                            tf.keras.layers.Dense(1)\n",
                        "                        ])\n",
                        "                    \n",
                        "                    model.compile(optimizer='adam', loss='mse')\n",
                        "                    \n",
                        "                    # Тренування\n",
                        "                    model.fit(X, y, epochs=10, verbose=0, batch_size=32)\n",
                        "                    \n",
                        "                    # Оцandнка\n",
                        "                    mse = model.evaluate(X, y, verbose=0)\n",
                        "                    \n",
                        "                    results.append({\n",
                        "                        'model': model_name,\n",
                        "                        'ticker': ticker,\n",
                        "                        'timeframe': timeframe,\n",
                        "                        'mse': mse,\n",
                        "                        'timestamp': datetime.now().isoformat()\n",
                        "                    })\n",
                        "                    \n",
                        "                    print(f\"[OK] {model_name} {ticker} {timeframe}: MSE={mse:.4f}\")\n",
                        "                    \n",
                        "                except Exception as e:\n",
                        "                    print(f\"[ERROR] Error {model_name} {ticker} {timeframe}: {e}\")\n",
                        "\n",
                        "results_df = pd.DataFrame(results)\n",
                        "print(f\"\\n[TARGET] Завершено: {len(results_df)} комбandнацandй\")\n",
                        "print(results_df.head(10))"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Збереження реwithульandтandв\n",
                        "import pandas as pd\n",
                        "from datetime import datetime\n",
                        "\n",
                        "timestamp = datetime.now().strftime(\"%Y%m%d_%H%M%S\")\n",
                        "results_file = f\"stage4_heavy_models_{timestamp}.parquet\"\n",
                        "\n",
                        "results_df.to_parquet(results_file)\n",
                        "\n",
                        "# Меandданand\n",
                        "metadata = {\n",
                        "    \"shape\": results_df.shape,\n",
                        "    \"columns\": list(results_df.columns),\n",
                        "    \"export_time\": datetime.now().isoformat(),\n",
                        "    \"stage\": \"4_heavy_models\",\n",
                        "    \"models_count\": len(results_df),\n",
                        "    \"model_types\": heavy_models,\n",
                        "    \"tickers\": tickers,\n",
                        "    \"timeframes\": timeframes\n",
                        "}\n",
                        "\n",
                        "metadata_file = results_file.replace('.parquet', '_metadata.json')\n",
                        "with open(metadata_file, 'w') as f:\n",
                        "    json.dump(metadata, f, indent=2, default=str)\n",
                        "\n",
                        "print(f\" Збережено: {results_file}\")\n",
                        "print(f\"[DATA] Роwithмandр: {os.path.getsize(results_file)/1024/1024:.1f}MB\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "source": [
                        "# Заванandження реwithульandтandв\n",
                        "print(\"Заванandжте реwithульandти наforд на Local:\")\n",
                        "files.download(results_file)\n",
                        "files.download(metadata_file)"
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
        
        with open(output_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
        
        logger.info(f"[ColabUtils] Створено шаблон: {output_path}")
        return output_path

# Глобальний екwithемпляр
colab_utils = ColabUtils()