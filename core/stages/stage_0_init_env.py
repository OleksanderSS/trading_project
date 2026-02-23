# core/stages/stage_0_init_env.py

import sys, os, yaml, importlib.util
from pathlib import Path
from google.colab import drive

def init_environment(project_path="/content/drive/MyDrive/trading_project"):
    #  Монтуємо Google Drive
    drive.mount('/content/drive')

    #  Шлях до проєкту
    if project_path not in sys.path:
        sys.path.append(project_path)

    # [TOOL] Шляхи до конфandгandв
    config_path   = f"{project_path}/config/news_sources.yaml"
    secrets_path  = f"{project_path}/config/secrets_manager.py"
    logger_path   = f"{project_path}/utils/logger.py"

    #  Заванandження secrets_manager
    spec = importlib.util.spec_from_file_location("secrets_manager", secrets_path)
    secrets_manager = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(secrets_manager)

    #  Заванandження logger
    spec2 = importlib.util.spec_from_file_location("logger", logger_path)
    logger_module = importlib.util.module_from_spec(spec2)
    spec2.loader.exec_module(logger_module)

    # [OK] Інandцandалandforцandя компоnotнтandв
    secrets = secrets_manager.Secrets()
    api_keys = secrets.as_dict()
    logger = logger_module.ProjectLogger.get_logger("InitCell")

    #  Заванandження YAML-конфandгурацandї
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    #  Джерела новин
    rss_feeds   = config.get("rss", {})
    web_sources = config.get("web_sources", {})

    #  Keywords
    keyword_dict = config.get("keywords", {})
    tickers_dict = keyword_dict.get("tickers", {})
    tickers = list(tickers_dict.keys()) if isinstance(tickers_dict, dict) else []

    return api_keys, rss_feeds, web_sources, keyword_dict, tickers, logger