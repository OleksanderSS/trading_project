

def recover_missing_data(config_manager, brain, logger):
    """Recover news and macro data using DataManager. Safe to call from stage."""
    try:
        from src.data.management.data_manager import DataManager

        db_manager = DataManager(config_manager)

        # News
        if brain.get("news_data") is None or (hasattr(brain.get("news_data"), "empty") and brain["news_data"].empty):
            logger.info("🔍 news_data missing from brain. Attempting recovery from DB...")
            news_df = db_manager.fetch_data_from_table("news_sentiment_cache")
            if news_df is not None and not news_df.empty:
                brain["news_data"] = news_df
                logger.info(f"✅ Recovered {len(news_df)} news records from DB")

        # Macro
        if brain.get("macro_data") is None or (hasattr(brain.get("macro_data"), "empty") and brain["macro_data"].empty):
            logger.info("🔍 macro_data missing from brain. Attempting recovery from DB...")
            macro_df = db_manager.fetch_data_from_table("macro_data_raw")
            if macro_df is not None and not macro_df.empty:
                brain["macro_data"] = macro_df
                logger.info(f"✅ Recovered {len(macro_df)} macro records from DB")

    except Exception as e:
        logger.error(f"❌ Adaptive data recovery failed: {e}", exc_info=True)
        raise
