# core/stages/incremental_pipeline.py - andнкременandльnot оновлення даandсету

from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from utils.news_harmonizer import harmonize_entry
from collectors.base_collector import _hash_item
import pandas as pd
import sqlite3
import logging

logger = logging.getLogger("incremental_pipeline")

class IncrementalDataUpdater:
    """Клас for andнкременandльного оновлення даandсету"""
    
    def __init__(self, db_path: str, table_name: str):
        self.db_path = db_path
        self.table_name = table_name
        self.conn = None
        
    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
    
    def get_last_update_time(self) -> Optional[datetime]:
        """Отримати час осandннього оновлення"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"""
                SELECT MAX(published_at) FROM {self.table_name}
                WHERE published_at IS NOT NULL
            """)
            result = cursor.fetchone()
            if result and result[0]:
                return pd.to_datetime(result[0])
            return None
        except Exception as e:
            logger.error(f"Error getting last update time: {e}")
            return None
    
    def check_duplicates(self, entries: list) -> list:
        """Check наявнandсть forписandв в баwithand"""
        if not entries:
            return []
            
        existing_hashes = set()
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT published_at, description, source, url FROM {self.table_name}")
            for row in cursor.fetchall():
                entry_hash = _hash_item({
                    'published_at': row[0],
                    'description': row[1], 
                    'source': row[2],
                    'url': row[3]
                })
                existing_hashes.add(entry_hash)
        except Exception as e:
            logger.error(f"Error checking existing entries: {e}")
            
        new_entries = []
        for entry in entries:
            entry_hash = _hash_item({
                'published_at': entry.get('published_at'),
                'description': entry.get('description'),
                'source': entry.get('source'),
                'url': entry.get('url')
            })
            
            if entry_hash not in existing_hashes:
                new_entries.append(entry)
                
        logger.info(f"Found {len(new_entries)} new entries out of {len(entries)} total")
        return new_entries
    
    def update_dataset(self, new_entries: list, source: str) -> int:
        """Оновити даandсет новими forписами"""
        if not new_entries:
            logger.info("No new entries to add")
            return 0
            
        harmonized = []
        for entry in new_entries:
            try:
                harmonized_entry = harmonize_entry(entry, source)
                harmonized.append(harmonized_entry)
            except Exception as e:
                logger.warning(f"Error harmonizing entry: {e}")
                continue
                
        if not harmonized:
            logger.warning("No harmonized entries to save")
            return 0
            
        # Зберandгаємо в DataFrame
        df = pd.DataFrame(harmonized)
        
        # Додаємо до баwithи data
        try:
            df.to_sql(self.table_name, self.conn, if_exists='append', index=False)
            self.conn.commit()
            logger.info(f"Added {len(df)} new entries to {self.table_name}")
            return len(df)
        except Exception as e:
            logger.error(f"Error saving to database: {e}")
            self.conn.rollback()
            return 0


def run_incremental_update(db_path: str, table_name: str, 
                        new_data_fetcher, source: str, 
                        days_back: int = 7) -> Dict[str, Any]:
    """
    Запустити andнкременandльnot оновлення даandсету
    
    Args:
        db_path: шлях до SQLite баwithи
        table_name: наwithва andблицand
        new_data_fetcher: функцandя for отримання нових data (start_date, end_date)
        source: наwithва джерела data
        days_back: скandльки днandв наforд перевandряти
        
    Returns:
        Dict with реwithульandandми оновлення
    """
    results = {
        'start_time': datetime.now(),
        'last_update': None,
        'new_entries_found': 0,
        'new_entries_added': 0,
        'errors': []
    }
    
    try:
        with IncrementalDataUpdater(db_path, table_name) as updater:
            # Отримуємо час осandннього оновлення
            last_update = updater.get_last_update_time()
            results['last_update'] = last_update
            
            # Виwithначаємо period for пошуку нових data
            if last_update:
                start_date = last_update - timedelta(days=1)  # with forпасом
                logger.info(f"Last update: {last_update}, fetching from: {start_date}")
            else:
                start_date = datetime.now() - timedelta(days=days_back)
                logger.info(f"No previous data, fetching from: {start_date}")
                
            end_date = datetime.now()
            
            # Отримуємо новand данand
            try:
                new_entries = new_data_fetcher(start_date, end_date)
                results['new_entries_found'] = len(new_entries)
                logger.info(f"Fetched {len(new_entries)} potential new entries")
            except Exception as e:
                error_msg = f"Error fetching new data: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
                return results
            
            # Перевandряємо на дублandкати
            unique_entries = updater.check_duplicates(new_entries)
            logger.info(f"After duplicate check: {len(unique_entries)} unique entries")
            
            # Оновлюємо даandсет
            added_count = updater.update_dataset(unique_entries, source)
            results['new_entries_added'] = added_count
            
    except Exception as e:
        error_msg = f"Error in incremental update: {e}"
        results['errors'].append(error_msg)
        logger.error(error_msg)
    
    results['end_time'] = datetime.now()
    results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
    
    return results


# Приклад викорисandння
if __name__ == "__main__":
    def sample_fetcher(start_date, end_date):
        """Приклад функцandї for отримання нових data"""
        # Тут повинна бути логandка отримання data with API
        return []
    
    results = run_incremental_update(
        db_path="data/databases/news_processed.db",
        table_name="processed_news", 
        new_data_fetcher=sample_fetcher,
        source="sample_source",
        days_back=7
    )
    
    logger.info("Incremental update results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value}")