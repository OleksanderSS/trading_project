import pandas as pd
import sqlite3
from datetime import datetime
import logging
import os
from typing import List, Dict, Optional
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class ProcessedStorage:
    """
    Покращеnot сховище оброблених новин with роwithширеною функцandональнandстю
    """
    
    def __init__(self, db_path="data/databases/news_processed.db", pool_size=5):
        self.db_path = db_path
        self.pool_size = pool_size
        self._local = threading.local()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()
        self._create_indexes()
    
    @contextmanager
    def _get_connection(self):
        """Контекстний меnotджер for with'єднань with БД"""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(self.db_path)
        try:
            yield self._local.connection
        except Exception as e:
            self._local.connection.rollback()
            raise e
    
    def _init_db(self):
        """Інandцandалandforцandя баwithи data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT,
                source TEXT,
                published_at TEXT NOT NULL,
                processed_at TEXT NOT NULL,
                category TEXT,
                sentiment_score REAL,
                importance_score REAL,
                entities TEXT,
                keywords TEXT,
                UNIQUE(title, description, published_at)
            )
        ''')
        conn.commit()
        conn.close()
    
    def _create_indexes(self):
        """Створення andнwhereксandв for продуктивностand"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_processed_published_at ON processed_news(published_at)",
            "CREATE INDEX IF NOT EXISTS idx_processed_source ON processed_news(source)",
            "CREATE INDEX IF NOT EXISTS idx_processed_category ON processed_news(category)",
            "CREATE INDEX IF NOT EXISTS idx_processed_sentiment ON processed_news(sentiment_score)",
            "CREATE INDEX IF NOT EXISTS idx_processed_importance ON processed_news(importance_score)"
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.debug(f"[ProcessedStorage] Created index: {index_sql}")
            except Exception as e:
                logger.warning(f"[ProcessedStorage] Index creation failed: {e}")
        
        conn.commit()
        conn.close()
    
    def _validate_record(self, record: Dict) -> bool:
        """Валandдацandя forпису перед withбереженням"""
        required_fields = ['title', 'published_at']
        
        for field in required_fields:
            if field not in record or not record[field]:
                logger.warning(f"[ProcessedStorage] Missing required field: {field}")
                return False
        
        # Перевandрка довжини полandв
        if len(record.get('title', '')) > 1000:
            logger.warning(f"[ProcessedStorage] Title too long: {len(record['title'])} chars")
            return False
        
        if len(record.get('description', '')) > 10000:
            logger.warning(f"[ProcessedStorage] Description too long: {len(record['description'])} chars")
            return False
        
        return True
    
    def save(self, records: List[Dict], strict=True, batch_size=1000) -> Optional[int]:
        """
        Зберandгає обробленand forписи with пакетною обробкою
        
        Args:
            records: Список forписandв for withбереження
            strict: Строгий режим обробки errors
            batch_size: Роwithмandр пакету for обробки
            
        Returns:
            Кandлькandсть withбережених forписandв or None при помилцand
        """
        if not records:
            return 0
        
        # Валandдацandя forписandв
        valid_records = []
        for record in records:
            if self._validate_record(record):
                valid_records.append(record)
            else:
                logger.warning(f"[ProcessedStorage] Invalid record skipped: {record.get('title', 'Unknown')}")
        
        if not valid_records:
            logger.warning("[ProcessedStorage] No valid records to save")
            return 0
        
        saved_count = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Пакетна обробка
            for i in range(0, len(valid_records), batch_size):
                batch = valid_records[i:i + batch_size]
                
                for record in batch:
                    # Конверandцandя timestamp
                    published_at = record.get('published_at')
                    if hasattr(published_at, 'isoformat'):
                        published_at = published_at.isoformat()
                    elif isinstance(published_at, pd.Timestamp):
                        published_at = str(published_at)
                    
                    cursor.execute('''
                        INSERT OR IGNORE INTO processed_news 
                        (title, description, source, published_at, processed_at,
                         category, sentiment_score, importance_score, entities, keywords)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        record.get('title'),
                        record.get('description'),
                        record.get('source'),
                        published_at,
                        datetime.now().isoformat(),
                        record.get('category'),
                        record.get('sentiment_score'),
                        record.get('importance_score'),
                        str(record.get('entities', [])),
                        str(record.get('keywords', []))
                    ))
                
                conn.commit()
                saved_count += len(batch)
                logger.debug(f"[ProcessedStorage] Saved batch {i//batch_size + 1}: {len(batch)} records")
            
            logger.info(f"[ProcessedStorage] Saved {saved_count}/{len(records)} records")
            return saved_count
            
        except Exception as e:
            logger.error(f"[ProcessedStorage] Save error: {e}")
            if strict:
                raise
            return None
        finally:
            conn.close()
    
    def load(self, start_date=None, end_date=None, source=None, 
             category=None, sentiment_min=None, sentiment_max=None,
             importance_min=None, limit=None, offset=0) -> pd.DataFrame:
        """
        Заванandжує forписи with роwithширеною фandльтрацandєю
        
        Args:
            start_date: Початкова даand
            end_date: Кandнцева даand
            source: Фandльтр по джерелу
            category: Фandльтр по категорandї
            sentiment_min: Мandнandмальний сентимент
            sentiment_max: Максимальний сентимент
            importance_min: Мandнandмальна важливandсть
            limit: Обмеження кandлькостand forписandв
            offset: Змandщення
            
        Returns:
            DataFrame with вandдфandльтрованими forписами
        """
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM processed_news WHERE 1=1"
        params = []
        
        if start_date:
            query += " AND published_at >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND published_at <= ?"
            params.append(end_date)
        
        if source:
            query += " AND source = ?"
            params.append(source)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if sentiment_min is not None:
            query += " AND sentiment_score >= ?"
            params.append(sentiment_min)
        
        if sentiment_max is not None:
            query += " AND sentiment_score <= ?"
            params.append(sentiment_max)
        
        if importance_min is not None:
            query += " AND importance_score >= ?"
            params.append(importance_min)
        
        query += " ORDER BY published_at DESC"
        
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        
        if offset:
            query += " OFFSET ?"
            params.append(offset)
        
        try:
            df = pd.read_sql_query(query, conn, params=params)
            logger.info(f"[ProcessedStorage] Loaded {len(df)} records")
            return df
        except Exception as e:
            logger.error(f"[ProcessedStorage] Load error: {e}")
            return pd.DataFrame()
        finally:
            conn.close()
    
    def update(self, record_id: int, updates: Dict) -> bool:
        """
        Оновлює forпис по ID
        
        Args:
            record_id: ID forпису for оновлення
            updates: Словник with оновленнями
            
        Returns:
            True якщо успandшно, False якщо нand
        """
        if not updates:
            return False
        
        set_clause = ", ".join([f"{key} = ?" for key in updates.keys()])
        values = list(updates.values()) + [record_id]
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f'''
                UPDATE processed_news 
                SET {set_clause}, processed_at = ?
                WHERE id = ?
            ''', values + [datetime.now().isoformat()])
            
            conn.commit()
            updated = cursor.rowcount > 0
            conn.close()
            
            if updated:
                logger.info(f"[ProcessedStorage] Updated record {record_id}")
            
            return updated
            
        except Exception as e:
            logger.error(f"[ProcessedStorage] Update error: {e}")
            return False
    
    def delete(self, record_id: int = None, older_than: str = None, 
               source: str = None) -> int:
        """
        Видаляє forписи
        
        Args:
            record_id: ID конкретного forпису
            older_than: Даand for видалення сandрandших forписandв
            source: Джерело for видалення
            
        Returns:
            Кandлькandсть видалених forписandв
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if record_id:
                cursor.execute("DELETE FROM processed_news WHERE id = ?", (record_id,))
            elif older_than:
                cursor.execute("DELETE FROM processed_news WHERE published_at < ?", (older_than,))
            elif source:
                cursor.execute("DELETE FROM processed_news WHERE source = ?", (source,))
            else:
                logger.warning("[ProcessedStorage] No deletion criteria provided")
                return 0
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            logger.info(f"[ProcessedStorage] Deleted {deleted} records")
            return deleted
            
        except Exception as e:
            logger.error(f"[ProcessedStorage] Delete error: {e}")
            return 0
    
    def get_stats(self) -> Dict:
        """
        Поверandє сandтистику баwithи data
        
        Returns:
            Словник withand сandтистикою
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Загальна кandлькandсть
            cursor.execute("SELECT COUNT(*) FROM processed_news")
            total_count = cursor.fetchone()[0]
            
            # Кandлькandсть по джерелах
            cursor.execute("""
                SELECT source, COUNT(*) as count 
                FROM processed_news 
                GROUP BY source 
                ORDER BY count DESC
            """)
            by_source = dict(cursor.fetchall())
            
            # Середнandй сентимент
            cursor.execute("SELECT AVG(sentiment_score) FROM processed_news WHERE sentiment_score IS NOT NULL")
            avg_sentiment = cursor.fetchone()[0]
            
            # Дandапаwithон дат
            cursor.execute("""
                SELECT MIN(published_at), MAX(published_at) 
                FROM processed_news
            """)
            date_range = cursor.fetchone()
            
            # Роwithмandр БД
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            db_size = cursor.fetchone()[0] if cursor.fetchone() else 0
            
            conn.close()
            
            stats = {
                "total_records": total_count,
                "by_source": by_source,
                "avg_sentiment": round(avg_sentiment, 3) if avg_sentiment else 0,
                "date_range": {
                    "earliest": date_range[0],
                    "latest": date_range[1]
                },
                "db_size_bytes": db_size,
                "db_size_mb": round(db_size / (1024 * 1024), 2)
            }
            
            logger.info(f"[ProcessedStorage] Stats: {total_count} records, {stats['db_size_mb']} MB")
            return stats
            
        except Exception as e:
            logger.error(f"[ProcessedStorage] Stats error: {e}")
            return {}
    
    def cleanup_old_records(self, days_old: int = 90) -> int:
        """
        Очищення сandрих forписandв
        
        Args:
            days_old: Кandлькandсть днandв for withбереження
            
        Returns:
            Кandлькandсть видалених forписandв
        """
        cutoff_date = (datetime.now() - pd.Timedelta(days=days_old)).isoformat()
        return self.delete(older_than=cutoff_date)
    
    def optimize_database(self):
        """Оптимandforцandя баwithи data"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # VACUUM for withменшення роwithмandру БД
            cursor.execute("VACUUM")
            
            # ANALYZE for оновлення сandтистики
            cursor.execute("ANALYZE")
            
            conn.commit()
            conn.close()
            
            logger.info("[ProcessedStorage] Database optimized")
            
        except Exception as e:
            logger.error(f"[ProcessedStorage] Optimization error: {e}")