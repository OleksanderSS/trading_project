#!/usr/bin/env python3
"""
Утиліта для імпорту книг та статей (PDF/TXT) в Базу Даних (Knowledge Base RAG).
Використовує sentence-transformers для створення векторів та зберігає в DuckDB.

Вимоги:
    pip install PyPDF2 sentence-transformers

Використання:
    python scripts/knowledge_ingestor.py
"""

import os
import sys
import argparse
from pathlib import Path

# Додаємо проект до шляху
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import PyPDF2
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("❌ Помилка: Не встановлені необхідні бібліотеки для RAG.")
    print("Виконайте: pip install PyPDF2 sentence-transformers")
    sys.exit(1)

from src.core.database.database_manager import DatabaseManager
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger(__name__)

BOOKS_DIR = PROJECT_ROOT / "data" / "knowledge_base" / "books"
TABLE_NAME = "knowledge_base_documents"
MODEL_NAME = "all-MiniLM-L6-v2"  # Швидка та легка модель

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Витягує текст з PDF файлу."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        logger.error(f"Помилка читання PDF {pdf_path.name}: {e}")
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Розбиває текст на чанки по словам."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks

def setup_database(db: DatabaseManager):
    """Створює таблицю та інсталює VSS для DuckDB."""
    conn = db.get_connection()
    try:
        # Встановлення VSS розширення (може зайняти час при першому запуску)
        conn.execute("INSTALL vss;")
        conn.execute("LOAD vss;")
        
        # Створення таблиці
        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id VARCHAR PRIMARY KEY,
                source VARCHAR,
                chunk_index INTEGER,
                content TEXT,
                embedding FLOAT[384]
            );
        """)
        logger.info("✅ База даних та таблиця для RAG успішно налаштовані.")
    except Exception as e:
        logger.error(f"Помилка налаштування DuckDB VSS: {e}")
        raise

def process_documents():
    """Основний цикл обробки документів."""
    if not BOOKS_DIR.exists():
        logger.warning(f"Директорія {BOOKS_DIR} не існує. Створюю...")
        BOOKS_DIR.mkdir(parents=True, exist_ok=True)
        return
        
    files = list(BOOKS_DIR.glob("*.pdf")) + list(BOOKS_DIR.glob("*.txt"))
    if not files:
        logger.info(f"📁 Немає файлів для обробки у {BOOKS_DIR}")
        return
        
    logger.info(f"Знайдено {len(files)} файлів. Завантажую AI модель '{MODEL_NAME}'...")
    model = SentenceTransformer(MODEL_NAME)
    
    config = UnifiedConfigManager()
    db = DatabaseManager(config)
    setup_database(db)
    conn = db.get_connection()
    
    for file_path in files:
        logger.info(f"📖 Читання {file_path.name}...")
        
        # 1. Читання тексту
        if file_path.suffix.lower() == '.pdf':
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
                
        if not text.strip():
            logger.warning(f"Файл {file_path.name} порожній або не вдалося прочитати.")
            continue
            
        # 2. Чанкінг
        chunks = chunk_text(text)
        logger.info(f"   Розбито на {len(chunks)} фрагментів. Векторизація...")
        
        # 3. Векторизація
        embeddings = model.encode(chunks)
        
        # 4. Збереження в базу
        source_name = file_path.name
        records_inserted = 0
        
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{source_name}_{i}"
            try:
                # Використовуємо ON CONFLICT DO NOTHING щоб не дублювати
                conn.execute(f"""
                    INSERT INTO {TABLE_NAME} (id, source, chunk_index, content, embedding)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO NOTHING;
                """, (chunk_id, source_name, i, chunk, emb.tolist()))
                records_inserted += 1
            except Exception as e:
                logger.error(f"Помилка вставки фрагмента {i}: {e}")
                
        logger.info(f"✅ Збережено {records_inserted} нових векторів з {source_name}.")

if __name__ == "__main__":
    logger.info("=" * 50)
    logger.info("📚 ЗАПУСК ІНГЕСТОРА БАЗИ ЗНАНЬ (RAG)")
    logger.info("=" * 50)
    process_documents()
    logger.info("Завершено.")
