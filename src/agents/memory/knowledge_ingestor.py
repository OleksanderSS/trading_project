import os
import glob
import logging
from typing import List, Dict
import json

try:
    from pypdf import PdfReader
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.error("Для роботи Knowledge Ingestor потрібні: pypdf, faiss-cpu, sentence-transformers")

logger = logging.getLogger(__name__)

class KnowledgeIngestor:
    def __init__(self, books_dir: str = "data/knowledge_base/books", index_path: str = "data/memory/faiss_index"):
        self.books_dir = books_dir
        self.index_path = index_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2') # Легка, швидка модель для ембедінгів
        self.dimension = self.model.get_sentence_embedding_dimension()
        
        os.makedirs(self.books_dir, exist_ok=True)
        os.makedirs(self.index_path, exist_ok=True)

        self.index_file = os.path.join(self.index_path, "knowledge.index")
        self.meta_file = os.path.join(self.index_path, "metadata.json")

        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Отримує сирий текст з PDF файлу."""
        text = ""
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        except Exception as e:
            logger.error(f"Помилка читання {pdf_path}: {e}")
        return text

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Розбиває текст на чанки з перекриттям."""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    def ingest_new_books(self):
        """Шукає нові PDF, витягує текст, робить ембедінги і зберігає у FAISS."""
        existing_sources = {m["source"] for m in self.metadata}
        
        pdf_files = glob.glob(os.path.join(self.books_dir, "*.pdf"))
        new_files = [f for f in pdf_files if os.path.basename(f) not in existing_sources]
        
        if not new_files:
            logger.info("Нових файлів для індексування не знайдено.")
            return

        logger.info(f"Знайдено {len(new_files)} нових файлів. Починаю індексування...")

        new_embeddings = []
        new_meta = []

        for file in new_files:
            logger.info(f"Обробка: {os.path.basename(file)}")
            text = self.extract_text_from_pdf(file)
            chunks = self.chunk_text(text)
            
            if not chunks:
                continue

            try:
                embeddings = self.model.encode(chunks, show_progress_bar=False)
                for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    new_embeddings.append(emb)
                    new_meta.append({
                        "source": os.path.basename(file),
                        "chunk_id": i,
                        "content": chunk
                    })
            except Exception as e:
                logger.error(f"Помилка векторизації файлу {file}: {e}")

        if new_embeddings:
            # Додаємо до FAISS
            embeddings_np = np.array(new_embeddings).astype('float32')
            self.index.add(embeddings_np)
            self.metadata.extend(new_meta)

            # Зберігаємо
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Успішно проіндексовано та додано {len(new_meta)} чанків у базу знань.")

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Шукає релевантні чанки тексту за запитом."""
        if self.index.ntotal == 0:
            return []
            
        query_emb = self.model.encode([query]).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                res = self.metadata[idx]
                res["distance"] = float(dist)
                results.append(res)
        return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = KnowledgeIngestor()
    # Закоментовано, щоб не почати випадково індексувати всі 260 файлів при імпорті
    # ingestor.ingest_new_books()
