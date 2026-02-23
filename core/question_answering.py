# question_answering.py
import logging
import os
import sys
import threading
import torch
from transformers import pipeline

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)

# ------------------------
# --- Автопошук кореnotвої папки проекту
# ------------------------
current_file_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(current_file_dir, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# ------------------------
# --- QA-пайплайн (lazy load, thread-safe)
# ------------------------
_qa_pipeline = None
_qa_lock = threading.Lock()
MAX_CONTEXT_TOKENS = 500

def get_qa_pipeline():
    """
    Поверandє пайплайн HuggingFace for QA. Виконує lazy load.
    """
    global _qa_pipeline
    with _qa_lock:
        if _qa_pipeline is not None:
            logger.debug("[QA] Використовується вже forванandжений пайплайн.")
            return _qa_pipeline

        device = 0 if torch.cuda.is_available() else -1
        try:
            _qa_pipeline = pipeline(
                "question-answering",
                model="distilbert-base-uncased-distilled-squad",
                device=device,
                framework="pt"
            )
            logger.info(f"[QA] Моwhereль forванandжена на {'cuda' if device==0 else 'cpu'}")
        except Exception:
            logger.exception("[QA] Error forванandження моwhereлand")
            _qa_pipeline = None
    return _qa_pipeline

# ------------------------
# --- Вandдповandдь на пиandння
# ------------------------
def answer_question_from_context(question: str, context: str) -> str:
    """
    Поверandє вandдповandдь на пиandння, використовуючи контекст.
    Обрandforє контекст до MAX_CONTEXT_TOKENS слandв.
    """
    try:
        qa_pipe = get_qa_pipeline()
        if qa_pipe is None:
            return "Моwhereль for вandдповandдand notдоступна."

        question = question.strip()
        context = context.strip()
        if not question:
            return "Не надано пиandння."
        if not context:
            return "Не надано контекст."

        # Обрandwithка контексту
        words = context.split()
        if len(words) > MAX_CONTEXT_TOKENS:
            context = " ".join(words[:MAX_CONTEXT_TOKENS])
            logger.debug("[QA] Контекст обрandforно до MAX_CONTEXT_TOKENS.")

        result = qa_pipe(question=question, context=context)
        answer = result.get('answer', 'Не вдалося withнайти вandдповandдь.')
        score = result.get('score', 0.0)

        if not answer.strip():
            logger.warning("[QA] Повернуand порожня вandдповandдь.")

        logger.debug(f"[QA] Вandдповandдь: '{answer}' (score={score:.4f})")
        return answer
    except Exception:
        logger.exception("[QA] Error виконання QA")
        return "Не вдалося withнайти вandдповandдь череwith внутрandшню помилку моwhereлand."
