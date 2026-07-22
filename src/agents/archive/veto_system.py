import logging
from typing import List, Dict, Any

from src.agents.memory.knowledge_ingestor import KnowledgeIngestor
from src.meta_learning.memory.diary_engine import DiaryEngine, DecisionRecord, DecisionType, DecisionOutcome
from src.features.enrichers.finbert_sentiment import FinBERTSentimentAnalyzer
from src.agents.templates.cognitive_extractor import get_cognitive_prompt

logger = logging.getLogger(__name__)

class AgenticVetoSystem:
    """
    Інвестиційний Комітет (Agentic Veto System).
    Отримує рекомендації від математичних моделей і приймає рішення, 
    чи варто їх пропустити на реальні торги, ґрунтуючись на макроекономічному 
    контексті (книги), поточному сентименті (FinBERT) та минулому досвіді.
    """
    def __init__(self):
        self.ingestor = KnowledgeIngestor()
        self.diary = DiaryEngine()
        # В режимі заглушки для уникнення затримки при старті, якщо FinBERT не потрібен відразу
        self.sentiment_analyzer = None 

    def _get_sentiment_analyzer(self):
        if self.sentiment_analyzer is None:
            self.sentiment_analyzer = FinBERTSentimentAnalyzer()
        return self.sentiment_analyzer

    async def review_recommendations(self, recommendations: List[Dict[str, Any]], latest_news: str = "") -> List[Dict[str, Any]]:
        """
        Переглядає масив рекомендацій (BUY/SELL) від математичних моделей.
        Повертає оновлений список з полями 'vetoed' та 'veto_reason'.
        """
        if not recommendations:
            return []

        # 1. Рахуємо загальний сентимент новин
        sentiment = {"composite_score": 0.0}
        if latest_news:
            try:
                sentiments = self._get_sentiment_analyzer().analyze(latest_news)
                if sentiments:
                    sentiment = sentiments[0]
            except Exception as e:
                logger.error(f"Помилка аналізу сентименту: {e}")

        # 2. Шукаємо макро-контекст у книгах (Аджемоглу, Даліо)
        context_chunks = []
        if latest_news:
            # Використовуємо новину як запит для пошуку аналогій у книгах
            results = self.ingestor.search(latest_news, top_k=2)
            context_chunks = [r['content'] for r in results]

        # 3. Читаємо минулий досвід (тимчасово заглушка, оскільки DiaryEngine поки не має get_past_lessons)
        past_lessons = "Поки що немає зафіксованих помилок."

        reviewed_recs = []
        for rec in recommendations:
            # Робимо копію, щоб не мутувати оригінал
            rec_copy = rec.copy()
            
            # Якщо модель впевнена менше ніж на 40%, ми автоматично накладаємо вето
            if rec_copy.get('confidence', 0) < 0.4:
                rec_copy['vetoed'] = True
                rec_copy['veto_reason'] = "Занадто низька впевненість математичної моделі (<40%)."
                reviewed_recs.append(rec_copy)
                continue

            # Емуляція виклику LLM (Cognitive Agent)
            # В реальності тут викликається OpenAI/Anthropic з промптом get_cognitive_prompt
            # і передається: rec_copy, sentiment, context_chunks, past_lessons.
            
            veto_decision, reason, causal_graph = self._simulate_llm_decision(rec_copy, sentiment, context_chunks, past_lessons)
            
            rec_copy['vetoed'] = veto_decision
            rec_copy['veto_reason'] = reason
            rec_copy['causal_graph'] = causal_graph

            # Записуємо рішення агента в Журнал Досвіду (DuckDB)
            decision_type = DecisionType.BUY if rec_copy['action'] == "BUY" else (DecisionType.SELL if rec_copy['action'] == "SELL" else DecisionType.HOLD)
            
            record = DecisionRecord(
                agent_id="CognitiveVetoAgent",
                ticker=rec_copy['ticker'],
                decision_type=decision_type,
                reasoning=f"{'VETO' if veto_decision else 'APPROVED'}: {reason}",
                market_context={
                    "sentiment_score": sentiment.get('composite_score', 0),
                    "latest_news": latest_news[:100] if latest_news else "",
                    "veto_graph": causal_graph
                },
                context_fingerprint="veto_system_fingerprint",
                model_prediction=rec_copy.get('prediction'),
                model_confidence=rec_copy.get('confidence'),
                outcome=DecisionOutcome.PENDING
            )
            self.diary.record_decision(record)

            reviewed_recs.append(rec_copy)

        return reviewed_recs

    def _simulate_llm_decision(self, rec: Dict, sentiment: Dict, context_chunks: List[str], past_lessons: str):
        """
        Тимчасова логіка до повноцінного підключення LLM-API.
        """
        ticker = rec.get('ticker')
        action = rec.get('action') # BUY або SELL
        comp_score = sentiment.get('composite_score', 0)

        # Логіка 1: Протиріччя сентименту
        if action == "BUY" and comp_score < -0.5:
            graph = [{"source": "Negative Sentiment", "target": ticker, "probability": 0.85, "impact_direction": "Negative", "rationale": "High panic overrides math signal."}]
            return True, f"Вето: Математика пропонує BUY, але загальний ринковий сентимент екстремально негативний ({comp_score}).", graph
        
        if action == "SELL" and comp_score > 0.5:
            graph = [{"source": "Positive Sentiment", "target": ticker, "probability": 0.85, "impact_direction": "Positive", "rationale": "High euphoria overrides math signal."}]
            return True, f"Вето: Математика пропонує SELL, але ринковий сентимент екстремально позитивний ({comp_score}).", graph

        # Логіка 2: Вплив книг (якби був справжній LLM)
        if context_chunks:
            # LLM могла б прочитати context_chunks і сказати "Це криза"
            pass

        graph = [{"source": "Baseline Market", "target": ticker, "probability": 0.60, "impact_direction": "Positive" if action == "BUY" else "Negative", "rationale": "Trend continuation expected."}]
        return False, "Схвалено: Макро-контекст та сентимент не суперечать математиці.", graph

# Синглтон
veto_system = AgenticVetoSystem()
