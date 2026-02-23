#!/usr/bin/env python3
"""
Textual Data Integrator - об'єднує всі текстові колектори
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

logger = logging.getLogger("trading_project.textual_integrator")


class TextualDataIntegrator:
    """Інтегратор текстових data з різних джерел"""
    
    def __init__(self):
        self.news_collectors = {}
        self.hf_collector = None
        self.sentiment_analyzers = {}
        
    def register_news_collector(self, name: str, collector):
        """Реєстрація новинного колектора"""
        self.news_collectors[name] = collector
        logger.info(f"[TextualIntegrator] Registered news collector: {name}")
    
    def register_hf_collector(self, collector):
        """Реєстрація HuggingFace колектора"""
        self.hf_collector = collector
        logger.info("[TextualIntegrator] Registered HF collector")
    
    def register_sentiment_analyzer(self, name: str, analyzer):
        """Реєстрація сентимент аналізатора"""
        self.sentiment_analyzers[name] = analyzer
        logger.info(f"[TextualIntegrator] Registered sentiment analyzer: {name}")
    
    def collect_all_textual_data(self, tickers: List[str], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Збір всіх текстових data"""
        textual_data = {
            'news': {},
            'hf_models': {},
            'sentiment_analysis': {},
            'metadata': {
                'tickers': tickers,
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'collection_time': datetime.now().isoformat()
            }
        }
        
        # 1. Збір новин з усіх джерел
        logger.info("[TextualIntegrator] Collecting news from all sources...")
        for name, collector in self.news_collectors.items():
            try:
                if hasattr(collector, 'collect'):
                    news_data = collector.collect()
                elif hasattr(collector, 'collect_data'):
                    news_data = collector.collect_data(
                        start_date=start_date,
                        end_date=end_date
                    )
                else:
                    logger.warning(f"[TextualIntegrator] Unknown collector type: {name}")
                    continue
                
                textual_data['news'][name] = news_data
                logger.info(f"[TextualIntegrator] Collected {len(news_data) if hasattr(news_data, '__len__') else 'unknown'} items from {name}")
                
            except Exception as e:
                logger.error(f"[TextualIntegrator] Failed to collect from {name}: {e}")
                textual_data['news'][name] = pd.DataFrame()
        
        # 2. Збір HF моделей/data
        if self.hf_collector:
            try:
                logger.info("[TextualIntegrator] Collecting HF data...")
                hf_data = self.hf_collector.collect()
                textual_data['hf_models'] = hf_data
                logger.info(f"[TextualIntegrator] Collected HF data: {type(hf_data)}")
            except Exception as e:
                logger.error(f"[TextualIntegrator] Failed to collect HF data: {e}")
                textual_data['hf_models'] = {}
        
        # 3. Сентимент аналіз всіх текстових data
        logger.info("[TextualIntegrator] Running sentiment analysis...")
        all_text_data = self._combine_all_text(textual_data['news'])
        
        if not all_text_data.empty:
            for name, analyzer in self.sentiment_analyzers.items():
                try:
                    logger.info(f"[TextualIntegrator] Running {name} sentiment analysis...")
                    sentiment_results = []
                    
                    for idx, row in all_text_data.iterrows():
                        text = str(row.get('title', '')) + ' ' + str(row.get('description', ''))
                        result = analyzer.analyze_sentiment(text)
                        sentiment_results.append({
                            'index': idx,
                            'source': row.get('source', 'unknown'),
                            'title': row.get('title', ''),
                            'description': row.get('description', ''),
                            f'{name}_label': result.get('label', 'NEUTRAL'),
                            f'{name}_score': result.get('score', 0.0),
                            f'{name}_confidence': result.get('confidence', 0.0),
                            'published_at': row.get('published_at', datetime.now().isoformat())
                        })
                    
                    textual_data['sentiment_analysis'][name] = sentiment_results
                    logger.info(f"[TextualIntegrator] {name} analysis completed: {len(sentiment_results)} items")
                    
                except Exception as e:
                    logger.error(f"[TextualIntegrator] {name} sentiment analysis failed: {e}")
                    textual_data['sentiment_analysis'][name] = []
        
        return textual_data
    
    def _combine_all_text(self, news_data: Dict[str, Any]) -> pd.DataFrame:
        """Об'єднання всіх текстових data в один DataFrame"""
        all_texts = []
        
        for source, data in news_data.items():
            if isinstance(data, pd.DataFrame) and not data.empty:
                # Додаємо джерело до кожного запису
                data_copy = data.copy()
                data_copy['source'] = source
                all_texts.append(data_copy)
        
        if all_texts:
            combined_df = pd.concat(all_texts, ignore_index=True)
            logger.info(f"[TextualIntegrator] Combined {len(combined_df)} text items from {len(news_data)} sources")
            return combined_df
        else:
            logger.warning("[TextualIntegrator] No text data to combine")
            return pd.DataFrame()
    
    def get_unified_textual_dataframe(self, textual_data: Dict[str, Any]) -> pd.DataFrame:
        """Отримати уніфікований DataFrame з усіма текстовими даними"""
        unified_data = []
        
        # Додаємо новини
        for source, news_df in textual_data.get('news', {}).items():
            if isinstance(news_df, pd.DataFrame) and not news_df.empty:
                for _, row in news_df.iterrows():
                    unified_data.append({
                        'type': 'news',
                        'source': source,
                        'title': row.get('title', ''),
                        'description': row.get('description', ''),
                        'published_at': row.get('published_at', ''),
                        'url': row.get('url', ''),
                        'sentiment_score': row.get('sentiment', 0.0),
                        'mention_score': row.get('mention_score', 0.0)
                    })
        
        # Додаємо HF дані
        hf_data = textual_data.get('hf_models', {})
        if hf_data:
            for key, value in hf_data.items():
                unified_data.append({
                    'type': 'hf_model',
                    'source': 'huggingface',
                    'title': key,
                    'description': str(value),
                    'published_at': datetime.now().isoformat(),
                    'url': '',
                    'sentiment_score': 0.0,
                    'mention_score': 0.0
                })
        
        # Додаємо сентимент аналіз
        for analyzer_name, sentiment_results in textual_data.get('sentiment_analysis', {}).items():
            for result in sentiment_results:
                unified_data.append({
                    'type': 'sentiment_analysis',
                    'source': result.get('source', 'unknown'),
                    'title': result.get('title', ''),
                    'description': result.get('description', ''),
                    'published_at': result.get('published_at', ''),
                    'url': '',
                    'sentiment_score': result.get(f'{analyzer_name}_score', 0.0),
                    'mention_score': result.get(f'{analyzer_name}_confidence', 0.0)
                })
        
        if unified_data:
            return pd.DataFrame(unified_data)
        else:
            return pd.DataFrame()
    
    def get_summary_stats(self, textual_data: Dict[str, Any]) -> Dict[str, Any]:
        """Отримати статистику по текстових data"""
        stats = {
            'total_news_items': 0,
            'news_sources': list(textual_data.get('news', {}).keys()),
            'hf_models_count': len(textual_data.get('hf_models', {})),
            'sentiment_analyzers': list(textual_data.get('sentiment_analysis', {}).keys()),
            'sentiment_items': 0
        }
        
        # Підраховуємо новини
        for source, news_df in textual_data.get('news', {}).items():
            if isinstance(news_df, pd.DataFrame):
                stats['total_news_items'] += len(news_df)
        
        # Підраховуємо сентимент аналіз
        for analyzer, results in textual_data.get('sentiment_analysis', {}).items():
            stats['sentiment_items'] += len(results)
        
        return stats
