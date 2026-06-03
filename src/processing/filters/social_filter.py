import pandas as pd
from typing import Dict, Tuple, Any, Optional
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("SocialFilter")

class SocialFilter:
    """Specialized filter for social media data (Reddit, etc.)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.reddit_score_threshold = config.get('reddit_score_threshold', 1)
        self.reddit_text_min_len = config.get('reddit_text_min_len', 10)

    def filter_reddit_data(self, reddit_data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Intelligently filters Reddit sentiment data."""
        if not isinstance(reddit_data, pd.DataFrame) or reddit_data.empty:
            return pd.DataFrame(), {'status': 'empty', 'posts': 0}

        initial_count = len(reddit_data)
        
        # 1. Score filter
        if 'score' in reddit_data.columns:
            reddit_data = reddit_data[reddit_data['score'] >= self.reddit_score_threshold]
            
        # 2. Text length filter
        if 'text' in reddit_data.columns:
            reddit_data = reddit_data[reddit_data['text'].str.len() >= self.reddit_text_min_len]
            
        return reddit_data, {
            'status': 'accepted',
            'initial_posts': initial_count,
            'final_posts': len(reddit_data),
            'removed': initial_count - len(reddit_data)
        }
