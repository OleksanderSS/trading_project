# collectors/google_news_collector.py

import feedparser
import pandas as pd
import time
import random
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from utils.logger_fixed import ProjectLogger
from utils.cache_utils import CacheManager
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = ProjectLogger.get_logger("GoogleNewsCollector")


class GoogleNewsCollector:
    """
    Advanced Google News RSS collector with multiple keyword feeds,
    strict filtering, and deduplication.
    """
    
    def __init__(
        self,
        keywords: List[str],
        source_quality_config: Dict[str, float],
        cache_manager: Optional[CacheManager] = None,
        similarity_threshold: float = 0.85,
        delay_range: tuple = (1.0, 3.0),
        days_back: int = 60
    ):
        self.keywords = keywords
        self.source_quality = source_quality_config
        self.cache_manager = cache_manager
        self.similarity_threshold = similarity_threshold
        self.delay_range = delay_range
        self.days_back = days_back
        
        # Single keywords for initial detection
        self.trigger_keywords = [
            # Companies/Tech
            "tesla", "tsla", "nvidia", "nvda", "apple", "aapl", "microsoft", "msft",
            "google", "googl", "meta", "amazon", "amzn", "intel", "intc", "ibm",
            "oracle", "orcl", "amd", "qualcomm", "qcom", "adobe", "adbe",
            "salesforce", "crm", "netflix", "nflx", "spotify", "spot",
            
            # Economic/Financial
            "fed", "federal", "reserve", "inflation", "recession", "economy", "gdp",
            "unemployment", "interest", "rates", "oil", "gas", "energy", "commodities",
            "gold", "silver", "bitcoin", "crypto", "etf", "nasdaq", "dow", "sp500",
            
            # Natural disasters
            "hurricane", "tornado", "earthquake", "flood", "wildfire", "drought",
            "storm", "tsunami", "volcano", "blizzard", "cyclone",
            
            # Sports/Entertainment entities
            "nfl", "nba", "mlb", "nhl", "olympics", "fifa", "uefa", "disney",
            "warner", "universal", "sony", "paramount", "spotify", "apple music"
        ]
        
        # Context patterns for determining importance
        self.importance_contexts = {
            # Financial contexts
            "financial": [
                "earnings", "revenue", "profit", "loss", "merger", "acquisition",
                "buyback", "dividend", "stock", "share", "price", "market", "investment",
                "funding", "valuation", "ipo", "offering", "deal", "sale", "purchase"
            ],
            
            # Economic impact contexts  
            "economic": [
                "prices", "supply", "demand", "production", "output", "export", "import",
                "shutdown", "disruption", "shortage", "surplus", "inflation", "growth",
                "decline", "crisis", "recovery", "impact", "affect", "influence"
            ],
            
            # Natural disaster economic impact
            "disaster": [
                "damage", "destruction", "facility", "plant", "refinery", "port",
                "infrastructure", "production", "operations", "shutdown", "evacuation",
                "billion", "million", "cost", "loss", "economic", "business"
            ],
            
            # Sports/Entertainment financial contexts
            "media_financial": [
                "rights", "deal", "acquisition", "merger", "buyout", "sale", "purchase",
                "billion", "million", "valuation", "contract", "broadcast", "streaming",
                "franchise", "team", "league", "studio", "catalog", "intellectual"
            ]
        }
        
        # True trash patterns (always filter out) - more specific
        self.trash_patterns = [
            "office party", "corporate event", "team building", "christmas party",
            "wedding", "birthday", "anniversary", "vacation", "recipe", "fashion",
            "beauty", "health tips", "dating", "relationships", "gardening", "pets",
            "local news", "community event", "school news", "traffic", "accident",
            "concert", "nightclub", "celebrity gossip", "reality tv", "religion",
            "church", "art", "museum", "crime", "police"
        ]
        
        # Important financial keywords
        self.important_keywords = [
            "earnings", "revenue", "profit", "loss", "merger", "acquisition",
            "sec filing", "regulation", "lawsuit", "patent", "approval",
            "launch", "recall", "expansion", "closure", "bankruptcy",
            "dividend", "buyback", "stock split", "ipo", "offering",
            "quarterly", "annual", "guidance", "forecast", "outlook"
        ]
        
        # Market-moving patterns
        self.market_moving_patterns = [
            "stock price", "share price", "market cap", "trading volume",
            "analyst rating", "price target", "upgrade", "downgrade",
            "beat estimates", "missed estimates", "surprise earnings"
        ]
        
        # Neutral but important patterns (contextual importance)
        self.neutral_important_patterns = [
            # Analyst predictions and expectations
            "analyst predicts", "analyst expects", "forecast suggests",
            "analyst forecast", "price target", "estimates show",
            "wall street expects", "analysts project",
            
            # CEO/Executive actions and statements
            "ceo announces", "ceo confirms", "ceo reveals",
            "executive decision", "management plans", "board approves",
            "leadership change", "executive departure", "ceo statement",
            
            # Expected corporate actions
            "expected to sell", "plans to sell", "considering sale",
            "rumored to sell", "speculation about", "market expects",
            "investors anticipate", "shareholders expect",
            
            # Economic indicators and consumer data
            "credit card debt", "consumer debt", "household debt",
            "consumer spending", "retail sales", "consumer confidence",
            "personal savings rate", "mortgage rates", "housing market",
            "unemployment claims", "jobless claims", "labor market",
            
            # Economic disasters and crises
            "economic crisis", "financial crisis", "market crash",
            "recession warning", "economic collapse", "banking crisis",
            "debt crisis", "currency crisis", "inflation crisis",
            "supply chain crisis", "energy crisis", "food crisis",
            
            # Climate and natural disasters with economic impact
            "hurricane", "tropical storm", "oil facilities", "refinery damage",
            "gulf coast", "oil production", "energy infrastructure", "port closure",
            "supply disruption", "commodity shortage", "agricultural losses",
            "flood damage", "wildfire", "drought impact", "energy prices spike",
            
            # Sports and entertainment with financial impact
            "sports franchise sale", "team acquisition", "stadium deal",
            "broadcast rights", "media rights deal", "sports league merger",
            "entertainment acquisition", "music catalog purchase", "film studio buyout",
            "streaming rights", "content acquisition", "intellectual property buyout",
            
            # Regulatory and legal developments
            "sec filing", "regulatory approval", "compliance deadline",
            "legal settlement", "court ruling", "antitrust review",
            "government investigation", "regulatory scrutiny",
            
            # Market structure changes
            "stock split", "dividend declaration", "buyback program",
            "share offering", "secondary offering", "dilution",
            "shareholder meeting", "proxy vote",
            
            # Major M&A and media deals (KEEP THESE)
            "merger talks", "acquisition talks", "buyout talks",
            "netflix acquire", "disney acquire", "media merger",
            "entertainment merger", "streaming merger", "studio acquisition",
            "tech merger", "media buyout", "entertainment buyout",
            
            # Partnership and collaboration deals
            "partnership announced", "joint venture", "strategic alliance",
            "collaboration deal", "technology sharing", "licensing agreement",
            
            # Product and service launches
            "product launch", "service rollout", "new product line",
            "technology upgrade", "platform update", "feature release",
            
            # Supply chain and operational changes
            "supply chain", "production halt", "factory closure",
            "operational changes", "restructuring", "cost cutting",
            
            # Market position and competition
            "market share", "competitive position", "industry ranking",
            "market leader", "competitive advantage", "market penetration"
        ]
    
    def fetch_raw_news(self) -> pd.DataFrame:
        """Fetch raw news from Google News RSS feeds without filtering."""
        return self.fetch_all_feeds()
    
    def fetch_all_feeds(self) -> pd.DataFrame:
        """Fetch news from all keyword feeds with 60-day range."""
        all_news = []
        cutoff_date = datetime.now() - timedelta(days=self.days_back)
        
        for keyword in self.keywords:
            try:
                logger.debug(f"Fetching Google News for keyword: {keyword}")
                
                # Generate RSS URL for single keyword
                rss_url = self._generate_google_news_url(keyword)
                
                # Fetch feed
                feed_data = self._fetch_single_feed(rss_url, keyword, cutoff_date)
                
                if not feed_data.empty:
                    all_news.append(feed_data)
                
                # Random delay to avoid rate limiting
                delay = random.uniform(*self.delay_range)
                try:
                    time.sleep(delay)
                except KeyboardInterrupt:
                    logger.info("Google News collection interrupted by user")
                    break
                
            except Exception as e:
                logger.error(f"Error fetching keyword {keyword}: {e}")
                continue
        
        if all_news:
            result = pd.concat(all_news, ignore_index=True)
            logger.info(f"Total Google News collected: {len(result)}")
            return result
        else:
            logger.warning("No news collected from any Google News feeds")
            return pd.DataFrame()
    
    def collect_data(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Основний метод збору data для сумісності з іншими колекторами
        
        Args:
            start_date: Початкова дата
            end_date: Кінцева дата
            
        Returns:
            pd.DataFrame: Зібрані дані з повною фільтрацією
        """
        # Використовуємо поточну дату, якщо не вказано
        if start_date is None:
            start_date = datetime.now() - timedelta(days=self.days_back)
        if end_date is None:
            end_date = datetime.now()
            
        # Збираємо дані з повною обробкою
        logger.info(f"[GoogleNewsCollector] Starting data collection with full filtering...")
        raw_data = self.fetch_all_feeds()
        
        if not raw_data.empty:
            # Застосовуємо повну фільтрацію
            filtered_data = self.filter_and_classify(raw_data)
            logger.info(f"[GoogleNewsCollector] Final filtered data: {len(filtered_data)} articles")
            return filtered_data
        else:
            return pd.DataFrame()
    
    def _generate_google_news_url(self, keyword: str) -> str:
        """Generate Google News RSS URL for single keyword."""
        encoded_keyword = keyword.replace(" ", "%20")
        return f"https://news.google.com/rss/search?q={encoded_keyword}&hl=en-US&gl=US&ceid=US:en"
    
    def _fetch_single_feed(self, rss_url: str, keyword: str, cutoff_date: datetime) -> pd.DataFrame:
        """Fetch and parse single RSS feed with date filtering."""
        try:
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                logger.warning(f"No entries found for keyword: {keyword}")
                return pd.DataFrame()
            
            news_items = []
            
            for entry in feed.entries:
                try:
                    # Extract basic info
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    link = entry.get("link", "")
                    published = entry.get("published", "")
                    
                    # Parse and filter by date
                    published_date = self._parse_date(published)
                    if published_date < cutoff_date:
                        continue
                    
                    # Extract source from title or summary
                    source = self._extract_source(title, summary)
                    
                    # Create news item
                    news_item = {
                        "title": title,
                        "summary": summary,
                        "url": link,
                        "source": source,
                        "keyword": keyword,
                        "published_at": published_date,
                        "collected_at": datetime.now()
                    }
                    
                    news_items.append(news_item)
                    
                except Exception as e:
                    logger.warning(f"Error parsing entry: {e}")
                    continue
            
            if news_items:
                df = pd.DataFrame(news_items)
                logger.debug(f"Fetched {len(df)} items for keyword: {keyword}")
                return df
            else:
                logger.warning(f"No valid items for keyword: {keyword}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching feed {rss_url}: {e}")
            return pd.DataFrame()
    
    def _extract_source(self, title: str, summary: str) -> str:
        """Extract source name from title or summary using HTML parsing."""
        import re
        
        # First try to extract from HTML font tags (Google News format)
        html_pattern = r'<font[^>]*color="#6f6f6f"[^>]*>([^<]+)</font>'
        html_match = re.search(html_pattern, f"{title} {summary}")
        
        if html_match:
            source_name = html_match.group(1).strip()
            return source_name
        
        # Fallback to text-based extraction
        text = f"{title} {summary}".lower()
        
        # Tier 1 - Economy & Finance (Highest weight)
        tier_1 = [
            "bloomberg", "bloomberg.com", "bbg",
            "reuters", "reuters.com", 
            "wall street journal", "wall street journal.com", "wsj", "wsj.com",
            "financial times", "financial times.com", "ft", "ft.com",
            "the economist", "economist.com", "economist",
            "cnbc", "cnbc.com"
        ]
        tier_2 = [
            "cnbc", "cnbc.com",
            "bbc", "bbc.com",
            "marketwatch", "marketwatch.com",
            "business insider", "business insider.com",
            "fortune", "fortune.com",
            "forbes", "forbes.com",
            "money", "money.com"
        ]
        tier_3 = [
            "yahoo finance", "yahoo finance.com", "finance.yahoo.com", "yahoo finance uk",
            "investing.com", "investing.com",
            "the motley fool", "motley fool", "fool.com",
            "morningstar", "morningstar.com"
        ]
        tier_4 = [
            "seeking alpha", "seekingalpha.com", "seeking alpha.com",
            "benzinga", "benzinga.com",
            "investor's business daily", "investors.com", "ibd",
            "thestreet", "thestreet.com",
            "nasdaq", "nasdaq.com"
        ]
        tier_5 = [
            "barron's", "barrons.com",
            "kiplinger", "kiplinger.com",
            "marketrealist", "marketrealist.com",
            "thestockadvisor", "stockadvisor.com",
            "tipranks", "tipranks.com",
            "zacks", "zacks.com",
            "ycharts", "ycharts.com"
        ]
        # Tier 2 - Technology & AI (For sector analysis)
        tier_6 = [
            "techcrunch", "techcrunch.com",
            "the information", "theinformation.com",
            "axios", "axios.com",
            "wired", "wired.com",
            "the verge", "theverge.com",
            "arstechnica", "arstechnica.com",
            "mit technology review", "technologyreview.com", "mit review",
            "arxiv", "arxiv.org",
            # Specialized Tesla sources
            "electrek", "electrek.co",
            "teslarati", "teslarati.com"
        ]
        # Tier 3 - Politics & Geopolitics (VIX & Gold impact)
        tier_7 = [
            "foreign affairs", "foreignaffairs.com",
            "politico", "politico.com",
            "al jazeera", "aljazeera.com", "al jazeera english"
        ]
        
        for source_list in [tier_1, tier_2, tier_3, tier_4, tier_5, tier_6, tier_7]:
            for source in source_list:
                if source in text:
                    return source.title()
        
        return "Unknown"
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse date from various formats."""
        if not date_str or date_str.strip() == "":
            return None
            
        try:
            formats = [
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S%z",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%a, %d %b %Y %H:%M:%S %Z",
                "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S.%fZ",
                # Додатковand формати for Google News
                "%Y-%m-%dT%H:%M:%S+00:00",
                "%Y-%m-%dT%H:%M:%S-00:00",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%d %H:%M:%S+00:00",
                "%Y-%m-%d %H:%M:%S-00:00"
            ]
            
            for fmt in formats:
                try:
                    parsed_date = datetime.strptime(date_str.strip(), fmt)
                    # Логування for whereбагу
                    logger.debug(f"Parsed date: '{date_str}' -> {parsed_date} with format: {fmt}")
                    return parsed_date
                except ValueError:
                    continue
            
            # Якщо жоwhereн формат not пandдandйшов, логуємо помилку
            logger.warning(f"Failed to parse date: '{date_str}'")
            return None  # Return None if no format matches
            
        except Exception:
            return None  # Return None on any error
    
    def filter_and_classify(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized pipeline: word parsing  trash filter  logic analysis  signal calculation."""
        if news_df.empty:
            return pd.DataFrame()
        
        try:
            logger.info(f"Starting optimized pipeline with {len(news_df)} news items")
            
            # Stage 1: Quick word parsing and trash filtering
            filtered_df = self._quick_trash_filter(news_df)
            logger.info(f"Stage 1 - Quick trash filter: {len(news_df)} -> {len(filtered_df)}")
            
            if filtered_df.empty:
                return filtered_df
            
            # Stage 2: Logic and context analysis
            analyzed_df = self._logic_analysis(filtered_df)
            logger.info(f"Stage 2 - Logic analysis: {len(filtered_df)} -> {len(analyzed_df)}")
            
            if analyzed_df.empty:
                return analyzed_df
            
            # Stage 3: Apply source quality weighting
            weighted_df = self._apply_source_quality_weighting(analyzed_df)
            
            # Stage 4: Classify news importance
            classified_df = self._classify_news_importance(weighted_df)
            
            # Stage 5: Calculate final signal strength
            final_df = self._calculate_signal_strength(classified_df)
            
            # Stage 6: Remove duplicates using TF-IDF
            result_df = self._remove_duplicates(final_df)
            
            logger.info(f"Final result: {len(news_df)} -> {len(result_df)} news items")
            return result_df
            
        except Exception as e:
            logger.error(f"Error in optimized pipeline: {e}")
            return pd.DataFrame()
    
    def _quick_trash_filter(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 1: Fast word parsing and immediate trash filtering."""
        if news_df.empty:
            return pd.DataFrame()
        
        def is_obvious_trash(title: str, summary: str) -> bool:
            # Clean text - remove HTML tags and URLs
            import re
            clean_title = re.sub(r'<[^>]+>', '', title)
            clean_summary = re.sub(r'<[^>]+>', '', summary)
            clean_summary = re.sub(r'https?://[^\s]+', '', clean_summary)
            
            text = f"{clean_title} {clean_summary}".lower()
            # Quick check for obvious trash patterns
            found_patterns = [pattern for pattern in self.trash_patterns if pattern in text]
            if found_patterns:
                logger.debug(f"Trash patterns found: {found_patterns} in: {clean_title[:50]}...")
                return True
            return False
        
        def has_trigger_keyword(title: str, summary: str) -> bool:
            # Clean text - remove HTML tags and URLs
            import re
            clean_title = re.sub(r'<[^>]+>', '', title)
            clean_summary = re.sub(r'<[^>]+>', '', summary)
            clean_summary = re.sub(r'https?://[^\s]+', '', clean_summary)
            
            text = f"{clean_title} {clean_summary}".lower()
            return any(keyword in text for keyword in self.trigger_keywords)
        
        # First filter obvious trash
        trash_mask = news_df.apply(lambda row: is_obvious_trash(row["title"], row["summary"]), axis=1)
        # Then keep only items with trigger keywords
        trigger_mask = news_df.apply(lambda row: has_trigger_keyword(row["title"], row["summary"]), axis=1)
        
        # Keep items that are NOT trash AND have trigger keywords
        mask = ~trash_mask & trigger_mask
        filtered_count = len(news_df[mask])
        logger.info(f"Quick filter stats: Total={len(news_df)}, "
            f"Trash={trash_mask.sum()}, "
            f"Trigger={trigger_mask.sum()}, "
            f"Result={filtered_count}")
        return news_df[mask].copy()
    
    def _logic_analysis(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 2: Deep logic and context analysis."""
        if news_df.empty:
            return pd.DataFrame()
        
        def has_economic_logic(title: str, summary: str) -> bool:
            # Clean text - remove HTML tags and URLs
            import re
            clean_title = re.sub(r'<[^>]+>', '', title)
            clean_summary = re.sub(r'<[^>]+>', '', summary)
            clean_summary = re.sub(r'https?://[^\s]+', '', clean_summary)
            
            text = f"{clean_title} {clean_summary}".lower()
            
            # Check for economic/financial context
            financial_context = any(context in text for context in self.importance_contexts["financial"])
            economic_context = any(context in text for context in self.importance_contexts["economic"])
            
            # Special context checks
            disaster_context = False
            if any(disaster in text for disaster in ["hurricane", "tornado", "earthquake", "flood", "wildfire"]):
                disaster_context = any(context in text for context in self.importance_contexts["disaster"])
            
            media_context = False
            if any(entity in text for entity in ["nfl", "nba", "disney", "warner", "sony", "universal"]):
                media_context = any(context in text for context in self.importance_contexts["media_financial"])
            
            return financial_context or economic_context or disaster_context or media_context
        
        mask = news_df.apply(lambda row: has_economic_logic(row["title"], row["summary"]), axis=1)
        return news_df[mask].copy()
    
    def _calculate_signal_strength(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Stage 5: Calculate final signal strength with cross-reference validation."""
        if news_df.empty:
            return pd.DataFrame()
        
        # Base signal strength = source_quality_weight * importance_weight
        news_df["signal_strength"] = (
            news_df["source_quality_weight"] * 
            news_df["importance_weight"]
        )
        
        # Cross-reference validation
        news_df = self._apply_cross_reference_validation(news_df)
        
        return news_df
    
    def _apply_cross_reference_validation(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply cross-reference validation to boost signal strength for confirmed news."""
        if news_df.empty:
            return news_df
        
        # Sort by published_at for time window analysis
        news_df = news_df.sort_values('published_at')
        
        # Define Tier 1 sources for cross-reference
        tier_1_sources = {
            'bloomberg', 'reuters', 'wall street journal', 'wsj', 
            'financial times', 'ft', 'the economist', 'cnbc'
        }
        
        # Create normalized title for matching similar news
        news_df['normalized_title'] = news_df['title'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        
        for idx, row in news_df.iterrows():
            source_lower = row['source'].lower()
            
            # Check if this is a Tier 1 source
            is_tier_1 = any(tier in source_lower for tier in tier_1_sources)
            
            if is_tier_1:
                # Look for similar news from other Tier 1 sources within 30 minutes
                time_window = pd.Timedelta(minutes=30)
                start_time = row['published_at'] - time_window
                end_time = row['published_at'] + time_window
                
                # Find similar news in time window
                similar_news = news_df[
                    (news_df['published_at'] >= start_time) &
                    (news_df['published_at'] <= end_time) &
                    (news_df.index != idx) &
                    (news_df['normalized_title'].str.contains(
                        self._extract_keywords(row['normalized_title']), 
                        na=False
                    ))
                ]
                
                # Check if any similar news from other Tier 1 sources
                confirmed_sources = similar_news[
                    similar_news['source'].str.lower().apply(
                        lambda x: any(tier in x for tier in tier_1_sources)
                    )
                ]
                
                if len(confirmed_sources) > 0:
                    # Boost signal strength for cross-referenced news
                    boost_factor = 1.5 if len(confirmed_sources) == 1 else 2.0
                    news_df.at[idx, 'signal_strength'] = min(
                        news_df.at[idx, 'signal_strength'] * boost_factor, 1.0
                    )
                    news_df.at[idx, 'cross_reference_confirmed'] = True
                else:
                    news_df.at[idx, 'cross_reference_confirmed'] = False
            else:
                news_df.at[idx, 'cross_reference_confirmed'] = False
        
        # Drop temporary column
        news_df = news_df.drop('normalized_title', axis=1)
        
        return news_df
    
    def _extract_keywords(self, text: str) -> str:
        """Extract key keywords from text for matching similar news."""
        # Simple keyword extraction - can be enhanced
        important_words = ['inflation', 'fed', 'tesla', 'oil', 'bitcoin', 'hurricane', 
                          'nvidia', 'apple', 'recession', 'economy', 'market']
        
        words = text.split()
        keywords = [word for word in words if word in important_words or len(word) > 6]
        
        return '|'.join(keywords[:3]) if keywords else ''
    
    def _filter_trash_content(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Filter out trash content using single-keyword detection with context analysis."""
        if news_df.empty:
            return pd.DataFrame()
        
        def analyze_news_importance(title: str, summary: str) -> bool:
            """Return True if news is important, False if trash."""
            text = f"{title} {summary}".lower()
            
            # First, filter out obvious trash patterns
            if any(pattern in text for pattern in self.trash_patterns):
                return False
            
            # Check if any trigger keyword is present
            trigger_found = any(keyword in text for keyword in self.trigger_keywords)
            if not trigger_found:
                return False
            
            # SIMPLIFIED: If trigger keyword found, consider it important
            # Remove complex context analysis to increase news volume
            return True
        
        mask = news_df.apply(lambda row: analyze_news_importance(row["title"], row["summary"]), axis=1)
        filtered_df = news_df[mask].copy()
        
        logger.info(f"Context filtering: {len(news_df)} -> {len(filtered_df)}")
        return filtered_df
    
    def _apply_source_quality_weighting(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Apply source quality weighting without aggressive deletion."""
        if news_df.empty:
            return pd.DataFrame()
        
        # 1. Отримуємо whereфолтну вагу with конфandгу (якщо ми її andм прописали) or сandвимо 0.35
        # 0.35  це "прохandдний бал", so that новина not видалялася
        default_w = self.source_quality.get('default_source_weight', 0.35)
        
        def get_source_weight(source: str) -> float:
            src_lower = str(source).lower()
            # Шукаємо в усandх тandрах (Tier 1-4)
            for i in range(1, 5):
                tier_dict = self.source_quality.get(f'tier_{i}_sources', {})
                if src_lower in tier_dict:
                    return tier_dict[src_lower]
            return default_w
        
        # Застосовуємо ваги
        news_df["source_quality_weight"] = news_df["source"].apply(get_source_weight)
        
        # 2. Фandльтрацandя: тепер ми НЕ видаляємо новини with вагою 0.35
        # Видаляємо тandльки якщо вага withоallм критична (for example, < 0.1)
        initial_count = len(news_df)
        news_df = news_df[news_df["source_quality_weight"] >= 0.2] 
        
        filtered_count = len(news_df)
        logger.info(f"Source quality filter: {initial_count} -> {filtered_count} (kept unknown sources with weight {default_w})")
        
        return news_df
    
    def _classify_news_importance(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Classify news importance."""
        if news_df.empty:
            return pd.DataFrame()
        
        def classify_news(title: str, summary: str) -> tuple:
            text = f"{title} {summary}".lower()
            
            # High importance keywords
            if any(keyword in text for keyword in self.important_keywords):
                return "important", 1.0
            
            # Market moving patterns
            if any(pattern in text for pattern in self.market_moving_patterns):
                return "market_moving", 0.8
            
            # Neutral but important patterns (contextual importance)
            if any(pattern in text for pattern in self.neutral_important_patterns):
                return "neutral_important", 0.7
            
            # Default neutral
            return "neutral", 0.5
        
        classifications = news_df.apply(
            lambda row: classify_news(row["title"], row["summary"]), axis=1
        )
        
        news_df["news_importance"] = classifications.apply(lambda x: x[0])
        news_df["importance_weight"] = classifications.apply(lambda x: x[1])
        
        return news_df
    
    def _remove_duplicates(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates using TF-IDF similarity."""
        if news_df.empty:
            return pd.DataFrame()
        
        try:
            texts = (news_df["title"] + " " + news_df["summary"]).astype(str).tolist()
            
            vectorizer = TfidfVectorizer().fit_transform(texts)
            sim_matrix = cosine_similarity(vectorizer)
            
            to_drop = set()
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    if sim_matrix[i, j] > self.similarity_threshold:
                        to_drop.add(j)
            
            filtered_df = news_df.drop(news_df.index[list(to_drop)]).reset_index(drop=True)
            
            logger.info(f"Deduplication: {len(news_df)} -> {len(filtered_df)}")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error in deduplication: {e}")
            return news_df
    
    def fetch(self) -> pd.DataFrame:
        """Main fetch method with full processing."""
        try:
            raw_news = self.fetch_all_feeds()
            
            if raw_news.empty:
                return pd.DataFrame()
            
            processed_news = self.filter_and_classify(raw_news)
            
            if processed_news.empty:
                return pd.DataFrame()
            
            # Calculate final signal strength
            processed_news["signal_strength"] = (
                processed_news["source_quality_weight"] * 
                processed_news["importance_weight"]
            )
            
            logger.info(f"Final processed Google News: {len(processed_news)} items")
            return processed_news
            
        except Exception as e:
            logger.error(f"Error in Google News fetch: {e}")
            return pd.DataFrame()
