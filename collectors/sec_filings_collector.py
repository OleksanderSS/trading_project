# collectors/sec_filings_collector.py

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .base_collector import BaseCollector

logger = logging.getLogger("trading_project.sec_filings_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class SECFilingsCollector(BaseCollector):
    """
    📄 SEC Filings Collector - Безкоштовний колектор SEC звітів
    
    Джерела:
    - SEC EDGAR API (офіційний)
    - SEC RSS feeds
    
    Типи звітів:
    - 10-K: Річні звіти
    - 10-Q: Квартальні звіти  
    - 8-K: Поточні звіти
    - 4: Інсайдерські транзакції
    - 13F: Звіти інституційних інвесторів
    
    Обмеження:
    - Rate limited: 10 запитів/секунду
    - Потрібно delays між запитами
    - Великі обсяги data
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        filing_types: Optional[List[str]] = None,
        days_back: int = 30,
        table_name: str = "sec_filings",
        db_path: str = ":memory:",
        strict: bool = True,
        **kwargs
    ):
        super().__init__(db_path=db_path, table_name=table_name, strict=strict, **kwargs)
        
        self.tickers = tickers or ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        self.filing_types = filing_types or ["10-K", "10-Q", "8-K", "4"]
        self.days_back = days_back
        
        # Додаємо session для HTTP запитів
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate'
        })
        
        # SEC EDGAR base URL
        self.sec_base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        self.company_url = "https://www.sec.gov/files/edgar/data/company_tickers.json"
        
        logger.info(f"[SECFilings] Initialized for {len(self.tickers)} tickers")
        self.sec_api_base = "https://data.sec.gov/api/xbrl"
        
        # Headers для SEC API
        self.headers = {
            'User-Agent': 'Trading Project (contact@example.com)',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
        }
        
        logger.info(f"[SECFilings] Initialized: {len(self.tickers)} tickers, {len(self.filing_types)} filing types")
    
    def _normalize_recent_filings(self, recent):
        """Normalize SEC 'recent' filings to list of dicts."""
        if isinstance(recent, dict):
            try:
                df = pd.DataFrame(recent)
                return df.to_dict(orient="records")
            except Exception:
                return []
        if isinstance(recent, list):
            return [r for r in recent if isinstance(r, dict)]
        return []

    def _get_company_info(self, ticker: str) -> Optional[Dict[str, Any]]:
        """
        Отримує інформацію про компанію з SEC EDGAR
        """
        try:
            # ВИПРАВЛЕНО: використовуємо правильний URL
            url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=exclude&count=10"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                # Парсимо HTML для отримання CIK
                import re
                cik_match = re.search(r'CIK=(\d{10})', response.text)
                if cik_match:
                    cik = cik_match.group(1)
                    return {
                        'ticker': ticker,
                        'cik': cik,
                        'company_name': self._extract_company_name(response.text)
                    }
            
        except Exception as e:
            logger.warning(f"[SECFilings] Error getting company info for {ticker}: {e}")
        
        return None
    
    def _extract_company_name(self, html_text: str) -> str:
        """
        Витягує назву компанії з HTML
        """
        try:
            import re
            name_match = re.search(r'Company Name[^>]*>([^<]+)', html_text)
            if name_match:
                return name_match.group(1).strip()
        except:
            pass
        return ""
    
    def _fetch_company_filings(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """
        Збирає звіти компанії
        """
        filings = []
        
        try:
            # Використовуємо SEC API для отримання останніх звітів
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # [OK] ДОДАНО - діагностика структури data
                    logger.debug(f"[SECFilings] {ticker}: Data keys: {list(data.keys())}")
                    filings_data = data.get('filings', {})
                    logger.debug(f"[SECFilings] {ticker}: Filings keys: {list(filings_data.keys())}")
                    
                    # Отримуємо останні звіти
                    recent_raw = data.get('filings', {}).get('recent', {})
                    recent_filings = self._normalize_recent_filings(recent_raw)
                    logger.info(f"[SECFilings] {ticker}: Got {len(recent_filings)} recent items")
                    if recent_filings:
                        first_item = recent_filings[0]
                        logger.debug(f"[SECFilings] {ticker}: First item type: {type(first_item)}")
                        if isinstance(first_item, dict):
                            logger.debug(f"[SECFilings] {ticker}: First item keys: {list(first_item.keys())[:5]}")
                    
                    for filing in recent_filings:
                        try:
                            # [OK] ВИПРАВЛЕНО - перевіряємо тип filing
                            if not isinstance(filing, dict):
                                logger.warning(f"[SECFilings] Invalid filing data type for {ticker}: {type(filing)}")
                                continue
                            
                            form_type = filing.get('form', '')
                            
                            # Фільтруємо за типами звітів
                            if form_type in self.filing_types:
                                filing_date = pd.to_datetime(filing.get('filingDate', ''))
                                
                                # Перевіряємо дату
                                if filing_date >= datetime.now() - timedelta(days=self.days_back):
                                    filing_data = {
                                        'ticker': ticker,
                                        'cik': cik,
                                        'form_type': form_type,
                                        'filing_date': filing_date,
                                        'reporting_date': pd.to_datetime(filing.get('reportDate', '')),
                                        'accession_number': filing.get('accessionNumber', ''),
                                        'file_number': filing.get('fileNumber', ''),
                                        'film_number': filing.get('filmNumber', ''),
                                        'items': filing.get('items', ''),
                                        'size': filing.get('size', 0),
                                        'is_xbrl': filing.get('isXBRL', False),
                                        'is_inline_xbrl': filing.get('isInlineXBRL', False),
                                        'primary_document': filing.get('primaryDocument', ''),
                                        'primary_doc_description': filing.get('primaryDocDescription', ''),
                                        'document_url': f"https://www.sec.gov/Archives/edgar/data/{cik}/{filing.get('accessionNumber', '').replace('-', '')}/{filing.get('primaryDocument', '')}",
                                        'source': 'SEC EDGAR',
                                        'collected_at': datetime.now()
                                    }
                                    
                                    filings.append(filing_data)
                                    
                        except Exception as e:
                            logger.warning(f"[SECFilings] Error parsing filing for {ticker}: {e}")
                            continue
                    
                    logger.info(f"[SECFilings] {ticker}: {len(filings)} filings collected")
                    
                except Exception as e:
                    logger.warning(f"[SECFilings] Error parsing JSON for {ticker}: {e}")
            else:
                logger.warning(f"[SECFilings] SEC API request failed for {ticker}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[SECFilings] Error fetching filings for {ticker}: {e}")
        
        return filings
    
    def _fetch_insider_trading(self, ticker: str, cik: str) -> List[Dict[str, Any]]:
        """
        Отримує дані про інсайдерські транзакції
        """
        insider_data = []
        
        try:
            # Спробуємо отримати дані про інсайдерські транзакції
            url = f"https://data.sec.gov/submissions/CIK{cik.zfill(10)}.json"
            
            try:
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                # Перевіряємо структуру data
                if 'filings' in data and 'recent' in data['filings']:
                    recent_raw = data['filings']['recent']
                    recent_filings = self._normalize_recent_filings(recent_raw)
                    
                    # Шукаємо Form 4 (інсайдерські транзакції)
                    for i, filing in enumerate(recent_filings):
                        try:
                            if isinstance(filing, dict) and filing.get('form') == '4':
                                filing_date = pd.to_datetime(filing.get('filingDate', ''), errors='coerce')
                                if pd.isna(filing_date):
                                    continue
                                if filing_date < datetime.now() - timedelta(days=self.days_back):
                                    continue
                                insider_info = {
                                    'ticker': ticker,
                                    'cik': cik,
                                    'form_type': '4',
                                    'filing_date': filing_date,
                                    'transaction_date': pd.to_datetime(filing.get('periodOfReport', ''), errors='coerce'),
                                    'insider_name': 'Unknown',  # Більше детальної інформації needed в іншому запиті
                                    'transaction_type': self._extract_transaction_type(filing),
                                    'shares_traded': 0,  # Більше детальної інформації needed в іншому запиті
                                    'transaction_price': 0.0,
                                    'source': 'SEC EDGAR',
                                    'collected_at': datetime.now()
                                }
                                
                                insider_data.append(insider_info)
                                
                        except Exception as e:
                            logger.debug(f"[SECFilings] Error parsing filing {i} for {ticker}: {e}")
                            continue
                    
                    logger.info(f"[SECFilings] {ticker}: {len(insider_data)} insider transactions collected")
                    
            except Exception as e:
                logger.warning(f"[SECFilings] Error parsing insider data for {ticker}: {e}")
                    
        except Exception as e:
            logger.error(f"[SECFilings] Error fetching insider data for {ticker}: {e}")
        
        return insider_data
    
    def _extract_transaction_type(self, filing: Dict[str, Any]) -> str:
        """
        Витягує тип транзакції з data звіту
        """
        # Спрощена логіка - в реальному проекті тут був би детальний парсинг
        return "Unknown"
    
    def collect_data(self) -> pd.DataFrame:
        """
        Основний метод збору data
        """
        logger.info(f"[SECFilings] Collecting data for {len(self.tickers)} tickers")
        
        all_filings = []
        all_insider = []
        
        for ticker in self.tickers:
            try:
                # Отримуємо інформацію про компанію
                company_info = self._get_company_info(ticker)
                
                if company_info:
                    cik = company_info['cik']
                    
                    # Збираємо звіти
                    filings = self._fetch_company_filings(ticker, cik)
                    all_filings.extend(filings)
                    
                    # Збираємо інсайдерські транзакції
                    insider = self._fetch_insider_trading(ticker, cik)
                    all_insider.extend(insider)
                    
                    # Delay між запитами
                    time.sleep(1)
                else:
                    logger.warning(f"[SECFilings] No company info found for {ticker}")
                    
            except Exception as e:
                logger.error(f"[SECFilings] Error processing {ticker}: {e}")
                continue
        
        # Об'єднуємо всі дані
        all_data = all_filings + all_insider
        
        # Створюємо DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Додаємо розрахункові колонки
            # ВИПРАВЛЕНО: конвертуємо дати перед операціями
            df['filing_date'] = pd.to_datetime(df['filing_date'])
            df['days_since_filing'] = (datetime.now() - df['filing_date']).dt.days
            df['filing_year'] = df['filing_date'].dt.year
            df['filing_quarter'] = df['filing_date'].dt.quarter
            
            # Категоризація типів звітів
            df['filing_category'] = df['form_type'].map({
                '10-K': 'Annual',
                '10-Q': 'Quarterly', 
                '8-K': 'Current',
                '4': 'Insider',
                '13F': 'Institutional'
            }).fillna('Other')
            
            # Сортуємо за датою
            df = df.sort_values('filing_date', ascending=False)
            
            logger.info(f"[SECFilings] Total filings collected: {len(df)}")
            logger.info(f"[SECFilings] Annual reports (10-K): {len(df[df['form_type'] == '10-K'])}")
            logger.info(f"[SECFilings] Quarterly reports (10-Q): {len(df[df['form_type'] == '10-Q'])}")
            logger.info(f"[SECFilings] Current reports (8-K): {len(df[df['form_type'] == '8-K'])}")
            logger.info(f"[SECFilings] Insider transactions (Form 4): {len(df[df['form_type'] == '4'])}")
            
            return df
        else:
            logger.warning("[SECFilings] No filings collected")
            return pd.DataFrame()
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Метод для сумісності з іншими колекторами
        """
        df = self.collect_data()
        return df.to_dict('records')
    
    def get_recent_filings(self, days: int = 7) -> pd.DataFrame:
        """
        Повертає звіти за останні N днів
        """
        df = self.collect_data()
        if df.empty:
            return df
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent = df[df['filing_date'] >= cutoff_date]
        
        return recent.sort_values('filing_date', ascending=False)
    
    def get_insider_activity(self, ticker: Optional[str] = None) -> pd.DataFrame:
        """
        Повертає інсайдерську activeсть
        """
        df = self.collect_data()
        if df.empty:
            return df
        
        insider_df = df[df['form_type'] == '4']
        
        if ticker:
            insider_df = insider_df[insider_df['ticker'] == ticker]
        
        return insider_df.sort_values('filing_date', ascending=False)
    
    def get_company_filings_summary(self, ticker: str) -> Dict[str, Any]:
        """
        Повертає звіт по звітах компанії
        """
        df = self.collect_data()
        if df.empty:
            return {}
        
        company_df = df[df['ticker'] == ticker]
        
        if company_df.empty:
            return {}
        
        summary = {
            'ticker': ticker,
            'total_filings': len(company_df),
            'latest_filing': company_df['filing_date'].max(),
            'filing_types': company_df['form_type'].value_counts().to_dict(),
            'recent_filings_30d': len(company_df[company_df['filing_date'] >= datetime.now() - timedelta(days=30)]),
            'insider_transactions': len(company_df[company_df['form_type'] == '4']),
            'last_updated': datetime.now()
        }
        
        return summary
