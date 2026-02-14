# collectors/crypto_price_collector.py

import requests
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from .base_collector import BaseCollector

logger = logging.getLogger("trading_project.crypto_price_collector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(ch)
logger.propagate = False


class CryptoPriceCollector(BaseCollector):
    """
    🪙 Crypto Price Collector - Безкоштовний колектор цін криптовалют
    
    Джерела:
    - CoinGecko API (withoutкоштовний)
    - CoinMarketCap API (withoutкоштовний tier)
    
    Обмеження:
    - Rate limited: 10-50 запитів/хвилину
    - Дані затримуються на 1-5 хвилин
    - Обмежена кількість криптовалют за запит
    """
    
    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        vs_currency: str = "usd",
        include_market_cap: bool = True,
        include_24hr_vol: bool = True,
        include_24hr_change: bool = True,
        table_name: str = "crypto_prices",
        db_path: str = ":memory:",
        strict: bool = True,
        **kwargs
    ):
        super().__init__(db_path=db_path, table_name=table_name, strict=strict, **kwargs)
        
        # Основні криптовалюти за замовчуванням
        self.symbols = symbols or [
            "bitcoin", "ethereum", "binancecoin", "cardano", "solana",
            "ripple", "polkadot", "dogecoin", "avalanche-2", "chainlink",
            "polygon", "uniswap", "litecoin", "stellar", "cosmos"
        ]
        
        self.vs_currency = vs_currency
        self.include_market_cap = include_market_cap
        self.include_24hr_vol = include_24hr_vol
        self.include_24hr_change = include_24hr_change
        
        # API endpoints
        self.coingecko_base = "https://api.coingecko.com/api/v3"
        
        # Headers для API
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        logger.info(f"[CryptoPrice] Initialized: {len(self.symbols)} symbols, {self.vs_currency} base")
    
    def _fetch_coingecko_prices(self, symbols: List[str]) -> List[Dict[str, Any]]:
        """
        Збирає ціни з CoinGecko API
        """
        prices = []
        
        try:
            # CoinGecko дозволяє до 250 символів за один запит
            # Розбиваємо на batch по 50 для надійності
            batch_size = 50
            
            for i in range(0, len(symbols), batch_size):
                batch_symbols = symbols[i:i + batch_size]
                
                # Формуємо URL
                url = f"{self.coingecko_base}/coins/markets"
                
                params = {
                    'vs_currency': self.vs_currency,
                    'ids': ','.join(batch_symbols),
                    'order': 'market_cap_desc',
                    'per_page': len(batch_symbols),
                    'page': 1,
                    'sparkline': False,
                    'price_change_percentage': '24h'
                }
                
                response = requests.get(url, params=params, headers=self.headers, timeout=15)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        for coin in data:
                            try:
                                price_data = {
                                    'symbol': coin.get('id', ''),
                                    'name': coin.get('name', ''),
                                    'symbol_upper': coin.get('symbol', '').upper(),
                                    'price': coin.get('current_price', 0.0),
                                    'market_cap': coin.get('market_cap', 0.0) if self.include_market_cap else None,
                                    'total_volume': coin.get('total_volume', 0.0) if self.include_24hr_vol else None,
                                    'price_change_24h': coin.get('price_change_24h', 0.0) if self.include_24hr_change else None,
                                    'price_change_percentage_24h': coin.get('price_change_percentage_24h', 0.0) if self.include_24hr_change else None,
                                    'circulating_supply': coin.get('circulating_supply', 0.0),
                                    'total_supply': coin.get('total_supply', 0.0),
                                    'max_supply': coin.get('max_supply', 0.0),
                                    'ath': coin.get('ath', 0.0),  # All Time High
                                    'ath_change_percentage': coin.get('ath_change_percentage', 0.0),
                                    'atl': coin.get('atl', 0.0),  # All Time Low
                                    'atl_change_percentage': coin.get('atl_change_percentage', 0.0),
                                    'market_cap_rank': coin.get('market_cap_rank', 0),
                                    'last_updated': coin.get('last_updated', ''),
                                    'data_type': 'current_price',  # [OK] ДОДАНО
                                    'source': 'CoinGecko',
                                    'collected_at': datetime.now()
                                }
                                
                                prices.append(price_data)
                                
                            except Exception as e:
                                logger.warning(f"[CryptoPrice] Error parsing coin data: {e}")
                                continue
                        
                        logger.info(f"[CryptoPrice] CoinGecko batch {i//batch_size + 1}: {len(data)} coins collected")
                        
                    except Exception as e:
                        logger.warning(f"[CryptoPrice] Error parsing JSON from CoinGecko: {e}")
                else:
                    logger.warning(f"[CryptoPrice] CoinGecko request failed: {response.status_code}")
                
                # Delay між batch запитами
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"[CryptoPrice] Error fetching CoinGecko prices: {e}")
        
        return prices
    
    def _fetch_coingecko_historical(self, symbol: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Збирає історичні дані для однієї криптовалюти
        """
        historical_data = []
        
        try:
            url = f"{self.coingecko_base}/coins/{symbol}/market_chart"
            
            params = {
                'vs_currency': self.vs_currency,
                'days': days,
                'interval': 'daily'
            }
            
            response = requests.get(url, params=params, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Prices data
                    for price_point in data.get('prices', []):
                        timestamp = price_point[0] / 1000  # Convert milliseconds to seconds
                        date = datetime.fromtimestamp(timestamp)
                        
                        historical_data.append({
                            'symbol': symbol,
                            'date': date,
                            'price': price_point[1],
                            'data_type': 'price',
                            'source': 'CoinGecko',
                            'collected_at': datetime.now()
                        })
                    
                    # Volume data
                    for volume_point in data.get('total_volumes', []):
                        timestamp = volume_point[0] / 1000
                        date = datetime.fromtimestamp(timestamp)
                        
                        historical_data.append({
                            'symbol': symbol,
                            'date': date,
                            'volume': volume_point[1],
                            'data_type': 'volume',
                            'source': 'CoinGecko',
                            'collected_at': datetime.now()
                        })
                    
                    logger.info(f"[CryptoPrice] Historical data for {symbol}: {len(historical_data)} points")
                    
                except Exception as e:
                    logger.warning(f"[CryptoPrice] Error parsing historical data for {symbol}: {e}")
            else:
                logger.warning(f"[CryptoPrice] Historical data request failed for {symbol}: {response.status_code}")
                
        except Exception as e:
            logger.error(f"[CryptoPrice] Error fetching historical data for {symbol}: {e}")
        
        return historical_data
    
    def collect_data(self, historical_days: Optional[int] = None) -> pd.DataFrame:
        """
        Основний метод збору data
        """
        logger.info(f"[CryptoPrice] Collecting data for {len(self.symbols)} symbols")
        
        all_data = []
        
        # Поточні ціни
        try:
            current_prices = self._fetch_coingecko_prices(self.symbols)
            all_data.extend(current_prices)
            
        except Exception as e:
            logger.error(f"[CryptoPrice] Error collecting current prices: {e}")
        
        # Історичні дані (якщо needed)
        if historical_days and historical_days > 0:
            logger.info(f"[CryptoPrice] Collecting historical data for {historical_days} days")
            
            # Обмежуємо кількість символів для історичних data
            historical_symbols = self.symbols[:5]  # Топ 5 для економії часу
            
            for symbol in historical_symbols:
                try:
                    hist_data = self._fetch_coingecko_historical(symbol, historical_days)
                    all_data.extend(hist_data)
                    time.sleep(1)  # Delay між запитами
                    
                except Exception as e:
                    logger.warning(f"[CryptoPrice] Error collecting historical data for {symbol}: {e}")
        
        # Створюємо DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Додаємо розрахункові колонки
            if 'price' in df.columns and 'market_cap' in df.columns:
                df['market_cap_to_price_ratio'] = df['market_cap'] / df['price'].replace(0, 1)
            
            if 'price_change_24h' in df.columns and 'price' in df.columns:
                df['volatility_24h'] = abs(df['price_change_24h']) / df['price'].replace(0, 1) * 100
            
            # Категоризація за капіталізацією
            if 'market_cap' in df.columns:
                df['market_cap_category'] = pd.cut(
                    df['market_cap'],
                    bins=[0, 1e9, 10e9, 100e9, float('inf')],
                    labels=['Small', 'Medium', 'Large', 'Mega']
                )
            
            logger.info(f"[CryptoPrice] Total data points collected: {len(df)}")
            logger.info(f"[CryptoPrice] Current prices: {len(df[df['data_type'] != 'historical'])}")
            logger.info(f"[CryptoPrice] Historical points: {len(df[df['data_type'] == 'historical'])}")
            
            return df
        else:
            logger.warning("[CryptoPrice] No data collected")
            return pd.DataFrame()
    
    def collect(self) -> List[Dict[str, Any]]:
        """
        Метод для сумісності з іншими колекторами
        """
        df = self.collect_data()
        return df.to_dict('records')
    
    def get_top_movers(self, min_volume: float = 1000000) -> pd.DataFrame:
        """
        Повертає криптовалюти з найбільшими рухами за 24 години
        """
        df = self.collect_data()
        if df.empty or 'price_change_percentage_24h' not in df.columns:
            return pd.DataFrame()
        
        # Фільтруємо за обсягом
        filtered = df[df['total_volume'] >= min_volume]
        
        # Сортуємо за зміною ціни
        top_gainers = filtered.nlargest(5, 'price_change_percentage_24h')
        top_losers = filtered.nsmallest(5, 'price_change_percentage_24h')
        
        return pd.concat([top_gainers, top_losers])
    
    def get_market_summary(self) -> Dict[str, Any]:
        """
        Повертає огляд ринку криптовалют
        """
        df = self.collect_data()
        if df.empty:
            return {}
        
        summary = {
            'total_cryptos': len(df),
            'total_market_cap': df['market_cap'].sum() if 'market_cap' in df.columns else 0,
            'total_24h_volume': df['total_volume'].sum() if 'total_volume' in df.columns else 0,
            'avg_24h_change': df['price_change_percentage_24h'].mean() if 'price_change_percentage_24h' in df.columns else 0,
            'gainers': len(df[df['price_change_24h'] > 0]) if 'price_change_24h' in df.columns else 0,
            'losers': len(df[df['price_change_24h'] < 0]) if 'price_change_24h' in df.columns else 0,
            'top_5_by_market_cap': df.nlargest(5, 'market_cap')['name'].tolist() if 'market_cap' in df.columns else [],
            'last_updated': datetime.now()
        }
        
        return summary
