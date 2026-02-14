#!/usr/bin/env python3
"""
Fama-French Factors Integration
Класичні фактори для акцій: Market, Size, Value, Momentum, Profitability, Investment
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class FamaFrenchFactors:
    """
    Fama-French фактори для акцій
    Інтегрує 6 факторів: Market, Size, Value, Momentum, Profitability, Investment
    """
    
    def __init__(self):
        self.logger = logging.getLogger("FamaFrenchFactors")
        
        # Фактори Fama-French
        self.factors = {
            'MKT': 'Market Factor (Excess return of market portfolio)',
            'SMB': 'Size Factor (Small minus Big)',  
            'HML': 'Value Factor (High minus Low book-to-market)',
            'UMD': 'Momentum Factor (Up minus Down)',
            'RMW': 'Profitability Factor (Robust minus Weak)',
            'CMA': 'Investment Factor (Conservative minus Aggressive)'
        }
        
        # Бенчмарк індекси для факторів
        self.benchmark_tickers = {
            'market': '^GSPC',      # S&P 500
            'size_small': '^RUT',   # Russell 2000 (small caps)
            'size_big': '^OEX',     # S&P 100 (large caps)
            'value': 'IWD',         # iShares Russell 1000 Value
            'growth': 'IWF',        # iShares Russell 1000 Growth
            'momentum_up': 'MTUM',  # iShares MSCI USA Momentum
            'momentum_down': 'MDLA', # iShares MSCI USA Minimum Volatility
            'profitability': 'IWP', # iShares Russell 1000 Growth proxy
            'investment': 'IJJ'     # iShares S&P 100 Value proxy
        }
        
        # Кешування факторів
        self.factor_cache = {}
        self.cache_duration = timedelta(hours=24)
        
        self.logger.info("Fama-French Factors initialized with 6 factors")
    
    def calculate_market_factor(self, prices: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.Series:
        """
        Розрахунок Market Factor (MKT)
        Excess return of market portfolio over risk-free rate
        """
        try:
            # Розраховуємо дохідність ринку
            market_returns = prices.pct_change().dropna()
            
            # Річна withoutризикова ставка в денних
            daily_rf_rate = risk_free_rate / 252
            
            # Excess return
            market_factor = market_returns - daily_rf_rate
            
            self.logger.info(f"Market Factor calculated: mean={market_factor.mean():.6f}")
            return market_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating market factor: {e}")
            return pd.Series()
    
    def calculate_size_factor(self, small_cap_prices: pd.DataFrame, 
                             large_cap_prices: pd.DataFrame) -> pd.Series:
        """
        Розрахунок Size Factor (SMB)
        Small caps minus Big caps
        """
        try:
            # Доходність малих компаній
            small_returns = small_cap_prices.pct_change().dropna()
            
            # Доходність великих компаній  
            large_returns = large_cap_prices.pct_change().dropna()
            
            # SMB = Small - Big
            smb_factor = small_returns - large_returns
            
            self.logger.info(f"Size Factor calculated: mean={smb_factor.mean():.6f}")
            return smb_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating size factor: {e}")
            return pd.Series()
    
    def calculate_value_factor(self, value_prices: pd.DataFrame, 
                             growth_prices: pd.DataFrame) -> pd.Series:
        """
        Розрахунок Value Factor (HML)
        High book-to-market minus Low book-to-market (Value minus Growth)
        """
        try:
            # Доходність value акцій
            value_returns = value_prices.pct_change().dropna()
            
            # Доходність growth акцій
            growth_returns = growth_prices.pct_change().dropna()
            
            # HML = Value - Growth
            hml_factor = value_returns - growth_returns
            
            self.logger.info(f"Value Factor calculated: mean={hml_factor.mean():.6f}")
            return hml_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating value factor: {e}")
            return pd.Series()
    
    def calculate_momentum_factor(self, momentum_prices: pd.DataFrame, 
                               min_vol_prices: pd.DataFrame) -> pd.Series:
        """
        Розрахунок Momentum Factor (UMD)
        Up minus Down (momentum winners minus losers)
        """
        try:
            # Доходність моментум акцій
            momentum_returns = momentum_prices.pct_change().dropna()
            
            # Доходність акцій з низькою волатильністю (proxy for losers)
            min_vol_returns = min_vol_prices.pct_change().dropna()
            
            # UMD = Momentum - Minimum Volatility
            umd_factor = momentum_returns - min_vol_returns
            
            self.logger.info(f"Momentum Factor calculated: mean={umd_factor.mean():.6f}")
            return umd_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum factor: {e}")
            return pd.Series()
    
    def calculate_profitability_factor(self, profitability_prices: pd.DataFrame, 
                                     investment_prices: pd.DataFrame) -> pd.Series:
        """
        Розрахунок Profitability Factor (RMW)
        Robust minus Weak profitability
        """
        try:
            # Proxy для високоприбуткових компаній
            robust_returns = profitability_prices.pct_change().dropna()
            
            # Proxy для низькоприбуткових компаній
            weak_returns = investment_prices.pct_change().dropna()
            
            # RMW = Robust - Weak
            rmw_factor = robust_returns - weak_returns
            
            self.logger.info(f"Profitability Factor calculated: mean={rmw_factor.mean():.6f}")
            return rmw_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating profitability factor: {e}")
            return pd.Series()
    
    def calculate_investment_factor(self, conservative_prices: pd.DataFrame, 
                                   aggressive_prices: pd.DataFrame) -> pd.Series:
        """
        Розрахунок Investment Factor (CMA)
        Conservative minus Aggressive investment
        """
        try:
            # Proxy для консервативних інвестицій
            conservative_returns = conservative_prices.pct_change().dropna()
            
            # Proxy для агресивних інвестицій
            aggressive_returns = aggressive_prices.pct_change().dropna()
            
            # CMA = Conservative - Aggressive
            cma_factor = conservative_returns - aggressive_returns
            
            self.logger.info(f"Investment Factor calculated: mean={cma_factor.mean():.6f}")
            return cma_factor
            
        except Exception as e:
            self.logger.error(f"Error calculating investment factor: {e}")
            return pd.Series()
    
    def get_all_factors(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Отримати всі Fama-French фактори
        
        Args:
            start_date: Початкова дата
            end_date: Кінцева дата
            
        Returns:
            pd.DataFrame: Всі фактори
        """
        cache_key = f"{start_date}_{end_date}"
        
        # Перевіряємо кеш
        if cache_key in self.factor_cache:
            cached_time, cached_data = self.factor_cache[cache_key]
            if datetime.now() - cached_time < self.cache_duration:
                return cached_data
        
        try:
            # Встановлюємо дати
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365*5)).strftime('%Y-%m-%d')
            
            # Завантажуємо дані
            data = {}
            
            # Market data
            market_data = yf.download(self.benchmark_tickers['market'], 
                                    start=start_date, end=end_date)['Adj Close']
            data['market'] = market_data
            
            # Size data
            small_cap_data = yf.download(self.benchmark_tickers['size_small'], 
                                       start=start_date, end=end_date)['Adj Close']
            large_cap_data = yf.download(self.benchmark_tickers['size_big'], 
                                       start=start_date, end=end_date)['Adj Close']
            data['small_cap'] = small_cap_data
            data['large_cap'] = large_cap_data
            
            # Value data
            value_data = yf.download(self.benchmark_tickers['value'], 
                                  start=start_date, end=end_date)['Adj Close']
            growth_data = yf.download(self.benchmark_tickers['growth'], 
                                   start=start_date, end=end_date)['Adj Close']
            data['value'] = value_data
            data['growth'] = growth_data
            
            # Momentum data
            momentum_data = yf.download(self.benchmark_tickers['momentum_up'], 
                                      start=start_date, end=end_date)['Adj Close']
            min_vol_data = yf.download(self.benchmark_tickers['momentum_down'], 
                                    start=start_date, end=end_date)['Adj Close']
            data['momentum'] = momentum_data
            data['min_vol'] = min_vol_data
            
            # Profitability/Investment data
            profitability_data = yf.download(self.benchmark_tickers['profitability'], 
                                           start=start_date, end=end_date)['Adj Close']
            investment_data = yf.download(self.benchmark_tickers['investment'], 
                                        start=start_date, end=end_date)['Adj Close']
            data['profitability'] = profitability_data
            data['investment'] = investment_data
            
            # Розраховуємо фактори
            factors_df = pd.DataFrame(index=market_data.index)
            
            # Market Factor
            factors_df['MKT'] = self.calculate_market_factor(data['market'])
            
            # Size Factor
            factors_df['SMB'] = self.calculate_size_factor(data['small_cap'], data['large_cap'])
            
            # Value Factor
            factors_df['HML'] = self.calculate_value_factor(data['value'], data['growth'])
            
            # Momentum Factor
            factors_df['UMD'] = self.calculate_momentum_factor(data['momentum'], data['min_vol'])
            
            # Profitability Factor
            factors_df['RMW'] = self.calculate_profitability_factor(data['profitability'], data['investment'])
            
            # Investment Factor
            factors_df['CMA'] = self.calculate_investment_factor(data['profitability'], data['investment'])
            
            # Видаляємо NaN
            factors_df = factors_df.dropna()
            
            # Кешуємо результат
            self.factor_cache[cache_key] = (datetime.now(), factors_df)
            
            self.logger.info(f"Fama-French factors calculated: {len(factors_df)} observations")
            
            return factors_df
            
        except Exception as e:
            self.logger.error(f"Error getting Fama-French factors: {e}")
            return pd.DataFrame()
    
    def calculate_factor_exposures(self, asset_returns: pd.Series, 
                                  factors: pd.DataFrame) -> Dict[str, float]:
        """
        Розрахувати експозицію активу на фактори
        
        Args:
            asset_returns: Доходність активу
            factors: Фактори
            
        Returns:
            Dict: Експозиції на фактори
        """
        try:
            # Об'єднуємо дані
            combined_data = pd.concat([asset_returns, factors], axis=1).dropna()
            
            if len(combined_data) < 30:  # Мінімум 30 спостережень
                return {}
            
            # Розділяємо на X (фактори) та y (доходність активу)
            X = combined_data.iloc[:, 1:]  # Фактори
            y = combined_data.iloc[:, 0]   # Доходність активу
            
            # Додаємо константу
            X = sm.add_constant(X)
            
            # Регресія
            model = sm.OLS(y, X).fit()
            
            # Експозиції на фактори (коефіцієнти регресії)
            exposures = {}
            for factor_name in factors.columns:
                if factor_name in model.params:
                    exposures[factor_name] = model.params[factor_name]
            
            # Alpha
            exposures['alpha'] = model.params['const']
            
            # R-squared
            exposures['r_squared'] = model.rsquared
            
            self.logger.info(f"Factor exposures calculated: R²={exposures['r_squared']:.3f}")
            
            return exposures
            
        except Exception as e:
            self.logger.error(f"Error calculating factor exposures: {e}")
            return {}
    
    def analyze_factor_performance(self, factors: pd.DataFrame) -> Dict[str, Dict]:
        """
        Аналізувати продуктивність факторів
        
        Args:
            factors: DataFrame з факторами
            
        Returns:
            Dict: Статистика факторів
        """
        try:
            factor_stats = {}
            
            for factor in factors.columns:
                factor_data = factors[factor].dropna()
                
                stats = {
                    'mean': factor_data.mean(),
                    'std': factor_data.std(),
                    'sharpe': factor_data.mean() / factor_data.std() * np.sqrt(252),
                    'skewness': factor_data.skew(),
                    'kurtosis': factor_data.kurtosis(),
                    'min': factor_data.min(),
                    'max': factor_data.max(),
                    'positive_days': (factor_data > 0).sum() / len(factor_data),
                    'annual_return': factor_data.mean() * 252,
                    'annual_volatility': factor_data.std() * np.sqrt(252)
                }
                
                factor_stats[factor] = stats
            
            self.logger.info(f"Factor performance analyzed for {len(factor_stats)} factors")
            
            return factor_stats
            
        except Exception as e:
            self.logger.error(f"Error analyzing factor performance: {e}")
            return {}


# Глобальний екземпляр
fama_french_factors = FamaFrenchFactors()


def get_fama_french_factors(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Отримати Fama-French фактори"""
    return fama_french_factors.get_all_factors(start_date, end_date)


def calculate_factor_exposures(asset_returns: pd.Series, factors: pd.DataFrame) -> Dict[str, float]:
    """Розрахувати експозиції на фактори"""
    return fama_french_factors.calculate_factor_exposures(asset_returns, factors)


if __name__ == "__main__":
    # Приклад використання
    logging.basicConfig(level=logging.INFO)
    
    print("[DATA] Fama-French Factors Test")
    print("="*50)
    
    # Отримуємо фактори
    factors = get_fama_french_factors("2020-01-01", "2024-01-01")
    
    if not factors.empty:
        print(f"[UP] Fama-French Factors ({len(factors)} observations):")
        print(factors.tail())
        
        # Аналізуємо продуктивність
        factor_stats = fama_french_factors.analyze_factor_performance(factors)
        
        print(f"\n[DATA] Factor Performance:")
        for factor, stats in factor_stats.items():
            print(f"   {factor}:")
            print(f"     Annual Return: {stats['annual_return']:.2%}")
            print(f"     Annual Vol: {stats['annual_volatility']:.2%}")
            print(f"     Sharpe: {stats['sharpe']:.2f}")
            print(f"     Positive Days: {stats['positive_days']:.1%}")
        
        print(f"\n[OK] Fama-French Factors working correctly!")
    else:
        print(f"[ERROR] No factors data available")
