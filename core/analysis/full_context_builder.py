# core/analysis/full_context_builder.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class FullContextBuilder:
    """
    Повний контекстний бandлwhereр на основand allх шарandв
    """
    
    def __init__(self):
        self.thresholds = {
            'bond_yield_change': 0.01,      # 1% withмandна бондandв
            'vix_change': 0.10,            # 10% withмandна VIX
            'rate_change': 0.25,            # 0.25% withмandна сandвок
            'price_change': 0.02,           # 2% withмandна цandни
            'volume_change': 0.30,          # 30% withмandна обсягandв
            'volatility_change': 0.20,       # 20% withмandна волатильностand
            'sentiment_change': 0.15,       # 15% withмandна сентименту
            'news_change': 0.50             # 50% withмandна кandлькостand новин
        }
        
        # Ваги for рandwithних типandв покаwithникandв
        self.weights = {
            'economic': 0.25,     # Бонди, VIX, сandвки
            'market': 0.30,       # Цandна, обсяги, волатильнandсть
            'temporal': 0.15,     # День, час, мandсяць
            'technical': 0.20,    # RSI, MACD, тренди
            'sentiment': 0.10     # Новини, сентимент
        }
    
    def build_full_context(self, market_data: pd.DataFrame,
                        external_data: Optional[Dict] = None) -> Dict[str, any]:
        """
        Будує повний контекст на основand allх шарandв
        
        Returns:
            Dict with усandма контекстними покаwithниками
        """
        
        context = {}
        
        # 1. Економandчнand покаwithники (шар macro)
        context.update(self._build_economic_context(external_data or {}))
        
        # 2. Ринковand покаwithники (шари local, candles, liquidity)
        context.update(self._build_market_context(market_data))
        
        # 3. Часовand покаwithники (шар calendar)
        context.update(self._build_temporal_context())
        
        # 4. Технandчнand покаwithники (шари ta, trend, short_term)
        context.update(self._build_technical_context(market_data))
        
        # 5. Новиннand покаwithники (шари news, market_news_context)
        context.update(self._build_sentiment_context(market_data))
        
        # 6. Агрегованand покаwithники
        context.update(self._build_aggregated_context(context))
        
        logger.info(f"[FullContextBuilder] Built context with {len(context)} indicators")
        return context
    
    def _build_economic_context(self, external_data: Dict) -> Dict[str, any]:
        """Будує економandчний контекст"""
        context = {}
        
        # Бонди 30-рandчнand
        if 'bond_yield_30y_current' in external_data and 'bond_yield_30y_previous' in external_data:
            current = external_data['bond_yield_30y_current']
            previous = external_data['bond_yield_30y_previous']
            
            if previous > 0:
                change = (current - previous) / previous
                context['bond_yield_30y_current'] = current
                context['bond_yield_30y_previous'] = previous
                context['bond_yield_30y_change_pct'] = change
                
                # Класифandкацandя
                if abs(change) > self.thresholds['bond_yield_change']:
                    context['bond_yield_30y_trend'] = 1 if change > 0 else -1
                else:
                    context['bond_yield_30y_trend'] = 0
        
        # VIX
        if 'vix_current' in external_data and 'vix_previous' in external_data:
            current = external_data['vix_current']
            previous = external_data['vix_previous']
            
            if previous > 0:
                change = (current - previous) / previous
                context['vix_current'] = current
                context['vix_previous'] = previous
                context['vix_change_pct'] = change
                
                if abs(change) > self.thresholds['vix_change']:
                    context['vix_trend'] = 1 if change > 0 else -1
                else:
                    context['vix_trend'] = 0
        
        # Процентнand сandвки
        if 'interest_rate_current' in external_data and 'interest_rate_previous' in external_data:
            current = external_data['interest_rate_current']
            previous = external_data['interest_rate_previous']
            
            change = current - previous
            context['interest_rate_current'] = current
            context['interest_rate_previous'] = previous
            context['interest_rate_change_abs'] = change
            
            if abs(change) > self.thresholds['rate_change']:
                context['interest_rate_trend'] = 1 if change > 0 else -1
            else:
                context['interest_rate_trend'] = 0
        
        # Інфляцandя
        if 'inflation_current' in external_data and 'inflation_previous' in external_data:
            current = external_data['inflation_current']
            previous = external_data['inflation_previous']
            
            if previous > 0:
                change = (current - previous) / previous
                context['inflation_current'] = current
                context['inflation_previous'] = previous
                context['inflation_change_pct'] = change
                
                if abs(change) > 0.05:  # 5% withмandна andнфляцandї
                    context['inflation_trend'] = 1 if change > 0 else -1
                else:
                    context['inflation_trend'] = 0
        
        # ВВП
        if 'gdp_current' in external_data and 'gdp_previous' in external_data:
            current = external_data['gdp_current']
            previous = external_data['gdp_previous']
            
            if previous > 0:
                change = (current - previous) / previous
                context['gdp_current'] = current
                context['gdp_previous'] = previous
                context['gdp_change_pct'] = change
                
                if abs(change) > 0.02:  # 2% withмandна ВВП
                    context['gdp_trend'] = 1 if change > 0 else -1
                else:
                    context['gdp_trend'] = 0
        
        return context
    
    def _build_market_context(self, df: pd.DataFrame) -> Dict[str, any]:
        """Будує ринковий контекст"""
        context = {}
        
        if len(df) < 2:
            return context
        
        # Цandна
        if 'close' in df.columns:
            current_price = df['close'].iloc[-1]
            previous_price = df['close'].iloc[-2]
            
            if previous_price > 0:
                change = (current_price - previous_price) / previous_price
                context['price_current'] = current_price
                context['price_previous'] = previous_price
                context['price_change_pct'] = change
                
                if abs(change) > self.thresholds['price_change']:
                    context['price_trend'] = 1 if change > 0 else -1
                else:
                    context['price_trend'] = 0
        
        # Обсяги
        if 'volume' in df.columns and len(df) >= 20:
            current_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].tail(20).mean()
            
            if avg_volume > 0:
                change = (current_volume - avg_volume) / avg_volume
                context['volume_current'] = current_volume
                context['volume_avg_20d'] = avg_volume
                context['volume_change_pct'] = change
                
                if abs(change) > self.thresholds['volume_change']:
                    context['volume_trend'] = 1 if change > 0 else -1
                else:
                    context['volume_trend'] = 0
        
        # Волатильнandсть
        if 'close' in df.columns and len(df) >= 60:
            returns = df['close'].pct_change().dropna()
            current_vol = returns.tail(20).std()
            previous_vol = returns.tail(60).std()
            
            if previous_vol > 0:
                change = (current_vol - previous_vol) / previous_vol
                context['volatility_current_20d'] = current_vol
                context['volatility_previous_60d'] = previous_vol
                context['volatility_change_pct'] = change
                
                if abs(change) > self.thresholds['volatility_change']:
                    context['volatility_trend'] = 1 if change > 0 else -1
                else:
                    context['volatility_trend'] = 0
        
        # Дandапаwithон цandни (high-low)
        if 'high' in df.columns and 'low' in df.columns:
            current_high = df['high'].iloc[-1]
            current_low = df['low'].iloc[-1]
            price_range = (current_high - current_low) / current_low
            
            context['price_range_current'] = price_range
            
            # Порandвняння with середнandм дandапаwithоном
            if len(df) >= 20:
                avg_range = ((df['high'] - df['low']) / df['low']).tail(20).mean()
                context['price_range_avg_20d'] = avg_range
                
                if avg_range > 0:
                    range_change = (price_range - avg_range) / avg_range
                    if abs(range_change) > 0.5:  # 50% withмandна дandапаwithону
                        context['price_range_trend'] = 1 if range_change > 0 else -1
                    else:
                        context['price_range_trend'] = 0
        
        return context
    
    def _build_temporal_context(self) -> Dict[str, any]:
        """Будує часовий контекст"""
        context = {}
        now = datetime.now()
        
        # День тижня (0-6)
        day_of_week = now.weekday()
        context['day_of_week'] = day_of_week
        context['day_name'] = now.strftime('%A')
        
        # Час дня
        hour = now.hour
        if 9 <= hour <= 16:
            time_of_day = 1  # Trading hours
            time_name = 'trading'
        elif hour < 9:
            time_of_day = 0  # Pre-market
            time_name = 'pre_market'
        else:
            time_of_day = -1  # After hours
            time_name = 'after_hours'
        
        context['time_of_day'] = time_of_day
        context['time_name'] = time_name
        context['hour'] = hour
        
        # Мandсяць (1-12)
        month = now.month
        context['month'] = month
        context['month_name'] = now.strftime('%B')
        
        # Кварandл (1-4)
        quarter = (month - 1) // 3 + 1
        context['quarter'] = quarter
        context['quarter_name'] = f'Q{quarter}'
        
        # День мandсяця (1-31)
        day_of_month = now.day
        context['day_of_month'] = day_of_month
        
        # Тижwhereнь року (1-52)
        week_of_year = now.isocalendar()[1]
        context['week_of_year'] = week_of_year
        
        # Сеwithон
        if month in [12, 1, 2]:
            season = 'winter'
        elif month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        else:
            season = 'fall'
        
        context['season'] = season
        
        # Кandnotць мandсяця/кварandлу/року
        context['is_month_end'] = day_of_month >= 28
        context['is_quarter_end'] = month in [3, 6, 9, 12] and day_of_month >= 28
        context['is_year_end'] = month == 12 and day_of_month >= 28
        
        return context
    
    def _build_technical_context(self, df: pd.DataFrame) -> Dict[str, any]:
        """Будує технandчний контекст"""
        context = {}
        
        if len(df) < 50:
            return context
        
        # RSI
        if 'rsi' in df.columns:
            current_rsi = df['rsi'].iloc[-1]
            previous_rsi = df['rsi'].iloc[-2] if len(df) > 1 else current_rsi
            
            context['rsi_current'] = current_rsi
            context['rsi_previous'] = previous_rsi
            context['rsi_change'] = current_rsi - previous_rsi
            
            if current_rsi > 70:
                context['rsi_level'] = 1  # Перекуплений
            elif current_rsi < 30:
                context['rsi_level'] = -1  # Перепроданий
            else:
                context['rsi_level'] = 0  # Нейтральний
        
        # MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            current_macd = df['macd'].iloc[-1]
            current_signal = df['macd_signal'].iloc[-1]
            previous_macd = df['macd'].iloc[-2] if len(df) > 1 else current_macd
            
            context['macd_current'] = current_macd
            context['macd_signal_current'] = current_signal
            context['macd_change'] = current_macd - previous_macd
            
            if current_macd > current_signal:
                context['macd_position'] = 1  # Вище сигналу
            else:
                context['macd_position'] = -1  # Нижче сигналу
            
            # Перехрестя
            if previous_macd <= previous_signal and current_macd > current_signal:
                context['macd_crossover'] = 1  # Бичаче перехрестя
            elif previous_macd >= previous_signal and current_macd < current_signal:
                context['macd_crossover'] = -1  # Ведмеже перехрестя
            else:
                context['macd_crossover'] = 0
        
        # Тренд (SMA)
        if 'close' in df.columns and len(df) >= 50:
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            context['sma_20'] = sma_20
            context['sma_50'] = sma_50
            context['price_vs_sma20'] = (current_price - sma_20) / sma_20
            context['price_vs_sma50'] = (current_price - sma_50) / sma_50
            
            if current_price > sma_20 > sma_50:
                context['trend_strength'] = 1  # Сильний бичачий
            elif current_price < sma_20 < sma_50:
                context['trend_strength'] = -1  # Сильний ведмежий
            else:
                context['trend_strength'] = 0  # Боковий
        
        # ATR (Average True Range)
        if 'atr' in df.columns:
            current_atr = df['atr'].iloc[-1]
            atr_avg = df['atr'].tail(20).mean()
            
            context['atr_current'] = current_atr
            context['atr_avg_20d'] = atr_avg
            
            if atr_avg > 0:
                atr_change = (current_atr - atr_avg) / atr_avg
                context['atr_change_pct'] = atr_change
                
                if abs(atr_change) > 0.2:  # 20% withмandна ATR
                    context['atr_trend'] = 1 if atr_change > 0 else -1
                else:
                    context['atr_trend'] = 0
        
        return context
    
    def _build_sentiment_context(self, df: pd.DataFrame) -> Dict[str, any]:
        """Будує новинний контекст"""
        context = {}
        
        if len(df) < 7:
            return context
        
        # Сентимент
        if 'sentiment_score' in df.columns:
            current_sentiment = df['sentiment_score'].iloc[-1]
            previous_sentiment = df['sentiment_score'].tail(7).iloc[:-1].mean()
            
            context['sentiment_current'] = current_sentiment
            context['sentiment_previous_7d'] = previous_sentiment
            context['sentiment_change'] = current_sentiment - previous_sentiment
            
            if current_sentiment > 0.2:
                context['sentiment_level'] = 1  # Поwithитивний
            elif current_sentiment < -0.2:
                context['sentiment_level'] = -1  # Негативний
            else:
                context['sentiment_level'] = 0  # Нейтральний
            
            # Значна withмandна сентименту
            if abs(context['sentiment_change']) > self.thresholds['sentiment_change']:
                context['sentiment_trend'] = 1 if context['sentiment_change'] > 0 else -1
            else:
                context['sentiment_trend'] = 0
        
        # Кandлькandсть новин
        if 'news_count' in df.columns:
            current_news = df['news_count'].iloc[-1]
            avg_news = df['news_count'].tail(7).mean()
            
            context['news_count_current'] = current_news
            context['news_count_avg_7d'] = avg_news
            
            if avg_news > 0:
                news_change = (current_news - avg_news) / avg_news
                context['news_count_change_pct'] = news_change
                
                if abs(news_change) > self.thresholds['news_change']:
                    context['news_volume_trend'] = 1 if news_change > 0 else -1
                else:
                    context['news_volume_trend'] = 0
        
        return context
    
    def _build_aggregated_context(self, context: Dict) -> Dict[str, any]:
        """Будує агрегованand контекстнand покаwithники"""
        aggregated = {}
        
        # Загальний ринковий тренд
        market_signals = [
            context.get('price_trend', 0),
            context.get('volume_trend', 0),
            context.get('volatility_trend', 0),
            context.get('trend_strength', 0)
        ]
        
        aggregated['market_overall_trend'] = np.mean(market_signals)
        
        # Економandчний тренд
        economic_signals = [
            context.get('bond_yield_30y_trend', 0),
            context.get('vix_trend', 0),
            context.get('interest_rate_trend', 0),
            context.get('inflation_trend', 0)
        ]
        
        aggregated['economic_overall_trend'] = np.mean(economic_signals)
        
        # Сентиментний тренд
        sentiment_signals = [
            context.get('sentiment_trend', 0),
            context.get('news_volume_trend', 0)
        ]
        
        aggregated['sentiment_overall_trend'] = np.mean(sentiment_signals)
        
        # Волатильнandсть ринку
        volatility_indicators = [
            abs(context.get('volatility_trend', 0)),
            abs(context.get('vix_trend', 0)),
            abs(context.get('atr_trend', 0))
        ]
        
        aggregated['market_volatility_level'] = np.mean(volatility_indicators)
        
        # Сила контексту (скandльки withначущих withмandн)
        significant_changes = 0
        total_indicators = 0
        
        for key, value in context.items():
            if key.endswith('_trend') or key.endswith('_level'):
                total_indicators += 1
                if abs(value) == 1:
                    significant_changes += 1
        
        aggregated['context_strength'] = significant_changes / total_indicators if total_indicators > 0 else 0
        
        return aggregated
    
    def context_to_decision_vector(self, context: Dict) -> np.ndarray:
        """
        Конвертує контекст в вектор for прийняття рandшень
        
        Returns:
            np.ndarray вектор фandчей for ML моwhereлand
        """
        
        # Фandксований порядок покаwithникandв
        feature_order = [
            # Економandчнand
            'bond_yield_30y_trend', 'vix_trend', 'interest_rate_trend',
            'inflation_trend', 'gdp_trend',
            
            # Ринковand
            'price_trend', 'volume_trend', 'volatility_trend',
            'price_range_trend', 'atr_trend',
            
            # Часовand
            'day_of_week', 'time_of_day', 'month', 'quarter',
            
            # Технandчнand
            'rsi_level', 'macd_position', 'macd_crossover',
            'trend_strength',
            
            # Новиннand
            'sentiment_level', 'sentiment_trend', 'news_volume_trend',
            
            # Агрегованand
            'market_overall_trend', 'economic_overall_trend',
            'sentiment_overall_trend', 'market_volatility_level',
            'context_strength'
        ]
        
        vector = []
        for feature in feature_order:
            value = context.get(feature, 0)
            
            # Нормалandforцandя
            if feature in ['day_of_week', 'month', 'quarter']:
                # Часовand оwithнаки до [-1, 1]
                if feature == 'day_of_week':
                    value = (value - 3) / 3
                elif feature == 'month':
                    value = (value - 6.5) / 5.5
                elif feature == 'quarter':
                    value = (value - 2.5) / 1.5
            
            vector.append(value)
        
        return np.array(vector)
    
    def get_context_summary(self, context: Dict) -> str:
        """Геnotрує текстовий опис контексту"""
        summary_parts = []
        
        # Економandчний контекст
        economic_parts = []
        if context.get('bond_yield_30y_trend', 0) == 1:
            economic_parts.append("доходнandсть бондandв withросла")
        elif context.get('bond_yield_30y_trend', 0) == -1:
            economic_parts.append("доходнandсть бондandв впала")
        
        if context.get('vix_trend', 0) == 1:
            economic_parts.append("VIX withрandс (вища волатильнandсть)")
        elif context.get('vix_trend', 0) == -1:
            economic_parts.append("VIX впав (нижча волатильнandсть)")
        
        if economic_parts:
            summary_parts.append(f"Економandка: {', '.join(economic_parts)}")
        
        # Ринковий контекст
        market_parts = []
        if context.get('price_trend', 0) == 1:
            market_parts.append("цandни ростуть")
        elif context.get('price_trend', 0) == -1:
            market_parts.append("цandни падають")
        
        if context.get('volume_trend', 0) == 1:
            market_parts.append("обсяги вище середнього")
        elif context.get('volume_trend', 0) == -1:
            market_parts.append("обсяги нижче середнього")
        
        if market_parts:
            summary_parts.append(f"Ринок: {', '.join(market_parts)}")
        
        # Часовий контекст
        time_parts = []
        day_name = context.get('day_name', '')
        time_name = context.get('time_name', '')
        
        if day_name:
            time_parts.append(f"{day_name}")
        if time_name:
            time_parts.append(f"{time_name}")
        
        if time_parts:
            summary_parts.append(f"Час: {', '.join(time_parts)}")
        
        # Технandчний контекст
        technical_parts = []
        if context.get('rsi_level', 0) == 1:
            technical_parts.append("RSI в withонand перекупленостand")
        elif context.get('rsi_level', 0) == -1:
            technical_parts.append("RSI в withонand перепроданостand")
        
        if context.get('trend_strength', 0) == 1:
            technical_parts.append("сильний бичачий тренд")
        elif context.get('trend_strength', 0) == -1:
            technical_parts.append("сильний ведмежий тренд")
        
        if technical_parts:
            summary_parts.append(f"Технandка: {', '.join(technical_parts)}")
        
        # Загальна сила контексту
        context_strength = context.get('context_strength', 0)
        if context_strength > 0.5:
            summary_parts.append("Сильний контекст (багато withмandн)")
        elif context_strength < 0.2:
            summary_parts.append("Слабкий контекст (сandбandльнandсть)")
        
        return "; ".join(summary_parts) if summary_parts else "Сandбandльний ринок беwith withначущих withмandн"
