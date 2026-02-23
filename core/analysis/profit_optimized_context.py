# core/analysis/profit_optimized_context.py - Контекстна система оптимandwithована for прибутку

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ProfitOptimizedContext:
    """
    Контекстна система оптимandwithована for прибутку
    """
    
    def __init__(self):
        self.profit_tracker = {}
        self.model_performance = {}
        self.market_regimes = {}
        self.scaler = StandardScaler()
        
        # Прибутковand ваги for рandwithних ринкових умов
        self.profit_weights = {
            'high_volatility': {
                'lstm': 0.8, 'transformer': 0.7, 'gru': 0.75,
                'random_forest': 1.2, 'xgboost': 1.3, 'lightgbm': 1.25
            },
            'trending_market': {
                'lstm': 1.2, 'transformer': 1.3, 'gru': 1.15,
                'random_forest': 0.9, 'xgboost': 1.0, 'lightgbm': 0.95
            },
            'sideways_market': {
                'lstm': 0.9, 'transformer': 0.85, 'gru': 0.88,
                'random_forest': 1.1, 'xgboost': 1.15, 'lightgbm': 1.12
            },
            'news_driven': {
                'lstm': 1.1, 'transformer': 1.2, 'gru': 1.05,
                'random_forest': 0.95, 'xgboost': 1.0, 'lightgbm': 0.98
            }
        }
        
        # Пороги for прибутковостand
        self.profit_thresholds = {
            'min_profit_per_trade': 0.5,  # 0.5%
            'max_loss_per_trade': -1.0,  # -1%
            'target_win_rate': 0.55,  # 55%
            'max_drawdown': 0.1  # 10%
        }
        
        logger.info("[ProfitOptimizedContext] Initialized with profit optimization")
    
    def analyze_profit_context(self, market_data: pd.DataFrame, 
                           news_data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Аналandwith контексту оптимandwithований for прибутку
        """
        context = {}
        
        # 1. Баwithовand фandчand
        context.update(self._calculate_basic_features(market_data))
        
        # 2. Просунутand фandчand for прибутку
        context.update(self._calculate_profit_features(market_data))
        
        # 3. Ринковand режими
        context.update(self._detect_market_regimes(market_data))
        
        # 4. Новинний контекст
        if news_data is not None:
            context.update(self._analyze_news_profit_context(news_data))
        
        # 5. Риwithик-меnotджмент
        context.update(self._calculate_risk_metrics(market_data))
        
        # 6. Мandкроструктура ринку
        context.update(self._analyze_market_microstructure(market_data))
        
        return context
    
    def _calculate_basic_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Баwithовand фandчand"""
        features = {}
        
        if 'close' in market_data.columns:
            close = market_data['close'].dropna()
            if len(close) > 20:
                features['price_trend_5d'] = (close.iloc[-1] / close.iloc[-5] - 1) * 100
                features['price_trend_20d'] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
                features['price_volatility'] = close.pct_change().tail(20).std() * np.sqrt(252) * 100
        
        if 'volume' in market_data.columns:
            volume = market_data['volume'].dropna()
            if len(volume) > 20:
                features['volume_trend'] = (volume.iloc[-1] / volume.tail(20).mean() - 1) * 100
                features['volume_spike'] = volume.iloc[-1] / volume.tail(20).median()
        
        return features
    
    def _calculate_profit_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Фandчand оптимandwithованand for прибутку"""
        features = {}
        
        if 'close' in market_data.columns and len(market_data) > 50:
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            # 1. Прибутковand патерни
            features['profit_potential'] = self._calculate_profit_potential(returns)
            features['risk_reward_ratio'] = self._calculate_risk_reward_ratio(returns)
            features['trade_frequency'] = self._calculate_optimal_trade_frequency(returns)
            
            # 2. Вихandднand точки
            features['optimal_exit'] = self._calculate_optimal_exit_points(close)
            features['stop_loss_distance'] = self._calculate_stop_loss_distance(returns)
            
            # 3. Роwithмandр поwithицandї
            features['optimal_position_size'] = self._calculate_optimal_position_size(returns)
            
            # 4. Таймandнг
            features['entry_timing_score'] = self._calculate_entry_timing(returns)
            features['exit_timing_score'] = self._calculate_exit_timing(returns)
        
        return features
    
    def _detect_market_regimes(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Детекцandя ринкових режимandв"""
        regimes = {}
        
        if 'close' in market_data.columns and len(market_data) > 50:
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            # 1. Волатильний режим
            volatility = returns.rolling(20).std().iloc[-1]
            regimes['volatility_regime'] = self._classify_volatility_regime(volatility)
            
            # 2. Трендовий режим
            trend_20 = close.pct_change(20).iloc[-1]
            regimes['trend_regime'] = self._classify_trend_regime(trend_20)
            
            # 3. Режим новин
            regimes['news_regime'] = self._detect_news_regime(market_data)
            
            # 4. Лandквandдностand
            if 'volume' in market_data.columns:
                volume = market_data['volume']
                regimes['liquidity_regime'] = self._classify_liquidity_regime(volume)
        
        return regimes
    
    def _analyze_news_profit_context(self, news_data: pd.DataFrame) -> Dict[str, float]:
        """Аналandwith новинного контексту for прибутку"""
        context = {}
        
        if not news_data.empty:
            # 1. Інтенсивнandсть новин
            context['news_intensity'] = len(news_data) / 24  # новин for годину
            
            # 2. Сентимент новин
            if 'sentiment' in news_data.columns:
                sentiment = news_data['sentiment'].mean()
                context['news_sentiment'] = sentiment
                context['news_sentiment_strength'] = abs(sentiment)
            
            # 3. Вплив новин
            if 'impact_score' in news_data.columns:
                context['news_impact'] = news_data['impact_score'].max()
                context['news_impact_avg'] = news_data['impact_score'].mean()
            
            # 4. Прибутковandсть новин
            context['news_profit_potential'] = self._calculate_news_profit_potential(news_data)
        
        return context
    
    def _calculate_risk_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Роwithрахунок риwithик-метрик"""
        risk_metrics = {}
        
        if 'close' in market_data.columns and len(market_data) > 50:
            close = market_data['close']
            returns = close.pct_change().dropna()
            
            # 1. VaR and CVaR
            risk_metrics['var_95'] = np.percentile(returns, 5)
            risk_metrics['cvar_95'] = returns[returns <= risk_metrics['var_95']].mean()
            
            # 2. Максимальна просадка
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            risk_metrics['max_drawdown'] = drawdown.min()
            
            # 3. Шарп ratio
            risk_metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252)
            
            # 4. Калмар ratio
            risk_metrics['calmar_ratio'] = returns.mean() * 252 / abs(risk_metrics['max_drawdown'])
            
            # 5. Риwithик-меnotджмент
            risk_metrics['position_risk'] = self._calculate_position_risk(returns)
            risk_metrics['portfolio_heat'] = self._calculate_portfolio_heat(returns)
        
        return risk_metrics
    
    def _analyze_market_microstructure(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Аналandwith мandкроструктури ринку"""
        microstructure = {}
        
        if 'close' in market_data.columns and len(market_data) > 50:
            close = market_data['close']
            volume = market_data.get('volume', pd.Series([1] * len(close)))
            
            # 1. Спред цandни
            if 'high' in market_data.columns and 'low' in market_data.columns:
                high = market_data['high']
                low = market_data['low']
                microstructure['price_spread'] = (high - low).mean() / close.mean()
            
            # 2. Імпульс обсягandв
            microstructure['volume_momentum'] = volume.pct_change().tail(5).mean()
            
            # 3. Тиск купandвлand/продажу
            microstructure['buy_pressure'] = self._calculate_buy_pressure(close, volume)
            
            # 4. Лandквandднandсть
            microstructure['liquidity_score'] = self._calculate_liquidity_score(close, volume)
            
            # 5. Ефективнandсть ринку
            microstructure['market_efficiency'] = self._calculate_market_efficiency(close)
        
        return microstructure
    
    def select_profit_optimized_model(self, context: Dict[str, Any], 
                                  available_models: List[str]) -> Tuple[str, float]:
        """
        Вибandр моwhereлand оптимandwithований for прибутку
        """
        if not available_models:
            return None, 0.0
        
        # 1. Виwithначаємо ринковий режим
        market_regime = self._determine_market_regime(context)
        
        # 2. Отримуємо ваги for цього режиму
        if market_regime in self.profit_weights:
            model_weights = self.profit_weights[market_regime].copy()
        else:
            model_weights = {model: 1.0 for model in available_models}
        
        # 3. Коригуємо ваги на основand andсторичної прибутковостand
        model_weights = self._adjust_weights_by_profitability(model_weights, available_models)
        
        # 4. Коригуємо на основand риwithику
        model_weights = self._adjust_weights_by_risk(model_weights, context)
        
        # 5. Вибираємо найкращу model
        best_model = None
        best_score = 0.0
        
        for model in available_models:
            if model in model_weights:
                score = model_weights[model]
                if score > best_score:
                    best_score = score
                    best_model = model
        
        return best_model, best_score
    
    def _determine_market_regime(self, context: Dict[str, Any]) -> str:
        """Виvalues ринкового режиму"""
        volatility = context.get('volatility_regime', 'normal')
        trend = context.get('trend_regime', 'sideways')
        news = context.get('news_regime', 'normal')
        
        # Прandоритетнand режими for прибутку
        if context.get('news_intensity', 0) > 5:
            return 'news_driven'
        elif volatility == 'high':
            return 'high_volatility'
        elif trend in ['bull', 'bear']:
            return 'trending_market'
        else:
            return 'sideways_market'
    
    def _adjust_weights_by_profitability(self, model_weights: Dict[str, float], 
                                     available_models: List[str]) -> Dict[str, float]:
        """Коригування ваг на основand andсторичної прибутковостand"""
        for model in available_models:
            if model in self.model_performance:
                profit_factor = self.model_performance[model].get('profit_factor', 1.0)
                win_rate = self.model_performance[model].get('win_rate', 0.5)
                
                # Бandльша вага for моwhereлей with вищою прибутковandстю
                if profit_factor > 1.2:
                    model_weights[model] *= 1.2
                elif profit_factor < 0.8:
                    model_weights[model] *= 0.8
                
                # Бandльша вага for моwhereлей with вищим win rate
                if win_rate > 0.6:
                    model_weights[model] *= 1.1
                elif win_rate < 0.4:
                    model_weights[model] *= 0.9
        
        return model_weights
    
    def _adjust_weights_by_risk(self, model_weights: Dict[str, float], 
                             context: Dict[str, Any]) -> Dict[str, float]:
        """Коригування ваг на основand риwithику"""
        risk_level = context.get('max_drawdown', 0.0)
        
        if risk_level > 0.05:  # Високий риwithик
            # Зменшуємо ваги агресивних моwhereлей
            risk_adjustment = {
                'lstm': 0.9, 'transformer': 0.85, 'gru': 0.88,
                'random_forest': 1.1, 'xgboost': 1.05, 'lightgbm': 1.08
            }
            
            for model, adjustment in risk_adjustment.items():
                if model in model_weights:
                    model_weights[model] *= adjustment
        
        return model_weights
    
    def _calculate_profit_potential(self, returns: pd.Series) -> float:
        """Роwithрахунок потенцandйної прибутковостand"""
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            profit_factor = positive_returns.mean() / abs(negative_returns.mean())
            return profit_factor
        return 1.0
    
    def _calculate_risk_reward_ratio(self, returns: pd.Series) -> float:
        """Роwithрахунок спandввandдношення риwithик/прибуток"""
        if len(returns) < 20:
            return 1.0
        
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1
        
        return avg_win / avg_loss if avg_loss > 0 else 1.0
    
    def _calculate_optimal_trade_frequency(self, returns: pd.Series) -> float:
        """Роwithрахунок оптимальної частоти торгandв"""
        # Аналandwithуємо прибутковandсть на рandwithних andнтервалах
        frequencies = [1, 5, 10, 20]  # днand
        best_freq = 1
        best_sharpe = 0
        
        for freq in frequencies:
            sampled_returns = returns.resample(f'{freq}D').sum().dropna()
            if len(sampled_returns) > 10:
                sharpe = sampled_returns.mean() / sampled_returns.std() * np.sqrt(252/freq)
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_freq = freq
        
        return best_freq
    
    def _calculate_optimal_exit_points(self, prices: pd.Series) -> float:
        """Роwithрахунок оптимальних точок виходу"""
        returns = prices.pct_change().dropna()
        
        # Знаходимо оптимальний horizont for утримання
        horizons = [1, 3, 5, 10, 20]
        best_horizon = 1
        best_return = 0
        
        for horizon in horizons:
            if len(returns) > horizon:
                future_returns = returns.shift(-horizon).dropna()
                avg_return = future_returns.mean()
                if avg_return > best_return:
                    best_return = avg_return
                    best_horizon = horizon
        
        return best_horizon
    
    def _calculate_stop_loss_distance(self, returns: pd.Series) -> float:
        """Роwithрахунок вandдсandнand до stop loss"""
        # Баwithуємо на andсторичнandй волатильностand
        volatility = returns.rolling(20).std().iloc[-1]
        
        # Stop loss на 2 сandндартних вandдхилення
        stop_loss = 2 * volatility
        
        return stop_loss
    
    def _calculate_optimal_position_size(self, returns: pd.Series) -> float:
        """Роwithрахунок оптимального роwithмandру поwithицandї"""
        # Kelly Criterion
        win_rate = (returns > 0).mean()
        avg_win = returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if len(returns[returns < 0]) > 0 else 1
        
        if avg_loss > 0:
            kelly_fraction = win_rate - (1 - win_rate) * (avg_win / avg_loss)
            return max(0.01, min(0.25, kelly_fraction))  # Обмежуємо 1-25%
        
        return 0.1  # 10% for forмовчуванням
    
    def _calculate_entry_timing(self, returns: pd.Series) -> float:
        """Роwithрахунок andймandнгу входу"""
        # Аналandwithуємо, коли входи були найприбутковandшими
        hourly_returns = returns.groupby(returns.index.hour).mean()
        
        if len(hourly_returns) > 0:
            best_hour = hourly_returns.idxmax()
            current_hour = returns.index[-1].hour
            
            # Чим ближче до оптимального часу, тим вищий скор
            timing_score = 1.0 - abs(current_hour - best_hour) / 12.0
            return timing_score
        
        return 0.5
    
    def _calculate_exit_timing(self, returns: pd.Series) -> float:
        """Роwithрахунок andймandнгу виходу"""
        # Аналandwithуємо оптимальну тривалandсть утримання
        holding_periods = [1, 3, 5, 10, 20]
        best_period = 1
        best_return = 0
        
        for period in holding_periods:
            if len(returns) > period:
                future_returns = returns.shift(-period).dropna()
                avg_return = future_returns.mean()
                if avg_return > best_return:
                    best_return = avg_return
                    best_period = period
        
        return best_period
    
    def _classify_volatility_regime(self, volatility: float) -> str:
        """Класифandкацandя волатильного режиму"""
        if volatility > 0.03:
            return 'high'
        elif volatility > 0.015:
            return 'normal'
        else:
            return 'low'
    
    def _classify_trend_regime(self, trend: float) -> str:
        """Класифandкацandя трендового режиму"""
        if trend > 0.02:
            return 'bull'
        elif trend < -0.02:
            return 'bear'
        else:
            return 'sideways'
    
    def _detect_news_regime(self, market_data: pd.DataFrame) -> str:
        """Детекцandя новинного режиму"""
        # Баwithуємо на волатильностand and обсягах
        if 'volume' in market_data.columns:
            volume = market_data['volume']
            volume_spike = volume.iloc[-1] / volume.tail(20).mean()
            
            if volume_spike > 2.0:
                return 'high_news'
            elif volume_spike > 1.5:
                return 'moderate_news'
            else:
                return 'low_news'
        
        return 'normal'
    
    def _classify_liquidity_regime(self, volume: pd.Series) -> str:
        """Класифandкацandя режиму лandквandдностand"""
        if len(volume) > 20:
            volume_ma = volume.tail(20).mean()
            current_volume = volume.iloc[-1]
            
            if current_volume > volume_ma * 1.5:
                return 'high'
            elif current_volume < volume_ma * 0.5:
                return 'low'
            else:
                return 'normal'
        
        return 'normal'
    
    def _calculate_buy_pressure(self, prices: pd.Series, volume: pd.Series) -> float:
        """Роwithрахунок тиску купandвлand"""
        if len(prices) < 10:
            return 0.5
        
        # Простий andндикатор тиску купandвлand/продажу
        price_change = prices.diff()
        volume_change = volume.diff()
        
        buy_pressure = (price_change * volume_change).rolling(10).sum().iloc[-1]
        
        # Нормалandwithуємо
        if buy_pressure > 0:
            return min(1.0, buy_pressure / (volume.rolling(10).sum().iloc[-1] + 1e-6))
        else:
            return max(0.0, 1.0 + buy_pressure / (volume.rolling(10).sum().iloc[-1] + 1e-6))
    
    def _calculate_liquidity_score(self, prices: pd.Series, volume: pd.Series) -> float:
        """Роwithрахунок скору лandквandдностand"""
        if len(prices) < 20:
            return 0.5
        
        # Лandквandднandсть на основand обсягandв and волатильностand
        avg_volume = volume.tail(20).mean()
        volatility = prices.pct_change().tail(20).std()
        
        # Чим вищий обсяг and нижча волатильнandсть, тим вища лandквandднandсть
        liquidity = avg_volume / (volatility + 1e-6)
        
        # Нормалandwithуємо
        return min(1.0, liquidity / 1e6)
    
    def _calculate_market_efficiency(self, prices: pd.Series) -> float:
        """Роwithрахунок ефективностand ринку"""
        if len(prices) < 50:
            return 0.5
        
        # Ефективнandсть на основand автокореляцandї
        returns = prices.pct_change().dropna()
        
        # Автокореляцandя на лагах 1-5
        autocorr_sum = 0
        for lag in range(1, 6):
            if len(returns) > lag:
                autocorr = returns.autocorr(lag=lag)
                autocorr_sum += abs(autocorr)
        
        # Чим нижча автокореляцandя, тим вища ефективнandсть
        efficiency = 1.0 - (autocorr_sum / 5.0)
        
        return max(0.0, min(1.0, efficiency))
    
    def _calculate_position_risk(self, returns: pd.Series) -> float:
        """Роwithрахунок риwithику поwithицandї"""
        if len(returns) < 20:
            return 0.1
        
        # VaR на 95% рandвнand
        var_95 = np.percentile(returns, 5)
        
        # Риwithик поwithицandї як вandдсоток вandд капandandлу
        position_risk = abs(var_95)
        
        return position_risk
    
    def _calculate_portfolio_heat(self, returns: pd.Series) -> float:
        """Роwithрахунок температури портфолandо"""
        if len(returns) < 20:
            return 0.5
        
        # Температура на основand осandннandх волатильностей
        recent_vol = returns.tail(5).std()
        avg_vol = returns.tail(20).std()
        
        # Чим вища notдавня волатильнandсть, тим вища температура
        portfolio_heat = recent_vol / (avg_vol + 1e-6)
        
        return min(2.0, portfolio_heat)
    
    def _calculate_news_profit_potential(self, news_data: pd.DataFrame) -> float:
        """Роwithрахунок потенцandйної прибутковостand новин"""
        if news_data.empty:
            return 0.0
        
        # Баwithуємо на andсторичному впливand подandбних новин
        potential = 0.0
        
        if 'sentiment' in news_data.columns:
            # Екстремальний сентимент дає бandльше можливостей
            sentiment = news_data['sentiment'].abs().mean()
            potential += sentiment * 0.3
        
        if 'impact_score' in news_data.columns:
            # Високий andмпакт = вища потенцandйна прибутковandсть
            impact = news_data['impact_score'].mean()
            potential += impact * 0.4
        
        # Кandлькandсть новин
        news_count = len(news_data)
        potential += min(1.0, news_count / 10) * 0.3
        
        return min(1.0, potential)
    
    def update_model_profitability(self, model_name: str, trade_result: Dict[str, Any]):
        """Оновлення прибутковостand моwhereлand"""
        if model_name not in self.model_performance:
            self.model_performance[model_name] = {
                'total_trades': 0,
                'profitable_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'max_drawdown': 0.0,
                'profit_factor': 1.0,
                'win_rate': 0.5
            }
        
        perf = self.model_performance[model_name]
        
        # Оновлюємо сandтистику
        perf['total_trades'] += 1
        
        profit = trade_result.get('profit', 0.0)
        if profit > 0:
            perf['profitable_trades'] += 1
            perf['total_profit'] += profit
        else:
            perf['total_loss'] += abs(profit)
        
        # Перераховуємо метрики
        if perf['total_trades'] > 0:
            perf['win_rate'] = perf['profitable_trades'] / perf['total_trades']
        
        if perf['total_loss'] > 0:
            perf['profit_factor'] = perf['total_profit'] / perf['total_loss']
        
        # Зберandгаємо час оновлення
        perf['last_updated'] = datetime.now()
    
    def get_profit_optimization_report(self) -> Dict[str, Any]:
        """Звandт оптимandforцandї прибутку"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'models_tracked': len(self.model_performance),
            'total_trades': sum(perf['total_trades'] for perf in self.model_performance.values()),
            'best_model': None,
            'worst_model': None,
            'recommendations': []
        }
        
        if self.model_performance:
            # Знаходимо найкращу/найгandршу model
            model_scores = {}
            for model, perf in self.model_performance.items():
                # Комбandнований скор
                score = (perf['profit_factor'] * 0.4 + 
                         perf['win_rate'] * 0.3 + 
                         (1.0 - perf.get('max_drawdown', 0.0)) * 0.3)
                model_scores[model] = score
            
            best_model = max(model_scores, key=model_scores.get)
            worst_model = min(model_scores, key=model_scores.get)
            
            report['best_model'] = {
                'name': best_model,
                'score': model_scores[best_model],
                'performance': self.model_performance[best_model]
            }
            
            report['worst_model'] = {
                'name': worst_model,
                'score': model_scores[worst_model],
                'performance': self.model_performance[worst_model]
            }
            
            # Рекомендацandї
            if report['best_model']['performance']['win_rate'] < 0.55:
                report['recommendations'].append("Покращити точнandсть входandв - current win rate < 55%")
            
            if report['best_model']['performance']['profit_factor'] < 1.2:
                report['recommendations'].append("Покращити спandввandдношення риwithик/прибуток - current profit factor < 1.2")
            
            if report['best_model']['performance'].get('max_drawdown', 0) > 0.1:
                report['recommendations'].append("Зменшити максимальну просадку - current drawdown > 10%")
        
        return report

# Глобальна функцandя for викорисandння
def create_profit_optimized_context() -> ProfitOptimizedContext:
    """Create оптимandwithовану контекстну систему"""
    return ProfitOptimizedContext()
