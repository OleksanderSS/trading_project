# core/trading_advisor.py

import pandas as pd
from typing import Dict, Optional, Callable, Any
import logging
from utils.trading_calendar import TradingCalendar
from config.thresholds import get_all_thresholds
from core.context_enricher import ContextEnricher
from utils.layer_signal_processor import LayerSignalProcessor
from utils.ensemble import ensemble_forecast
import numpy as np

logger = logging.getLogger("TradingProjectLogger")

SIGNAL_TO_NUM = {"BUY": 1, "HOLD": 0, "SELL": -1}
NUM_TO_SIGNAL = {1: "BUY", 0: "HOLD", -1: "SELL"}

def numeric_to_signal(score: float) -> str:
    if score > 0.3:
        return "BUY"
    elif score < -0.3:
        return "SELL"
    else:
        return "HOLD"

class TradingAdvisor:
    """
    Клас for формування сигналandв BUY/HOLD/SELL for одного or allх тикерandв.
    Інкапсулює operation моwhereлей, ансамблювання, RSI and сентименту.
    """
    
    def __init__(
        self,
        ticker: str = "SPY",
        interval: str = "1d",
        forecast_thresholds: Optional[Dict[str, float]] = None,
        rsi_thresholds: Optional[Dict[str, float]] = None,
        sentiment_thresholds: Optional[Dict[str, float]] = None,
        weights: Optional[Dict[str, float]] = None,
        calendar: Optional[TradingCalendar] = None
    ):
        thresholds = get_all_thresholds(ticker, interval)

        self.forecast_thresholds = forecast_thresholds or {
            "bullish": thresholds["forecast_bullish"],
            "bearish": thresholds["forecast_bearish"]
        }
        self.rsi_thresholds = rsi_thresholds or {
            "oversold": thresholds["rsi"][0],
            "overbought": thresholds["rsi"][1]
        }
        self.sentiment_thresholds = sentiment_thresholds or {
            "positive": thresholds["sentiment_positive"],
            "negative": thresholds["sentiment_negative"]
        }
        self.weights = weights or {"forecast": 0.5, "rsi": 0.3, "sentiment": 0.2}
        self.calendar = calendar
        self.layer_processor = LayerSignalProcessor()
        self.context_enricher = ContextEnricher()

    def get_ensemble_predictions(self, df_features: pd.DataFrame, debug: bool = False) -> Dict[str, list]:
        if "prediction_proba" in df_features.columns:
            preds = df_features["prediction_proba"].tolist()
        else:
            preds = [0.0] * len(df_features)
        return {"ensemble": preds}

    def get_signals_for_ticker(
        self,
        df_features: pd.DataFrame,
        ticker: str,
        interval: str = "1d",
        avg_sentiment: Optional[dict] = None,
        news_analyzer=None,
        process_news_func: Optional[Callable] = None,
        debug: bool = False
    ) -> Dict[str, str]:

        signals_default = {
            "forecast_signal": "HOLD",
            "rsi_signal": "HOLD",
            "sentiment_signal": "HOLD",
            "final_signal": "HOLD",
            "type": "ensemble"
        }

        if df_features.empty:
            return signals_default

        df_features = df_features.copy()
        df_features.index = pd.to_datetime(df_features.index, errors='coerce')
        df_features = df_features[df_features.index.notna()]
        if df_features.empty:
            return signals_default
        
        #  Застосовуємо ваги шарandв до фandчей (поки all = 1.0)
        df_features = self.layer_processor.apply_layer_weights_to_features(df_features)
        
        if debug:
            layer_breakdown = self.layer_processor.get_layer_signal_breakdown(df_features, {})
            logger.info(f"[Advisor] Layer breakdown for {ticker}: {layer_breakdown}")

        # Перевandрка торгового дня
        if self.calendar:
            last_date = df_features.index[-1].date()
            if not self.calendar.is_trading_day(last_date):
                logger.info(f"[Advisor] {last_date} not є торговим дnotм  сигнал HOLD")
                return signals_default

        forecast_signal, rsi_signal, sentiment_signal = "HOLD", "HOLD", "HOLD"

        # --- Forecast ---
        try:
            preds = self.get_ensemble_predictions(df_features, debug=debug)
            last_pred = preds.get('ensemble', [0.0])[-1]
            try:
                last_pred = float(last_pred)
            except (ValueError, TypeError):
                last_pred = 0.0
            if last_pred >= self.forecast_thresholds["bullish"]:
                forecast_signal = "BUY"
            elif last_pred <= self.forecast_thresholds["bearish"]:
                forecast_signal = "SELL"
        except Exception as e:
            logger.warning(f"[Forecast] Failed for {ticker} {interval}: {e}")

        # --- RSI ---
        try:
            for col in ["RSI_14", "RSI_day"]:
                if col in df_features.columns and not df_features[col].isna().all():
                    last_rsi = df_features[col].iloc[-1]
                    try:
                        last_rsi = float(last_rsi)
                    except (ValueError, TypeError):
                        last_rsi = 50.0
                    if last_rsi <= self.rsi_thresholds["oversold"]:
                        rsi_signal = "BUY"
                    elif last_rsi >= self.rsi_thresholds["overbought"]:
                        rsi_signal = "SELL"
        except Exception as e:
            logger.warning(f"[RSI] Failed for {ticker} {interval}: {e}")

        # --- Sentiment ---
        try:
            if avg_sentiment is None:
                if news_analyzer:
                    avg_sentiment = news_analyzer.get_latest_news_sentiment(df_features.index[-1])
                elif process_news_func:
                    _, avg_sentiment, _ = process_news_func()
                else:
                    avg_sentiment = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

            if "sentiment_score" in df_features.columns:
                sentiment_val = df_features["sentiment_score"].iloc[-1]
                try:
                    sentiment_val = float(sentiment_val)
                except (ValueError, TypeError):
                    sentiment_val = 0.0
                if sentiment_val > self.sentiment_thresholds["positive"]:
                    sentiment_signal = "BUY"
                elif sentiment_val < self.sentiment_thresholds["negative"]:
                    sentiment_signal = "SELL"
            else:
                if avg_sentiment.get("positive", 0.0) >= self.sentiment_thresholds["positive"]:
                    sentiment_signal = "BUY"
                elif avg_sentiment.get("negative", 0.0) <= self.sentiment_thresholds["negative"]:
                    sentiment_signal = "SELL"
        except Exception as e:
            logger.warning(f"[Sentiment] Failed for {ticker} {interval}: {e}")

        # --- Final Signal ---
        try:
            score = (
                SIGNAL_TO_NUM.get(forecast_signal, 0) * self.weights["forecast"] +
                SIGNAL_TO_NUM.get(rsi_signal, 0) * self.weights["rsi"] +
                SIGNAL_TO_NUM.get(sentiment_signal, 0) * self.weights["sentiment"]
            )

            if "market_phase" in df_features.columns:
                phase = df_features["market_phase"].iloc[-1]
                if phase == "bull":
                    score *= 1.1
                elif phase == "bear":
                    score *= 0.9
                elif phase == "sideways":
                    score *= 0.8

            final_signal = numeric_to_signal(score)
        except Exception as e:
            logger.warning(f"[FinalSignal] Failed for {ticker} {interval}: {e}")
            final_signal = "HOLD"

        # Логування промandжних сигналandв with andнформацandєю про шари
        active_layers = [name for name, weight in self.layer_processor.layer_weights.items() if weight != 1.0]
        layer_info = f" | Active layers: {active_layers}" if active_layers else " | All layers neutral (1.0)"
        
        # Створюємо whereandльний лог сигналandв
        signal_details = {
            'timestamp': pd.Timestamp.now(),
            'ticker': ticker,
            'interval': interval,
            'forecast_signal': forecast_signal,
            'rsi_signal': rsi_signal,
            'sentiment_signal': sentiment_signal,
            'final_signal': final_signal,
            'confidence': self._calculate_confidence(forecast_signal, rsi_signal, sentiment_signal),
            'actual_result': None,  # Буwhere forповnotно пandwithнandше
            'accuracy': None  # Буwhere роwithраховано пandwithнandше
        }
        
        # Зберandгаємо whereandльний сигнал
        if not hasattr(self, '_detailed_signals_log'):
            self._detailed_signals_log = []
        self._detailed_signals_log.append(signal_details)
        
        # Рекомендацandя for користувача
        if final_signal == "BUY":
            action = "КУПУЙМО"
        elif final_signal == "SELL":
            action = "ПРОДАЙМО"
        else:
            action = "HOLD"
            
        # Логуємо with роwithширеною andнформацandєю
        logger.info(f"[SIGNAL] {ticker} {interval}: {action} (Confidence: {signal_details['confidence']:.2f})")
        logger.info(f"[Advisor] {ticker} {interval} -> forecast={forecast_signal}, rsi={rsi_signal}, sentiment={sentiment_signal}, final={final_signal}{layer_info}")
        # Виводимо andблицю сигналandв for allх моwhereлей
        self._print_signal_table(ticker, interval)
        
        # Додаємо сигнал до глобального списку for фandнального withвandту
        if not hasattr(self, '_all_signals_log'):
            self._all_signals_log = []
        
        self._all_signals_log.append({
            'ticker': ticker,
            'interval': interval,
            'forecast': forecast_signal,
            'rsi': rsi_signal,
            'sentiment': sentiment_signal,
            'final': final_signal,
            'layer_info': layer_info
        })

        # Prepare layer info for context enrichment
        layer_info_dict = {
            "active_layers": active_layers,
            "total_layers": len(self.layer_processor.layer_weights),
            "neutral_layers": len([w for w in self.layer_processor.layer_weights.values() if w == 1.0]),
            "layer_weights": self.layer_processor.layer_weights
        }
        
        # Enrich signal with context
        base_signal = {
            "forecast_signal": forecast_signal,
            "rsi_signal": rsi_signal,
            "sentiment_signal": sentiment_signal,
            "final_signal": final_signal,
            "type": "ensemble",
            "layer_info": layer_info_dict
        }
        
        enriched_signal = self.context_enricher.enrich_signal_with_context(
            base_signal, layer_info_dict
        )

        return enriched_signal

    def get_signals_for_all_tickers(
            self,
            df_features_all: Dict[str, Dict[str, pd.DataFrame]],
            avg_sentiment: Optional[dict] = None,
            news_analyzer=None,
            process_news_func: Optional[Callable] = None,
            debug: bool = False
    ) -> Dict[str, Dict[str, Dict[str, str]]]:
        all_signals = {}
        for ticker, tf_dict in df_features_all.items():
            ticker_signals = {}
            for interval, df_feat in tf_dict.items():
                try:
                    advisor = TradingAdvisor(ticker=ticker, interval=interval, calendar=self.calendar)
                    sig = advisor.get_signals_for_ticker(
                        df_features=df_feat,
                        ticker=ticker,
                        interval=interval,
                        avg_sentiment=avg_sentiment,
                        news_analyzer=news_analyzer,
                        process_news_func=process_news_func,
                        debug=debug
                    )
                    ticker_signals[interval] = sig
                except Exception as e:
                    logger.warning(f"[SignalGen] Failed for {ticker} {interval}: {e}")
                    ticker_signals[interval] = {
                        "forecast_signal": "HOLD",
                        "rsi_signal": "HOLD",
                        "sentiment_signal": "HOLD",
                        "final_signal": "HOLD",
                        "type": "ensemble"
                    }
            all_signals[ticker] = ticker_signals
        return all_signals
    
    def update_layer_weight(self, layer_name: str, new_weight: float):
        """Оновлює вагу шару for тюнandнгу сигналandв"""
        self.layer_processor.update_layer_weight(layer_name, new_weight)
        logger.info(f"[Advisor] Оновлено вагу шару '{layer_name}' до {new_weight}")
    
    def get_layer_summary(self) -> Dict[str, Dict]:
        """Поверandє withвеwhereння по allх шарах"""
        return self.layer_processor.get_layer_summary()
    
    def reset_layer_weights(self):
        """Скидає all ваги шарandв до notйтрального values 1.0"""
        self.layer_processor.reset_all_weights()
        logger.info("[Advisor] Всand ваги шарandв скинуто до 1.0")
    
    def get_all_signals_summary(self) -> Dict[str, any]:
        """Поверandє withвеwhereння по allх сигналах for фandнального withвandту"""
        if not hasattr(self, '_all_signals_log'):
            return {"total_signals": 0, "signals": []}
        
        summary = {
            "total_signals": len(self._all_signals_log),
            "signals": self._all_signals_log,
            "tickers": list(set(sig['ticker'] for sig in self._all_signals_log)),
            "timeframes": list(set(sig['interval'] for sig in self._all_signals_log)),
            "final_signals_count": {
                "BUY": len([s for s in self._all_signals_log if s['final'] == 'BUY']),
                "SELL": len([s for s in self._all_signals_log if s['final'] == 'SELL']),
                "HOLD": len([s for s in self._all_signals_log if s['final'] == 'HOLD'])
            }
        }
        return summary


    def _calculate_confidence(self, forecast_signal, rsi_signal, sentiment_signal):
        """Роwithрахунок впевnotностand сигналу"""
        try:
            # Конвертуємо сигнали в числа
            forecast_val = signal_to_numeric(forecast_signal)
            rsi_val = signal_to_numeric(rsi_signal)
            sentiment_val = signal_to_numeric(sentiment_signal)
        
        # Роwithрахуємо консенсус
            total = abs(forecast_val) + abs(rsi_val) + abs(sentiment_val)
            if total == 0:
                return 0.0
            
            consensus = (abs(forecast_val) + abs(rsi_val) + abs(sentiment_val)) / 3
            return min(consensus, 1.0)
        except:
            return 0.0

    def _print_signal_table(self, ticker, interval):
        """Виводить andблицю сигналandв for поточного тandкера"""
        if not hasattr(self, '_detailed_signals_log'):
            return
        
        # Фandльтруємо сигнали for поточного тandкера/andнтервалу
        current_signals = [s for s in self._detailed_signals_log 
                          if s['ticker'] == ticker and s['interval'] == interval]
        
        if current_signals:
            latest = current_signals[-1]
            logger.info(f"\n=== SIGNAL TABLE {ticker} {interval} ===")
            logger.info(f"Forecast: {latest['forecast_signal']} | RSI: {latest['rsi_signal']} | Sentiment: {latest['sentiment_signal']}")
            logger.info(f"Final: {latest['final_signal']} | Confidence: {latest['confidence']:.2f}")
            logger.info(f"Actual result: {latest['actual_result'] or 'Waiting...'}")
            logger.info("=" * 50)
    
    
    def tune_layers_for_ticker(self, ticker: str, performance_data: Dict[str, float]):
        """Автоматичnot тюнandнгування шарandв на основand реwithульandтandв моwhereлей
        
        Args:
            ticker: Тandкер for тюнandнгу
            performance_data: Данand про продуктивнandсть {layer_name: performance_score}
        """
        logger.info(f"[Advisor] Початок тюнandнгу шарandв for {ticker}")
        
        # Поки що логandка просand - в майбутньому can роwithширити
        for layer_name, score in performance_data.items():
            if score > 0.7:  # Високий скор - пandдсилюємо
                new_weight = min(1.5, 1.0 + (score - 0.7) * 2)
            elif score < 0.3:  # Ниwithький скор - ослаблюємо
                new_weight = max(0.5, 1.0 - (0.3 - score) * 2)
            else:  # Середнandй скор - forлишаємо notйтрально
                new_weight = 1.0
            
            if new_weight != 1.0:
                self.update_layer_weight(layer_name, new_weight)
        
        logger.info(f"[Advisor] Тюнandнг шарandв for {ticker} forвершено")
    
    def update_layer_performance(self, layer_name: str, performance_metrics: Dict[str, float]):
        """Update layer performance for context learning"""
        self.context_enricher.update_layer_performance(layer_name, performance_metrics)
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get complete context summary"""
        return self.context_enricher.get_context_summary()
    
    def export_context_data(self, filepath: str):
        """Export context data for analysis"""
        self.context_enricher.export_context_data(filepath)