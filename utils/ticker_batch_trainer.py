# ticker_batch_trainer.py - Пакетне тренування по тікерах

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class TickerBatchTrainer:
    """
    Пакетне тренування по тікерах
    Батч = 1 тікер + всі таргети + всі таймфрейми
    """
    
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "ticker_batch_results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def create_ticker_batches(self, features_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Створює батчі по тікерах
        
        Args:
            features_df: DataFrame з фічами та таргетами
            
        Returns:
            Dict: {ticker: batch_df}
        """
        logger.info(f"[TickerBatch] Створення батчів з {features_df.shape}")
        
        if features_df.empty:
            logger.warning("[TickerBatch] DataFrame порожній")
            return {}
        
        # Визначаємо колонку тікера
        ticker_col = None
        for col in ['ticker', 'symbol', 'asset']:
            if col in features_df.columns:
                ticker_col = col
                break
        
        if ticker_col is None:
            logger.warning("[TickerBatch] Не found колонку тікера")
            return {}
        
        # Створюємо батчі
        ticker_batches = {}
        tickers = features_df[ticker_col].unique()
        
        logger.info(f"[TickerBatch] Знайдено тікерів: {len(tickers)}")
        
        for ticker in tickers:
            ticker_df = features_df[features_df[ticker_col] == ticker].copy()
            
            if not ticker_df.empty:
                ticker_batches[ticker] = ticker_df
                logger.info(f"[TickerBatch] Батч {ticker}: {ticker_df.shape}")
        
        logger.info(f"[TickerBatch] Створено {len(ticker_batches)} батчів")
        return ticker_batches
    
    def prepare_batch_for_training(self, batch_df: pd.DataFrame, 
                                 ticker: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Готує батч для тренування
        
        Args:
            batch_df: DataFrame для одного тікера
            ticker: Назва тікера
            
        Returns:
            Tuple: (X, y, feature_columns)
        """
        logger.info(f"[TickerBatch] Підготовка батчу {ticker}: {batch_df.shape}")
        
        # Визначаємо фічі та таргети
        feature_cols = []
        target_cols = []
        
        for col in batch_df.columns:
            col_lower = col.lower()
            
            # Таргети
            if any(keyword in col_lower for keyword in ['target', 'label', 'y_']):
                target_cols.append(col)
            # Фічі (окрім службових)
            elif col not in ['date', 'timestamp', 'trade_date', 'ticker', 'symbol', 'asset']:
                feature_cols.append(col)
        
        logger.info(f"[TickerBatch] {ticker}: {len(feature_cols)} фіч, {len(target_cols)} таргетів")
        
        if len(target_cols) == 0:
            logger.warning(f"[TickerBatch] {ticker}: Немає таргетів")
            return pd.DataFrame(), pd.DataFrame(), []
        
        # Готуємо дані
        X = batch_df[feature_cols].copy()
        y = batch_df[target_cols].copy()
        
        # Очищення data
        X = X.fillna(0)
        y = y.fillna(0)
        
        # Видаляємо рядки з NaN
        valid_mask = ~(X.isnull().any(axis=1) | y.isnull().any(axis=1))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"[TickerBatch] {ticker}: Підготовлено {len(X)} рядків")
        
        return X, y, feature_cols
    
    def train_ticker_batch(self, ticker: str, X: pd.DataFrame, y: pd.DataFrame, 
                          feature_cols: List[str], models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Тренує моделі для одного тікера
        
        Args:
            ticker: Назва тікера
            X: Фічі
            y: Таргети
            feature_cols: Список фіч
            models_to_train: Список моделей для тренування
            
        Returns:
            Dict: Результати тренування
        """
        if models_to_train is None:
            models_to_train = ['random_forest', 'xgboost', 'lightgbm']
        
        logger.info(f"[TickerBatch] Тренування {ticker}: {models_to_train}")
        
        results = {
            'ticker': ticker,
            'data_shape': X.shape,
            'feature_count': len(feature_cols),
            'target_count': y.shape[1],
            'models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, classification_report
            
            # Розділення data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Тренуємо кожну модель
            for model_name in models_to_train:
                logger.info(f"[TickerBatch] Тренування {ticker} - {model_name}")
                
                try:
                    if model_name == 'random_forest':
                        model = RandomForestClassifier(n_estimators=100, random_state=42)
                        model.fit(X_train, y_train.iloc[:, 0])  # Перший таргет
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test.iloc[:, 0], predictions)
                        
                        results['models'][model_name] = {
                            'accuracy': accuracy,
                            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                            'status': 'success'
                        }
                        
                    elif model_name == 'xgboost':
                        import xgboost as xgb
                        model = xgb.XGBClassifier(random_state=42)
                        model.fit(X_train, y_train.iloc[:, 0])
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test.iloc[:, 0], predictions)
                        
                        results['models'][model_name] = {
                            'accuracy': accuracy,
                            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                            'status': 'success'
                        }
                        
                    elif model_name == 'lightgbm':
                        import lightgbm as lgb
                        model = lgb.LGBMClassifier(random_state=42)
                        model.fit(X_train, y_train.iloc[:, 0])
                        predictions = model.predict(X_test)
                        accuracy = accuracy_score(y_test.iloc[:, 0], predictions)
                        
                        results['models'][model_name] = {
                            'accuracy': accuracy,
                            'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                            'status': 'success'
                        }
                    
                    logger.info(f"[TickerBatch] {ticker} - {model_name}: accuracy={accuracy:.4f}")
                    
                except Exception as e:
                    logger.error(f"[TickerBatch] {ticker} - {model_name}: error {e}")
                    results['models'][model_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }
            
        except Exception as e:
            logger.error(f"[TickerBatch] {ticker}: error тренування {e}")
            results['error'] = str(e)
        
        return results
    
    def train_all_ticker_batches(self, features_df: pd.DataFrame, 
                               models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Тренує всі батчі по тікерах
        
        Args:
            features_df: DataFrame з фічами
            models_to_train: Список моделей
            
        Returns:
            Dict: Загальні результати
        """
        logger.info(f"[TickerBatch] Початок тренування всіх батчів: {features_df.shape}")
        
        # Створюємо батчі
        ticker_batches = self.create_ticker_batches(features_df)
        
        if not ticker_batches:
            logger.error("[TickerBatch] Не вдалося створити батчі")
            return {}
        
        # Тренуємо кожен батч
        all_results = {
            'summary': {
                'total_tickers': len(ticker_batches),
                'models_trained': models_to_train or ['random_forest', 'xgboost', 'lightgbm'],
                'timestamp': datetime.now().isoformat()
            },
            'ticker_results': {}
        }
        
        for ticker, batch_df in ticker_batches.items():
            logger.info(f"[TickerBatch] Обробка тікера {ticker}")
            
            # Готуємо дані
            X, y, feature_cols = self.prepare_batch_for_training(batch_df, ticker)
            
            if X.empty or y.empty:
                logger.warning(f"[TickerBatch] Пропускаємо {ticker} - порожні дані")
                continue
            
            # Тренуємо моделі
            ticker_results = self.train_ticker_batch(ticker, X, y, feature_cols, models_to_train)
            all_results['ticker_results'][ticker] = ticker_results
        
        # Зберігаємо результати
        self.save_batch_results(all_results)
        
        # Створюємо звіт
        self.create_batch_report(all_results)
        
        logger.info(f"[TickerBatch] Завершено тренування {len(all_results['ticker_results'])} тікерів")
        return all_results
    
    def save_batch_results(self, results: Dict[str, Any]):
        """Зберігає результати тренування"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Зберігаємо JSON
        results_file = self.results_dir / f"ticker_batch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"[TickerBatch] Результати saved: {results_file}")
    
    def create_batch_report(self, results: Dict[str, Any]):
        """Створює звіт по тренуванню"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.results_dir / f"ticker_batch_report_{timestamp}.md"
        
        # Аналізуємо результати
        successful_tickers = []
        failed_tickers = []
        model_accuracies = {model: [] for model in results['summary']['models_trained']}
        
        for ticker, ticker_result in results['ticker_results'].items():
            if 'error' in ticker_result:
                failed_tickers.append(ticker)
            else:
                successful_tickers.append(ticker)
                for model_name, model_result in ticker_result['models'].items():
                    if model_result.get('status') == 'success':
                        model_accuracies[model_name].append(model_result['accuracy'])
        
        # Створюємо звіт
        report_content = f"""# Ticker Batch Training Report
        
## Summary
- **Total Tickers**: {results['summary']['total_tickers']}
- **Successful**: {len(successful_tickers)}
- **Failed**: {len(failed_tickers)}
- **Models Trained**: {', '.join(results['summary']['models_trained'])}
- **Timestamp**: {results['summary']['timestamp']}

## Model Performance
"""
        
        for model_name, accuracies in model_accuracies.items():
            if accuracies:
                avg_acc = np.mean(accuracies)
                max_acc = np.max(accuracies)
                min_acc = np.min(accuracies)
                report_content += f"""
### {model_name.title()}
- Average Accuracy: {avg_acc:.4f}
- Best Accuracy: {max_acc:.4f}
- Worst Accuracy: {min_acc:.4f}
- Models Trained: {len(accuracies)}
"""
        
        report_content += f"""
## Successful Tickers ({len(successful_tickers)})
{', '.join(successful_tickers[:10])}{'...' if len(successful_tickers) > 10 else ''}

## Failed Tickers ({len(failed_tickers)})
{', '.join(failed_tickers)}
"""
        
        # Зберігаємо звіт
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"[TickerBatch] Звіт created: {report_file}")

# Глобальний екземпляр
ticker_batch_trainer = TickerBatchTrainer()
