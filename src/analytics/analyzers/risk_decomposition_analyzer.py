"""
Risk Decomposition Analyzer - Аналізатор декомпозиції ризику.

Виконує декомпозицію ризику портфеля на компоненти:
- Systematic risk (ринковий ризик)
- Idiosyncratic risk (специфічний ризик)
- Factor risk (ризик факторів)
- Liquidity risk (ризик ліквідності)
- Concentration risk (ризик концентрації)

Використовує:
- Risk factor models (Fama-French, etc.)
- Principal Component Analysis
- Risk attribution techniques
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats

from ..interfaces import IAnalyzer
from src.core.logging.logger import ProjectLogger

class RiskDecompositionAnalyzer(IAnalyzer):
    """
    Аналізатор декомпозиції ризику портфеля.

    Розкладає загальний ризик на компоненти для кращого розуміння
    джерел волатильності та прийняття рішень.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = ProjectLogger.get_logger("RiskDecompositionAnalyzer")
        self.config = config or {}

        # Параметри аналізу
        self.use_pca = self.config.get('use_pca', True)
        self.n_factors = self.config.get('n_factors', 5)
        self.factor_model = self.config.get('factor_model', 'fama_french')  # 'fama_french', 'pca', 'custom'

        # Ризик метрики
        self.risk_metrics = ['volatility', 'var_95', 'cvar_95', 'max_drawdown', 'sharpe_ratio']

        # Фактори ризику (якщо використовуємо custom)
        self.custom_factors = self.config.get('custom_factors', [])

    def analyze(self, data: Dict[str, pd.DataFrame], **kwargs) -> Dict[str, Any]:
        """
        Виконує декомпозицію ризику портфеля.

        Args:
            data: Словник з даними портфеля та факторів
                - 'portfolio_returns': pd.DataFrame з поверненнями активів
                - 'factor_returns': pd.DataFrame з факторними поверненнями (опціонально)
                - 'weights': Dict з вагами активів (опціонально)

        Returns:
            Dict з результатами декомпозиції ризику
        """
        try:
            # Отримання даних
            portfolio_returns = data.get('portfolio_returns')
            factor_returns = data.get('factor_returns')
            weights = data.get('weights', {})

            if portfolio_returns is None or portfolio_returns.empty:
                return {"error": "portfolio_returns is required and cannot be empty"}

            # Розрахунок загального ризику портфеля
            portfolio_risk = self._calculate_portfolio_risk(portfolio_returns, weights)

            # Декомпозиція на систематичний та ідіосинкратичний ризик
            systematic_risk, idiosyncratic_risk = self._decompose_systematic_idiosyncratic(
                portfolio_returns, factor_returns
            )

            # Факторна декомпозиція
            factor_decomposition = self._factor_risk_decomposition(
                portfolio_returns, factor_returns
            )

            # Концентраційний ризик
            concentration_risk = self._calculate_concentration_risk(portfolio_returns, weights)

            # Ліквідність ризик
            liquidity_risk = self._calculate_liquidity_risk(portfolio_returns)

            # Агрегація результатів
            result = {
                'portfolio_risk': portfolio_risk,
                'systematic_risk': systematic_risk,
                'idiosyncratic_risk': idiosyncratic_risk,
                'factor_decomposition': factor_decomposition,
                'concentration_risk': concentration_risk,
                'liquidity_risk': liquidity_risk,
                'risk_attribution': self._calculate_risk_attribution(
                    systematic_risk, idiosyncratic_risk, factor_decomposition,
                    concentration_risk, liquidity_risk
                ),
                'recommendations': self._generate_recommendations(
                    systematic_risk, idiosyncratic_risk, factor_decomposition,
                    concentration_risk, liquidity_risk
                )
            }

            self.logger.info("Risk decomposition analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error in risk decomposition analysis: {e}")
            return {"error": str(e)}

    def _calculate_portfolio_risk(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, float]:
        """Розрахунок основних метрик ризику портфеля"""
        try:
            # Якщо ваги не задані, використовуємо рівні ваги
            if not weights:
                n_assets = len(returns.columns)
                weights = {col: 1.0 / n_assets for col in returns.columns}

            # Перетворення у numpy arrays
            returns_array = returns.values
            weights_array = np.array([weights.get(col, 0) for col in returns.columns])

            # Портфельні повернення
            portfolio_returns = returns_array @ weights_array

            # Метрики ризику
            volatility = np.std(portfolio_returns) * np.sqrt(252)  # Annualized
            var_95 = np.percentile(portfolio_returns, 5)  # VaR 95%
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean()  # CVaR 95%

            # Maximum drawdown
            cumulative = np.cumprod(1 + portfolio_returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            # Sharpe ratio (припускаємо безризикову ставку 0.02)
            excess_returns = portfolio_returns - 0.02/252
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)

            return {
                'volatility': float(volatility),
                'var_95': float(var_95),
                'cvar_95': float(cvar_95),
                'max_drawdown': float(max_drawdown),
                'sharpe_ratio': float(sharpe_ratio)
            }

        except Exception as e:
            self.logger.warning(f"Error calculating portfolio risk: {e}")
            return {}

    def _decompose_systematic_idiosyncratic(self, returns: pd.DataFrame,
                                          factor_returns: Optional[pd.DataFrame]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Декомпозиція на систематичний та ідіосинкратичний ризик"""
        try:
            systematic_risk = {}
            idiosyncratic_risk = {}

            for asset in returns.columns:
                asset_returns = returns[asset].dropna()

                if factor_returns is not None and not factor_returns.empty:
                    # Використання факторної моделі
                    common_index = asset_returns.index.intersection(factor_returns.index)
                    if len(common_index) < 30:
                        # Недостатньо даних для факторної моделі
                        systematic_risk[asset] = 0.7 * asset_returns.std()
                        idiosyncratic_risk[asset] = 0.3 * asset_returns.std()
                        continue

                    X = factor_returns.loc[common_index].values
                    y = asset_returns.loc[common_index].values

                    # Регресія на фактори
                    reg = LinearRegression()
                    reg.fit(X, y)

                    # Систематичний ризик
                    systematic_var = reg.predict(X).var()
                    systematic_risk[asset] = np.sqrt(systematic_var)

                    # Ідіосинкратичний ризик
                    residuals = y - reg.predict(X)
                    idiosyncratic_risk[asset] = np.sqrt(residuals.var())

                else:
                    # Спрощена декомпозиція (70% систематичний, 30% ідіосинкратичний)
                    total_vol = asset_returns.std()
                    systematic_risk[asset] = 0.7 * total_vol
                    idiosyncratic_risk[asset] = 0.3 * total_vol

            return systematic_risk, idiosyncratic_risk

        except Exception as e:
            self.logger.warning(f"Error in systematic/idiosyncratic decomposition: {e}")
            return {}, {}

    def _factor_risk_decomposition(self, returns: pd.DataFrame,
                                 factor_returns: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Факторна декомпозиція ризику"""
        try:
            if factor_returns is None or factor_returns.empty:
                if self.use_pca:
                    # Використання PCA для створення факторів
                    returns_clean = returns.dropna()
                    pca = PCA(n_components=min(self.n_factors, len(returns_clean.columns)))
                    pca.fit(returns_clean.T)  # Транспонуємо для аналізу активів

                    factor_loadings = pca.components_
                    explained_variance = pca.explained_variance_ratio_

                    return {
                        'method': 'pca',
                        'factor_loadings': factor_loadings.tolist(),
                        'explained_variance': explained_variance.tolist(),
                        'n_factors': len(explained_variance)
                    }
                else:
                    return {'method': 'unavailable', 'reason': 'no factor data and PCA disabled'}

            # Аналіз факторних навантажень
            factor_loadings = {}
            factor_contributions = {}

            for asset in returns.columns:
                asset_returns = returns[asset].dropna()
                common_index = asset_returns.index.intersection(factor_returns.index)

                if len(common_index) >= 30:
                    X = factor_returns.loc[common_index].values
                    y = asset_returns.loc[common_index].values

                    reg = LinearRegression()
                    reg.fit(X, y)

                    factor_loadings[asset] = reg.coef_.tolist()
                    factor_contributions[asset] = (reg.coef_ ** 2).tolist()

            return {
                'method': 'factor_model',
                'factor_loadings': factor_loadings,
                'factor_contributions': factor_contributions,
                'factor_names': factor_returns.columns.tolist()
            }

        except Exception as e:
            self.logger.warning(f"Error in factor risk decomposition: {e}")
            return {'method': 'error', 'error': str(e)}

    def _calculate_concentration_risk(self, returns: pd.DataFrame, weights: Dict[str, float]) -> Dict[str, Any]:
        """Розрахунок ризику концентрації"""
        try:
            if not weights:
                n_assets = len(returns.columns)
                weights = {col: 1.0 / n_assets for col in returns.columns}

            weights_array = np.array([weights.get(col, 0) for col in returns.columns])

            # Herfindahl-Hirschman Index (HHI)
            hhi = np.sum(weights_array ** 2)

            # Effective number of assets
            effective_n = 1.0 / hhi

            # Concentration ratio (top 3 assets)
            sorted_weights = np.sort(weights_array)[::-1]
            concentration_ratio = np.sum(sorted_weights[:3])

            # Gini coefficient для ваг
            sorted_weights = np.sort(weights_array)
            n = len(sorted_weights)
            cumsum = np.cumsum(sorted_weights)
            gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

            return {
                'hhi': float(hhi),
                'effective_n_assets': float(effective_n),
                'concentration_ratio_top3': float(concentration_ratio),
                'gini_coefficient': float(gini),
                'is_concentrated': hhi > 0.25  # HHI > 0.25 вважається високою концентрацією
            }

        except Exception as e:
            self.logger.warning(f"Error calculating concentration risk: {e}")
            return {}

    def _calculate_liquidity_risk(self, returns: pd.DataFrame) -> Dict[str, Any]:
        """Розрахунок ризику ліквідності"""
        try:
            # Amihud illiquidity measure (спрощена версія)
            # У реальному випадку потрібні дані про обсяг та ціну
            liquidity_risk = {}

            for asset in returns.columns:
                asset_returns = returns[asset].dropna()

                # Спрощена метрика: волатильність як proxy для ліквідності
                # (менш ліквідні активи мають вищу волатильність)
                vol = asset_returns.std()

                # Turnover ratio (спрощений, без реальних даних)
                # Припускаємо середній turnover
                avg_turnover = 0.1  # Placeholder

                # Liquidity risk score
                liquidity_score = vol / (avg_turnover + 0.01)  # Вищий score = вищий ризик

                liquidity_risk[asset] = {
                    'volatility': float(vol),
                    'liquidity_score': float(liquidity_score),
                    'is_illiquid': liquidity_score > 1.0
                }

            # Портфельний ліквідність ризик
            portfolio_liquidity = np.mean([v['liquidity_score'] for v in liquidity_risk.values()])

            return {
                'asset_liquidity': liquidity_risk,
                'portfolio_liquidity_risk': float(portfolio_liquidity),
                'illiquid_assets_count': sum(1 for v in liquidity_risk.values() if v['is_illiquid'])
            }

        except Exception as e:
            self.logger.warning(f"Error calculating liquidity risk: {e}")
            return {}

    def _calculate_risk_attribution(self, systematic_risk: Dict[str, float],
                                  idiosyncratic_risk: Dict[str, float],
                                  factor_decomposition: Dict[str, Any],
                                  concentration_risk: Dict[str, Any],
                                  liquidity_risk: Dict[str, Any]) -> Dict[str, Any]:
        """Розрахунок атрибуції ризику"""
        try:
            # Агрегація ризиків по активах
            total_systematic = np.mean(list(systematic_risk.values()))
            total_idiosyncratic = np.mean(list(idiosyncratic_risk.values()))

            # Відносні вклади
            total_risk = total_systematic + total_idiosyncratic
            if total_risk > 0:
                systematic_pct = total_systematic / total_risk
                idiosyncratic_pct = total_idiosyncratic / total_risk
            else:
                systematic_pct = 0.5
                idiosyncratic_pct = 0.5

            return {
                'systematic_risk_contribution': float(systematic_pct),
                'idiosyncratic_risk_contribution': float(idiosyncratic_pct),
                'concentration_risk_impact': concentration_risk.get('hhi', 0),
                'liquidity_risk_impact': liquidity_risk.get('portfolio_liquidity_risk', 0),
                'diversification_benefit': 1.0 - concentration_risk.get('hhi', 1.0)
            }

        except Exception as e:
            self.logger.warning(f"Error in risk attribution: {e}")
            return {}

    def _generate_recommendations(self, systematic_risk: Dict[str, float],
                                idiosyncratic_risk: Dict[str, float],
                                factor_decomposition: Dict[str, Any],
                                concentration_risk: Dict[str, Any],
                                liquidity_risk: Dict[str, Any]) -> List[str]:
        """Генерація рекомендацій на основі аналізу ризику"""
        recommendations = []

        # Концентрація
        if concentration_risk.get('hhi', 0) > 0.25:
            recommendations.append("Висока концентрація портфеля. Рекомендується диверсифікація.")

        if concentration_risk.get('effective_n_assets', 10) < 5:
            recommendations.append("Низька ефективна кількість активів. Додайте більше різноманітних активів.")

        # Систематичний vs ідіосинкратичний ризик
        systematic_pct = np.mean(list(systematic_risk.values()))
        idiosyncratic_pct = np.mean(list(idiosyncratic_risk.values()))

        if systematic_pct > 0.8:
            recommendations.append("Переважний систематичний ризик. Розгляньте хеджування ринкового ризику.")
        elif idiosyncratic_pct > 0.8:
            recommendations.append("Переважний ідіосинкратичний ризик. Можна зменшити через диверсифікацію.")

        # Ліквідність
        if liquidity_risk.get('portfolio_liquidity_risk', 0) > 1.0:
            recommendations.append("Високий ризик ліквідності. Перегляньте позиції в менш ліквідних активах.")

        illiquid_count = liquidity_risk.get('illiquid_assets_count', 0)
        if illiquid_count > 0:
            recommendations.append(f"{illiquid_count} активів мають низьку ліквідність. Моніторте ці позиції.")

        # Факторний ризик
        if factor_decomposition.get('method') == 'pca':
            explained_var = sum(factor_decomposition.get('explained_variance', []))
            if explained_var < 0.5:
                recommendations.append("Низьке пояснення варіації факторами. Можливо потрібні додаткові фактори ризику.")

        if not recommendations:
            recommendations.append("Ризик портфеля добре збалансований. Продовжуйте моніторинг.")

        return recommendations