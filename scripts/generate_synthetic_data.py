#!/usr/bin/env python3
"""
Synthetic Data Generation Script
Генерування синтетичних даних для тренування моделей

Types:
1. Typical Scenarios (Monte Carlo) - типові ринкові умови
2. Shock Scenarios (MonsterTest) - ринкові шоки
3. Context Scenarios (DEAN) - різні ринкові режими
"""

import sys
import os
import logging
import json
from datetime import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# Create numpy random generator
rng = np.random.default_rng(42)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger
from src.data.management.data_manager import DataManager
from src.simulation.simulation_engine import SimulationEngine, SimulationContext
from src.main.modes.monster_test import MonsterTestMode
from src.models.dean.dean_bootstrap_system import DeanBootstrapSystem, ModelRole

logger = ProjectLogger.get_logger("SyntheticDataGeneration")

class SyntheticDataGenerator:
    """Генератор синтетичних даних"""
    
    def __init__(self, config_manager: UnifiedConfigManager):
        self.config_manager = config_manager
        self.data_manager = DataManager(config_manager)
        self.simulation_engine = SimulationEngine()
        self.dean_system = DeanBootstrapSystem()
        self.results = {
            'typical_scenarios': [],
            'shock_scenarios': [],
            'context_scenarios': []
        }
    
    def run(self, scenario_types: List[str] = None):
        """
        Запустити генерування синтетичних даних
        
        Args:
            scenario_types: Типи сценаріїв для генерування
                           ['typical', 'shock', 'context'] або None для всіх
        """
        logger.info("=" * 80)
        logger.info("🎲 SYNTHETIC DATA GENERATION - Starting")
        logger.info("=" * 80)
        
        scenario_types = scenario_types or ['typical', 'shock', 'context']
        
        try:
            if 'typical' in scenario_types:
                logger.info("\n[Type 1] Generating Typical Scenarios (Monte Carlo)")
                self._generate_typical_scenarios()
            
            if 'shock' in scenario_types:
                logger.info("\n[Type 2] Generating Shock Scenarios (MonsterTest)")
                self._generate_shock_scenarios()
            
            if 'context' in scenario_types:
                logger.info("\n[Type 3] Generating Context Scenarios (DEAN)")
                self._generate_context_scenarios()
            
            # Save results
            logger.info("\n[Storage] Saving synthetic data")
            self._save_results()
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ SYNTHETIC DATA GENERATION - Completed Successfully")
            logger.info("=" * 80)
            
            return {
                'status': 'success',
                'typical_scenarios': len(self.results['typical_scenarios']),
                'shock_scenarios': len(self.results['shock_scenarios']),
                'context_scenarios': len(self.results['context_scenarios']),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.exception(f"❌ Error during synthetic data generation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_typical_scenarios(self):
        """Type 1: Typical Scenarios (Monte Carlo)"""
        logger.info("Generating typical market scenarios using Monte Carlo...")
        
        try:
            # Get historical data for context
            enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
            if enriched_df is None or enriched_df.empty:
                logger.warning("⚠️  No enriched data found. Using synthetic baseline.")
                enriched_df = self._create_baseline_data()
            
            # Extract returns for Monte Carlo
            if 'close' in enriched_df.columns:
                returns = enriched_df['close'].pct_change().dropna()
            else:
                returns = pd.Series(rng.normal(0.0005, 0.02, 1000))
            
            # Run Monte Carlo simulations
            n_simulations = self.config_manager.get_config('simulation', {}).get('defaults', {}).get('monte_carlo_runs', 100)
            logger.info(f"Running {n_simulations} Monte Carlo simulations...")
            
            for i in range(n_simulations):
                # Generate price path
                price_path = self._generate_price_path(returns, horizon=252)  # 1 year
                
                # Calculate metrics
                metrics = self._calculate_path_metrics(price_path)
                
                scenario = {
                    'scenario_id': f'typical_{i:04d}',
                    'type': 'monte_carlo',
                    'price_path': price_path.tolist(),
                    'returns': np.diff(price_path) / price_path[:-1],
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results['typical_scenarios'].append(scenario)
                
                if (i + 1) % 20 == 0:
                    logger.info(f"  ✓ Generated {i + 1}/{n_simulations} scenarios")
            
            logger.info(f"✓ Generated {len(self.results['typical_scenarios'])} typical scenarios")
            
        except Exception as e:
            logger.error(f"❌ Typical scenario generation failed: {e}")
    
    def _generate_shock_scenarios(self):
        """Type 2: Shock Scenarios (MonsterTest)"""
        logger.info("Generating shock scenarios...")
        
        try:
            # Define shock types
            shock_types = {
                'flash_crash': {'magnitude': -0.10, 'duration': 1},
                'volatility_spike': {'magnitude': 0.50, 'duration': 5},
                'liquidity_crisis': {'magnitude': -0.05, 'duration': 10},
                'black_swan': {'magnitude': -0.20, 'duration': 1},
                'circuit_breaker': {'magnitude': -0.07, 'duration': 15}
            }
            
            # Get baseline data
            enriched_df = self.data_manager.fetch_data_from_table('enriched_features')
            if enriched_df is None or enriched_df.empty:
                enriched_df = self._create_baseline_data()
            
            baseline_price = enriched_df['close'].iloc[-1] if 'close' in enriched_df.columns else 100
            
            # Generate shock scenarios
            for shock_name, shock_params in shock_types.items():
                logger.info(f"  Generating {shock_name} scenario...")
                
                # Create shocked price path
                price_path = self._generate_shocked_price_path(
                    baseline_price,
                    shock_params['magnitude'],
                    shock_params['duration'],
                    horizon=252
                )
                
                # Calculate impact metrics
                metrics = self._calculate_shock_metrics(price_path, shock_params)
                
                scenario = {
                    'scenario_id': f'shock_{shock_name}',
                    'type': 'shock',
                    'shock_type': shock_name,
                    'shock_magnitude': shock_params['magnitude'],
                    'shock_duration': shock_params['duration'],
                    'price_path': price_path.tolist(),
                    'metrics': metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results['shock_scenarios'].append(scenario)
            
            logger.info(f"✓ Generated {len(self.results['shock_scenarios'])} shock scenarios")
            
        except Exception as e:
            logger.error(f"❌ Shock scenario generation failed: {e}")
    
    def _generate_context_scenarios(self):
        """Type 3: Context Scenarios (DEAN)"""
        logger.info("Generating context scenarios using DEAN...")
        
        try:
            # Define market regimes
            regimes = {
                'trending_up': {
                    'trend': 0.001,
                    'volatility': 0.015,
                    'description': 'Восходящий тренд з низькою волатильністю'
                },
                'trending_down': {
                    'trend': -0.001,
                    'volatility': 0.020,
                    'description': 'Нисходящий тренд з підвищеною волатильністю'
                },
                'ranging': {
                    'trend': 0.0,
                    'volatility': 0.010,
                    'description': 'Бічний рух з низькою волатильністю'
                },
                'volatile': {
                    'trend': 0.0,
                    'volatility': 0.040,
                    'description': 'Висока волатильність без чіткого тренду'
                },
                'crisis': {
                    'trend': -0.002,
                    'volatility': 0.060,
                    'description': 'Паніка з різким падінням'
                }
            }
            
            # Generate scenario for each regime
            for regime_name, regime_params in regimes.items():
                logger.info(f"  Generating {regime_name} scenario...")
                
                # Generate price path for this regime
                price_path = self._generate_regime_price_path(
                    regime_params['trend'],
                    regime_params['volatility'],
                    horizon=252
                )
                
                # Simulate DEAN bootstrap
                dean_simulation = self._simulate_dean_bootstrap(regime_name)
                
                scenario = {
                    'scenario_id': f'context_{regime_name}',
                    'type': 'context',
                    'regime': regime_name,
                    'regime_description': regime_params['description'],
                    'regime_characteristics': {
                        'trend': regime_params['trend'],
                        'volatility': regime_params['volatility']
                    },
                    'price_path': price_path.tolist(),
                    'dean_simulation': dean_simulation,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.results['context_scenarios'].append(scenario)
            
            logger.info(f"✓ Generated {len(self.results['context_scenarios'])} context scenarios")
            
        except Exception as e:
            logger.error(f"❌ Context scenario generation failed: {e}")
    
    def _generate_price_path(self, returns: pd.Series, horizon: int) -> np.ndarray:
        """Generate synthetic price path using historical returns"""
        prices = [100.0]
        for _ in range(horizon):
            ret = rng.choice(returns.values)
            prices.append(prices[-1] * (1 + ret))
        return np.array(prices)
    
    def _generate_shocked_price_path(self, baseline: float, shock_mag: float, shock_dur: int, horizon: int) -> np.ndarray:
        """Generate price path with shock"""
        prices = [baseline]
        
        for i in range(horizon):
            if i < shock_dur:
                # Apply shock
                ret = shock_mag / shock_dur + rng.normal(0, 0.01)
            else:
                # Recovery
                ret = rng.normal(0.0005, 0.02)
            
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices)
    
    def _generate_regime_price_path(self, trend: float, volatility: float, horizon: int) -> np.ndarray:
        """Generate price path for specific market regime"""
        prices = [100.0]
        
        for _ in range(horizon):
            ret = rng.normal(trend, volatility)
            prices.append(prices[-1] * (1 + ret))
        
        return np.array(prices)
    
    def _calculate_path_metrics(self, price_path: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for price path"""
        returns = np.diff(price_path) / price_path[:-1]
        
        return {
            'total_return': (price_path[-1] - price_path[0]) / price_path[0],
            'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(price_path),
            'volatility': np.std(returns) * np.sqrt(252),
            'final_price': float(price_path[-1])
        }
    
    def _calculate_shock_metrics(self, price_path: np.ndarray, shock_params: Dict) -> Dict[str, float]:
        """Calculate metrics for shocked price path"""
        metrics = self._calculate_path_metrics(price_path)
        
        # Add shock-specific metrics
        shock_impact = (price_path[shock_params['duration']] - price_path[0]) / price_path[0]
        recovery_time = self._calculate_recovery_time(price_path, shock_params['duration'])
        
        metrics['shock_impact'] = shock_impact
        metrics['recovery_time'] = recovery_time
        
        return metrics
    
    def _calculate_max_drawdown(self, price_path: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cummax = np.maximum.accumulate(price_path)
        drawdown = (price_path - cummax) / cummax
        return float(np.min(drawdown))
    
    def _calculate_recovery_time(self, price_path: np.ndarray, shock_end: int) -> int:
        """Calculate time to recover from shock"""
        shock_price = price_path[shock_end]
        for i in range(shock_end, len(price_path)):
            if price_path[i] >= shock_price:
                return i - shock_end
        return len(price_path) - shock_end
    
    def _simulate_dean_bootstrap(self, regime_name: str) -> Dict[str, Any]:
        """Simulate DEAN bootstrap for regime"""
        return {
            'regime': regime_name,
            'actor_actions': [
                {'action_type': 'buy', 'confidence': 0.7},
                {'action_type': 'hold', 'confidence': 0.8},
                {'action_type': 'sell', 'confidence': 0.6}
            ],
            'critic_feedback': [
                {'critique_score': 0.5, 'points': ['Good entry point', 'Adequate risk management']},
                {'critique_score': 0.3, 'points': ['Timing could be better']},
                {'critique_score': -0.2, 'points': ['High risk in this regime']}
            ],
            'simulation_steps': 5
        }
    
    def _create_baseline_data(self) -> pd.DataFrame:
        """Create baseline synthetic data if no real data exists"""
        logger.info("Creating baseline synthetic data...")
        
        dates = pd.date_range(end=datetime.now(), periods=252, freq='D')
        prices = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, 252)))
        
        return pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + rng.normal(0, 0.01, 252)),
            'high': prices * (1 + np.abs(rng.normal(0, 0.02, 252))),
            'low': prices * (1 - np.abs(rng.normal(0, 0.02, 252))),
            'close': prices,
            'volume': rng.integers(1000000, 10000000, 252)
        })
    
    def _save_results(self):
        """Save synthetic data to files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save typical scenarios
        if self.results['typical_scenarios']:
            filename = f'results/synthetic_typical_scenarios_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(self.results['typical_scenarios'], f, indent=2, default=str)
            logger.info(f"✓ Saved typical scenarios to {filename}")
        
        # Save shock scenarios
        if self.results['shock_scenarios']:
            filename = f'results/synthetic_shock_scenarios_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(self.results['shock_scenarios'], f, indent=2, default=str)
            logger.info(f"✓ Saved shock scenarios to {filename}")
        
        # Save context scenarios
        if self.results['context_scenarios']:
            filename = f'results/synthetic_context_scenarios_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(self.results['context_scenarios'], f, indent=2, default=str)
            logger.info(f"✓ Saved context scenarios to {filename}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic data for training')
    parser.add_argument('--types', nargs='+', choices=['typical', 'shock', 'context'],
                       default=['typical', 'shock', 'context'],
                       help='Types of scenarios to generate')
    parser.add_argument('--config-path', default='src/config', help='Path to config directory')
    
    args = parser.parse_args()
    
    # Initialize config
    config_manager = UnifiedConfigManager(config_dir=args.config_path)
    
    # Run generation
    generator = SyntheticDataGenerator(config_manager)
    result = generator.run(scenario_types=args.types)
    
    # Print result
    logger.info("\n" + "=" * 80)
    logger.info("RESULT:")
    logger.info(json.dumps(result, indent=2, default=str))
    logger.info("=" * 80)
    
    return 0 if result['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
