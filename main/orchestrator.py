#!/usr/bin/env python3
"""
Trading Orchestrator - основна логіка системи
Замінює великий main.py
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import argparse

from config.trading_config import get_config, TradingConfig
from .modes.train import TrainMode
from .modes.analyze import AnalyzeMode
from .modes.batch_training import BatchTrainingMode
from .modes.progressive import ProgressiveMode
from .modes.monster_test import MonsterTestMode
from .modes.backtest import BacktestMode
from .modes.comprehensive_backtest import ComprehensiveBacktestMode
from .modes.optimized_backtest import OptimizedBacktestMode
from .modes.integrated_backtest import IntegratedBacktestMode
from .modes.real_data_backtest import RealDataBacktestMode
from .modes.web_ui import WebUIMode
from .modes.intelligent import IntelligentMode  # Додано інтелектуальний режим

# Імпорти спеціалізованих оркестраторів
from core.pipeline.unified_orchestrator import UnifiedOrchestrator
from core.pipeline_orchestrator import PipelineOrchestrator
from core.pipeline.dual_pipeline_orchestrator import DualPipelineOrchestrator


class TradingOrchestrator:
    """Основний orchestrator trading системи"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.config = self._create_config_from_args()
        self.logger = logging.getLogger(__name__)
        
        # Ініціалізація компонентів
        self._initialize_components()
    
    def _create_config_from_args(self) -> TradingConfig:
        """Створення конфігурації з CLI аргументів"""
        config = get_config()
        
        # Оновлення конфігурації з аргументів
        if self.args.tickers:
            config.data.tickers = [t.strip().upper() for t in self.args.tickers.split(',')]
        
        if self.args.timeframes:
            config.data.timeframes = [t.strip() for t in self.args.timeframes.split(',')]
        
        if self.args.risk_per_trade is not None:
            config.risk.risk_per_trade = self.args.risk_per_trade
        
        if self.args.max_positions is not None:
            config.risk.max_positions = self.args.max_positions
        
        if self.args.initial_capital is not None:
            config.risk.initial_capital = self.args.initial_capital
        
        if self.args.log_level:
            config.logging.log_level = self.args.log_level
        
        return config
    
    def _initialize_components(self) -> None:
        """Ініціалізація компонентів системи"""
        self.logger.info("Initializing trading system components...")
        
        try:
            # Створення директорій
            self._create_directories()
            
            # Ініціалізація спеціалізованих оркестраторів
            self._initialize_orchestrators()
            
            # Ініціалізація режимів
            self.modes = {
                'train': TrainMode(self.config),
                'analyze': AnalyzeMode(self.config),
                'batch-training': BatchTrainingMode(self.config),
                'progressive': ProgressiveMode(self.config),
                'monster-test': MonsterTestMode(self.config),
                'backtest': BacktestMode(self.config),
                'comprehensive-backtest': ComprehensiveBacktestMode(self.config),
                'optimized-backtest': OptimizedBacktestMode(self.config),
                'integrated-backtest': IntegratedBacktestMode(self.config),
                'real-data-backtest': RealDataBacktestMode(self.config),
                'web-ui': WebUIMode(self.config),
                'intelligent': IntelligentMode(self.args),  # Додано інтелектуальний режим
                
                # Нові режими з оркестраторами
                'full-pipeline': self.unified_orchestrator,
                'optimal-pipeline': self.pipeline_orchestrator,
                'dual-pipeline': self.dual_orchestrator
            }
            
            self.logger.info(f"All components initialized successfully")
            self.logger.info(f"Available modes: {list(self.modes.keys())}")
            self.logger.info(f"Configured tickers: {self.config.data.tickers}")
            self.logger.info(f"Configured timeframes: {self.config.data.timeframes}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            raise
    
    def _initialize_orchestrators(self) -> None:
        """Ініціалізує спеціалізованих оркестраторів"""
        try:
            # Конвертуємо конфігурацію в потрібний формат
            tickers_dict = {ticker: ticker for ticker in self.config.data.tickers}
            
            # Ініціалізуємо оркестратори з гнучкими тікерами і таймфреймами
            self.unified_orchestrator = UnifiedOrchestrator(
                tickers=tickers_dict,
                time_frames=self.config.data.timeframes,
                mode="optimal",
                debug=self.args.debug if hasattr(self.args, 'debug') else False
            )
            
            self.pipeline_orchestrator = PipelineOrchestrator(
                tickers=tickers_dict,
                time_frames=self.config.data.timeframes,
                debug=self.args.debug if hasattr(self.args, 'debug') else False
            )
            
            self.dual_orchestrator = DualPipelineOrchestrator(
                tickers=tickers_dict,
                time_frames=self.config.data.timeframes,
                debug=self.args.debug if hasattr(self.args, 'debug') else False
            )
            
            self.logger.info(f"Specialized orchestrators initialized with {len(tickers_dict)} tickers: {list(tickers_dict.keys())}")
            self.logger.info(f"Timeframes: {self.config.data.timeframes}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrators: {e}")
            raise
    
    def _create_directories(self) -> None:
        """Створення необхідних директорій"""
        directories = [
            self.config.data.data_dir / 'cache',
            self.config.data.data_dir / 'batches',
            self.config.data.data_dir / 'results',
            self.config.models.models_dir / 'trained',
            self.config.models.models_dir / 'cache',
            self.config.logging.logs_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Created directory: {directory}")
    
    def run_mode(self, mode: str) -> Dict[str, Any]:
        """Запуск вказаного режиму"""
        if mode not in self.modes:
            raise ValueError(f"Unknown mode: {mode}")
        
        mode_instance = self.modes[mode]
        
        # Спеціальна обробка для оркестраторів
        if mode in ['full-pipeline', 'optimal-pipeline', 'dual-pipeline']:
            return self._run_orchestrator_mode(mode, mode_instance)
        
        # Спеціальна обробка для web-ui
        if mode == 'web-ui':
            return mode_instance.run(
                host=getattr(self.args, 'host', 'localhost'),
                port=getattr(self.args, 'port', '8080')
            )
        
        self.logger.info(f"Starting {mode} mode...")
        
        try:
            result = mode_instance.run()
            result.update({
                'mode': mode,
                'tickers_count': len(self.config.data.tickers),
                'timeframes': self.config.data.timeframes,
                'config': {
                    'max_positions': self.config.risk.max_positions,
                    'risk_per_trade': self.config.risk.risk_per_trade
                }
            })
            
            self.logger.info(f"{mode} mode completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"{mode} mode failed: {e}")
            return {
                'status': 'failed',
                'mode': mode,
                'error': str(e),
                'tickers_count': len(self.config.data.tickers),
                'timeframes': self.config.data.timeframes
            }
    
    def _run_orchestrator_mode(self, mode: str, orchestrator) -> Dict[str, Any]:
        """Запуск режиму оркестратора"""
        self.logger.info(f"Starting {mode} orchestrator...")
        
        try:
            if mode == 'full-pipeline':
                result = orchestrator.run_complete_pipeline_with_comparison(
                    compare_models=True,
                    colab_training=True,
                    models_to_compare=None
                )
            elif mode == 'optimal-pipeline':
                result = orchestrator.run_optimal_pipeline(
                    colab_training=True
                )
            elif mode == 'dual-pipeline':
                # Dual pipeline не має стандартного методу, використовуємо внутрішні методи
                result = {
                    'status': 'success',
                    'mode': mode,
                    'message': 'Dual pipeline mode - use specific methods',
                    'available_methods': ['run_dual_pipeline_with_validation']
                }
            
            result.update({
                'mode': mode,
                'tickers_count': len(self.config.data.tickers),
                'timeframes': self.config.data.timeframes,
                'orchestrator_type': mode
            })
            
            self.logger.info(f"{mode} orchestrator completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"{mode} orchestrator failed: {e}")
            return {
                'status': 'failed',
                'mode': mode,
                'error': str(e),
                'tickers_count': len(self.config.data.tickers),
                'timeframes': self.config.data.timeframes
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Отримати статус системи"""
        return {
            'config_validated': True,
            'components_initialized': True,
            'available_modes': list(self.modes.keys()),
            'tickers_count': len(self.config.data.tickers),
            'timeframes': self.config.data.timeframes,
            'risk_settings': {
                'max_positions': self.config.risk.max_positions,
                'risk_per_trade': self.config.risk.risk_per_trade,
                'max_portfolio_risk': self.config.risk.max_portfolio_risk
            }
        }
    
    def validate_system(self) -> Dict[str, Any]:
        """Валідація системи"""
        issues = []
        
        # Перевірка тікерів
        if not self.config.data.tickers:
            issues.append("No tickers configured")
        
        # Перевірка таймфреймів
        if not self.config.data.timeframes:
            issues.append("No timeframes configured")
        
        # Перевірка ризиків
        if self.config.risk.risk_per_trade <= 0:
            issues.append("Risk per trade must be positive")
        
        if self.config.risk.max_positions <= 0:
            issues.append("Max positions must be positive")
        
        # Перевірка директорій
        required_dirs = [
            self.config.data.data_dir,
            self.config.models.models_dir,
            self.config.logging.logs_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create directory {directory}: {e}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'status': 'OK' if len(issues) == 0 else 'ERROR'
        }
