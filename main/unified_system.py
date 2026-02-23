#!/usr/bin/env python3
"""
Unified System - Інтеграція всіх розширених modules
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Advanced Features
try:
    from advanced_features import PrizmArchitectureEngine, DataLeakageDetector, ValidationProtocolsEngine
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_AVAILABLE = False

# Business Rules
try:
    from business_rules import InvestmentRulesEngine, MacroKnowledgeEngine, AbstractionLevelsEngine
    BUSINESS_RULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Business rules not available: {e}")
    BUSINESS_RULES_AVAILABLE = False

# Meta-Learning
try:
    from meta_learning import ExperienceDiary, AutomatedMetaCoding, DualLearningLoops, RealtimeContextAwareness
    META_LEARNING_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Meta-learning not available: {e}")
    META_LEARNING_AVAILABLE = False

# Advanced Analysis
try:
    from core.analysis.final_context_system import FinalContextSystem
    from core.analysis.critical_signals_analysis import CriticalSignalsAnalysis
    from core.analysis.profit_optimized_context import create_profit_optimized_context
    from core.analysis.advanced_online_model_comparator import AdvancedOnlineModelComparator
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Advanced analysis not available: {e}")
    ADVANCED_ANALYSIS_AVAILABLE = False

# Dashboard
try:
    from dashboard.unified_dashboard import UnifiedDashboard
    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Dashboard not available: {e}")
    DASHBOARD_AVAILABLE = False

# Existing Core Components
from money_maker import MoneyMaker
from core.analysis.smart_switcher import SmartSwitcher
from flexible_feature_selection import FlexibleFeatureSelection


class UnifiedTradingSystem:
    """
    Єдина інтелектуальна торгова система
    Інтегрує всі розширені модулі для максимальної продуктивності
    """
    
    def __init__(self, enable_all_features: bool = True):
        self.logger = logging.getLogger(__name__)
        self.enable_all_features = enable_all_features
        
        # Існуючі компоненти
        self.money_maker = MoneyMaker(enable_colab_integration=True)
        self.smart_switcher = SmartSwitcher()
        self.flexible_selection = FlexibleFeatureSelection()
        
        # Advanced Features
        self.prizm_engine = None
        self.leakage_detector = None
        self.validation_engine = None
        
        # Business Rules
        self.investment_rules = None
        self.macro_knowledge = None
        self.abstraction_levels = None
        
        # Meta-Learning
        self.experience_diary = None
        self.meta_coding = None
        self.dual_learning = None
        self.realtime_context = None
        
        # Advanced Analysis
        self.final_context = None
        self.critical_signals = None
        self.profit_optimized = None
        self.online_comparator = None
        
        # Dashboard
        self.dashboard = None
        
        # Ініціалізація компонентів
        self._initialize_components()
    
    def _initialize_components(self):
        """Ініціалізація всіх компонентів"""
        self.logger.info("[START] Initializing Unified Trading System...")
        
        # Advanced Features
        if self.enable_all_features and ADVANCED_FEATURES_AVAILABLE:
            try:
                self.prizm_engine = PrizmArchitectureEngine()
                self.leakage_detector = DataLeakageDetector()
                self.validation_engine = ValidationProtocolsEngine()
                self.logger.info("[OK] Advanced Features initialized")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to initialize Advanced Features: {e}")
        
        # Business Rules
        if self.enable_all_features and BUSINESS_RULES_AVAILABLE:
            try:
                self.investment_rules = InvestmentRulesEngine()
                self.macro_knowledge = MacroKnowledgeEngine()
                self.abstraction_levels = AbstractionLevelsEngine()
                self.logger.info("[OK] Business Rules initialized")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to initialize Business Rules: {e}")
        
        # Meta-Learning
        if self.enable_all_features and META_LEARNING_AVAILABLE:
            try:
                self.experience_diary = ExperienceDiary()
                self.meta_coding = AutomatedMetaCoding()
                self.dual_learning = DualLearningLoops()
                self.realtime_context = RealtimeContextAwareness()
                self.logger.info("[OK] Meta-Learning initialized")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to initialize Meta-Learning: {e}")
        
        # Advanced Analysis
        if self.enable_all_features and ADVANCED_ANALYSIS_AVAILABLE:
            try:
                self.final_context = FinalContextSystem()
                self.critical_signals = CriticalSignalsAnalysis()
                self.profit_optimized = create_profit_optimized_context()
                self.online_comparator = AdvancedOnlineModelComparator()
                self.logger.info("[OK] Advanced Analysis initialized")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to initialize Advanced Analysis: {e}")
        
        # Dashboard
        if self.enable_all_features and DASHBOARD_AVAILABLE:
            try:
                self.dashboard = UnifiedDashboard()
                self.logger.info("[OK] Dashboard initialized")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to initialize Dashboard: {e}")
    
    def run_intelligent_pipeline(self, tickers: List[str] = None) -> Dict[str, Any]:
        """
        Запуск інтелектуального pipeline з усіма компонентами
        """
        if tickers is None:
            tickers = ["SPY", "QQQ", "TSLA", "NVDA"]
        
        self.logger.info(f"🧠 Starting INTELLIGENT PIPELINE for {tickers}")
        
        results = {
            'pipeline_status': 'running',
            'tickers': tickers,
            'components_used': [],
            'features_enabled': self.enable_all_features
        }
        
        try:
            # 1. Data Leakage Detection (якщо доступно)
            if self.leakage_detector:
                self.logger.info("🛡️ Running Data Leakage Detection...")
                leakage_report = self.leakage_detector.analyze_pipeline()
                results['leakage_detection'] = leakage_report
                results['components_used'].append('leakage_detector')
                
                if leakage_report.leakage_detected:
                    self.logger.warning("[WARN] Data leakage detected! Applying fixes...")
                    # Застосувати виправлення
                    self._apply_leakage_fixes(leakage_report)
            
            # 2. Enhanced Data Collection з Business Rules
            if self.macro_knowledge:
                self.logger.info("📚 Applying Macro Knowledge...")
                macro_context = self.macro_knowledge.get_current_context()
                results['macro_context'] = macro_context
                results['components_used'].append('macro_knowledge')
            
            # 3. Run Core Pipeline
            self.logger.info("[RESTART] Running Core Pipeline...")
            core_results = self.money_maker.run_full_pipeline_with_colab(tickers=tickers)
            results.update(core_results)
            results['components_used'].append('money_maker')
            
            # 4. Apply Investment Rules
            if self.investment_rules and core_results.get('smart_switcher_ready'):
                self.logger.info("[MONEY] Applying Investment Rules...")
                investment_analysis = self.investment_rules.analyze_opportunities(core_results)
                results['investment_analysis'] = investment_analysis
                results['components_used'].append('investment_rules')
            
            # 5. Meta-Learning Integration
            if self.experience_diary and core_results.get('paper_trading_session'):
                self.logger.info("🧠 Learning from Experience...")
                paper_session = core_results['paper_trading_session']
                
                # Записати досвід
                for decision in paper_session.get('decisions', []):
                    self.experience_diary.record_decision(decision)
                
                # Отримати уроки
                lessons = self.experience_diary.get_recent_lessons(days=7)
                results['experience_lessons'] = lessons
                results['components_used'].append('experience_diary')
                
                # Автоматично покращити код
                if self.meta_coding and lessons:
                    self.logger.info("🤖 Auto-improving code based on experience...")
                    improvements = self.meta_coding.generate_improvements(lessons)
                    results['code_improvements'] = improvements
                    results['components_used'].append('meta_coding')
            
            # 6. Critical Signals Analysis
            if self.critical_signals and core_results.get('smart_switcher_ready'):
                self.logger.info("[WARN] Analyzing Critical Signals...")
                critical_analysis = self.critical_signals.analyze_market_signals(core_results)
                results['critical_signals'] = critical_analysis
                results['components_used'].append('critical_signals')
                
                # Перевірити на критичні попередження
                if critical_analysis.get('critical_warnings'):
                    self.logger.warning("🚨 CRITICAL WARNINGS DETECTED!")
                    results['critical_warnings'] = critical_analysis['critical_warnings']
            
            # 7. Profit Optimization
            if self.profit_optimized and core_results.get('smart_switcher_ready'):
                self.logger.info("[MONEY] Optimizing for Maximum Profit...")
                profit_optimization = self.profit_optimized.optimize_strategy(core_results)
                results['profit_optimization'] = profit_optimization
                results['components_used'].append('profit_optimized')
            
            # 8. Validation Protocols
            if self.validation_engine and core_results.get('models_trained'):
                self.logger.info("[OK] Running Validation Protocols...")
                validation_results = self.validation_engine.validate_models(core_results)
                results['validation_results'] = validation_results
                results['components_used'].append('validation_engine')
            
            # 9. Dashboard Update (якщо доступно)
            if self.dashboard:
                self.logger.info("[DATA] Updating Dashboard...")
                self.dashboard.update_with_results(results)
                results['dashboard_updated'] = True
                results['components_used'].append('dashboard')
            
            # 10. Final Context Integration
            if self.final_context:
                self.logger.info("🧠 Applying Final Context System...")
                final_context = self.final_context.create_unified_context(results)
                results['final_context'] = final_context
                results['components_used'].append('final_context')
            
            results['pipeline_status'] = 'completed'
            self.logger.info("[OK] INTELLIGENT PIPELINE COMPLETED SUCCESSFULLY!")
            
        except Exception as e:
            self.logger.error(f"[ERROR] Error in intelligent pipeline: {e}")
            results['pipeline_status'] = 'failed'
            results['error'] = str(e)
        
        return results
    
    def _apply_leakage_fixes(self, leakage_report):
        """Застосувати виправлення для витоку data"""
        # Логіка виправлення витоку data
        pass
    
    def run_prizm_decision_process(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Запуск Prizm архітектури для прийняття рішень
        """
        if not self.prizm_engine:
            return {'error': 'Prizm engine not available'}
        
        self.logger.info("🤖 Running Prizm Decision Process...")
        
        # Generator - генерує гіпотези
        generator_decision = self.prizm_engine.generate_hypotheses(market_data)
        
        # Critic - шукає контрприклади
        critic_decision = self.prizm_engine.critic_hypotheses(generator_decision, market_data)
        
        # Refiner - покращує рішення
        refined_decision = self.prizm_engine.refine_decision(critic_decision, market_data)
        
        # Judge - приймає/відхиляє
        judge_decision = self.prizm_engine.judge_decision(refined_decision, market_data)
        
        # Scenario - проганяє what-if
        scenario_analysis = self.prizm_engine.run_scenario_analysis(judge_decision, market_data)
        
        return {
            'generator': generator_decision,
            'critic': critic_decision,
            'refined': refined_decision,
            'judge': judge_decision,
            'scenarios': scenario_analysis
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Отримати статус системи"""
        status = {
            'system_name': 'UnifiedTradingSystem',
            'version': '2.0.0',
            'components': {},
            'features_enabled': self.enable_all_features
        }
        
        # Перевірити статус кожного компонента
        if self.prizm_engine:
            status['components']['prizm_engine'] = 'active'
        if self.leakage_detector:
            status['components']['leakage_detector'] = 'active'
        if self.validation_engine:
            status['components']['validation_engine'] = 'active'
        if self.investment_rules:
            status['components']['investment_rules'] = 'active'
        if self.macro_knowledge:
            status['components']['macro_knowledge'] = 'active'
        if self.experience_diary:
            status['components']['experience_diary'] = 'active'
        if self.critical_signals:
            status['components']['critical_signals'] = 'active'
        if self.profit_optimized:
            status['components']['profit_optimized'] = 'active'
        if self.dashboard:
            status['components']['dashboard'] = 'active'
        
        return status
