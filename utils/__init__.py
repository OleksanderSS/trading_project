"""
Enhanced Utils package - роwithширений пакет утилandт for allєї system
"""

# Новand модулand монandторингу and аналandwithу
from .results_manager import ResultsManager, HeavyLightModelComparator, ComprehensiveReporter
from .automated_reporting import AutomatedReporting, RealTimeMonitor, HistoricalAnalytics
from .ci_cd_integration import CICDIntegration
from .ml_analytics import MLAnalytics

# Новand меnotджери and конфandгурацandя
from .utils_manager import UtilsManager
from .utils_config import UtilsConfig
from .utils_optimizer import UtilsOptimizer

# Ключові існуючі утиліти - використовуємо виправлений логер
from .logger_fixed import ProjectLogger
from .performance_tracker import PerformanceTracker

# Додатковand утилandти (якщо доступнand)
try:
    from .feature_selector import select_features
    _FEATURE_SELECTOR_AVAILABLE = True
except ImportError:
    _FEATURE_SELECTOR_AVAILABLE = False

try:
    from .feature_scaler import FeatureScaler
    _FEATURE_SCALER_AVAILABLE = True
except ImportError:
    _FEATURE_SCALER_AVAILABLE = False

try:
    from .data_utils import generate_content_hash
    _DATA_UTILS_AVAILABLE = True
except ImportError:
    _DATA_UTILS_AVAILABLE = False

try:
    from .features_utils import FeatureUtils
    _FEATURES_UTILS_AVAILABLE = True
except ImportError:
    _FEATURES_UTILS_AVAILABLE = False

try:
    from .advanced_features import add_advanced_features
    _ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    _ADVANCED_FEATURES_AVAILABLE = False

try:
    from .technical_features import add_technical_indicators
    _TECHNICAL_FEATURES_AVAILABLE = True
except ImportError:
    _TECHNICAL_FEATURES_AVAILABLE = False

try:
    from .macro_features import add_macro_features
    _MACRO_FEATURES_AVAILABLE = True
except ImportError:
    _MACRO_FEATURES_AVAILABLE = False

__all__ = [
    # Новand модулand
    'ResultsManager', 'HeavyLightModelComparator', 'ComprehensiveReporter',
    'AutomatedReporting', 'RealTimeMonitor', 'HistoricalAnalytics',
    'CICDIntegration', 'MLAnalytics',
    
    # Новand меnotджери
    'UtilsManager', 'UtilsConfig', 'UtilsOptimizer',
    
    # Ключові утиліти
    'ProjectLogger', 'PerformanceTracker'
]

# Додаємо опцandональнand утилandти якщо доступнand
if _FEATURE_SELECTOR_AVAILABLE:
    __all__.append('select_features')

if _FEATURE_SCALER_AVAILABLE:
    __all__.append('FeatureScaler')

if _DATA_UTILS_AVAILABLE:
    __all__.append('generate_content_hash')

if _FEATURES_UTILS_AVAILABLE:
    __all__.append('FeatureUtils')

if _ADVANCED_FEATURES_AVAILABLE:
    __all__.append('add_advanced_features')

if _TECHNICAL_FEATURES_AVAILABLE:
    __all__.append('add_technical_indicators')

if _MACRO_FEATURES_AVAILABLE:
    __all__.append('add_macro_features')

__version__ = "2.0.0"
__author__ = "Trading System Utils Team"