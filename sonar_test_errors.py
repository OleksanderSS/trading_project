Run pytest --cov=src --cov-report=xml --cov-report=lcov
============================= test session starts ==============================
platform linux -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0
rootdir: /home/runner/work/trading_project/trading_project
configfile: pytest.ini (WARNING: ignoring pytest config in pyproject.toml!)
testpaths: tests
plugins: dash-4.2.0, cov-7.1.0, anyio-4.13.0
collected 246 items / 39 errors

==================================== ERRORS ====================================
________ ERROR collecting tests/models/ensemble/test_dynamic_weights.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/ensemble/test_dynamic_weights.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/ensemble/test_dynamic_weights.py:8: in <module>
    from src.models.ensemble.dynamic_weights import DynamicWeightCalculator
src/models/__init__.py:12: in <module>
    from .ensemble.confidence_calibrator import ConfidenceCalibrator, calibrate_confidence_quick, get_confidence_calibrator
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
------------------------------- Captured stdout --------------------------------
2026-06-06 16:33:11,311 - matplotlib - DEBUG - matplotlib data path: /opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/matplotlib/mpl-data
2026-06-06 16:33:11,333 - matplotlib - DEBUG - CONFIGDIR=/home/runner/.config/matplotlib
2026-06-06 16:33:11,338 - matplotlib - DEBUG - interactive is False
2026-06-06 16:33:11,338 - matplotlib - DEBUG - platform is linux
2026-06-06 16:33:11,640 - matplotlib - DEBUG - CACHEDIR=/home/runner/.cache/matplotlib
2026-06-06 16:33:11,641 - matplotlib.font_manager - DEBUG - font search path [PosixPath('/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf'), PosixPath('/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/afm'), PosixPath('/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/pdfcorefonts')]
2026-06-06 16:33:12,048 - matplotlib.font_manager - INFO - Failed to extract font properties from /usr/share/fonts/truetype/noto/NotoColorEmoji.ttf: Can not load face (unknown file format; error code 0x2)
2026-06-06 16:33:12,272 - matplotlib.font_manager - INFO - generated new fontManager
2026-06-06 16:33:13,174 - src.core.security.secure_secrets_manager - DEBUG - Skipping config env paths while UnifiedConfigManager is loading.
2026-06-06 16:33:13,174 - src.core.security.secure_secrets_manager - WARNING - No .env configuration file found in project local paths: ['.env']. Utilizing existing environment variables.
____ ERROR collecting tests/models/model_selector/test_adaptive_selector.py ____
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/model_selector/test_adaptive_selector.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/model_selector/test_adaptive_selector.py:8: in <module>
    from src.models.model_selector.adaptive_selector import AdaptiveModelSelector
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__________ ERROR collecting tests/models/prototypes/test_prototype.py __________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/prototypes/test_prototype.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/prototypes/test_prototype.py:10: in <module>
    from src.models.prototypes.prototype import ModelPrototype
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__________ ERROR collecting tests/models/prototypes/test_registry.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/prototypes/test_registry.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/prototypes/test_registry.py:12: in <module>
    from src.models.prototypes.prototype import ModelPrototype
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
___________ ERROR collecting tests/models/quality/test_controller.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/quality/test_controller.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/quality/test_controller.py:6: in <module>
    from src.models.quality.controller import ModelQualityController
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/models/test_persistent_pool.py _____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/models/test_persistent_pool.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/models/test_persistent_pool.py:8: in <module>
    from src.models.persistent_pool import PersistentModelPool
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__________ ERROR collecting tests/test_fix_p1_performance_tracker.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/test_fix_p1_performance_tracker.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_fix_p1_performance_tracker.py:2: in <module>
    from src.analytics.arena.performance_tracker import get_performance_tracker
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_____________ ERROR collecting tests/test_secure_model_loading.py ______________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/test_secure_model_loading.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/test_secure_model_loading.py:8: in <module>
    from src.predictions.models_predict import predict_from_parquet
src/predictions/models_predict.py:11: in <module>
    from src.ensembling.stacked_ensemble import ensemble_forecast
src/ensembling/stacked_ensemble.py:10: in <module>
    from src.meta_learning.memory.diary_engine import DiaryEngine
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__ ERROR collecting tests/unit/models/analysis/test_model_health_analyzer.py ___
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/models/analysis/test_model_health_analyzer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/models/analysis/test_model_health_analyzer.py:5: in <module>
    from src.models.analysis.model_health_analyzer import ModelHealthAnalyzer
src/models/analysis/model_health_analyzer.py:11: in <module>
    from src.models.monitoring.prediction_drift_monitor import PredictionDriftMonitor
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____ ERROR collecting tests/unit/models/test_weight_stability_visualizer.py ____
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/models/test_weight_stability_visualizer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/models/test_weight_stability_visualizer.py:5: in <module>
    from src.models.ensemble.weight_stability.visualizer import WeightStabilityVisualizer
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/unit/test_artifact_security.py _____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_artifact_security.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_artifact_security.py:6: in <module>
    from src.core.error_handling.error_handler import ModelLoadingError
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
________ ERROR collecting tests/unit/test_autoencoder_routing_policy.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_autoencoder_routing_policy.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_autoencoder_routing_policy.py:9: in <module>
    from src.pipeline.hybrid.final_stages_executor import FinalStagesExecutor
src/pipeline/hybrid/__init__.py:19: in <module>
    from .orchestrator_config import OrchestratorConfigManager, PipelineConfig
src/pipeline/hybrid/orchestrator_config.py:16: in <module>
    GDRIVE_AVAILABLE = all(
src/pipeline/hybrid/orchestrator_config.py:17: in <genexpr>
    find_spec(module_name) is not None
    ^^^^^^^^^^^^^^^^^^^^^^
E   ModuleNotFoundError: No module named 'googleapiclient'
____________ ERROR collecting tests/unit/test_constraint_engine.py _____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_constraint_engine.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_constraint_engine.py:2: in <module>
    from src.meta_learning.security.constraint_engine import SecurityConstraintEngine, ConstraintType, ConstraintSeverity
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_______ ERROR collecting tests/unit/test_context_pattern_sequence_knn.py _______
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_context_pattern_sequence_knn.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_context_pattern_sequence_knn.py:9: in <module>
    from src.meta_learning.memory.diary_engine import DiaryEngine
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__________ ERROR collecting tests/unit/test_data_manager_security.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_data_manager_security.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_data_manager_security.py:8: in <module>
    from src.data.management.data_manager import DataManager
src/data/management/data_manager.py:12: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
______________ ERROR collecting tests/unit/test_drift_analyzer.py ______________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_drift_analyzer.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_drift_analyzer.py:4: in <module>
    from src.models.monitoring.drift.analyzer import DriftAnalyzer
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
______________ ERROR collecting tests/unit/test_drift_history.py _______________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_drift_history.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_drift_history.py:4: in <module>
    from src.models.monitoring.drift.history import HistoryManager
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/unit/test_exception_handling.py ____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_exception_handling.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_exception_handling.py:6: in <module>
    from src.analytics.unified_analytics_engine import UnifiedAnalyticsEngine
src/analytics/unified_analytics_engine.py:14: in <module>
    from src.analytics.data_managers.model_results_manager import ModelResultsManager
src/analytics/data_managers/model_results_manager.py:13: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_______ ERROR collecting tests/unit/test_feature_engineering_monitor.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_feature_engineering_monitor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_feature_engineering_monitor.py:3: in <module>
    from src.pipeline.stages.monitoring.feature_monitoring import FeatureEngineeringMonitor
src/pipeline/stages/monitoring/feature_monitoring.py:3: in <module>
    from src.monitoring.data_freshness_monitor import get_data_freshness_monitor
src/monitoring/__init__.py:47: in <module>
    from .health_hub import HealthHub
src/monitoring/health_hub.py:12: in <module>
    from src.analytics.data_managers.model_results_manager import ModelResultsManager
src/analytics/data_managers/model_results_manager.py:13: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
------------------------------- Captured stdout --------------------------------
2026-06-06 16:33:18,784 - FeatureDriftMonitor - WARNING - ⚠️ Evidently AI not installed. Install with: pip install evidently
_____________ ERROR collecting tests/unit/test_feature_leakage.py ______________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_feature_leakage.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_feature_leakage.py:4: in <module>
    from src.models.feature_selector import ModelFeatureSelector
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____ ERROR collecting tests/unit/test_financial_ratio_denominator_policy.py ____
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_financial_ratio_denominator_policy.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_financial_ratio_denominator_policy.py:6: in <module>
    from src.meta_learning.memory.diary_engine import DiaryEngine
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_________ ERROR collecting tests/unit/test_hybrid_pipeline_manager.py __________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_hybrid_pipeline_manager.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_hybrid_pipeline_manager.py:10: in <module>
    from src.pipeline.hybrid.pipeline_config import FinalStagesParams, PipelineParams
src/pipeline/hybrid/__init__.py:19: in <module>
    from .orchestrator_config import OrchestratorConfigManager, PipelineConfig
src/pipeline/hybrid/orchestrator_config.py:16: in <module>
    GDRIVE_AVAILABLE = all(
src/pipeline/hybrid/orchestrator_config.py:17: in <genexpr>
    find_spec(module_name) is not None
    ^^^^^^^^^^^^^^^^^^^^^^
E   ModuleNotFoundError: No module named 'googleapiclient'
_________ ERROR collecting tests/unit/test_hyperparameter_searcher.py __________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_hyperparameter_searcher.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_hyperparameter_searcher.py:3: in <module>
    from src.optimization.hyperparameter_searcher import HyperparameterSearcher
src/optimization/__init__.py:9: in <module>
    from .factory import OptimizationFactory
src/optimization/factory.py:6: in <module>
    from src.optimization.portfolio.optimizer import PortfolioOptimizer
src/optimization/portfolio/optimizer.py:13: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/unit/test_ml_analytics_drift.py ____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_ml_analytics_drift.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_ml_analytics_drift.py:5: in <module>
    from src.monitoring.ml_analytics import MLAnalytics
src/monitoring/__init__.py:47: in <module>
    from .health_hub import HealthHub
src/monitoring/health_hub.py:12: in <module>
    from src.analytics.data_managers.model_results_manager import ModelResultsManager
src/analytics/data_managers/model_results_manager.py:13: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
________ ERROR collecting tests/unit/test_pattern_and_synthetic_math.py ________
tests/unit/test_pattern_and_synthetic_math.py:6: in <module>
    from src.optimization.portfolio.optimizer import PortfolioOptimizer
<frozen importlib._bootstrap>:1360: in _find_and_load
    ???
<frozen importlib._bootstrap>:1322: in _find_and_load_unlocked
    ???
<frozen importlib._bootstrap>:1262: in _find_spec
    ???
<frozen importlib._bootstrap_external>:1532: in find_spec
    ???
<frozen importlib._bootstrap_external>:1501: in _get_spec
    ???
<frozen importlib._bootstrap_external>:1372: in __iter__
    ???
<frozen importlib._bootstrap_external>:1359: in _recalculate
    ???
<frozen importlib._bootstrap_external>:1355: in _get_parent_path
    ???
E   KeyError: 'src.optimization'
__________ ERROR collecting tests/unit/test_pct_change_fill_policy.py __________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_pct_change_fill_policy.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_pct_change_fill_policy.py:11: in <module>
    from src.pipeline.stages.trading.recommendation_engine import TradingRecommendationEngine
src/pipeline/stages/trading/__init__.py:1: in <module>
    from src.pipeline.stages.trading.data_io import TradingDataIO
src/pipeline/stages/trading/data_io.py:8: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
___________ ERROR collecting tests/unit/test_performance_tracker.py ____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_performance_tracker.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_performance_tracker.py:2: in <module>
    from src.analytics.arena.performance_tracker import ModelPerformanceTracker
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____ ERROR collecting tests/unit/test_pipeline_orchestrator_execute_sync.py ____
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_pipeline_orchestrator_execute_sync.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_pipeline_orchestrator_execute_sync.py:5: in <module>
    from src.pipeline.pipeline_orchestrator import PipelineOrchestrator
src/pipeline/pipeline_orchestrator.py:10: in <module>
    from src.analytics.data_managers.model_results_manager import ModelResultsManager
src/analytics/data_managers/model_results_manager.py:13: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/unit/test_portfolio_manager.py _____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_portfolio_manager.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_portfolio_manager.py:1: in <module>
    from src.trading.portfolio_manager import PortfolioManager
src/trading/__init__.py:6: in <module>
    from .virtual_portfolio import VirtualPortfolio
src/trading/virtual_portfolio.py:14: in <module>
    from src.core.error_handling.error_handler import get_error_handler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_____ ERROR collecting tests/unit/test_prediction_stage_model_selection.py _____
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_prediction_stage_model_selection.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_prediction_stage_model_selection.py:5: in <module>
    from src.pipeline.stages.stage_5_prediction import PredictionStage
src/pipeline/stages/stage_5_prediction.py:31: in <module>
    from src.ensembling.stacked_ensemble import StackedEnsemble
src/ensembling/stacked_ensemble.py:10: in <module>
    from src.meta_learning.memory.diary_engine import DiaryEngine
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
______ ERROR collecting tests/unit/test_progressive_trainer_resources.py _______
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_progressive_trainer_resources.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_progressive_trainer_resources.py:4: in <module>
    from src.training.progressive_trainer import ProgressiveTrainer
src/training/progressive_trainer.py:19: in <module>
    from src.training.base_trainer import BaseTrainer, TrainerConfig
src/training/base_trainer.py:22: in <module>
    from src.factories.model_factory import ModelFactory
src/factories/model_factory.py:8: in <module>
    from src.factories.tree_model_factory import TreeModelFactory
src/factories/tree_model_factory.py:5: in <module>
    from src.models.interfaces import BaseModel
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
________ ERROR collecting tests/unit/test_real_time_learning_metrics.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_real_time_learning_metrics.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_real_time_learning_metrics.py:3: in <module>
    from src.meta_learning.real_time_learning import RealTimeLearning
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
________ ERROR collecting tests/unit/test_reddit_sentiment_collector.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_reddit_sentiment_collector.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_reddit_sentiment_collector.py:3: in <module>
    from src.data.collectors.reddit_sentiment_collector import RedditSentimentCollector
src/data/collectors/reddit_sentiment_collector.py:10: in <module>
    from src.core.cache.cache_manager import CacheManager
src/core/cache/__init__.py:1: in <module>
    from .cache_manager import CacheManager
src/core/cache/cache_manager.py:15: in <module>
    from src.data.management.data_manager import DataManager
src/data/management/data_manager.py:12: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
________ ERROR collecting tests/unit/test_sample_fallback_collectors.py ________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_sample_fallback_collectors.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_sample_fallback_collectors.py:5: in <module>
    from src.data.collectors.cftc_collector import CFTCCollector
src/data/collectors/cftc_collector.py:10: in <module>
    from src.core.cache.cache_manager import CacheManager
src/core/cache/__init__.py:1: in <module>
    from .cache_manager import CacheManager
src/core/cache/cache_manager.py:15: in <module>
    from src.data.management.data_manager import DataManager
src/data/management/data_manager.py:12: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
_____________ ERROR collecting tests/unit/test_signal_processor.py _____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_signal_processor.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_signal_processor.py:1: in <module>
    from src.trading.signal_processor import SignalProcessor
src/trading/__init__.py:6: in <module>
    from .virtual_portfolio import VirtualPortfolio
src/trading/virtual_portfolio.py:14: in <module>
    from src.core.error_handling.error_handler import get_error_handler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
__________ ERROR collecting tests/unit/test_stage3_data_contracts.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_stage3_data_contracts.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_stage3_data_contracts.py:7: in <module>
    from src.pipeline.stages.feature_engineering.orchestrator import FeatureEngineeringStage
src/pipeline/stages/feature_engineering/__init__.py:3: in <module>
    from .orchestrator import FeatureEngineeringStage
src/pipeline/stages/feature_engineering/orchestrator.py:7: in <module>
    from src.core.error_handling.error_handler import ErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
___________ ERROR collecting tests/unit/test_stage6_diary_logging.py ___________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_stage6_diary_logging.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_stage6_diary_logging.py:5: in <module>
    from src.meta_learning.memory.diary_engine import DecisionOutcome, DecisionType
src/meta_learning/__init__.py:14: in <module>
    from .evolution.dual_loops import LearningLoopsEngine, TradingRule
src/meta_learning/evolution/dual_loops.py:29: in <module>
    from src.analytics.arena.arena_battle import get_trading_arena
src/analytics/arena/__init__.py:5: in <module>
    from .arena_battle import BattleMetrics, BattleResult, TradingModelArena, get_trading_arena
src/analytics/arena/arena_battle.py:17: in <module>
    from .battle_groups import get_battle_group_manager
src/analytics/arena/battle_groups.py:4: in <module>
    from src.models.registry.model_registry import ModelRegistry
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
______ ERROR collecting tests/unit/test_target_orchestrator_alignment.py _______
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_target_orchestrator_alignment.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_target_orchestrator_alignment.py:3: in <module>
    from src.pipeline.stages.feature_engineering.targets import TargetGenerator
src/pipeline/stages/feature_engineering/__init__.py:3: in <module>
    from .orchestrator import FeatureEngineeringStage
src/pipeline/stages/feature_engineering/orchestrator.py:7: in <module>
    from src.core.error_handling.error_handler import ErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
____________ ERROR collecting tests/unit/test_tree_model_factory.py ____________
ImportError while importing test module '/home/runner/work/trading_project/trading_project/tests/unit/test_tree_model_factory.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/hostedtoolcache/Python/3.12.13/x64/lib/python3.12/importlib/__init__.py:90: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
tests/unit/test_tree_model_factory.py:2: in <module>
    from src.factories.tree_model_factory import TreeModelFactory
src/factories/tree_model_factory.py:5: in <module>
    from src.models.interfaces import BaseModel
src/models/__init__.py:13: in <module>
    from .ensemble.model_correlation_analyzer import (
src/models/ensemble/__init__.py:18: in <module>
    from .model_correlation_analyzer import (
src/models/ensemble/model_correlation_analyzer.py:11: in <module>
    from .correlation.correlation_engine import get_correlation_engine
src/models/ensemble/correlation/__init__.py:7: in <module>
    from .correlation_engine import CorrelationEngine, get_correlation_engine
src/models/ensemble/correlation/correlation_engine.py:15: in <module>
    from src.core.error_handling.error_handler import ErrorHandler, IErrorHandler
src/core/error_handling/error_handler.py:13: in <module>
    from src.core.logging.notifier import UniversalNotifier
src/core/logging/notifier.py:7: in <module>
    import aiofiles
E   ModuleNotFoundError: No module named 'aiofiles'
=============================== warnings summary ===============================
tests/test_async_timeout.py:9
  /home/runner/work/trading_project/trading_project/tests/test_async_timeout.py:9: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/overfitting_detection/test_analyzer.py:21
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/overfitting_detection/test_analyzer.py:21: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/overfitting_detection/test_analyzer.py:39
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/overfitting_detection/test_analyzer.py:39: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/overfitting_detection/test_manager.py:12
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/overfitting_detection/test_manager.py:12: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/overfitting_detection/test_manager.py:34
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/overfitting_detection/test_manager.py:34: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/test_baseline_dominance_detector.py:20
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/test_baseline_dominance_detector.py:20: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/models/analysis/test_baseline_dominance_detector.py:39
  /home/runner/work/trading_project/trading_project/tests/unit/models/analysis/test_baseline_dominance_detector.py:39: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/test_feature_engineering_stage_no_target_leakage.py:16
  /home/runner/work/trading_project/trading_project/tests/unit/test_feature_engineering_stage_no_target_leakage.py:16: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

tests/unit/test_overfitting_detector.py:7
  /home/runner/work/trading_project/trading_project/tests/unit/test_overfitting_detector.py:7: PytestUnknownMarkWarning: Unknown pytest.mark.asyncio - is this a typo?  You can register custom marks to avoid this warning - for details, see https://docs.pytest.org/en/stable/how-to/mark.html
    @pytest.mark.asyncio

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
=========================== short test summary info ============================
ERROR tests/models/ensemble/test_dynamic_weights.py
ERROR tests/models/model_selector/test_adaptive_selector.py
ERROR tests/models/prototypes/test_prototype.py
ERROR tests/models/prototypes/test_registry.py
ERROR tests/models/quality/test_controller.py
ERROR tests/models/test_persistent_pool.py
ERROR tests/test_fix_p1_performance_tracker.py
ERROR tests/test_secure_model_loading.py
ERROR tests/unit/models/analysis/test_model_health_analyzer.py
ERROR tests/unit/models/test_weight_stability_visualizer.py
ERROR tests/unit/test_artifact_security.py
ERROR tests/unit/test_autoencoder_routing_policy.py
ERROR tests/unit/test_constraint_engine.py
ERROR tests/unit/test_context_pattern_sequence_knn.py
ERROR tests/unit/test_data_manager_security.py
ERROR tests/unit/test_drift_analyzer.py
ERROR tests/unit/test_drift_history.py
ERROR tests/unit/test_exception_handling.py
ERROR tests/unit/test_feature_engineering_monitor.py
ERROR tests/unit/test_feature_leakage.py
ERROR tests/unit/test_financial_ratio_denominator_policy.py
ERROR tests/unit/test_hybrid_pipeline_manager.py
ERROR tests/unit/test_hyperparameter_searcher.py
ERROR tests/unit/test_ml_analytics_drift.py
ERROR tests/unit/test_pattern_and_synthetic_math.py - KeyError: 'src.optimization'
ERROR tests/unit/test_pct_change_fill_policy.py
ERROR tests/unit/test_performance_tracker.py
ERROR tests/unit/test_pipeline_orchestrator_execute_sync.py
ERROR tests/unit/test_portfolio_manager.py
ERROR tests/unit/test_prediction_stage_model_selection.py
ERROR tests/unit/test_progressive_trainer_resources.py
ERROR tests/unit/test_real_time_learning_metrics.py
ERROR tests/unit/test_reddit_sentiment_collector.py
ERROR tests/unit/test_sample_fallback_collectors.py
ERROR tests/unit/test_signal_processor.py
ERROR tests/unit/test_stage3_data_contracts.py
ERROR tests/unit/test_stage6_diary_logging.py
ERROR tests/unit/test_target_orchestrator_alignment.py
ERROR tests/unit/test_tree_model_factory.py
!!!!!!!!!!!!!!!!!!! Interrupted: 39 errors during collection !!!!!!!!!!!!!!!!!!!
======================= 9 warnings, 39 errors in 13.08s ========================
Error: Process completed with exit code 2.