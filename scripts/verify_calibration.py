
import numpy as np

from src.calibration.adaptive_confidence_calibrator import AdaptiveConfidenceCalibrator
from src.calibration.calibration_engine import CalibrationEngine
from src.config.unified_config_manager import UnifiedConfigManager
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("CalibrationValidation")

def test_calibration():
    logger.info("Starting Calibration Module validation...")
    
    # 1. Test AdaptiveConfidenceCalibrator
    logger.info("Testing AdaptiveConfidenceCalibrator...")
    calibrator = AdaptiveConfidenceCalibrator()
    
    # Simulate data
    raw_confs = np.random.rand(100)
    outcomes = (raw_confs > 0.5).astype(int)
    
    for r, o in zip(raw_confs, outcomes):
        calibrator.update_with_outcome(r, o)
        
    report = calibrator.get_calibration_report()
    logger.info(f"✅ Calibrator report: {report}")

    # 2. Test CalibrationEngine (Optuna check)
    logger.info("Testing CalibrationEngine...")
    try:
        config_manager = UnifiedConfigManager()
        engine = CalibrationEngine(config_manager, n_trials=2)
        logger.info("✅ CalibrationEngine initialized successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to initialize CalibrationEngine: {e}")

if __name__ == "__main__":
    test_calibration()
