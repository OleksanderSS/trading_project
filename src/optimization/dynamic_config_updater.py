# DynamicConfigurationUpdater Implementation
# Minimal dependencies - uses only psutil, built-in modules

import datetime
import json
from pathlib import Path

import psutil

from src.core.logging.logger import ProjectLogger


class DynamicConfigurationUpdater:
    """
    Адаптує конфігурацію під час тренування на основі реальних ресурсів та перформансу.
    """

    def __init__(self, base_config, project_path=None):
        """
        Args:
            base_config: Словник з базовою конфігурацією (batch_size, epochs, learning_rate, etc.)
            project_path: Шлях для збереження логу змін
        """
        self.base_config = base_config.copy()
        self.current_config = base_config.copy()
        self.project_path = Path(project_path) if project_path else Path.cwd()
        self.adjustment_log = []
        self.logger = ProjectLogger.get_logger("DynamicConfigUpdater")

        # Пороги для адаптації
        self.memory_critical_threshold = 90  # %
        self.memory_warning_threshold = 80   # %
        self.memory_safe_threshold = 50      # %

    def update_batch_size_for_memory(self, current_batch_size, memory_percent=None):
        """
        Адаптує batch_size на основі використання пам'яті.
        """
        if memory_percent is None:
            memory_percent = psutil.virtual_memory().percent

        previous_batch = current_batch_size

        if memory_percent >= self.memory_critical_threshold:
            new_batch = max(int(current_batch_size * 0.5), 4)
            adjustment = "critical_memory"
        elif memory_percent >= self.memory_warning_threshold:
            new_batch = max(int(current_batch_size * 0.75), 8)
            adjustment = "high_memory"
        elif memory_percent < self.memory_safe_threshold and current_batch_size < self.base_config.get('batch_size', 32):
            new_batch = min(int(current_batch_size * 1.1), self.base_config.get('batch_size', 32))
            adjustment = "safe_memory_increase"
        else:
            new_batch = current_batch_size
            adjustment = "stable"

        if new_batch != previous_batch:
            self._log_adjustment("batch_size", previous_batch, new_batch, adjustment, memory_percent)
            self.current_config['batch_size'] = new_batch
            self.logger.info(f"Batch size adjusted: {previous_batch} → {new_batch} ({adjustment}, mem={memory_percent:.1f}%)")

        return new_batch

    def update_epochs_for_convergence(self, current_epoch, patience_counter, patience_limit):
        """
        Адаптує epochs на основі збіжності моделі.
        """
        if patience_counter >= patience_limit:
            reduction = 0.8
            reason = "patience_exceeded"
        elif patience_counter >= patience_limit * 0.8:
            reduction = 0.9
            reason = "approaching_patience"
        else:
            return current_epoch

        new_epochs = max(int(self.base_config.get('epochs', 100) * reduction), current_epoch)
        self._log_adjustment("epochs", self.base_config.get('epochs', 100), new_epochs, reason, None)
        self.current_config['epochs'] = new_epochs
        self.logger.info(f"Epochs reduced: {self.base_config.get('epochs', 100)} → {new_epochs} ({reason})")

        return new_epochs

    def update_learning_rate_for_loss(self, current_loss, previous_loss, improvement_rate=0.01):
        """
        Адаптує learning_rate на основі змін loss.
        """
        base_lr = self.base_config.get('learning_rate', 0.001)
        current_lr = self.current_config.get('learning_rate', base_lr)

        if previous_loss is None:
            return current_lr

        loss_improvement = (previous_loss - current_loss) / previous_loss if previous_loss != 0 else 0

        if loss_improvement > 0.05:
            new_lr = current_lr * 1.05
            reason = "good_improvement"
        elif loss_improvement < -0.01:  # Loss погіршується
            new_lr = current_lr * 0.8
            reason = "loss_worsened"
        elif loss_improvement < improvement_rate:
            new_lr = current_lr * 0.9
            reason = "slow_improvement"
        else:
            return current_lr

        new_lr = max(new_lr, base_lr * 0.0001)
        self._log_adjustment("learning_rate", current_lr, new_lr, reason, loss_improvement)
        self.current_config['learning_rate'] = new_lr
        self.logger.info(f"Learning rate adjusted: {current_lr:.6f} → {new_lr:.6f} ({reason})")

        return new_lr

    def update_max_features_for_memory(self, current_features, memory_percent=None):
        """
        Адаптує max_features (кількість ознак) при дефіциті пам'яті.
        """
        if memory_percent is None:
            memory_percent = psutil.virtual_memory().percent

        base_max_features = self.base_config.get('max_features', 100)

        if memory_percent > 85 and current_features > int(base_max_features * 0.5):
            new_features = max(int(current_features * 0.75), 20)
            self._log_adjustment("max_features", current_features, new_features, "memory_pressure", memory_percent)
            self.current_config['max_features'] = new_features
            self.logger.info(f"Max features reduced: {current_features} → {new_features} (mem={memory_percent:.1f}%)")
            return new_features

        return current_features

    def suggest_early_stopping(self, patience_counter, patience_limit):
        """
        Рекомендує зупинити тренування якщо надто багато епох без поліпшень.
        """
        if patience_counter >= patience_limit * 1.2:
            self.logger.warning(f"Early stopping recommended: patience_counter={patience_counter} >= {patience_limit * 1.2}")
            self._log_adjustment("training_status", "active", "early_stopping", "patience_exceeded", None)
            return True
        return False

    def _log_adjustment(self, param_name, old_value, new_value, reason, context=None):
        """Залогувати зміну конфігурації"""
        log_entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'parameter': param_name,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason,
            'context': context
        }
        self.adjustment_log.append(log_entry)

    def get_config(self):
        """Отримати поточну конфігурацію"""
        return self.current_config.copy()

    def get_adjustment_log(self):
        """Отримати лог всіх змін"""
        return self.adjustment_log.copy()

    def save_adjustment_log(self, filepath=None):
        """Зберегти лог змін конфігурації"""
        if filepath is None:
            filepath = self.project_path / "config_adjustments.json"

        with open(filepath, 'w') as f:
            json.dump({
                'base_config': self.base_config,
                'final_config': self.current_config,
                'adjustments': self.adjustment_log
            }, f, indent=2)

        self.logger.info(f"Configuration adjustment log saved to {filepath}")
        return filepath

    def reset_to_base(self):
        """Скинути конфігурацію до базової"""
        self.current_config = self.base_config.copy()
        self.logger.info("Configuration reset to base values")


# EXAMPLE USAGE
if __name__ == "__main__":
    ProjectLogger.setup_logging()
    logger = ProjectLogger.get_logger("ConfigUpdaterRunner")

    # Базова конфігурація
    base_config = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'max_features': 100,
        'patience': 15
    }

    # Ініціалізація адаптера
    updater = DynamicConfigurationUpdater(base_config)

    # Симуляція тренування з адаптацією
    logger.info("=== Simulation of Dynamic Configuration Updates ===")

    # Цикл 1: Нормальна пам'ять
    logger.info("Step 1: Normal memory (60%)")
    new_batch = updater.update_batch_size_for_memory(32, memory_percent=60)

    # Цикл 2: Висока пам'ять
    logger.info("Step 2: High memory (85%)")
    new_batch = updater.update_batch_size_for_memory(32, memory_percent=85)

    # Цикл 3: Критична пам'ять
    logger.info("Step 3: Critical memory (92%)")
    new_batch = updater.update_batch_size_for_memory(new_batch, memory_percent=92)

    # Цикл 4: Адаптація learning rate
    logger.info("Step 4: Loss improvement detected")
    updater.update_learning_rate_for_loss(current_loss=0.045, previous_loss=0.050)

    # Цикл 5: Адаптація epochs
    logger.info("Step 5: Patience counter approaching limit")
    updater.update_epochs_for_convergence(current_epoch=30,
                                          patience_counter=12, patience_limit=15)

    # Зберегти лог
    updater.save_adjustment_log()
    logger.info("Final Configuration:")
    logger.info(json.dumps(updater.get_config(), indent=2))
