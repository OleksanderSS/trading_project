"""Runtime configuration loader for Colab training"""

import json
import sys
from pathlib import Path

RUNTIME_PARAMS_FILE = "runtime_params.json"


class RuntimeConfigLoader:
    """Завантаження та управління runtime конфігурацією"""

    FULL_MODE_MESSAGE = "📊 Режим: full (всі тікери, всі таргети, стандартні епохи)"

    def __init__(self, project_path, force_full_mode=False):
        self.project_path = Path(project_path)
        self.runtime_params = {}
        self.test_mode = {}
        self.force_full_mode = force_full_mode

        self.TEST_TICKER = None
        self.TEST_TARGET = None
        self.REDUCED_EPOCHS = None
        self.MAX_ITERATIONS = 100

    def load_runtime_params(self):
        """Завантаження runtime параметрів з файлів"""
        if self.force_full_mode:
            config_path = None
            print("🗑️ FORCE FULL MODE: ігноруємо всі config.json файли")
        else:
            config_path = self._find_config_file()

        determined_batch_dir = self._determine_batch_directory(config_path)
        self._load_batch_params(determined_batch_dir, config_path)
        self._print_loaded_params()

    def _find_config_file(self):
        """Знаходить шлях до файлу config.json в batch_dir"""
        batch_name = self._get_batch_name_from_args() or "main_database"
        batch_dir = self._build_batch_dir_path(batch_name)

        config_path = batch_dir / "config.json"

        if config_path.exists():
            return config_path
        else:
            return None

    def _get_batch_name_from_args(self):
        """Отримує batch_name з аргументів командного рядка"""
        if '--batch-name' in sys.argv:
            try:
                idx = sys.argv.index('--batch-name')
                return sys.argv[idx + 1]
            except (ValueError, IndexError):
                pass
        return None

    def _determine_batch_directory(self, config_path):
        """Визначає директорію батча на основі config.json"""
        if self.force_full_mode:
            batch_name = "main_database"
            print("🗑️ FORCE FULL MODE: використовуємо main_database")
        elif config_path:
            batch_name = config_path.parent.name
            print(f"🧪 Тестовий режим: використовуємо batch_name: {batch_name}")
        else:
            batch_name = "main_database"
            print("📊 Повноцінний режим: використовуємо main_database")

        return self._build_batch_dir_path(batch_name)

    def _build_batch_dir_path(self, batch_name):
        """Будує шлях до директорії батча"""
        batch_dir = self.project_path / "data" / "colab" / "accumulated" / batch_name
        print(f"🔧 Шукаємо дані в {batch_dir}")
        return batch_dir

    def _normalize_batch_name(self, batch_name):
        """Нормалізує назву батча"""
        batch_name = batch_name.replace('target_target_', 'target_')
        if not batch_name.startswith('test_') and batch_name != 'main_database':
            batch_name = 'test_' + batch_name
        return batch_name

    def _print_batch_source(self, runtime_params_path, batch_name):
        """Виводить джерело назви батча"""
        legacy_path = self.project_path / "src" / "config" / RUNTIME_PARAMS_FILE
        if runtime_params_path == legacy_path:
            print("⚠️ Batch name з legacy src/config/runtime_params.json: ")
        else:
            print("📦 Batch name з data/runtime/runtime_params.json:")
        print(f"{batch_name}")

    def _load_batch_params(self, batch_dir, config_path=None):
        """Завантаження параметрів з директорії батча"""
        is_main_database = batch_dir.name == "main_database"

        if self.force_full_mode:
            self._print_force_full_mode()
            return

        if is_main_database and config_path and config_path.exists():
            self._handle_config_in_main_database(config_path)
            return

        if config_path and config_path.exists():
            self._load_config_file(config_path)
        else:
            self._print_full_mode_message()

    def _print_force_full_mode(self):
        """Вивід повідомлення про force full mode"""
        print("🗑️ FORCE FULL MODE: ігноруємо config.json")
        print(self.FULL_MODE_MESSAGE)

    def _handle_config_in_main_database(self, config_path):
        """Обробка config.json в main_database"""
        print("⚠️ Знайдено config.json в main_database - це не коректно")
        print("🗑️ Видаляємо config.json з main_database")
        try:
            config_path.unlink()
            print("✅ config.json видалено з main_database")
        except Exception as e:
            print(f"⚠️ Не вдалося видалити: {e}")

    def _load_config_file(self, config_path):
        """Завантаження параметрів з config.json"""
        try:
            # Validate path to prevent path traversal
            if not isinstance(config_path, Path):
                config_path = Path(config_path)

            # Ensure path is within expected directory
            config_path = config_path.resolve()
            if not str(config_path).startswith(str(Path.cwd().resolve())):
                raise ValueError("Config path outside working directory not allowed")

            with open(config_path) as f:
                content = f.read().strip()
                if content:
                    self._parse_config_data(json.loads(content))
                else:
                    print("⚠️ config.json порожній, використовуємо ПОВНОЦІННІ параметри за замовчуванням")
                    print(self.FULL_MODE_MESSAGE)
        except json.JSONDecodeError as e:
            print(f"⚠️ Помилка читання config.json: {e}")
            print("📊 Використовуємо ПОВНОЦІННІ параметри за замовчуванням")

    def _parse_config_data(self, config_data):
        """Парсинг даних з config.json"""
        test_mode = config_data.get('test_mode', {})

        if test_mode.get('enabled'):
            self.TEST_TICKER = test_mode.get('test_ticker')
            self.TEST_TARGET = test_mode.get('test_target')
            self.REDUCED_EPOCHS = test_mode.get('epochs') or 5
            self.MAX_ITERATIONS = test_mode.get('max_iterations') or 5
            self.test_mode = test_mode
            print(f"🧪 ТЕСТОВИЙ РЕЖИМ: {self.TEST_TICKER} | {self.TEST_TARGET} | epochs={self.REDUCED_EPOCHS}")
        else:
            print("⚠️ config.json порожній або test_mode disabled, використовуємо ПОВНОЦІННІ параметри")
            print(self.FULL_MODE_MESSAGE)

    def _print_full_mode_message(self):
        """Вивід повідомлення про full mode"""
        print("📊 Повноцінний режим: немає config.json")
        print("   → Тренуємо всі тікери, всі таргети, повні епохи")

    def _print_loaded_params(self):
        """Вивід завантажених параметрів"""
        print("\n" + "=" * 80)
        print("📥 ПАРАМЕТРИ ЗАВАНТАЖЕНО")
        print("=" * 80)
        print(f"  Режим: {'TEST' if self.test_mode else 'FULL'}")

        if self.test_mode:
            print(f"  🧪 TEST_TICKER: {self.TEST_TICKER}")
            print(f"  🧪 TEST_TARGET: {self.TEST_TARGET}")
            print(f"  ⚡ REDUCED_EPOCHS: {self.REDUCED_EPOCHS}")
            print(f"  ⚡ MAX_ITERATIONS: {self.MAX_ITERATIONS}")
        else:
            print("  📊 Тікери: всі з конфігу")
            print("  📊 Таргети: всі з даних")
            print("  📊 Епохи: 50 (повноцінний режим)")
            print("  📊 Ітерації: 100 (повноцінний режим)")
        print("=" * 80 + "\n")
