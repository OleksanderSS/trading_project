"""Colab environment setup"""

import os
import sys
from pathlib import Path

COLAB_BASE_PATH = "/content"


class ColabEnvironment:
    """Налаштування середовища Colab"""

    def __init__(self):
        # Автовизначення шляху для Colab
        
        # Спробуємо різні варіанти шляхів
        possible_paths = [
            Path(f"{COLAB_BASE_PATH}/drive/MyDrive/trading_project"),  # Google Drive
            Path(f"{COLAB_BASE_PATH}/trading_project"),                # Local Colab
            Path("G:/Мій диск/trading_project"),              # Windows path
            Path.cwd() / "trading_project"                    # Relative
        ]
        
        self.PROJECT_PATH = None
        for path in possible_paths:
            if path.exists():
                self.PROJECT_PATH = path
                break
        
        if not self.PROJECT_PATH:
            self.PROJECT_PATH = Path.cwd()  # Fallback
            print("⚠️ Використовуємо поточну директорію як PROJECT_PATH")
        
        # Додаємо src в sys.path
        src_path = self.PROJECT_PATH / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        print(f"📂 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📂 SRC_PATH: {src_path}")
        print(f"📂 SRC існує: {src_path.exists()}")
        
        self.SRC_PATH = src_path
        self.BATCH_NAME = "main_database"
        self.batch_dir = None
        self.models_dir = None

    def setup_paths(self):
        """Налаштування шляхів для Colab або локального середовища"""
        try:
            from google.colab import drive
            import signal

            # Таймаут 5 секунд для попытки подключения
            def timeout_handler(signum, frame):
                raise TimeoutError("Таймаут при подключении к Google Colab")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(5)

            try:
                drive.mount(f'{COLAB_BASE_PATH}/drive')
                signal.alarm(0)  # Отмена таймаута
                self.PROJECT_PATH = (
                    f'{COLAB_BASE_PATH}/drive/MyDrive/trading_project')
                print("✅ Google Drive підключено")
            except TimeoutError:
                signal.alarm(0)  # Отмена таймаута
                print(
                    "⚠️ Таймаут при подключении к Google Drive, "
                    "працюємо локально"
                )
                raise ImportError("Таймаут Colab")
        except (ImportError, TimeoutError):
            self.PROJECT_PATH = str(Path.cwd())
            print(
                "⚠️ Працюємо локально"
            )
        self.SRC_PATH = os.path.join(self.PROJECT_PATH, "src")
        for p in [self.PROJECT_PATH, self.SRC_PATH]:
            if p not in sys.path:
                sys.path.insert(0, p)

        print(f"📁 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📁 Current dir: {os.getcwd()}")

    def setup_batch_directory(self, batch_name=None):
        """Налаштування директорії для батча"""
        if batch_name:
            self.BATCH_NAME = batch_name

        self.batch_dir = Path(self.PROJECT_PATH) / "data" / \
            "colab" / "accumulated" / self.BATCH_NAME
        self.models_dir = self.batch_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        return self.batch_dir, self.models_dir
