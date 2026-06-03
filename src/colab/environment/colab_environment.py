"""Colab environment setup and configuration"""

import os
import sys
from pathlib import Path

COLAB_BASE_PATH = "/content"


class ColabEnvironment:
    """Налаштування середовища Colab"""

    def __init__(self):
        """Ініціалізація середовища"""
        possible_paths = [
            Path(f"{COLAB_BASE_PATH}/drive/MyDrive/trading_project"),
            Path(f"{COLAB_BASE_PATH}/trading_project"),
            Path("G:/Мій диск/trading_project"),
            Path.cwd() / "trading_project"
        ]

        self.PROJECT_PATH: Path | None = None
        for path in possible_paths:
            if path.exists():
                self.PROJECT_PATH = path
                break

        if not self.PROJECT_PATH:
            self.PROJECT_PATH = Path.cwd()
            print("⚠️ Використовуємо поточну директорію як PROJECT_PATH")

        src_path = self.PROJECT_PATH / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))

        print(f"📂 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📂 SRC_PATH: {src_path}")
        print(f"📂 SRC існує: {src_path.exists()}")

        self.SRC_PATH: Path = src_path
        self.BATCH_NAME = "main_database"
        self.batch_dir = None
        self.models_dir = None

    def setup_paths(self):  # noqa: C901
        """
        Налаштування шляхів для Colab або локального середовища

        CodeScene/Ruff: Complex Method acceptable - Platform-specific setup requires
        multiple conditional branches to handle Colab vs local environments, signal
        availability on different platforms (Windows/Linux), and timeout handling.
        """
        try:
            from google.colab import drive

            # Platform-specific signal handling
            try:
                import signal

                def timeout_handler(_signum, _frame):  # type: ignore[no-untyped-def]
                    raise TimeoutError("Таймаут при подключении к Google Colab")

                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(5)  # type: ignore[attr-defined]

                try:
                    drive.mount(f'{COLAB_BASE_PATH}/drive')
                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)  # type: ignore[attr-defined]
                    self.PROJECT_PATH = Path(f'{COLAB_BASE_PATH}/drive/MyDrive/trading_project')
                    print("✅ Google Drive підключено")
                except TimeoutError:
                    if hasattr(signal, 'alarm'):
                        signal.alarm(0)  # type: ignore[attr-defined]
                    print(
                        "⚠️ Таймаут при подключении к Google Drive, "
                        "працюємо локально"
                    )
                    raise ImportError("Таймаут Colab") from None
            except ImportError:
                # No signal support (Windows), try direct mount
                drive.mount(f'{COLAB_BASE_PATH}/drive')
                self.PROJECT_PATH = Path(f'{COLAB_BASE_PATH}/drive/MyDrive/trading_project')
                print("✅ Google Drive підключено")
        except (ImportError, TimeoutError):
            self.PROJECT_PATH = Path.cwd()
            print("⚠️ Працюємо локально")

        if self.PROJECT_PATH:
            self.SRC_PATH = self.PROJECT_PATH / "src"
            for p in [str(self.PROJECT_PATH), str(self.SRC_PATH)]:
                if p not in sys.path:
                    sys.path.insert(0, p)

        print(f"📁 PROJECT_PATH: {self.PROJECT_PATH}")
        print(f"📁 Current dir: {os.getcwd()}")

    def setup_batch_directory(self, batch_name: str | None = None) -> tuple[Path, Path]:
        """Налаштування директорії для батча"""
        if batch_name:
            self.BATCH_NAME = batch_name

        if not self.PROJECT_PATH:
            self.PROJECT_PATH = Path.cwd()

        self.batch_dir = self.PROJECT_PATH / "data" / "colab" / "accumulated" / self.BATCH_NAME
        self.models_dir = self.batch_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        return self.batch_dir, self.models_dir
