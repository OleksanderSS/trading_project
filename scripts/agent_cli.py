#!/usr/bin/env python3
"""
Центральний CLI для запуску агентів (Agent CLI Router).
Дозволяє запускати агентів без засмічення кореневої папки.

Використання:
    python scripts/agent_cli.py <ім'я_агента_без_run_agent_>
Приклад:
    python scripts/agent_cli.py memory
    (це еквівалентно старому запуску python run_agent_memory.py)
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

# Визначаємо шляхи
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = PROJECT_ROOT / ".archive_temp" / "agent_scripts"

def main():
    parser = argparse.ArgumentParser(description="DEAN-OS Agent CLI Router")
    parser.add_argument("agent_name", help="Назва агента для запуску (наприклад: memory, ops, orchestrator)")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Додаткові аргументи для агента")
    
    args = parser.parse_args()
    
    # Формуємо назву файлу
    script_name = f"run_agent_{args.agent_name}.py"
    
    # Шукаємо скрипт в архіві (якщо він там є)
    script_path = ARCHIVE_DIR / script_name
    
    if not script_path.exists():
        print(f"❌ Помилка: Агент '{args.agent_name}' не знайдений.")
        print(f"Очікуваний шлях: {script_path}")
        sys.exit(1)
        
    print(f"🚀 Запуск агента: {args.agent_name}...")
    
    # Запускаємо архівний скрипт з переданими аргументами
    cmd = [sys.executable, str(script_path)] + args.args
    
    try:
        # Встановлюємо PYTHONPATH, щоб скрипти з архіву могли імпортувати src
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        
        result = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⏹️ Виконання перервано користувачем")
        sys.exit(130)

if __name__ == "__main__":
    main()
