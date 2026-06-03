import sys
from pathlib import Path

# Ensure project root is on sys.path so `src` imports resolve during tests
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
