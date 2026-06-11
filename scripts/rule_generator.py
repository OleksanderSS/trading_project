# scripts/rule_generator.py

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
sys.path.append(str(Path(__file__).parent.parent))


# This script is a wrapper to use the consolidated ContextRuleGenerator.
# Usage (if running from root):
#   python scripts/rule_generator.py
