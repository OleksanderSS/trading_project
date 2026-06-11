"""
Script to check HF_KEY availability through different methods.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("HF_KEY AVAILABILITY CHECK")
print("=" * 80)

# 1. Check without loading .env
print("\n1. Checking HF_KEY without loading .env...")
hf_key = os.getenv('HF_KEY')
if hf_key:
    print(f"✅ HF_KEY found: {hf_key[:10]}...{hf_key[-4:]}")
else:
    print("❌ HF_KEY not found")

# 2. Load .env and check
print("\n2. Loading .env and checking HF_KEY...")
env_path = project_root / '.env'
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded .env from {env_path}")
    hf_key = os.getenv('HF_KEY')
    if hf_key:
        print(f"✅ HF_KEY found: {hf_key[:10]}...{hf_key[-4:]}")
    else:
        print("❌ HF_KEY not found after loading .env")
else:
    print(f"⚠️ .env file not found at {env_path}")

# 3. Check through Secure Secrets Manager
print("\n3. Checking HF_KEY through Secure Secrets Manager...")
try:
    from src.core.security.secure_secrets_manager import SecureSecretsManager
    
    secrets_manager = SecureSecretsManager()
    secrets_manager.load_env_from_file(env_path)
    
    hf_key = os.getenv('HF_KEY')
    if hf_key:
        print(f"✅ HF_KEY found: {hf_key[:10]}...{hf_key[-4:]}")
    else:
        print("❌ HF_KEY not found through Secure Secrets Manager")
except Exception as e:
    print(f"❌ Failed to check through Secure Secrets Manager: {e}")

# 4. Check all environment variables
print("\n4. Checking all environment variables...")
env_vars = {k: v for k, v in os.environ.items() if 'KEY' in k or 'HF' in k}
for key, value in env_vars.items():
    if value:
        print(f"   {key}: {value[:10]}...{value[-4:]}")
    else:
        print(f"   {key}: (empty)")

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)
