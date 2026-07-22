import yaml

# Read the YAML file
with open('src/config/collectors.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Enable VIX collector
if 'collectors' in config and 'vix' in config['collectors']:
    config['collectors']['vix']['enabled'] = True
    print("Enabled VIX collector")
else:
    print("VIX collector not found in config")

# Write back the YAML file
with open('src/config/collectors.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("Configuration updated successfully")
