import yaml

# Read the YAML file
with open('src/config/collectors.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Enable yahoo_finance collector
if 'collectors' in config and 'yahoo_finance' in config['collectors']:
    config['collectors']['yahoo_finance']['enabled'] = True
    print("Enabled yahoo_finance collector")
else:
    print("yahoo_finance collector not found in config")

# Write back the YAML file
with open('src/config/collectors.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("Configuration updated successfully")
