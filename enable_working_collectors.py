import yaml

# Read the YAML file
with open('src/config/collectors.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# List of working collectors to enable
working_collectors = [
    'vix',
    'put_call_ratio',
    'fear_greed',
    'aaii_sentiment',
    'cftc',
    'economic_calendar',
    'insider'
]

# Enable each working collector
for collector_name in working_collectors:
    if 'collectors' in config and collector_name in config['collectors']:
        config['collectors'][collector_name]['enabled'] = True
        print(f"Enabled {collector_name} collector")
    else:
        print(f"{collector_name} collector not found in config")

# Write back the YAML file
with open('src/config/collectors.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("\nConfiguration updated successfully")
