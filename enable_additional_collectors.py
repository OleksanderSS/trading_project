import yaml

# Read the YAML file
with open('src/config/collectors.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# List of additional working collectors to enable
additional_collectors = [
    'fred',
    'google_news',
    'huggingface',
    'reddit_sentiment',
    'rss',
    'sec_filings'
]

# Enable each additional working collector
for collector_name in additional_collectors:
    if 'collectors' in config and collector_name in config['collectors']:
        config['collectors'][collector_name]['enabled'] = True
        print(f"Enabled {collector_name} collector")
    else:
        print(f"{collector_name} collector not found in config")

# Write back the YAML file
with open('src/config/collectors.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

print("\nConfiguration updated successfully")
