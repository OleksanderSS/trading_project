
with open('src/data/management/data_manager.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Відновлюємо втрачене визначення класу DataManager
if "class DataManager(IDatabaseManager):" not in content:
    # Знаходимо де закінчується IDatabaseManager
    insertion_point = content.find("    def _validate_table_name")
    if insertion_point != -1:
        new_content = content[:insertion_point] + "class DataManager(IDatabaseManager):\n    \"\"\"Implementation of IDatabaseManager using DuckDB.\"\"\"\n\n" + content[insertion_point:]
        with open('src/data/management/data_manager.py', 'w', encoding='utf-8') as f:
            f.write(new_content)
