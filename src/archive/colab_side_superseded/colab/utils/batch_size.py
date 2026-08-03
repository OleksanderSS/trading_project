"""Batch size optimization based on memory"""


def get_optimal_batch_size(memory_percent, base_batch_size=32):
    """
    Розрахувати оптимальний розмір батча на основі доступної пам'яті

    Логіка:
    - Якщо пам'ять < 50%: використовуємо base_batch_size
    - Якщо пам'ять 50-75%: зменшуємо до base_batch_size // 2
    - Якщо пам'ять 75-90%: зменшуємо до base_batch_size // 4
    - Якщо пам'ять > 90%: зменшуємо до base_batch_size // 8 (мінімум 2)
    """
    if memory_percent < 50:
        return base_batch_size
    elif memory_percent < 75:
        return max(base_batch_size // 2, 8)
    elif memory_percent < 90:
        return max(base_batch_size // 4, 4)
    else:
        return max(base_batch_size // 8, 2)
