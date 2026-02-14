import gc
import sys

def clear_context():
    """Очистити контекст and пам'ять"""
    # Очистити великand withмandннand
    for name in list(globals().keys()):
        if name.startswith('_') or name in ['gc', 'sys']:
            continue
        obj = globals()[name]
        if sys.getsizeof(obj) > 1024 * 1024:  # > 1MB
            del globals()[name]
    
    # Збирання смandття
    gc.collect()
    print("Контекст очищено")

def optimize_imports():
    """Оптимandwithувати andмпорти"""
    import pandas as pd
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 50)

if __name__ == "__main__":
    clear_context()
    optimize_imports()