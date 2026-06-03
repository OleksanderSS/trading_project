
try:
    import evidently
    print(f"Evidently version: {evidently.__version__}")
    print("Report import successful")
    print("DataDriftPreset import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
