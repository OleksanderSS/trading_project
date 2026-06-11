
try:
    import evidently
    print(f"Evidently version: {evidently.__version__}")
    from evidently.report import Report
    print("Report import successful")
    from evidently.metric_preset import DataDriftPreset
    print("DataDriftPreset import successful")
except Exception as e:
    print(f"Import failed: {e}")
    import traceback
    traceback.print_exc()
