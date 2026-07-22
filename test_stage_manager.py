from src.pipeline.stages.stage_manager import StageManager

def test_manager():
    try:
        manager = StageManager()
        print("StageManager facade successfully instantiated!")
    except Exception as e:
        print(f"Error instantiating StageManager: {e}")

if __name__ == "__main__":
    test_manager()
