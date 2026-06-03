from src.colab.models.model_factory import create_model

try:
    model = create_model('mlp', 10)
    print("Successfully created MLP model")
    
    model_lstm = create_model('lstm', 10)
    print("Successfully created LSTM model")
    
    print("Model creation verification passed.")
except Exception as e:
    print(f"Model creation failed: {e}")
