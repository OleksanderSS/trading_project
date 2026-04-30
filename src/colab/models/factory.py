"""Model factory for creating different model types"""

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False


def create_model(model_type, input_size):
    """Create model based on type with fallback to sklearn if torch unavailable"""
    if not TORCH_AVAILABLE:
        from .sklearn_fallback import create_sklearn_fallback_model
        return create_sklearn_fallback_model(model_type, input_size)
    
    from .torch_models import create_torch_model
    return create_torch_model(model_type, input_size)
