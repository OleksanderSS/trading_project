def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safely divide two numbers, returning a default value if denominator is 0."""
    return numerator / denominator if denominator != 0 else default
