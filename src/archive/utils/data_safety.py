

def safe_rolling(series, window, min_periods=None, agg="mean", **kwargs):
    """
    Apply rolling window calculation with shift(1)
    to prevent including the current (future) observation.
    """
    rolling_obj = series.rolling(window=window, min_periods=min_periods, **kwargs)
    if agg == "mean":
        return rolling_obj.mean().shift(1)
    elif agg == "std":
        return rolling_obj.std().shift(1)
    elif agg == "sum":
        return rolling_obj.sum().shift(1)
    elif agg == "max":
        return rolling_obj.max().shift(1)
    elif agg == "min":
        return rolling_obj.min().shift(1)
    elif agg == "apply":
        func = kwargs.pop("func", None)
        if func:
            return rolling_obj.apply(func, **kwargs).shift(1)
        return rolling_obj.mean().shift(1)
    return rolling_obj.mean().shift(1)


def safe_expanding(series, min_periods=1, agg="mean", **kwargs):
    """
    Apply expanding window calculation. For expanding, we shift 1
    as closed='left' isn't supported for expanding windows.
    """
    expanding_obj = series.expanding(min_periods=min_periods, **kwargs)
    if agg == "mean":
        return expanding_obj.mean().shift(1)
    elif agg == "std":
        return expanding_obj.std().shift(1)
    return expanding_obj.mean().shift(1)
