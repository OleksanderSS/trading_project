from src.risk.max_exposure_monitor import MaxExposureMonitor


def test_max_exposure_breach_calculation():
    config = {'risk_limits': {'max_total_exposure': 0.8}}
    monitor = MaxExposureMonitor(config)
    
    analysis = {'total_exposure': 0.9}
    breaches = monitor._check_exposure_breaches(analysis)
    assert len(breaches) == 1
    assert breaches[0] == "Total exposure breach"
    
    analysis_safe = {'total_exposure': 0.5}
    breaches_safe = monitor._check_exposure_breaches(analysis_safe)
    assert len(breaches_safe) == 0

def test_most_frequent_breach():
    monitor = MaxExposureMonitor()
    events = [
        {'type': 'asset'},
        {'type': 'sector'},
        {'type': 'asset'}
    ]
    frequent = monitor._get_most_frequent_breach(events)
    assert frequent == 'asset'
