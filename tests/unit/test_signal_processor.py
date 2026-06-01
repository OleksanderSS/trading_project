from src.trading.signal_processor import SignalProcessor


class DummyReport:
    def __init__(self, final_signal, confidence=0.0):
        self.final_signal = final_signal
        self.confidence = confidence


class DummyConsensusEngine:
    def __init__(self, report):
        self.report = report

    def generate_consensus(self, model_predictions=None, context_data=None, knn_results=None):
        return self.report


class DummyFilter:
    def apply(self, df):
        return df


def test_signal_processor_returns_raw_when_no_filter():
    processor = SignalProcessor(consensus_engine=DummyConsensusEngine(DummyReport('HOLD')))
    predictions = [{'ticker': 'TEST', 'predictions': [1.2]}]
    assert processor.prepare_predictions(predictions) == predictions


def test_signal_processor_generates_consensus_signal():
    processor = SignalProcessor(consensus_engine=DummyConsensusEngine(DummyReport('BUY', confidence=0.75)))
    predictions = [{'ticker': 'TEST', 'predictions': [1.2], 'selected_primary_model': 'base'}]
    signals = processor.generate_consensus_signals(predictions)
    assert len(signals) == 1
    assert signals[0]['ticker'] == 'TEST'
    assert signals[0]['final_signal'] == 'BUY'
    assert signals[0]['confidence'] == 0.75


def test_signal_processor_filters_predictions():
    dummy_filter = DummyFilter()
    processor = SignalProcessor(consensus_engine=DummyConsensusEngine(DummyReport('HOLD')), post_filter=dummy_filter)
    raw_predictions = [{'ticker': 'TEST', 'predictions': [1.2], 'metadata': {'foo': 'bar'}}]
    filtered = processor.prepare_predictions(raw_predictions)
    assert isinstance(filtered, list)
    assert filtered[0]['ticker'] == 'TEST'
    assert 'metadata' not in filtered[0]
