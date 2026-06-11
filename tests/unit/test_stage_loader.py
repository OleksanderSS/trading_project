
from src.pipeline.stage_loader import StageLoader


class FakeConfigManager:
    def __init__(self, stages):
        self._stages = stages

    def get_config(self, key, default=None):
        if key == 'training_pipeline':
            return self._stages
        return default


def test_stage_loader_loads_enabled_stage():
    # Prepare config with one enabled stage pointing to our test module
    stages = [
        {
            'name': 'dummy',
            'module': 'tests.unit.test_stage_module',
            'class': 'DummyStage',
            'enabled': True
        }
    ]

    config_manager = FakeConfigManager(stages)
    loader = StageLoader(config_manager)

    deps = {'config_manager': config_manager}
    loaded = loader.load_stages(None, deps)

    assert len(loaded) == 1
    assert hasattr(loaded[0], '_pipeline_stage_index')
