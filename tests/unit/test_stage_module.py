"""Test stage module used by StageLoader unit tests."""

class DummyStage:
    def __init__(self, config_manager=None, db_manager=None, http_client_factory=None, normalizer=None, error_handler=None, brain=None):
        self.config_manager = config_manager
        self.db_manager = db_manager
        self.http_client_factory = http_client_factory
        self.normalizer = normalizer
        self.error_handler = error_handler
        self.brain = brain

    async def run(self, **kwargs):
        return {'status': 'completed', 'outputs': {}}
