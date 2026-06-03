import pytest
from src.meta_learning.security.constraint_engine import SecurityConstraintEngine, ConstraintType, ConstraintSeverity
from src.core.error_handling.error_handler import IErrorHandler

class MockErrorHandler(IErrorHandler):
    def __init__(self):
        self.errors = []
    
    def handle_error(self, error: Exception, context: dict = None):
        self.errors.append((error, context))

def test_constraint_engine_initialization():
    engine = SecurityConstraintEngine()
    assert engine is not None
    assert len(engine._constraints) > 0

def test_constraint_validation_error_handling():
    mock_handler = MockErrorHandler()
    engine = SecurityConstraintEngine(error_handler=mock_handler)
    
    # Створюємо валідатор, який завжди викликає помилку
    def failing_validator(context):
        raise ValueError("Simulated validation error")
    
    # Додаємо його як новий constraint
    from src.meta_learning.security.constraint_engine import Constraint
    engine.add_constraint(Constraint(
        name="failing_constraint",
        constraint_type=ConstraintType.POSITION_SIZE,
        validator=failing_validator,
        severity=ConstraintSeverity.ERROR,
        description="Failing constraint for testing"
    ))
    
    # Запускаємо валідацію
    result = engine.validate_action("agent1", {"position_size": 100})
    
    # Перевіряємо, чи помилка була оброблена через ErrorHandler
    assert len(mock_handler.errors) == 1
    assert isinstance(mock_handler.errors[0][0], ValueError)
    assert not result['allowed']
