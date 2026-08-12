"""A saved model's own input shape decides how many timesteps it is fed.

The predictor kept a list of type names -- ['lstm', 'gru', 'transformer'] --
to decide which models needed real sequences. The Colab trainer keeps its own:
`_SEQUENCE_MODEL_TYPES = {'cnn', 'lstm', 'gru', 'transformer'}` with
`_SEQUENCE_WINDOW = 20`. When the trainer's list grew and the predictor's did
not, Colab-trained CNNs were built on twenty timesteps and served one:

    Negative dimension size caused by subtracting 3 from 1
    inputs=tf.Tensor(shape=(32, 1, 64))

Twenty such failures in the 2026-08-11 run, every one a CNN. The list was not
wrong when written -- the LOCAL CNNModel really does use a single timestep --
it was wrong to exist twice.

CNN was the lucky case: Conv1D with a kernel of 3 cannot span a length of 1,
so it crashed and the defect surfaced. An LSTM handed one timestep of a
twenty-step window returns a plausible number instead.
"""
import numpy as np

from src.models.loader import shape_input_for_keras
from src.models.neural.sequence_builder import SequenceBuilder


def _builder():
    return SequenceBuilder(strategy='sliding_window')


def test_a_model_built_on_twenty_timesteps_receives_twenty():
    x = np.arange(50 * 8, dtype=np.float32).reshape(50, 8)

    out = shape_input_for_keras(x, (None, 20, 8), sequence_builder=_builder())

    assert out.ndim == 3
    assert out.shape[1] == 20
    assert out.shape[2] == 8


def test_a_model_built_on_one_timestep_receives_one():
    x = np.zeros((50, 8), dtype=np.float32)

    out = shape_input_for_keras(x, (None, 1, 8), sequence_builder=_builder())

    assert out.shape == (50, 1, 8)


def test_the_decision_does_not_depend_on_the_model_name():
    """The whole point: no type list, so no list to fall out of step."""
    x = np.zeros((50, 8), dtype=np.float32)

    windowed = shape_input_for_keras(x, (None, 20, 8), sequence_builder=_builder())
    flat = shape_input_for_keras(x, (None, 1, 8), sequence_builder=_builder())

    assert windowed.shape[1] == 20
    assert flat.shape[1] == 1


def test_short_history_is_padded_to_the_declared_window():
    """Fewer rows than the window must not silently become a shorter sequence."""
    x = np.arange(5 * 8, dtype=np.float32).reshape(5, 8)

    out = shape_input_for_keras(x, (None, 20, 8), sequence_builder=_builder())

    assert out.shape == (1, 20, 8)
    # Padding repeats the most recent row, not the oldest.
    assert np.allclose(out[0, -1], x[-1])
    assert np.allclose(out[0, 0], x[-1])


def test_exactly_one_window_of_history_is_enough():
    x = np.arange(20 * 8, dtype=np.float32).reshape(20, 8)

    out = shape_input_for_keras(x, (None, 20, 8), sequence_builder=_builder())

    assert out.shape[1] == 20
    assert out.shape[0] >= 1


def test_a_missing_timestep_declaration_falls_back_to_one():
    x = np.zeros((10, 4), dtype=np.float32)

    out = shape_input_for_keras(x, (None, None, 4), sequence_builder=_builder())

    assert out.shape == (10, 1, 4)
