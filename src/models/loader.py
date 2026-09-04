"""
ModelLoaderStrategy: Strategy pattern for loading models from various sources

This module encapsulates model loading logic with multiple fallback strategies,
eliminating the 60+ lines of nested try-except blocks in Stage 5 Prediction.

Usage:
    loader = ModelLoaderStrategy(logger)
    model = loader.load_model(model_metadata)
    if model:
        predictions = model.predict(features)
"""
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from src.config.unified_config_manager import get_current_config
from src.core.error_handling.error_handler import ModelLoadingError
from src.core.logging.logger import ProjectLogger
from src.models.neural.sequence_builder import SequenceBuilder
from src.utils.artifact_security import resolve_trusted_artifact_path

KERAS_EXTENSION = '.keras'


def shape_input_for_keras(x_input, model_input_shape, sequence_builder=None):
    """Shape a 2-D feature frame for a model that declares a 3-D input.

    The number of timesteps is read from THE MODEL, not from a list of type
    names. That list -- ['lstm', 'gru', 'transformer'] -- was a second copy of
    a fact the model already carries, and it drifted: the Colab trainer's
    `_SEQUENCE_MODEL_TYPES` is {'cnn', 'lstm', 'gru', 'transformer'} with
    `_SEQUENCE_WINDOW = 20`, so a Colab-trained CNN was built on twenty
    timesteps and served one:

        Negative dimension size caused by subtracting 3 from 1
        inputs=tf.Tensor(shape=(32, 1, 64))

    Twenty such failures in the 2026-08-11 run, every one a CNN. The list was
    not wrong when written -- the local CNNModel really does use a single
    timestep -- it was wrong to exist twice.

    CNN was the lucky case. Conv1D with a kernel of 3 cannot span a length of
    1, so it crashed and we found out. An LSTM handed one timestep of a
    twenty-step window returns a plausible number instead.
    """
    expected_timesteps = model_input_shape[1]

    if not expected_timesteps or expected_timesteps <= 1:
        # The model itself declares a single timestep: (n, 1, n_features),
        # which is how the local CNNModel shapes its data -- NOT
        # (n, n_features, 1). An earlier version expanded on the last axis
        # for 'cnn' and fed it a transposed, incompatible shape.
        return np.expand_dims(x_input, axis=1)

    if len(x_input) >= expected_timesteps and sequence_builder is not None:
        return sequence_builder.build_sequences(
            x_input, window_size=expected_timesteps, step_size=1
        )

    # Too little history for one full window: repeat the most recent row
    # rather than quietly handing over a shorter sequence.
    last_row = x_input[-1:]
    repeated = np.repeat(last_row, expected_timesteps, axis=0)
    return repeated.reshape(1, expected_timesteps, -1)


class ModelLoaderStrategy:
    """
    Encapsulates model loading logic with multiple fallback strategies.

    Tries different strategies in order:
    1. Load from local filesystem (joblib)
    2. Load from Colab mounted drive
    3. Load consensus meta-model (fallback)
    4. Load stacked ensemble (final fallback)

    Each strategy is tried in order, and the first successful one is returned.
    """

    def __init__(self, logger: Any | None=None):
        """
        Initialize ModelLoaderStrategy.

        Args:
            logger: Optional logger instance (creates new if not provided)
        """
        self.logger = logger or ProjectLogger.get_logger('ModelLoaderStrategy')
        self.loaders: list[Callable] = [self._load_local_model, self.
            _load_colab_model, self._load_consensus_model, self.
            _load_stacked_ensemble]

    def _try_direct_load(self, model_path: str, model_meta: dict[str, Any], model_id: str) -> Any | None:
        """Try loading model directly from path."""
        if not model_path:
            return None
        try:
            model = self.load_path(model_path, model_meta)
            if model is not None:
                self.logger.info(f'✅ Loaded model {model_id} directly from path')
                return model
        except ModelLoadingError as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Direct load failed for {model_path}: {e}')
        except Exception as e:  # noqa: BLE001 - multi-strategy loader fallback, deliberately broad, always logged
            self.logger.warning(f'Direct load error for model {model_id} at {model_path}: {e}. Trying fallbacks.')
        return None

    def _try_loader(self, loader, model_path: str, model_meta: dict[str, Any], model_id: str) -> Any | None:
        """Try loading model with a specific loader."""
        try:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Trying loader: {loader.__name__}')
            model = loader(model_path, model_meta)
            if model is not None:
                self.logger.info(f'✅ Loaded model {model_id} using {loader.__name__}')
                return model
        except ModelLoadingError as e:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Loader {loader.__name__} failed: {e}')
        except Exception as e:  # noqa: BLE001 - multi-strategy loader fallback, deliberately broad, always logged
            self.logger.error(f'Помилка завантажувача {loader.__name__} для моделі {model_id}: {e}', exc_info=True)
        return None

    def load_model(self, model_meta: dict[str, Any]) ->Any | None:
        """
        Try loading model using multiple strategies.

        Attempts each strategy in order until one succeeds.
        Returns None if all strategies fail.

        Args:
            model_meta: Model metadata dictionary with 'model_path', 'model_id', etc.

        Returns:
            Loaded model instance, or None if all strategies failed
        """
        model_path = model_meta.get('model_path', '')
        model_id = model_meta.get('model_id', 'unknown')
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Attempting to load model {model_id} from {model_path}')

        # Try direct load first
        model = self._try_direct_load(model_path, model_meta, model_id)
        if model is not None:
            return model

        # Try fallback loaders
        for loader in self.loaders:
            model = self._try_loader(loader, model_path, model_meta, model_id)
            if model is not None:
                return model

        self.logger.warning(f'❌ All loaders failed for model {model_id} (path: {model_path})')
        return None

    def load_path(self, model_path: str, meta: dict[str, Any]) ->Any | None:
        """
        Load a model directly from a given file path.
        Supports .joblib, .pkl, and .pt file formats.
        """
        if not model_path:
            return None
        try:
            path = resolve_trusted_artifact_path(model_path, must_exist=True)
            if path.suffix.lower() == '.joblib':
                return self._load_joblib(path)
            if path.suffix.lower() == '.pkl':
                return self._load_pickle(path)
            if path.suffix.lower() == '.pt':
                return self._load_torch_model(path, meta)
            if path.suffix.lower() in (KERAS_EXTENSION, '.h5'):
                return self._load_keras_model(path, meta)
        except ModelLoadingError:
            raise
        except (FileNotFoundError, ValueError) as e:
            raise ModelLoadingError(
                f'Unsafe or missing model artifact path {model_path}: {e}') from e
        except (TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise ModelLoadingError(
                f'Failed to load model from path {model_path}: {e}') from e
        raise ModelLoadingError(f'Unsupported model file suffix: {path.suffix}'
            )

    def _load_local_model(self, model_path: str, meta: dict[str, Any]
        ) ->Any | None:
        """
        Load model from local filesystem.

        Returns None if path is not local (e.g., Colab path).
        Raises FileNotFoundError if local path doesn't exist.
        """
        if not model_path or '/content/drive/' in model_path:
            return None
        try:
            resolve_trusted_artifact_path(model_path, must_exist=True)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Model not found at local path: {model_path}') from None
        except ValueError as e:
            raise ModelLoadingError(f'Unsafe local model path {model_path}: {e}'
                ) from e
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading local model from {model_path}')
        return self.load_path(model_path, meta)

    def _load_colab_model(self, model_path: str, meta: dict[str, Any]
        ) ->Any | None:
        """
        Load model from Colab mounted drive.

        Returns None if path is not a Colab path.
        """
        if '/content/drive/' not in model_path:
            return None
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading Colab model from {model_path}')
        return self.load_path(model_path, meta)

    def _load_consensus_model(self, model_path: str, meta: dict[str, Any]
        ) ->Any | None:
        """
        Fallback: Load consensus meta-model.
        """
        config = get_current_config()
        registry = config.get('models.trained_models_registry', {})
        consensus_path_str = registry.get('consensus_meta_model', 'data/trained_models/consensus_meta_model.pkl')
        consensus_path = Path(consensus_path_str)

        if not consensus_path.exists():
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'Consensus model not found at {consensus_path}')
            return None
        try:
            self.logger.info(f'Using consensus meta-model from registry: {consensus_path}')
            trusted_path = resolve_trusted_artifact_path(consensus_path,
                allowed_suffixes={'.pkl'}, must_exist=True)
            return joblib.load(str(trusted_path))  # NOSONAR
        except Exception as e:  # noqa: BLE001 - multi-strategy loader fallback, deliberately broad, always logged
            self.logger.error(f'Помилка завантаження consensus моделі: {e}', exc_info=True)
            raise RuntimeError(f"Failed to load consensus model: {e}") from e

    def _load_stacked_ensemble(self, model_path: str, meta: dict[str, Any]
        ) ->Any | None:
        """
        Final fallback: Create default stacked ensemble.
        """
        try:
            self.logger.info(
                'Creating default stacked ensemble as final fallback')
            from src.ensembling.stacked_ensemble import StackedEnsemble
            return StackedEnsemble()
        except Exception as e:  # noqa: BLE001 - multi-strategy loader fallback, deliberately broad, always logged
            self.logger.error(f'Помилка створення default ensemble: {e}', exc_info=True)
            raise RuntimeError(f"Failed to create default ensemble: {e}") from e

    def _load_joblib(self, path: Path) ->Any:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading joblib model from {path}')
        try:
            trusted_path = resolve_trusted_artifact_path(path,
                allowed_suffixes={'.joblib'}, must_exist=True)
            return joblib.load(str(trusted_path))  # NOSONAR
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise ModelLoadingError(
                f'Failed to load joblib model at {path}: {e}') from e

    def _load_pickle(self, path: Path) ->Any:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading pickle model from {path}')
        try:
            trusted_path = resolve_trusted_artifact_path(path,
                allowed_suffixes={'.pkl', '.pickle'}, must_exist=True)
            return joblib.load(str(trusted_path))  # NOSONAR
        except (joblib.externals.loky.process_executor.TerminatedWorkerError, EOFError, ImportError, AttributeError) as e1:
            try:
                import pickle
                trusted_path = resolve_trusted_artifact_path(path,
                    allowed_suffixes={'.pkl', '.pickle'}, must_exist=True)
                with open(trusted_path, 'rb') as f:
                    return pickle.load(f)  # NOSONAR
            except (pickle.UnpicklingError, EOFError, ImportError, AttributeError) as e2:
                raise ModelLoadingError(
                    f'Failed to load pickle model at {path} via both joblib ({e1}) and standard pickle ({e2})'
                    ) from e2

    def _load_keras_model(self, path: Path, meta: dict[str, Any]) ->Any:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading Keras model from {path}')
        try:
            from tensorflow.keras.models import load_model
            trusted_path = resolve_trusted_artifact_path(path,
                allowed_suffixes={KERAS_EXTENSION, '.h5'}, must_exist=True)
            try:
                model = load_model(str(trusted_path), compile=False,
                    safe_mode=True)
            except TypeError:
                # `safe_mode` is refused by older Keras, and the fallback is
                # not the same operation: without it, deserialisation may
                # execute lambdas stored in the artifact. Falling back is
                # right; falling back in SILENCE is not, because the log then
                # cannot distinguish a model loaded safely from one that was
                # not.
                self.logger.warning(
                    "Keras refused safe_mode for %s (older version); loading "
                    "WITHOUT it, which permits code embedded in the artifact.",
                    path.name,
                )
                model = load_model(str(trusted_path), compile=False)
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(f'✅ Keras model loaded directly: {path.name}')
            return self._wrap_keras_model(model, meta.get('model_type',
                path.stem))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            self.logger.warning(
                f'⚠️ Keras model deserialization failed for {path.name}: {str(e)[:100]}...'
                )
            self.logger.info(f'🔄 Creating fallback model for {path.name}')
            try:
                fallback_model = self._create_fallback_model(meta.get(
                    'model_type', path.stem))
                self.logger.info(f'✅ Fallback model created for {path.name}')
                return self._wrap_keras_model(fallback_model, meta.get(
                    'model_type', path.stem))
            except Exception as fallback_error:
                self.logger.error(
                    f'❌ Even fallback model creation failed for {path.name}: {fallback_error}'
                    )
                raise ModelLoadingError(
                    f'Keras model and fallback creation failed for {path.name}'
                ) from fallback_error

    def _try_standard_load(self, path: Path, custom_objects: dict):
        """Standard Keras model loading"""
        from tensorflow.keras.models import load_model
        trusted_path = resolve_trusted_artifact_path(path,
            allowed_suffixes={KERAS_EXTENSION, '.h5'}, must_exist=True)
        return load_model(str(trusted_path), compile=False, custom_objects=
            custom_objects)

    def _try_safe_mode_load(self, path: Path, custom_objects: dict):
        """Load with safe_mode=False for more permissive loading"""
        import tensorflow as tf
        trusted_path = resolve_trusted_artifact_path(path,
            allowed_suffixes={KERAS_EXTENSION, '.h5'}, must_exist=True)
        return tf.keras.models.load_model(str(trusted_path), compile=False,
            custom_objects=custom_objects, safe_mode=True)

    def _try_minimal_load(self, path: Path):
        """Minimal loading without custom objects"""
        from tensorflow.keras.models import load_model
        trusted_path = resolve_trusted_artifact_path(path,
            allowed_suffixes={KERAS_EXTENSION, '.h5'}, must_exist=True)
        return load_model(str(trusted_path), compile=False)

    def _create_fallback_model(self, model_type: str):
        """Create a simple fallback model when loading fails"""
        import numpy as np
        import tensorflow as tf
        self.logger.warning(
            f'Creating fallback model for {model_type} due to loading failure')
        model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation=
            'relu', input_shape=(10,)), tf.keras.layers.Dropout(0.2), tf.
            keras.layers.Dense(32, activation='relu'), tf.keras.layers.
            Dense(1, activation='sigmoid')])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics
            =['accuracy'])
        dummy_input = np.random.random((1, 10))
        dummy_output = np.random.random((1, 1))
        model.fit(dummy_input, dummy_output, epochs=1, verbose=0)
        return model

    def _wrap_keras_model(self, model, model_type: str):


        class KerasPredictor:

            def __init__(self, keras_model, model_type):
                self.model = keras_model
                self.model_type = model_type
                self.sequence_builder = SequenceBuilder(strategy='sliding_window')
                self.logger = ProjectLogger.get_logger('KerasPredictor')

            def predict(self, X):
                if hasattr(X, 'values'):
                    x_input = X.values
                else:
                    x_input = X
                # Convert to float to avoid "Invalid dtype: object" errors
                x_input = np.asarray(x_input, dtype=np.float32)

                if len(self.model.input_shape) == 3 and len(x_input.shape) == 2:
                    x_input = shape_input_for_keras(
                        x_input,
                        self.model.input_shape,
                        sequence_builder=self.sequence_builder,
                    )
                elif len(self.model.input_shape) == 2 and len(x_input.shape) == 3:
                    x_input = x_input.squeeze(axis=1)

                predictions = self.model.predict(x_input, verbose=0)
                return predictions.flatten()
        return KerasPredictor(model, model_type)

    def _load_torch_model(self, path: Path, meta: dict[str, Any]) ->Any | None:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f'Loading PyTorch model from {path}')
        try:
            import torch
            trusted_path = resolve_trusted_artifact_path(path,
                allowed_suffixes={'.pt', '.pth'}, must_exist=True)
            try:
                # Use weights_only=True for safety if supported by torch version
                # trust path is verified by resolve_trusted_artifact_path
                loaded_obj = torch.load(
                    trusted_path, map_location='cpu', weights_only=True)
            except TypeError:
                # Fallback for older torch versions; trusted_path is validated
                # by resolve_trusted_artifact_path.
                #
                # But dropping `weights_only=True` is a downgrade, not a
                # detail: it goes from reading tensors to unpickling whatever
                # the file contains, which can run code. The fallback is
                # correct and it must not be silent -- an artifact loaded this
                # way and one loaded safely looked identical in every log this
                # project has.
                self.logger.warning(
                    "torch refused weights_only for %s (older version); "
                    "loading WITHOUT it, which unpickles arbitrary objects "
                    "from the artifact.", path.name,
                )
                loaded_obj = torch.load(trusted_path, map_location='cpu')  # NOSONAR
            except (EOFError, ImportError, AttributeError, RuntimeError, OSError) as e:
                self.logger.warning(f'Initial torch.load failed: {e}')
                if not meta.get('allow_full_torch_object_load', False):
                    raise
                # Explicitly allowed full load for trusted internal artifacts
                # trust path is verified by resolve_trusted_artifact_path
                loaded_obj = torch.load(
                    trusted_path, map_location='cpu', weights_only=False)  # NOSONAR
            if isinstance(loaded_obj, dict):
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f'Loaded dict from {path}, resolving state_dict or wrapper'
                        )
                state_dict = None
                scaler = None
                model_type = meta.get('model_type', path.stem)
                if 'model_state_dict' in loaded_obj:
                    state_dict = loaded_obj['model_state_dict']
                    model_type = loaded_obj.get('model_type', model_type)
                    scaler = loaded_obj.get('scaler')
                else:
                    state_dict = loaded_obj
                input_size = self._extract_input_size(state_dict)
                pytorch_model = self._create_pytorch_model(model_type,
                    input_size)
                pytorch_model.load_state_dict(state_dict)
                return self._wrap_pytorch_model(pytorch_model, model_type,
                    scaler)
            else:
                return self._wrap_pytorch_model(loaded_obj, meta.get(
                    'model_type', path.stem))
        except (ValueError, TypeError, AttributeError, KeyError, ZeroDivisionError) as e:
            raise ModelLoadingError(
                f'Failed to load PyTorch model from {path}: {e}') from e

    def _extract_input_size(self, state_dict: dict[str, Any]) ->int:
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug('Extracting input size from state_dict')
        for key, val in state_dict.items():
            if 'weight' in key and hasattr(val, 'shape') and len(val.shape
                ) >= 2:
                return int(val.shape[1])
        self.logger.warning(
            'Unable to determine input size from state_dict, using fallback 47'
            )
        return 47

    def _create_light_model(self, input_size: int):
        """Create a light PyTorch model."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 1)
        )

    def _create_mlp_model(self, input_size: int):
        """Create an MLP PyTorch model."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def _create_lstm_model(self, input_size: int):
        """Create an LSTM PyTorch model."""
        import torch.nn as nn

        class LSTMModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.lstm = nn.LSTM(input_sz, 64, 2, batch_first=True)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                out, _ = self.lstm(x.unsqueeze(1))
                return self.fc(out[:, -1, :])
        return LSTMModel(input_size)

    def _create_gru_model(self, input_size: int):
        """Create a GRU PyTorch model."""
        import torch.nn as nn

        class GRUModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.gru = nn.GRU(input_sz, 64, 2, batch_first=True)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                out, _ = self.gru(x.unsqueeze(1))
                return self.fc(out[:, -1, :])
        return GRUModel(input_size)

    def _create_cnn_model(self, input_size: int):
        """Create a CNN PyTorch model."""
        import torch
        import torch.nn as nn

        class CNNModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                x = x.unsqueeze(1)
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                return self.fc(self.pool(x).squeeze(-1))
        return CNNModel(input_size)

    def _create_transformer_model(self, input_size: int):
        """Create a Transformer PyTorch model."""
        import torch.nn as nn

        class TransformerModel(nn.Module):
            def __init__(self, input_sz):
                super().__init__()
                self.embedding = nn.Linear(input_sz, 64)
                encoder_layer = nn.TransformerEncoderLayer(64, 4, dim_feedforward=128, batch_first=True)
                self.transformer = nn.TransformerEncoder(encoder_layer, 2)
                self.fc = nn.Linear(64, 1)

            def forward(self, x):
                x = self.embedding(x.unsqueeze(1))
                x = self.transformer(x)
                return self.fc(x[:, -1, :])
        return TransformerModel(input_size)

    def _create_autoencoder_model(self, input_size: int):
        """Create an Autoencoder PyTorch model."""
        import torch.nn as nn

        class AutoencoderModel(nn.Module):  # audit-ignore: AUTOENCODER_ROUTING_REVIEW
            def __init__(self, input_sz):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(input_sz, 64), nn.ReLU(), nn.Linear(64, 32))
                self.decoder = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

            def forward(self, x):
                encoded = self.encoder(x)
                return self.decoder(encoded)
        return AutoencoderModel(input_size)  # audit-ignore: AUTOENCODER_ROUTING_REVIEW

    def _create_default_model(self, input_size: int):
        """Create a default PyTorch model."""
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 64),
            nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1)
        )

    def _create_pytorch_model(self, model_type: str, input_size: int):
        from src.models.registry.model_registry import ModelRegistry

        # Determine model group dynamically from registry
        model_config = ModelRegistry.get_model_config(model_type.lower())
        model_group = model_config.get('type', 'light') if model_config else 'light'

        if model_group == 'light' or model_type == 'tabnet':
            return self._create_light_model(input_size)
        elif model_type == 'mlp':
            return self._create_mlp_model(input_size)
        elif model_type == 'lstm':
            return self._create_lstm_model(input_size)
        elif model_type == 'gru':
            return self._create_gru_model(input_size)
        elif model_type == 'cnn':
            return self._create_cnn_model(input_size)
        elif model_type == 'transformer':
            return self._create_transformer_model(input_size)
        elif model_type == 'autoencoder':
            return self._create_autoencoder_model(input_size)
        else:
            return self._create_default_model(input_size)

    def _wrap_pytorch_model(self, model, model_type: str, scaler=None):
        import torch


        class PyTorchPredictor:

            def __init__(self, pytorch_model, model_type, scaler=None):
                self.model = pytorch_model
                self.model_type = model_type
                self.scaler = scaler
                self.model.eval()

            def predict(self, X):
                if hasattr(X, 'values'):
                    if self.scaler is not None:
                        x_normalized = self.scaler.transform(X)
                    else:
                        x_normalized = X
                    x_tensor = torch.FloatTensor(x_normalized)
                    with torch.no_grad():
                        output = self.model(x_tensor)
                    if isinstance(output, torch.Tensor):
                        return output.cpu().numpy().flatten()
                    return output
        return PyTorchPredictor(model, model_type, scaler)

    def add_loader(self, loader: Callable, position: int=-1):
        """
        Add a custom loader strategy.

        Allows extending ModelLoaderStrategy with custom loaders.

        Args:
            loader: Callable that takes (model_path, metadata) and returns model or None
            position: Position to insert (default: -1 = before final fallback)
        """
        if position == -1:
            self.loaders.insert(len(self.loaders) - 1, loader)
        else:
            self.loaders.insert(position, loader)
        self.logger.info(f'Added custom loader: {loader.__name__}')


class ModelLoaderFactory:
    """
    Factory for creating ModelLoaderStrategy instances.

    Can be extended to create specialized loaders for different scenarios.
    """

    @staticmethod
    def create_loader(logger: logging.Logger | None=None
        ) ->ModelLoaderStrategy:
        """
        Create a standard ModelLoaderStrategy instance.

        Args:
            logger: Optional logger instance

        Returns:
            ModelLoaderStrategy instance
        """
        return ModelLoaderStrategy(logger)

    @staticmethod
    def create_colab_loader(logger: logging.Logger | None=None
        ) ->ModelLoaderStrategy:
        """
        Create a loader optimized for Colab environment.

        Prioritizes Colab mount points over local filesystem.

        Args:
            logger: Optional logger instance

        Returns:
            ModelLoaderStrategy optimized for Colab
        """
        loader = ModelLoaderStrategy(logger)
        loader.loaders = [loader._load_colab_model, loader.
            _load_local_model, loader._load_consensus_model, loader.
            _load_stacked_ensemble]
        return loader
