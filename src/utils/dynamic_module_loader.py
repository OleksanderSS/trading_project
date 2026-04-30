# utils/dynamic_module_loader.py

import importlib
from src.core.logging.logger import ProjectLogger

logger = ProjectLogger.get_logger("DynamicModuleLoader")

class DynamicModuleLoader:
    """
    Critical utility implementing the Inversion of Control (IoC) principle.

    This class allow the system to avoid hardcoded dependencies. Instead of 
    code creating instances of specific classes directly (e.g., `obj = MyClass()`), 
    it delegates instance creation to this loader. The specific classes to be 
    loaded are defined in YAML configuration files. This makes the system 
    extremely flexible: replacing, adding, or removing a component (e.g., a new 
    data collector or analyzer) requires only changing the configuration, 
    without modifying the core code.
    """

    @staticmethod
    def load_class(class_path: str):
        """
        Loads a class object by its full string path.

        Args:
            class_path: Full path to the class (e.g., 'src.collectors.yf_collector.YFCollector').

        Returns:
            The class object ready for instantiation, or None if not found.

        Raises:
            ImportError, AttributeError: If the module or class cannot be found.
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            logger.debug(f"Module '{module_path}' successfully loaded.")
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(f"Error dynamically loading class '{class_path}': {e}", exc_info=True)
            raise

    @staticmethod
    def load_instance(config: dict, *args, **kwargs):
        """
        Creates a class instance based on a configuration dictionary.

        Expects the dictionary to contain:
        - 'class_path': Path to the class for loading.
        - 'params' (optional): Dictionary with parameters for the class constructor.

        Args:
            config: Configuration dictionary.
            *args: Additional positional arguments passed to the constructor.
            **kwargs: Additional keyword arguments passed to the constructor.
                      These have higher priority and can overwrite values from `config['params']`.

        Returns:
            An instance of the configured class.
        """
        class_path = config.get('class_path')
        if not class_path:
            raise ValueError("Dynamic loading configuration must contain the 'class_path' key.")

        loaded_class = DynamicModuleLoader.load_class(class_path)
        
        # Merge parameters: kwargs passed to the method have priority
        constructor_params = config.get('params', {})
        final_params = {**constructor_params, **kwargs}
        
        logger.info(f"Creating instance of '{class_path}' with parameters: {list(final_params.keys())}")
        return loaded_class(*args, **final_params)
