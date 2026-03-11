
import importlib
import logging

logger = logging.getLogger(__name__)

class DynamicModuleLoader:
    """
    Критично важлива утиліта, що реалізує принцип інверсії контролю (IoC).

    Цей клас дозволяє системі уникати жорстко закодованих залежностей. Замість того, щоб
    код створював екземпляри конкретних класів напряму (напр., `obj = MyClass()`), він
    делегує створення цих екземплярів цьому завантажувачу. Конкретні класи, які потрібно
    завантажити, визначаються в конфігураційних файлах YAML. Це робить систему надзвичайно
    гнучкою: для заміни, додавання або видалення компонента (напр., нового збирача даних
    або аналізатора) достатньо змінити конфігурацію, не торкаючись основного коду.
    """

    @staticmethod
    def load_class(class_path: str):
        """
        Завантажує об'єкт класу за його повним рядковим шляхом.

        Args:
            class_path: Повний шлях до класу (наприклад, 'src.collectors.yf_collector.YFCollector').

        Returns:
            Об'єкт класу, готовий до створення екземпляра, або None, якщо клас не знайдено.

        Raises:
            ImportError, AttributeError: Якщо модуль або клас не може бути знайдений.
        """
        try:
            module_path, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            logger.debug(f"Модуль '{module_path}' успішно завантажено.")
            return getattr(module, class_name)
        except (ImportError, AttributeError, ValueError) as e:
            logger.error(f"Помилка динамічного завантаження класу '{class_path}': {e}", exc_info=True)
            raise

    @staticmethod
    def load_instance(config: dict, *args, **kwargs):
        """
        Створює екземпляр класу на основі конфігураційного словника.

        Очікує, що словник містить:
        - 'class_path': Шлях до класу для завантаження.
        - 'params' (опціонально): Словник з параметрами для конструктора класу.

        Args:
            config: Словник конфігурації.
            *args: Додаткові позиційні аргументи, що передаються в конструктор.
            **kwargs: Додаткові іменовані аргументи, що передаються в конструктор.
                      Вони мають вищий пріоритет і можуть перезаписати значення з `config['params']`.

        Returns:
            Екземпляр сконфігурованого класу.
        """
        class_path = config.get('class_path')
        if not class_path:
            raise ValueError("Конфігурація для динамічного завантаження повинна містити ключ 'class_path'.")

        Class_ = DynamicModuleLoader.load_class(class_path)
        
        # Об'єднуємо параметри: пріоритет мають kwargs, передані в метод
        constructor_params = config.get('params', {})
        final_params = {**constructor_params, **kwargs}
        
        logger.info(f"Створення екземпляра класу '{class_path}' з параметрами: {list(final_params.keys())}")
        return Class_(*args, **final_params)

