class LabelGenerator:
    _instance = None  # Статичне поле для збереження єдиного екземпляру

    def __new__(cls, max_db_label=0):
        if cls._instance is None:
            cls._instance = super(LabelGenerator, cls).__new__(cls)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self, max_db_label):
        if self.__initialized:
            return
        self.__current_value = max_db_label
        self.__initialized = True

    def generate(self):
        self.__current_value += 1
        return self.__current_value
