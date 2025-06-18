import random
from typing import Dict, Tuple, Optional
from fragment import Fragment
import numpy as np

class FragmentsStorage:
    def __init__(self):
        self._fragments: Dict[int, Fragment] = {}
        self._next_id: int = 1

    def add_fragment(self, image_data:  np.ndarray, features: np.ndarray) -> int:
        """Додає новий фрагмент та повертає унікальний ID"""
        fragment_id = self._next_id
        self._fragments[fragment_id] = Fragment(img=image_data, features=features)
        self._next_id += 1
        return fragment_id

    def add_fragment_with_id(self, fragment_id: int, image_data:  np.ndarray, features: np.ndarray) -> None:
        """Додає фрагмент за заданим ID. Не змінює лічильник."""
        if fragment_id in self._fragments:
            raise ValueError(f"Fragment with ID {fragment_id} already exists.")
        self._fragments[fragment_id] = Fragment(img=image_data, features=features)

        # гарантуємо унікальність для майбутніх ID
        if fragment_id >= self._next_id:
            self._next_id = fragment_id + 1

    def get_fragment(self, fragment_id: int) -> Optional[Fragment]:
        """Повертає зображення і вектор ознак за ID"""
        return self._fragments.get(fragment_id)

    def get_random_fragment(self) -> Optional[Fragment]:
        return random.choice(list(self._fragments.values()))

    def get_fragment_as_dict(self, fragment_id: int) -> Optional[Dict]:
        """Повертає фрагмент у вигляді словника з image та feature"""
        fragment = self._fragments.get(fragment_id)
        return fragment.to_dict() if fragment else None

    def items(self):
        return self._fragments.items()

    def get_dict(self):
        return self._fragments

    def __len__(self):
        return len(self._fragments)
