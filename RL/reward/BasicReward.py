from abc import abstractmethod
from typing import Tuple

class BasicReward:
    def __init__(self):
        pass

    def __call__(self, state=None, action=None, next_state=None) -> Tuple[float, bool]:
        return self.evaluate(state, action, next_state)

    @abstractmethod
    def evaluate(self, state=None, action=None, next_state=None) -> Tuple[float, bool]:
        pass