from abc import abstractmethod
from typing import Tuple

class BaseReward:
    def __init__(self):
        pass


    @abstractmethod
    def evaluate_log(self, observation, plan, next_observation) -> Tuple[float, bool]:
        pass

    @abstractmethod
    def evaluate_exp(self, state, action, next_action) -> Tuple[float, bool]:
        pass

