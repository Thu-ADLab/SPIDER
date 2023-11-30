from abc import abstractmethod

class BaseConstraintChecker:
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def check(self, *args) -> bool:
        pass

