from abc import abstractmethod
from .common import Objective, Constraints


class BaseOptimizer:
    def __init__(self, objective=None, constraints=None):
        self.objective = objective
        self.constraints = constraints
        pass

    def set_constraints(self, constraints):
        self.constraints = constraints

    def set_objective(self, objective):
        self.objective = objective

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass
