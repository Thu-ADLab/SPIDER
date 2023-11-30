'''
笛卡尔坐标下的约束检查，利用的是笛卡尔坐标下的参数
'''

from spider.param import *
from spider.constraints.constraint_checker import BaseConstraintChecker
from spider.constraints.ConstraintCollection import ConstraintCollection

from spider.elements.trajectory import Trajectory


class CartConstriantChecker(BaseConstraintChecker):
    def __init__(self, config:dict, collision_checker):
        super(CartConstriantChecker, self).__init__(config)

        if not ("constraint_flags" in config):
            self.config.update({
                "constraint_flags":{
                    CONSTRIANT_SPEED_UB,
                    CONSTRIANT_SPEED_LB,
                    CONSTRIANT_ACCELERATION,
                    CONSTRIANT_DECELERATION,
                    CONSTRIANT_CURVATURE}
            })

        self.kinematics_feasibility_check = ConstraintCollection(self.config).aggregate()  # 这是一个函数
        self.collision_checker = collision_checker  # 这是一个对象

    def check(self, trajectory:Trajectory, predicted_obstacles) -> bool:
        feasible: bool = self.kinematics_feasibility_check(trajectory)
        collision: bool = self.collision_checker.check_trajectory(trajectory, predicted_obstacles)

        return feasible and (not collision)



