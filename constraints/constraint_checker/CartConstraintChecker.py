'''
笛卡尔坐标下的约束检查，利用的是笛卡尔坐标下的参数
'''

from spider.param import *
from spider.constraints.constraint_checker import BaseConstraintChecker
from spider.constraints.ConstraintCollection import ConstraintCollection

from spider.elements.trajectory import Trajectory


class CartConstriantChecker(BaseConstraintChecker):
    def __init__(self, config:dict, collision_checker=None):
        super(CartConstriantChecker, self).__init__(config)
        # todo:config的存在很混乱，以后要修改constraintchecker的输入定义

        if not ("constraint_flags" in config):
            self.config.update({
                "constraint_flags": set()
            })
            # self.config.update({
            #     "constraint_flags":{
            #         CONSTRIANT_SPEED_UB,
            #         CONSTRIANT_SPEED_LB,
            #         CONSTRIANT_ACCELERATION,
            #         CONSTRIANT_DECELERATION,
            #         CONSTRIANT_CURVATURE}
            # })

        self.kinematics_feasibility_check = ConstraintCollection(self.config).aggregate()  # 这是一个函数
        self.collision_checker = collision_checker

    def check_kinematics(self, trajectory:Trajectory):
        return self.kinematics_feasibility_check(trajectory)

    def check_collision(self, trajectory:Trajectory, predicted_perception):
        if self.collision_checker is None:
            collision = False
        else:
            collision: bool = self.collision_checker.check_trajectory(trajectory, predicted_perception)
        return collision


    def check(self, trajectory:Trajectory, predicted_perception) -> bool:
        feasible: bool = self.kinematics_feasibility_check(trajectory)
        collision: bool = self.check_collision(trajectory, predicted_perception)

        return feasible and (not collision)



