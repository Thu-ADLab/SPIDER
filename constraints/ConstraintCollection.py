from spider.param import *
from spider.elements.trajectory import Trajectory
import numpy as np



class ConstraintCollection:
    '''
    todo: 未来改个形式，变成cvt/transform那种的变换的叠加的形式
    '''
    # qzl: 这种形式有个弊端，只能处理上下界约束，如果是过程中可变约束怎么办
    traj_constraint_functions = {
        CONSTRIANT_SPEED_LB: lambda traj, config: np.all(np.array(traj.v) >= config['min_speed']),
        CONSTRIANT_SPEED_UB: lambda traj, config: np.all(np.array(traj.v) <= config['max_speed']),
        CONSTRIANT_ACCELERATION: lambda traj, config: np.all(np.array(traj.a) <= config["max_acceleration"]),
        CONSTRIANT_DECELERATION: lambda traj, config: np.all(np.array(traj.a) >= -config["max_deceleration"]),
        CONSTRIANT_CURVATURE: lambda traj, config: np.all(np.array(traj.curvature) <= config["max_curvature"]),
        CONSTRIANT_LATERAL_JERK: lambda traj, config: np.all(np.abs(np.array(traj.l_3dot)) <= config["max_lateral_jerk"]),
        CONSTRIANT_LONGITUDINAL_JERK: lambda traj, config: np.all(np.abs(np.array(traj.s_3dot)) <= config["max_longitudinal_jerk"]),
    }

    control_constraint_functions = {
    }


    def __init__(self, config: dict):
        self.config = config

        assert "constraint_flags" in config
        self.constraint_flags: set = config["constraint_flags"]

        # if "constraint_flags" in config:
        #     self.constraint_flags:set = config["constraint_flags"]
        # else:
        #     self.constraint_flags:set = {
        #         CONSTRIANT_SPEED_UB,
        #         CONSTRIANT_SPEED_LB,
        #         CONSTRIANT_ACCELERATION,
        #         CONSTRIANT_DECELERATION,
        #         CONSTRIANT_CURVATURE
        #     } # default

    def aggregate(self):
        '''
        把多个判断可行性的函数聚合成一个大函数
        todo: 想想看碰撞检测能不能融进来
        '''

        if self.config.get("output", OUTPUT_TRAJECTORY) == OUTPUT_TRAJECTORY:
            all_funcs = self.traj_constraint_functions
        else:
            all_funcs = self.control_constraint_functions

        funcs = [all_funcs[key] for key in self.constraint_flags]
        feasibility_function = lambda traj: np.all([func(traj, self.config) for func in funcs])
        return feasibility_function


    def formulate(self):
        '''
        形成优化算法里面的约束条件
        '''
        pass



if __name__ == '__main__':
    config = {
        "output": OUTPUT_TRAJECTORY,
        "steps": 50,
        "dt": 0.1,
        "ego_veh_length": 5.0,
        "ego_veh_width": 2.0,
        "max_speed": 60 / 3.6,
        "min_speed": 0,
        "max_acceleration": 10,
        "max_deceleration": 10,
        # "max_centripetal_acceleration" : 100,
        "max_curvature": 100,
        "end_s_candidates": (10, 20, 40, 60),
        "end_l_candidates": (-0.8, 0, 0.8),  # s,d采样生成横向轨迹 (-3.5, 0, 3.5), #
        "end_v_candidates": tuple(i * 60 / 3.6 / 3 for i in range(4)),  # 改这一项的时候，要连着限速一起改了
        "end_T_candidates": (1, 2, 4, 8)  # s_dot, T采样生成纵向轨迹
    }

    temp = ConstraintCollection(config)
    func = temp.aggregate()

    traj = Trajectory
    traj.v = [3,4,5000,6,7,8,6]

    x = func(traj)
    print(x)



    #
    # import timeit
    #
    # my_dict = {'name': 'John', 'age': 30, 'city': 'New York'}
    # my_dict.update({key:0 for key in range(10000)})
    # class my_object: name = "John"
    #
    #
    # # 使用字符串进行字典键索引
    # def dict_index():
    #     return my_dict['name']
    #
    #
    # # 使用对象属性访问
    # def object_attr():
    #     return my_object.name
    #
    #
    # # 测试性能
    # dict_time = timeit.timeit(dict_index, number=1000000)
    # object_time = timeit.timeit(object_attr, number=1000000)
    #
    # print(f"字典键索引耗时: {dict_time} 秒")
    # print(f"对象属性访问耗时: {object_time} 秒")
