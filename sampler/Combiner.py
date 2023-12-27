from spider.elements.trajectory import Trajectory, FrenetTrajectory
from spider.elements.curves import ParametricCurve, ExplicitCurve
import numpy as np

class LatLonCombiner:
    '''
    这里准确来讲不能叫PVD，只能叫LatLon解耦
    PVD本质是路径和速度（标量速度）的分解
    换句话讲，分解为路径曲线的生成 和 速度曲线的生成

    在frenet框架下，也应该是l(s) 和 v(x) ，然后一起生成轨迹, v(x) 决定了每一时刻的里程数， 去l(s)上面去找速度积分得到对应的里程数
    里程是什么？里程是速度的积分，是曲线上长度的累积，不是s！s只是横轴！
    然而这里的代码，是l(s) 和 s(x)， s(x)本质上不是严格的由速度导出的里程数。
    命名来讲，横纵向解耦更符合
    '''
    def __init__(self, steps, dt):
        self.steps = steps
        self.dt = dt

    def combine(self, lat_generators, lon_generators):
        # todo: 这个函数特别耗时！！！
        ts = np.arange(self.steps) * self.dt
        candidate_trajectories = []
        for long_generator in lon_generators:
            ss, dss, ddss, dddss = [ long_generator(ts, order) for order in range(4)] ## ss是绝对坐标
            # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
            s0 = ss[0]
            ss_relative = ss - s0
            for lat_generator in lat_generators:
                ls, dls, ddls, dddls = [lat_generator(ss_relative, order) for order in range(4)]

                traj = FrenetTrajectory(self.steps, self.dt)
                traj.t = ts
                traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = ss, dss, ddss, dddss
                traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = ls, dls, ddls, dddls
                candidate_trajectories.append(traj)

        return candidate_trajectories

class PVDCombiner:
    def __init__(self, steps, dt):
        # print("")
        self.steps = steps
        self.dt = dt

    def combine(self, path_generators, displacement_generators):
        '''
        path_generators: list of 参数化曲线，以displacement（累积曲线长度）为参数的x,y曲线参数方程
        displacement_generator: list of 显式曲线方程，表示displacement关于时间的函数关系，如果是speed profile,需要先进行积分
        这里有个点需要说明的是，这里的displacement不能直接理解为frenet坐标里面的s，因为s是沿着referenceline的里程，
        而displacement是沿着已规划好的path的里程，积累曲线长度
        这里很容易犯一个误区，就比如生成的path是frenet坐标下的，然后displacement_generator是speed_profile生成的
        然而speedprofile积分生成的是笛卡尔坐标下的距离，不是frenet下的path的距离！
        '''
        if not all([isinstance(path_generator, ParametricCurve) for path_generator in path_generators]):
            raise NotImplementedError("Only support ParametricCurve for path in PVDCombiner for now...")
        if not all([isinstance(displacement_generator, ParametricCurve) for displacement_generator in displacement_generators]):
            raise NotImplementedError("Only support Curve for displacement in PVDCombiner for now...")

        ts = [self.dt * i for i in range(self.steps)]
        candidate_trajectories = []

        for displacement_generator in displacement_generators:
            # todo: 改向量化的ts
            ss, dss, ddss, dddss = [[displacement_generator(t, order) for t in ts] for order in range(4)] ## ss是绝对坐标
            # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
            s0 = ss[0]
            for path_generator in path_generators:
                # 可以把ss一起放进去，不用一个个扔进去
                xys, dxys, ddxys, dddxys = [[path_generator(s-s0, order) for s in ss] for order in range(4)]

                traj = Trajectory(self.steps, self.dt)
                traj.t = ts
                traj.x, traj.y = xys
                traj.v = np.linalg.norm(dxys, axis=1)
                traj.heading = np.arctan2(dxys[:,1], dxys[:,0])
                traj.a = np.linalg.norm(ddxys, axis=1)

                traj.steer = []
                traj.curvature = []
                traj.centripetal_acceleration = []


                # traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = ss, dss, ddss, dddss
                # traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = ls, dls, ddls, dddls
                candidate_trajectories.append(traj)

        return candidate_trajectories

