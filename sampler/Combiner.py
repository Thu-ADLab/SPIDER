from spider.elements.trajectory import Trajectory, FrenetTrajectory
from spider.elements.curves import ParametricCurve, ExplicitCurve, BezierCurve
from spider.sampler.common import LazyList
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

    def combine(self, lat_generators, lon_generators, calc_by_need=False):
        ts = np.arange(self.steps) * self.dt

        if not calc_by_need: # calculation directly
            candidate_trajectories = []
            for lon_generator in lon_generators:
                ss, dss, ddss, dddss = [lon_generator(ts, order) for order in range(4)]  ## ss是绝对坐标
                # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
                s0 = ss[0]
                ss_relative = ss - s0
                for lat_generator in lat_generators:
                    ls, dls, ddls, dddls = [lat_generator(ss_relative, order) for order in range(4)]

                    traj = FrenetTrajectory(self.steps, self.dt)
                    traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = ss, dss, ddss, dddss
                    traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = ls, dls, ddls, dddls
                    candidate_trajectories.append(traj)

        else:

            ###### define temp function to wrap as closure function #######
            # temp_func_s = lambda gens, idx, ts, order: gens[idx](ts, order)
            # temp_func_l = lambda gens, idx, s_kine, order: gens[idx](s_kine[0] - s_kine[0][0], order)  # ss_rel = ss - s0
            def temp_func_s(gens, idx, ts, order):
                # print("inside temp_func_s")
                return gens[idx](ts, order)

            def temp_func_l(gens, idx, s_kine, order):
                # print("inside temp_func_l")
                return gens[idx](s_kine[0] - s_kine[0][0], order)

            def _get_traj(steps, dt, s_kine, l_kine):
                # print("inside _get_traj")
                traj = FrenetTrajectory(steps, dt)
                traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = s_kine
                traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = l_kine
                return traj

            ###########################################

            candidate_trajectories = LazyList()
            for i in range(len(lon_generators)):
                s_kinematics = LazyList(
                    [LazyList.wrap_generator(temp_func_s, lon_generators, i, ts, order) for order in range(4)]
                )
                for j in range(len(lat_generators)):
                    l_kinematics = LazyList(
                        [LazyList.wrap_generator(temp_func_l, lat_generators, j, s_kinematics, order) for order in range(4)]
                    )
                    candidate_trajectories.append(
                        LazyList.wrap_generator(_get_traj, self.steps, self.dt, s_kinematics, l_kinematics)
                    )

        return candidate_trajectories

    def combine_one(self, lat_generator, lon_generator):
        ts = np.arange(self.steps) * self.dt
        ss, dss, ddss, dddss = [lon_generator(ts, order) for order in range(4)]
        s0 = ss[0]
        ss_relative = ss - s0

        ls, dls, ddls, dddls = [lat_generator(ss_relative, order) for order in range(4)]
        traj = FrenetTrajectory(self.steps, self.dt)
        traj.t = ts
        traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = ss, dss, ddss, dddss
        traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = ls, dls, ddls, dddls
        return traj


class PVDCombiner:
    def __init__(self, steps, dt):
        # print("")
        self.steps = steps
        self.dt = dt

    def combine(self, path_generators, displacement_generators, calc_by_need=False):
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
        if not all([isinstance(displacement_generator, ExplicitCurve) for displacement_generator in displacement_generators]):
            raise NotImplementedError("Only support Curve for displacement in PVDCombiner for now...")

        ts = np.arange(self.steps) * self.dt

        if not calc_by_need:
            candidate_trajectories = []

            for displacement_generator in displacement_generators:
                ss, vs, accs, jerks = [displacement_generator(ts, order) for order in range(4)] ## ss是绝对坐标
                # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
                s0 = ss[0]
                displacement = ss - s0
                vvs = vs ** 2
                for path_generator in path_generators:
                    # 可以把ss一起放进去，不用一个个扔进去
                    xys, dxys, ddxys = [path_generator(displacement, order) for order in range(3)]

                    traj = Trajectory(self.steps, self.dt)
                    traj.x, traj.y = xys.T
                    dx, dy = dxys.T  # the derivative of x,y to s(displacement)
                    traj.heading = np.arctan2(dy, dx)
                    ddx, ddy = ddxys.T
                    traj.curvature = np.abs(ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5

                    traj.v = vs
                    traj.a = accs
                    traj.jerk = jerks

                    # traj.steer = []
                    traj.centripetal_acceleration = vvs * traj.curvature

                    # debug_info
                    if isinstance(path_generator, BezierCurve):
                        traj.debug_info["control_points"] = path_generator.control_points

                    candidate_trajectories.append(traj)
        else:
            ###### define temp function to wrap as closure function #######
            def temp_func_d(gens, idx, ts, order):
                # print("inside temp_func_s")
                return gens[idx](ts, order)

            def temp_func_p(gens, idx, d_info, order):
                # print("inside temp_func_l")
                displacement = d_info[0] - d_info[0][0]
                return gens[idx](displacement, order)

            def temp_func_cp(gens, idx):
                return getattr(gens[idx], 'control_points', None) # None by default

            def _get_traj(steps, dt, d_info, p_info):
                ss, vs, accs, jerks = d_info
                xys, dxys, ddxys, control_points = p_info

                # print("inside _get_traj")
                traj = Trajectory(steps, dt)
                traj.x, traj.y = xys.T
                dx, dy = dxys.T  # the derivative of x,y to s(displacement)
                traj.heading = np.arctan2(dy, dx)
                ddx, ddy = ddxys.T
                traj.curvature = np.abs(ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
                traj.v = vs
                traj.a = accs
                traj.jerk = jerks
                traj.centripetal_acceleration = vs ** 2 * traj.curvature
                # debug_info
                if control_points is not None:
                    traj.debug_info["control_points"] = control_points

                return traj

            ###########################################

            candidate_trajectories = LazyList()
            for i in range(len(displacement_generators)):
                d_info = LazyList(
                    [LazyList.wrap_generator(temp_func_d, displacement_generators, i, ts, order) for order in range(4)]
                )
                for j in range(len(path_generators)):
                    p_info = LazyList(
                        [LazyList.wrap_generator(temp_func_p,path_generators, j, d_info, order) for order in range(3)]
                    ) # xy, dxy, ddxy
                    p_info.append(LazyList.wrap_generator(temp_func_cp, path_generators, j)) # control points
                    candidate_trajectories.append(
                        LazyList.wrap_generator(_get_traj, self.steps, self.dt, d_info, p_info)
                    )
        return candidate_trajectories

    def combine_one(self, path_generator, displacement_generator):
        if not isinstance(path_generator, ParametricCurve):
            raise NotImplementedError("Only support ParametricCurve for path in PVDCombiner for now...")
        if not isinstance(displacement_generator, ExplicitCurve):
            raise NotImplementedError("Only support Curve for displacement in PVDCombiner for now...")

        ts = np.arange(self.steps) * self.dt
        ss, vs, accs, jerks = [displacement_generator(ts, order) for order in range(4)]  ## ss是绝对坐标
        # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
        s0 = ss[0]
        displacement = ss - s0
        vvs = vs ** 2
        xys, dxys, ddxys = [path_generator(displacement, order) for order in range(3)]
        traj = Trajectory(self.steps, self.dt)
        traj.x, traj.y = xys.T
        dx, dy = dxys.T  # the derivative of x,y to s(displacement)
        traj.heading = np.arctan2(dy, dx)
        ddx, ddy = ddxys.T
        traj.curvature = np.abs(ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
        traj.v = vs
        traj.a = accs
        traj.jerk = jerks
        # traj.steer = []
        traj.centripetal_acceleration = vvs * traj.curvature
        # debug_info
        if isinstance(path_generator, BezierCurve):
            traj.debug_info["control_points"] = path_generator.control_points

        return traj
