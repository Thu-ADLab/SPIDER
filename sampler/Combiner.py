from spider.elements.trajectory import Trajectory, FrenetTrajectory


class PVDCombiner:
    '''
    这里准确来讲不能叫PVD，只能叫LatLon解耦
    PVD本质是路径和速度（标量速度）的分解
    换句话讲，分解为路径曲线的生成 和 速度曲线的生成

    在frenet框架下，也应该是l(s) 和 v(t) ，然后一起生成轨迹, v(t) 决定了每一时刻的里程数， 去l(s)上面去找速度积分得到对应的里程数
    里程是什么？里程是速度的积分，是曲线上长度的累积，不是s！s只是横轴！
    然而这里的代码，是l(s) 和 s(t)， s(t)本质上不是严格的由速度导出的里程数。
    命名来讲，横纵向解耦更符合
    '''
    def __init__(self, steps, dt):
        self.steps = steps
        self.dt = dt

    def combine(self, lat_generators, long_generators):
        ts = [self.dt * i for i in range(self.steps)]
        candidate_trajectories = []
        for long_generator in long_generators:
            ss, dss, ddss, dddss = [ [long_generator(t, order) for t in ts] for order in range(4)] ## ss是绝对坐标
            # ss_abs = [s + ego_s0 for s in ss]  # 注意，加上ego_s0这一步非常重要,从相对变为绝对frenet坐标
            s0 = ss[0]
            for lat_generator in lat_generators:
                ls, dls, ddls, dddls = [[lat_generator(s-s0, order) for s in ss] for order in range(4)]

                traj = FrenetTrajectory(self.steps, self.dt)
                traj.t = ts
                traj.s, traj.s_dot, traj.s_2dot, traj.s_3dot = ss, dss, ddss, dddss
                traj.l, traj.l_prime, traj.l_2prime, traj.l_3prime = ls, dls, ddls, dddls
                candidate_trajectories.append(traj)

        return candidate_trajectories
