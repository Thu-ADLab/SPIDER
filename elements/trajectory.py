import numpy as np
from spider.vehicle_model.bicycle import Bicycle


# 考虑一下要不要改成 traj = [pose]
class Path:
    def __init__(self):
        self.x = []
        self.y = []
        self.heading = []
        self.curvature = []
        self.s = []
        self.l = []

        self.offset_center = 0 # 轨迹点到车辆OBB几何中心的偏差，尤其注意如果是后轴中心则需要设置此值。 正值在几何中心前方，负值在几何中心后方
        self.csp = None
        pass

    def offsetAdjust(self,lon_offset):
        '''
        纵向调整，用于计算车辆纵向中轴线上任一点的path，主要用于计算几何中心/近似Disk的path

        :param lon_offset: 目标点到车辆几何中点的偏差。正值在几何中心前方，负值在几何中心后方
        :return:
        '''
        adjust = lon_offset-self.offset_center
        self.x = np.array(self.x) + adjust * np.cos(self.heading)
        self.y = np.array(self.y) + adjust * np.sin(self.heading)
        # TODO:完成

    def densify(self):
        pass

class Trajectory(Path):
    def __init__(self, steps, dt=0.1):
        super(Trajectory, self).__init__()
        self.steps = steps
        self.dt = dt
        self.t = []
        self.v = []
        self.a = []
        self.steer = []
        self.steer_velocity = []
        pass

    def densify(self):
        # 时间维度的
        pass

    def step(self, veh_model:Bicycle, accs, steers=None, steer_velocities=None):
        # 要考虑轨迹第一个点是不是t=1*dt，这里认为是的
        self.clearState()
        self.appendStateFromModel(veh_model)
        for a, st in zip(accs,steers):
            veh_model.step(a,st,dt=self.dt)
            self.appendStateFromModel(veh_model)

    def derivative(self,veh_model:Bicycle,xs,ys):
        # 要考虑轨迹第一个点是不是t=0，这里认为是的
        assert np.linalg.norm([veh_model.x-xs[0], veh_model.y-ys[0]]) <= 0.2
        self.clearState()
        self.appendStateFromModel(veh_model)
        for x, y in zip(xs[1:],ys[1:]):
            veh_model.derivative(x, y, self.dt)
            self.appendStateFromModel(veh_model)


    def convert_to_acc_steer(self, wheelbase=3.0):
        # qzl: 还不完善，勉强能用
        if len(self.a) == 0:
            acc = np.diff(self.v) / self.dt
        else:
            acc = np.array(self.a)

        if len(self.steer) == 0:
            tansteer = np.array(self.curvature) * wheelbase
            steer = np.arctan(tansteer)
        else:
            steer = np.array(self.steer)

        assert len(steer) == len(acc)
        return np.column_stack((acc,steer))


    def appendStateFromModel(self,veh_model):
        self.x.append(veh_model.x)
        self.y.append(veh_model.y)
        self.v.append(veh_model.velocity)
        self.a.append(veh_model.acceleration)
        self.steer.append(veh_model.steer)
        self.heading.append(veh_model.heading)
        self.steer_velocity.append(veh_model.steer_velocity)
        if len(self.t) == 0:
            self.t.append(0.)
        else:
            self.t.append(self.t[-1]+self.dt)

    def clearState(self):
        self.x = []
        self.y = []
        self.v = []
        self.a = []
        self.steer = []
        self.heading = []
        self.steer_velocity = []

        self.curvature = []
        self.s = []
        self.l = []
        self.t = []

    @classmethod
    def from_trajectory_array(cls, trajectory_array):
        pass


class FrenetTrajectory(Trajectory):
    def __init__(self, steps, dt=0.1):
        super(FrenetTrajectory, self).__init__(steps, dt)
        self.s, self.s_dot, self.s_2dot, self.s_3dot = [], [], [], []
        self.l, self.l_dot, self.l_2dot, self.l_3dot = [], [], [], []
        self.l, self.l_prime, self.l_2prime, self.l_3prime = [], [], [], []
        # self.s, self.ds, self.dds, self.ddds = [],[],[],[]
        # self.l, self.dl, self.ddl, self.dddl = [], [], [], []

    def clearFrenetState(self):
        self.s, self.s_dot, self.s_2dot, self.s_3dot = [], [], [], []
        self.l, self.l_dot, self.l_2dot, self.l_3dot = [], [], [], []
        self.l, self.l_prime, self.l_2prime, self.l_3prime = [], [], [], []

    def clearState(self):
        super(FrenetTrajectory, self).clearState()
        self.clearFrenetState()


if __name__ == '__main__':
    a = Trajectory(50)
    b = isinstance(a, Path)
    pass