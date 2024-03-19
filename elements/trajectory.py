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

    _sequential_properties = [
        'a', 'centripetal_acceleration', 'curvature', 'heading', 'jerk', 'l',
        's', 'steer', 'steer_velocity', 'v', 'x', 'y' , 'debug_info',
        #'t' # 注意，t不应该被裁剪、串联等等
    ]

    def __init__(self, steps, dt=0.1):
        super(Trajectory, self).__init__()
        self.steps = steps
        self.dt = dt
        self.t = np.arange(steps) * dt
        self.v = []
        self.a = []
        self.steer = []
        self.steer_velocity = []
        self.jerk = []
        self.centripetal_acceleration = []

        self.debug_info: dict = {} # to store some necessary debug information
        pass

    @classmethod
    def from_trajectory_array(cls, trajectory_array: np.ndarray, dt, calc_derivative: bool = False,
                              v0=0., heading0=0., a0=0., steer0=0., steer_velocity0=0.) -> 'Trajectory':
        '''
        从x y点序列变换trajectory类，并且可以计算高阶导数
        trajectory_array: shape of [N,2], ndarray([[x,y],[x,y]])
        calc_derivative: if True, then calculate the Higher-order kinematics parameters
        todo: 以后要加入自定义vehicle model的接口
        '''
        VehModel = Bicycle
        xs, ys = trajectory_array[:, 0], trajectory_array[:, 1]
        traj = cls(len(trajectory_array), dt)

        if calc_derivative:
            veh_model = VehModel(xs[0], ys[0], v0, a0, heading0, steer0, steer_velocity0)
            traj.derivative(veh_model, xs, ys)
        else:
            traj.x = list(xs)
            traj.y = list(ys)
        return traj

    @property
    def trajectory_array(self):
        return np.column_stack((self.x, self.y))

    def densify(self):
        # 时间维度的
        pass

    def truncate(self, steps_num):
        self.steps = steps_num # 有没有必要呢？
        for prop_name in self._sequential_properties:
            seq = getattr(self, prop_name)
            if isinstance(seq, dict):
                for key in seq:
                    seq[key] = seq[key][:steps_num]
                setattr(self, prop_name, seq)
            else:
                setattr(self, prop_name, seq[:steps_num])
        return self

    def __add__(self, other):
        if isinstance(other, Trajectory):
            return self.concat(other)
        else:
            raise ValueError("Addition not supported between Trajectory and non-Trajectory")

    def concat(self, trajectory):
        assert self.dt == trajectory.dt
        # self.steps = self.steps + trajectory.steps # 有没有必要呢？

        for prop_name in self._sequential_properties:
            seq1, seq2 = getattr(self, prop_name), getattr(trajectory, prop_name)
            if isinstance(seq1, dict):
                for key in seq1:
                    seq1[key] = np.concatenate((seq1[key], seq2[key]))
                setattr(self, prop_name, seq1)
            else:
                setattr(self, prop_name, np.concatenate((seq1, seq2)))
        return self



    def step(self, veh_model:Bicycle, accs, steers=None):
        # 要考虑轨迹第一个点是不是t=1*dt，这里认为是的
        self.clearState()
        self.appendStateFromModel(veh_model)
        for a, st in zip(accs,steers):
            veh_model.step(a,st,dt=self.dt)
            self.appendStateFromModel(veh_model)

    def derivative(self,veh_model:Bicycle,xs,ys): # todo:这一块整个逻辑写的很乱，后面可以考虑改一改
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
        self.curvature.append(veh_model.curvature)
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

    @classmethod
    def from_sl_array(cls, sl_array, dt, calc_derivative=True):
        pass
        # todo: 想一想怎么用插值曲线的方式计算s和l的导数。三次样条插值的话，边界条件是个问题。

    @property
    def trajectory_sl_array(self):
        return np.column_stack((self.s, self.l))


if __name__ == '__main__':
    a = Trajectory(50)
    b = isinstance(a, Path)
    pass