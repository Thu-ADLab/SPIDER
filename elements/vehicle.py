from spider.elements.vector import Vector3D
import numpy as np


class Location:
    def __init__(self,x=0.0,y=0.0,z=0.0):
        self.x, self.y, self.z = x,y,z


class Rotation:
    def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
        self.pitch, self.yaw, self.roll = pitch,yaw,roll
        # todo: 加入四元数及其转换


class Transform:  # qzl:是叫transform好还是叫pose好呢？
    def __init__(self, location=None, rotation=None):
        self.location = Location() if location is None else location
        self.rotation = Rotation() if rotation is None else rotation


class VehicleState:
    # length = 5.0
    # width = 2.0
    # 2D ONLY for now
    def __init__(self, transform:Transform=None, velocity:Vector3D=None, acceleration:Vector3D=None,
                 length=5.0, width=2.0):

        self.transform = transform if not (transform is None) else Transform()
        self.velocity = velocity if not (velocity is None) else Vector3D()
        self.acceleration = acceleration if not (acceleration is None) else Vector3D()
        # todo: 标量值和矢量值如何在名字上区分呢？
        # self.speed = np.linalg.norm([velocity.x, velocity.y])
        # self.acc = np.linalg.norm([acceleration.x, acceleration.y])
        # self.curvature = (acceleration.y * velocity.x - acceleration.x * velocity.y) / self.speed
        self.kinematics = KinematicState()
        self.calc_kinematics()

        self.length = length
        self.width = width

    def calc_kinematics(self):
        self.kinematics.x = self.transform.location.x
        self.kinematics.y = self.transform.location.y
        self.kinematics.speed = np.linalg.norm([self.velocity.x, self.velocity.y])
        self.kinematics.yaw = self.transform.rotation.yaw
        self.kinematics.acceleration = np.linalg.norm([self.acceleration.x, self.acceleration.y])
        self.kinematics.curvature = 0.0 if self.kinematics.speed == 0.0 else \
            (self.acceleration.y * self.velocity.x - self.acceleration.x * self.velocity.y) / self.kinematics.speed

    # todo:这里说实话kinematics的引入造成了一定混乱，想办法优化一下

    def x(self): return self.transform.location.x
    # todo: 把这些getter全部变为property!

    def y(self): return self.transform.location.y

    def yaw(self): return self.transform.rotation.yaw

    def v(self): return self.kinematics.speed

    def a(self): return self.kinematics.acceleration

    def kappa(self): return self.kinematics.curvature

    @property
    def obb(self):
        return (self.x(), self.y(), self.length, self.width, self.yaw())

    @classmethod
    def from_kine_states(cls, x, y, yaw, vx=0., vy=0., ax=0., ay=0., length=5., width=2.):
        return cls(
            transform=Transform(
                location=Location(x, y, 0.),
                rotation=Rotation(yaw=yaw)
            ),
            velocity=Vector3D(vx, vy, 0.),
            acceleration=Vector3D(ax, ay, 0.),
            length=length,
            width=width
        )

    @classmethod
    def from_traj_step(cls, trajectory, step_index, length=5.0, width=2.0):
        ego_state = cls(length=length, width=width)
        idx = step_index
        traj = trajectory
        ego_state.transform.location.x, ego_state.transform.location.y, ego_state.transform.rotation.yaw \
            = traj.x[idx], traj.y[idx], traj.heading[idx]
        ego_state.kinematics.speed, ego_state.kinematics.acceleration, ego_state.kinematics.curvature \
            = traj.v[idx], traj.a[idx], traj.curvature[idx]
        return ego_state

    def to_dict(self):
        return { # 暂时没有存3D的高度信息
            "type": self.__class__.__name__,
            "location": [self.x(), self.y()],
            "size": [self.length, self.width],
            "yaw": self.yaw(),
            "velocity": [self.velocity.x, self.velocity.y],
            "acceleration": [self.acceleration.x, self.acceleration.y]
        }



class KinematicState:
    # def __init__(self, x=None, y=None, speed=None, yaw=None, acceleration=None, curvature=None):
    #     self.x = x
    #     self.y = y
    #     self.speed = speed
    #     self.yaw = yaw
    #     self.acceleration = acceleration
    #     self.curvature = curvature  # dtheta/ds
    def __init__(self):
        self.x = None
        self.y = None
        self.speed = None
        self.yaw = None
        self.acceleration = None
        self.curvature = None  # dtheta/ds



class FrenetKinematicState(KinematicState):
    def __init__(self):
        super(FrenetKinematicState, self).__init__()
        self.s = None
        self.l = None
        self.s_dot = None
        self.l_prime = None
        self.l_dot = None
        self.s_2dot = None
        self.l_2prime = None
        self.l_2dot = None


if __name__ == '__main__':
    temp = VehicleState()
    a = 3
    pass
