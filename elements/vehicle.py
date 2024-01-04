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
    def __init__(self, transform:Transform, velocity:Vector3D, acceleration:Vector3D, length=5.0, width=2.0):
        self.transform = transform
        self.velocity = velocity
        self.acceleration = acceleration
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

