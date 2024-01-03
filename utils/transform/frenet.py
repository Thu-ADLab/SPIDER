"""
可以思考一下如何加速
"""
import numpy as np
import math
from spider.elements.curves import ParametricCubicSpline
from spider.utils.geometry import find_nearest_point, point_to_segment_distance, resample_polyline
from spider.elements.trajectory import FrenetTrajectory
from spider.elements.vehicle import KinematicState, FrenetKinematicState
from spider.vehicle_model.bicycle import curvature2steer

# def cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa):
#     """
#     rs,rx,ry,rtheta,rkappa,rdkappa: 投影点的信息
#     x, y, v, a, theta, kappa：目标点的信息
#     """
#
#     # kappa是曲率
#     dx = x - rx
#     dy = y - ry
#
#     cos_theta_r = np.cos(rtheta)
#     sin_theta_r = np.sin(rtheta)
#
#     cross_rd_nd = cos_theta_r * dy - sin_theta_r * dx
#     d0 = np.copysign(np.sqrt(dx * dx + dy * dy), cross_rd_nd)
#
#     delta_theta = theta - rtheta
#     tan_delta_theta = np.tan(delta_theta)
#     cos_delta_theta = np.cos(delta_theta)
#
#     one_minus_kappa_r_d = 1 - rkappa * d0
#     d1 = one_minus_kappa_r_d * tan_delta_theta
#
#     kappa_r_d_prime = rdkappa * d0 + rkappa * d1
#
#     d2 = -kappa_r_d_prime * tan_delta_theta + one_minus_kappa_r_d / (cos_delta_theta * cos_delta_theta) * (kappa * one_minus_kappa_r_d / cos_delta_theta - rkappa)
#
#     s0 = rs
#     s1 = v * cos_delta_theta / one_minus_kappa_r_d
#     delta_theta_prime = one_minus_kappa_r_d / cos_delta_theta * kappa - rkappa
#     s2 = (a * cos_delta_theta - s1 * s1 * (d1 * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_d
#
#     return [s0, s1, s2], [d0, d1, d2]


class FrenetCoordinateTransformer:
    def __init__(self):
        self.refer_line_arr = None # N行2列，[[x,y]*N]
        self.refer_line_csp = None

    def set_reference_line(self, reference_line:np.ndarray, reference_line_csp=None, resample=False, resample_resolution=1.0):
        if resample:
            reference_line = resample_polyline(reference_line, resample_resolution)
            reference_line_csp = ParametricCubicSpline(reference_line[:, 0], reference_line[:, 1])

        self.refer_line_arr = reference_line
        if reference_line_csp is None:
            self.refer_line_csp = ParametricCubicSpline(self.refer_line_arr[:, 0], self.refer_line_arr[:, 1])
        else:
            self.refer_line_csp = reference_line_csp

    def __check_validity(self,order:int):
        if self.refer_line_csp is None:
            raise ValueError("Reference line not set")
        if order<0 or order>2:
            raise ValueError("Invalid order!")

    def __cart2frenet_order0(self, x, y):
        nearest_idx, min_dist = find_nearest_point(np.array([x, y]), self.refer_line_arr)

        # 计算两个线段之间的距离并选择最短的作为l
        if 0 < nearest_idx < len(self.refer_line_arr) - 1:
            proj1, dist1 = point_to_segment_distance(np.array([x, y]), self.refer_line_arr[nearest_idx - 1],
                                                     self.refer_line_arr[nearest_idx],allow_extension=False)
            proj2, dist2 = point_to_segment_distance(np.array([x, y]), self.refer_line_arr[nearest_idx],
                                                     self.refer_line_arr[nearest_idx + 1],allow_extension=False)

            # 选择距离更短的投影点及距离
            if abs(dist1) < abs(dist2):
                s_end, l = proj1, dist1 # s_end指的是最后一段segment上面s的距离
                segment_start_idx = nearest_idx - 1
            else:
                s_end, l = proj2, dist2
                segment_start_idx = nearest_idx
        # 处理最近点在参考线起始点的情况
        elif nearest_idx == 0:
            s_end, l = point_to_segment_distance(np.array([x, y]), self.refer_line_arr[nearest_idx],
                                                 self.refer_line_arr[nearest_idx + 1])
            segment_start_idx = nearest_idx
        # 处理最近点在参考线末尾的情况
        else:
            s_end, l = point_to_segment_distance(np.array([x, y]), self.refer_line_arr[nearest_idx - 1],
                                                 self.refer_line_arr[nearest_idx])
            segment_start_idx = nearest_idx - 1

        # 计算纵向位置s和横向导数l_prime
        s = np.sum(
            np.linalg.norm(self.refer_line_arr[1:segment_start_idx + 1] - self.refer_line_arr[:segment_start_idx],
                           axis=1)) + s_end
        return s,l

    def cart2frenet(self, x, y, speed=None, yaw=None, acc=None, kappa=None, *, order:int) -> FrenetKinematicState:
        """
        一般认为沿着参考线s增加方向的左边为正，右边为负

        order =0 : x,y -> s,l
        order =1 : x,y,speed,yaw -> s,l, s_dot, l_prime, l_dot
        order =2 : x,y,speed,yaw,acc,curvature -> s,l, s_dot, l_prime/l_dot, s_2dot, l_2prime, l_2dot
        """
        self.__check_validity(order)

        frenet_state = FrenetKinematicState()

        # 零阶导
        s, l = self.__cart2frenet_order0(x,y)
        frenet_state.s, frenet_state.l = s, l
        if order == 0:
            return frenet_state

        # 一阶导
        if speed is None or yaw is None:
            raise ValueError("Lack of 1-order information")

        # rx, ry = self.refer_line_csp.calc_position(s)
        rtheta = self.refer_line_csp.calc_yaw(s)
        rkappa = self.refer_line_csp.calc_curvature(s)

        dtheta = yaw - rtheta
        cos_dtheta = np.cos(dtheta)
        tan_dtheta = np.tan(dtheta)
        one_minus_kappa_r_l = 1 - rkappa * l
        s_dot = speed * cos_dtheta / one_minus_kappa_r_l
        l_prime = one_minus_kappa_r_l * tan_dtheta # l对s的导数
        l_dot = l_prime * s_dot

        frenet_state.s_dot, frenet_state.l_dot, frenet_state.l_prime = s_dot, l_dot, l_prime
        if order == 1:
            return frenet_state

        # 二阶导
        if speed is None or yaw is None:
            raise ValueError("Lack of 2-order information")

        rdkappa = 0 # todo: 需要在ParametricCubicSpline中加入计算曲率对于s的变化率的计算公式
        delta_theta_prime = one_minus_kappa_r_l / cos_dtheta * kappa - rkappa
        kappa_r_d_prime = rdkappa * l + rkappa * l_prime
        s_2dot = (acc * cos_dtheta - s_dot ** 2 * (l_prime * delta_theta_prime - kappa_r_d_prime)) / one_minus_kappa_r_l
        l_2prime = -kappa_r_d_prime * tan_dtheta + one_minus_kappa_r_l / cos_dtheta ** 2 * \
                   (kappa * one_minus_kappa_r_l / cos_dtheta - rkappa)
        l_2dot = l_2prime * s_dot ** 2 + l_prime * s_2dot

        frenet_state.s_2dot, frenet_state.l_2dot, frenet_state.l_2prime = s_2dot, l_2dot, l_2prime
        return frenet_state

    def frenet2cart(self, s, l, s_dot=None, l_dot=None, l_prime=None, s_2dot=None, l_2dot=None,
                    l_2prime=None, *, order:int) -> FrenetKinematicState:
        """
        order =0 : s,l -> x,y
        order =1 : s,l, s_dot, l_prime/l_dot -> x,y,speed,yaw
        order =2 : s,l, s_dot, l_prime/l_dot, s_2dot, l_2prime/l_2dot -> x,y,speed,yaw,acc,curvature
        """

        self.__check_validity(order)

        frenet_state = FrenetKinematicState()

        # 零阶导
        rx, ry = self.refer_line_csp.calc_point(s)
        rtheta = self.refer_line_csp.calc_yaw(s)
        x = rx - l * np.sin(rtheta)
        y = ry + l * np.cos(rtheta)
        frenet_state.x, frenet_state.y = x,y
        if order == 0:
            return frenet_state

        # 一阶导
        if s_dot is None or ((l_prime is None) and (l_dot is None)):
            raise ValueError("Lack of 1-order information")

        if l_prime is None:
            l_prime = l_dot / s_dot

        rkappa = self.refer_line_csp.calc_curvature(s)
        one_minus_kappa_r_l = 1 - rkappa * l
        yaw = rtheta + np.arctan2(l_prime, one_minus_kappa_r_l)
        speed = np.linalg.norm([s_dot * one_minus_kappa_r_l, s_dot*l_prime])
        frenet_state.yaw, frenet_state.speed = yaw, speed
        if order == 1:
            return frenet_state

        # 二阶导
        if s_2dot is None or ((l_2prime is None) and (l_2dot is None)):
            raise ValueError("Lack of 2-order information")

        dtheta = yaw - rtheta
        cos_dtheta = np.cos(dtheta)
        tan_dtheta = np.tan(dtheta)
        rdkappa = 0  # todo: 需要在ParametricCubicSpline中加入计算曲率对于s的变化率的计算公式
        kappa_r_d_prime = rdkappa * l + rkappa * l_prime
        temp = cos_dtheta / one_minus_kappa_r_l
        kappa = ((l_2prime + kappa_r_d_prime * tan_dtheta) * cos_dtheta * temp + rkappa) * temp

        delta_theta_prime = kappa / temp - rkappa
        acc = s_2dot / temp + s_dot ** 2 / cos_dtheta * (l_prime * delta_theta_prime - kappa_r_d_prime)

        frenet_state.acceleration, frenet_state.curvature = acc, kappa
        return frenet_state

    def cart2frenet4state(self, state: KinematicState,  order: int):
        self.__check_validity(order)
        return self.cart2frenet(state.x, state.y, state.speed, state.yaw,
                                state.acceleration, state.curvature, order=order)

    def frenet2cart4state(self, state: FrenetKinematicState, order: int):
        self.__check_validity(order)
        return self.frenet2cart(state.s, state.l, state.s_dot, state.l_dot, state.l_prime,
                                state.s_2dot, state.l_2dot, state.l_2prime, order=order)

    def cart2frenet4traj(self, traj:FrenetTrajectory, order:int):
        self.__check_validity(order)

        if order == 0:
            for i in range(traj.steps):
                temp_state = self.cart2frenet(traj.x[i],traj.y[i],order=order)
                traj.s.append(temp_state.s)
                traj.l.append(temp_state.l)

        elif order == 1:
            if len(traj.v) != traj.steps or len(traj.heading) != traj.steps:
                raise ValueError("Lack of 1-order information")

            for i in range(traj.steps):
                temp_state = self.cart2frenet(traj.x[i], traj.y[i], traj.v[i], traj.heading[i], order=order)
                traj.s.append(temp_state.s)
                traj.l.append(temp_state.l)
                traj.s_dot.append(temp_state.s_dot)
                traj.l_dot.append(temp_state.l_dot)
                traj.l_prime.append(temp_state.l_prime)

        elif order == 2:
            if len(traj.v) != traj.steps or len(traj.heading) != traj.steps:
                raise ValueError("Lack of 1-order information")
            if len(traj.a) != traj.steps or len(traj.curvature) != traj.steps:
                raise ValueError("Lack of 2-order information")

            for i in range(traj.steps):
                temp_state = self.cart2frenet(traj.x[i], traj.y[i], traj.v[i], traj.heading[i],
                                              traj.a[i], traj.curvature[i], order=order)
                traj.s.append(temp_state.s)
                traj.l.append(temp_state.l)
                traj.s_dot.append(temp_state.s_dot)
                traj.l_dot.append(temp_state.l_dot)
                traj.l_prime.append(temp_state.l_prime)
                traj.s_2dot.append(temp_state.s_2dot)
                traj.l_2dot.append(temp_state.l_2dot)
                traj.l_2prime.append(temp_state.l_2prime)

        return traj

    def frenet2cart4traj(self, traj:FrenetTrajectory, order:int):
        self.__check_validity(order)

        if order == 0:
            for i in range(traj.steps):
                temp_state = self.frenet2cart(traj.s[i],traj.l[i],order=order)
                traj.x.append(temp_state.x)
                traj.y.append(temp_state.y)

        elif order == 1:
            if len(traj.l_prime) == traj.steps:
                traj.l_dot = [None] * traj.steps
            elif len(traj.l_dot) == traj.steps:
                traj.l_prime = [None] * traj.steps
            else:
                raise ValueError("Lack of 1-order information") # 其实还少了一项对s_dot的判断

            for i in range(traj.steps):
                temp_state = self.frenet2cart(traj.s[i], traj.l[i], traj.s_dot[i], traj.l_dot[i], traj.l_prime[i],
                                              order=order)
                traj.x.append(temp_state.x)
                traj.y.append(temp_state.y)
                traj.v.append(temp_state.speed)
                traj.heading.append(temp_state.yaw)

        elif order == 2:
            if len(traj.l_prime) == traj.steps:
                traj.l_dot = [None] * traj.steps
            elif len(traj.l_dot) == traj.steps:
                traj.l_prime = [None] * traj.steps
            else:
                raise ValueError("Lack of 1-order information")
            if len(traj.l_2prime) == traj.steps:
                traj.l_2dot = [None] * traj.steps
            elif len(traj.l_2dot) == traj.steps:
                traj.l_2prime = [None] * traj.steps
            else:
                raise ValueError("Lack of 2-order information")

            for i in range(traj.steps):
                temp_state = self.frenet2cart(traj.s[i], traj.l[i], traj.s_dot[i], traj.l_dot[i], traj.l_prime[i],
                                              traj.s_2dot[i], traj.l_2dot[i], traj.l_2prime[i], order=order)
                traj.x.append(temp_state.x)
                traj.y.append(temp_state.y)
                traj.v.append(temp_state.speed)
                traj.heading.append(temp_state.yaw)
                traj.a.append(temp_state.acceleration)
                traj.curvature.append(temp_state.curvature)
                traj.steer.append(curvature2steer(temp_state.curvature))

        return traj

    # @classmethod
    # def


if __name__ == '__main__':
    xs = np.linspace(0,100,101)
    ys = np.zeros_like(xs)
    centerline = np.column_stack((xs, ys))
    transformer = FrenetCoordinateTransformer()
    transformer.set_reference_line(centerline)


    # 转某个点的坐标
    x, y, speed, yaw, acc, kappa = 50, 1, 5, np.pi/4, 3, 0.
    state = transformer.cart2frenet(x, y, speed, yaw, acc, kappa, order=2)
    print("s,l,s_dot, l_dot, l_prime,s_2dot, l_2dot, l_2prime")
    print(state.s, state.l, state.s_dot, state.l_dot, state.l_prime,state.s_2dot, state.l_2dot, state.l_2prime)

    print("======================")
    s, l, s_dot, l_dot, l_prime, s_2dot, l_2dot, l_2prime = \
        state.s, state.l, state.s_dot, state.l_dot, state.l_prime, state.s_2dot, state.l_2dot, state.l_2prime
    state = transformer.frenet2cart(s, l, s_dot, l_dot, l_prime, s_2dot, l_2dot, l_2prime, order=2)
    print("x, y, speed, yaw, acc, kappa")
    print(state.x, state.y, state.speed, state.yaw, state.acceleration, state.curvature)


    # 转某个轨迹的坐标
    xs = np.linspace(0,50,50)
    ys = np.random.rand(50)
    from spider.elements.trajectory import FrenetTrajectory

    traj = FrenetTrajectory(steps=50, dt=0.1)
    traj.x = xs
    traj.y = ys

    print("======================")
    frenet_traj = transformer.cart2frenet4traj(traj,order=0)
    print(frenet_traj.s, frenet_traj.l)

    frenet_traj = FrenetTrajectory(steps=50, dt=0.1)
    frenet_traj.s = xs
    frenet_traj.l = ys

    print("======================")
    cart_traj = transformer.frenet2cart4traj(frenet_traj, order=0)
    print(cart_traj.x, cart_traj.y)

    pass





    # rs = 0.0
    # rx = 0.0
    # ry = 0.0
    # rtheta = 0.0
    # rkappa = 0.0
    # rdkappa = 0.0
    # x = 10.0
    # y = 5.0
    # v = 10.0
    # a = 2.0
    # theta = 0.0
    # kappa = 0.0
    #
    #
    # s_condition, d_condition = cartesian_to_frenet(rs, rx, ry, rtheta, rkappa, rdkappa, x, y, v, a, theta, kappa)
    #
    # print("s condition:", s_condition)
    # print("d condition:", d_condition)

