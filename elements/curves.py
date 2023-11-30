"""
曲线类
目前只完成了2d平面曲线类

主要分两大类，
一类是显式方程表达的曲线，现在命名为Curve1d【这个命名实际上不对】，形如 y = f(x)
一类是参数方程表达的曲线，命名为ParametricCurve， 形如 y=f(s), x=f(s)， s默认认为是里程。

"""

import math
import numpy as np
import bisect
from abc import abstractmethod

import scipy
import cv2




################## 基础的一维曲线类(抽象类) #####################
class Curve1d:
    def __init__(self):
        pass

    def __call__(self, t, order:int=0):
        return self.evaluate(t, order)

    def evaluate(self, t, order=0):
        if order == 0:
            return self.calc_point(t)
        elif order == 1:
            return self.calc_first_derivative(t)
        elif order == 2:
            return self.calc_second_derivative(t)
        elif order == 3:
            return self.calc_third_derivative(t)
        else:
            raise ValueError("Order too high! Third derivative is supported at most")

    @abstractmethod
    def calc_point(self, t):
        pass

    @abstractmethod
    def calc_first_derivative(self,t):
        pass

    @abstractmethod
    def calc_second_derivative(self,t):
        pass

    @abstractmethod
    def calc_third_derivative(self, t):
        pass


################ 多项式曲线 ################

class BasicPolynomial:
    def __init__(self,coef):
        self.coef = np.array(coef)
        self.order = self.coef.shape[0] - 1

    def evaluate(self, t):
        t_items = np.array([t**i for i in range(self.order, -1, -1)])
        return self.coef @ t_items

    def fit(self):
        # 最小二乘拟合
        pass

    def interp(self):
        pass

# TODO:把四次和五次多项式继承basicpoly
class QuarticPolynomial(Curve1d):

    def __init__(self, xs, vxs, axs, vxe, axe, te):
        # calc coefficient of quartic polynomial
        super(QuarticPolynomial, self).__init__()
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * te ** 2, 4 * te ** 3],
                      [6 * te, 12 * te ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * te,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.t_range = te

    def calc_point(self, t):
        if t > self.t_range:
            xt = self.calc_point(self.t_range) + (t-self.t_range) * self.calc_first_derivative(self.t_range)
            return xt

        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        if t > self.t_range:
            xt = self.calc_first_derivative(self.t_range)
            return xt

        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        if t > self.t_range:
            xt = 0
            return xt

        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        if t > self.t_range:
            xt = 0
            return xt

        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt

class QuinticPolynomial(Curve1d):
    def __init__(self, xs, vxs, axs, xe, vxe, axe, te):
        # calc coefficient of quintic polynomial
        super(QuinticPolynomial, self).__init__()
        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[te ** 3, te ** 4, te ** 5],
                      [3 * te ** 2, 4 * te ** 3, 5 * te ** 4],
                      [6 * te, 12 * te ** 2, 20 * te ** 3]])
        b = np.array([xe - self.a0 - self.a1 * te - self.a2 * te ** 2,
                      vxe - self.a1 - 2 * self.a2 * te,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]
        self.t_range = te

    def calc_point(self, t):
        if t > self.t_range:
            xt = self.calc_point(self.t_range) + (t-self.t_range) * self.calc_first_derivative(self.t_range)
            return xt

        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4 + self.a5 * t ** 5

        return xt

    def calc_first_derivative(self, t):
        if t > self.t_range:
            xt = self.calc_first_derivative(self.t_range)
            return xt

        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3 + 5 * self.a5 * t ** 4

        return xt

    def calc_second_derivative(self, t):
        if t > self.t_range:
            xt = 0
            return xt
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2 + 20 * self.a5 * t ** 3

        return xt

    def calc_third_derivative(self, t):
        if t > self.t_range:
            xt = 0
            return xt
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t ** 2

        return xt



################ 线性插值曲线 ################


################ 三次样条曲线 ################
#todo: 很重要：给Spline继承Curve1d,加上__call__方法直接evaluate,使其能够成为轨迹的generator，这样子可以在combiner里用一些离散轨迹点生成连续轨迹

class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position

        if t is outside of the input x, return None

        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None
        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative

        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)

        return B


class Spline2D:
    """
    2D Cubic Spline class
    本质是关于s的x,y二维参数化曲线，以后集成到参数曲线类里面去
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw



################## 参数化曲线(抽象类) #####################
class ParametricCurve:
    '''
    用参数方程表达的二维曲线形式，
    参数默认与里程相关
    '''
    def __init__(self):
        pass

    def __call__(self, s, order: int=0):
        return self.evaluate(s, order)

    def evaluate(self, s, order=0):
        if order == 0:
            return self.calc_point(s)
        elif order == 1:
            return self.calc_first_derivative(s)
        elif order == 2:
            return self.calc_second_derivative(s)
        elif order == 3:
            return self.calc_third_derivative(s)
        else:
            raise ValueError("Order too high! Third derivative is supported at most")

    @abstractmethod
    def calc_point(self, s):
        pass

    @abstractmethod
    def calc_first_derivative(self, s):
        pass

    @abstractmethod
    def calc_second_derivative(self, s):
        pass

    @abstractmethod
    def calc_third_derivative(self, s):
        pass


################# 贝塞尔曲线 #####################
class BezierCurve(ParametricCurve):
    # 类变量的定义语句在类被定义时就会执行，并且只执行一次, 多次初始化多个类的实例时，类变量的定义语句不会被重新执行
    _binom_coeff_dict = {n: scipy.special.binom(n, np.arange(n+1)) for n in range(1, 9)} # 预先算至多8个控制点

    def __init__(self, control_points, pre_calculation=True):
        super(BezierCurve, self).__init__()
        assert len(control_points) >= 2
        self.n = len(control_points) - 1 # 贝塞尔曲线的阶数
        self.control_points = np.array(control_points)

        # self._bezier_coeff_dict = {}
        # self._binom_coeff_dict = {}

        self._derivative_bezier_curve = None
        self._arclength = None

        # if pre_calculation:
        #     # 提前计算二项式系数
        #     values = scipy.special.binom(self.n, np.arange(self.n+1))
        #     for i, val in enumerate(values):
        #         self._binom_coeff_dict[(self.n, i)] = val
        #     # 提前计算

    @property
    def derivative_bezier_curve(self):
        if self._derivative_bezier_curve is None:
            derivative_control_points = self.n * np.diff(self.control_points, axis=0)
            self._derivative_bezier_curve = BezierCurve(derivative_control_points, pre_calculation=False)
        return self._derivative_bezier_curve

    @property
    def arclength(self):
        # qzl: 贝塞尔曲线无法求解解析解，这里采用采样点间距求和代替积分。
        # 这里要再考虑一下，如果弧线长度很长，但采样的点还是500个，会不会造成弧长的近似严重不符合事实？
        if self._arclength is None:
            ts = np.linspace(0, 1, 500)
            pts = self.calc_point_t(ts).astype(dtype=np.float32)
            self._arclength = cv2.arcLength(pts, closed=False) # 用cv2的函数会比自己写的稍快一些
            # delta_xy = np.diff(pts, axis=0)
            # delta = np.linalg.norm(delta_xy,axis=1)
            # self._arclength = np.sum(delta)
        return self._arclength

    # @property
    # def arclength2(self):
    #     # qzl: 这里要再考虑一下，如果弧线长度很长，但采样的点还是500个，会不会造成弧长的近似严重不符合事实？
    #     if self._arclength is None:
    #         ts = np.linspace(0, 1, 500)
    #         pts = self.calc_point_t(ts)
    #         delta_xy = np.diff(pts, axis=0)
    #         delta = np.linalg.norm(delta_xy,axis=1)
    #         self._arclength = np.sum(delta)
    #     return self._arclength


    def _binom(self, n, i):
        assert i >= 0 and i <= n
        if n in self._binom_coeff_dict:
            return self._binom_coeff_dict[n][i]
        else:
            self._binom_coeff_dict[n] = scipy.special.binom(n, np.arange(n+1))
            return self._binom_coeff_dict[n][i]

    def _isscalar(self, x):
        return np.isscalar(x) and np.ndim(x)==0

    def calc_point_t(self, t):
        '''
        输入贝塞尔曲线的参数t，输出对应的x,y
        若t是一个数，输出(1,2)的点
        若t是n个数组成的数组，输出(n,2)的点
        '''
        # 关于贝塞尔曲线本身的参数
        # todo:加入处理t是否在0-1以外的函数，即extrapolate

        isscalar = self._isscalar(t)
        if isscalar:
            point = np.zeros((1,2), dtype=float)
        else:
            num_pts = len(t)
            point = np.zeros((num_pts, 2), dtype=float)

        t = np.array(t)
        assert np.all(t <= 1) and np.all(t >= 0)

        for i in range(self.n+1):
            point += np.outer(self._binom(self.n, i) *
                              t ** i * (1 - t) ** (self.n - i),
                              self.control_points[i])

        if isscalar: point = point[0]

        return point

    def calc_first_derivative_t(self, t):
        '''
        输入贝塞尔曲线的参数t，输出对应的x,y关于u的导数
        '''
        # 关于贝塞尔曲线本身的参数

        # todo:加入处理t是否在0-1以外的函数，即extrapolate
        if self.n == 1:
            # 一阶贝塞尔曲线就是一条直线，直接计算即可
            dval = (self.control_points[-1] - self.control_points[0]) / 1.
            if not self._isscalar(t):
                dval = np.tile(dval, (len(t), 1))
            return dval
        else:
            return self.derivative_bezier_curve.calc_point_t(t)

    def calc_second_derivative_t(self, t):
        if self.n == 1:
            # 一阶贝塞尔曲线就是一条直线，二阶导为0
            shape = (2,) if self._isscalar(t) else (len(t), 2)
            return np.zeros(shape, dtype=float)
        else:
            return self.derivative_bezier_curve.calc_first_derivative_t(t)

    def calc_third_derivative_t(self, t):
        if self.n <= 2:
            # 一阶或二阶贝塞尔曲线的3阶导是0
            shape = (2,) if self._isscalar(t) else (len(t), 2)
            return np.zeros(shape, dtype=float)
        else:
            return self.derivative_bezier_curve.calc_second_derivative_t(t)

    def calc_point(self, s):
        return self.calc_point_t( np.array(s) / self.arclength )

    def calc_first_derivative(self, s):
        return self.calc_first_derivative_t( np.array(s) / self.arclength )

    def calc_second_derivative(self, s):
        return self.calc_second_derivative_t( np.array(s) / self.arclength )

    def calc_third_derivative(self, s):
        return self.calc_third_derivative_t( np.array(s) / self.arclength )

    # def calc_point(self, t):
    #     '''
    #     qzl:
    #     非常不推荐使用这个函数，
    #     贝塞尔曲线的计算本质上是在描述点 (x,y) 关于参数 t 的函数关系
    #     这个函数将其视作描述y关于x的函数关系（这里采用采样点插值处理）
    #     在不是函数映射的曲线（比如竖着的半圆曲线）无法处理
    #     '''
    #     uu = np.linspace(0, 1, 100)



if __name__ == '__main__':
    cp = np.array([[0, 0], [2, 3], [4, 4],[8,10]])
    t = 0.1
    c = BezierCurve(cp)
    x = c.calc_point_t(t)
    dx = c.calc_first_derivative_t(t)
    ddx = c.calc_second_derivative_t(t)
    dddx = c.calc_third_derivative_t(t)

    from time import time
    t1 = time()
    for _ in range(10000):
        a = c.arclength
    t2 = time()
    for _ in range(10000):
        # a = {n: scipy.special.binom(n, np.arange(n + 1)) for n in range(1, 9)}
        b = c.arclength3
    t3 = time()

    print(t2-t1)
    print(t3-t2)
    print(a)
    print(b)


    pass
