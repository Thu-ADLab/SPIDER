"""
曲线类
只包含2d平面曲线类

主要分三大类，
一类是显式方程表达的曲线，现在命名ExplicitCurve，形如 y = f(x)
一类是参数方程表达的曲线，命名为ParametricCurve， 形如 y=f(s), x=f(s)， s默认认为是里程。
一类是隐式方程表达的曲线

1. **Explicit Curves:**
   - *Representation:* \(y = f(x)\)
   - *Characteristics:* Direct expression of \(y\) as a function of \(x\).

2. **Parametric Curves:**
   - *Representation:* \(x = f(s)\) and \(y = g(s)\), where \(s\) is a parameter.
   - *Characteristics:* Position given by parametric equations. Useful for complex shapes.

3. **Implicit Curves:**
   - *Representation:* \(F(x, y) = 0\)
   - *Characteristics:* Relation between \(x\) and \(y\) expressed as an equation without explicitly solving for \(y\).

"""

import math
import warnings

import numpy as np
import bisect
from abc import abstractmethod
from typing import Union, List

from scipy.special import binom
import scipy
import scipy.interpolate

try:
    from cv2 import arcLength as _arclength  # 用cv2的函数会比自己写的稍快一些
except (ModuleNotFoundError, ImportError):
    print("opencv-python module not found.")
    def _arclength(pts, closed=False):
        if closed: pts = np.vstack((pts, pts[0]))
        delta_xy = np.diff(pts, axis=0)
        delta = np.linalg.norm(delta_xy, axis=1)
        return np.sum(delta)

try:
    from . import _poly_calc
except (ModuleNotFoundError, ImportError):
    print("Cython module not found. Use pyximport temporally")
    import pyximport
    import os, sys
    pyxroot = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(pyxroot)
    pyximport.install(language_level=3)  # '../elements/_poly_calc.pyx',
    # _poly_calc = pyximport.load_module('_poly_calc', pyxroot+'\\__poly_calc.pyx', language_level=3)
    import _poly_calc


################## 基础的一维曲线类(抽象类) 现在已经弃用！#####################
class Curve1d:
    def __init__(self):
        raise AssertionError('Curve1d has been replaced by ExplicitCurve for clear description. Please use ExplicitCurve Instead!')



################## 基础的显式方程表达的曲线类 #####################
class ExplicitCurve:
    def __init__(self):
        pass

    def __call__(self, x, order:int=0):
        return self.evaluate(x, order)

    def evaluate(self, x, order=0):
        if order == 0:
            return self.calc_point(x)
        elif order == 1:
            return self.calc_first_derivative(x)
        elif order == 2:
            return self.calc_second_derivative(x)
        elif order == 3:
            return self.calc_third_derivative(x)
        else:
            raise ValueError("Order too high! Third derivative is supported at most")

    @abstractmethod
    def calc_point(self, x):
        pass

    @abstractmethod
    def calc_first_derivative(self,x):
        pass

    @abstractmethod
    def calc_second_derivative(self,x):
        pass

    @abstractmethod
    def calc_third_derivative(self, x):
        pass

    def calc_yaw(self, x):
        y_prime = self.calc_first_derivative(x)
        yaw = np.arctan(y_prime)
        return yaw

    def calc_curvature(self, x, absolute:bool=False):
        y_2prime = self.calc_second_derivative(x)
        y_prime = self.calc_first_derivative(x)
        if absolute:
            curvature = np.abs(y_2prime) / (1 + y_prime**2)**1.5
        else:
            curvature = y_2prime / (1 + y_prime ** 2) ** 1.5
        return curvature

    def _isscalar(self, x):
        return np.isscalar(x) or np.ndim(x)==0


################ 多项式曲线 ################

class BasePolynomial(ExplicitCurve):
    def __init__(self, coef=None, valid_x_range=None):
        super(BasePolynomial, self).__init__()

        self._derivative_coefs = {}

        if coef is None:
            self.coef = None
            self.order = -1
        else:
            self.set_coef(coef)
            # self.coef = np.array(coef)
            # self.order = self.coef.shape[0] - 1
            # for i in range(1,4): # 先求1-3阶导数
            #     self.derivative_coef(order=i)

        if valid_x_range is None:
            self.valid_x_range = [-np.inf, np.inf]
        else:
            self.valid_x_range = valid_x_range

        self._lower_boundary_value = [None for _ in range(4)] # 分别对应valid_x_range上下界情况下的0-3阶导数
        self._upper_boundary_value = [None for _ in range(4)]

    def _get_boundary_value(self, order):
        if (self._lower_boundary_value[order] is None) or (self._upper_boundary_value[order] is None):
            x = np.ascontiguousarray(self.valid_x_range, dtype=np.float64)
            coef = self.coef if order == 0 else self.derivative_coef(order)
            c = np.ascontiguousarray(coef, dtype=np.float64)
            val = _poly_calc.evaluate(x, c, self.valid_x_range[0], self.valid_x_range[1], 0, 0.,0.,0.,0.)
            val = np.asarray(val)
            self._lower_boundary_value[order] = val[0] #self.evaluate(self.valid_x_range[0],order)
            self._upper_boundary_value[order] = val[1] #self.evaluate(self.valid_x_range[1],order)
        return self._lower_boundary_value[order], self._upper_boundary_value[order]

    def _out_of_range_flag(self, x) -> Union[bool, np.ndarray]:
        '''
        if x is a scalar, return bool
        if x is an array, return a boolean array with the same size
        '''
        return (x<self.valid_x_range[0]) | (x>self.valid_x_range[1]) # 不可以用or， 因为要考虑array情况

    def extrapolate(self, x, order):
        '''
        在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
        please notice that, if any x in the valid_x_range, the corresponding y maintains 0 value
        '''
        val = np.zeros_like(x)
        if val.size == 0:
            return val

        x_0, x_end = self.valid_x_range
        if order == 0:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            y_0 = self.evaluate(x_0, order=0)
            y_end = self.evaluate(x_end, order=0)
            val[x > x_end] = y_end + y_prime_end * (x[x>x_end] - x_end)
            val[x < x_0] = y_0 + y_prime_0 * (x[x < x_end] - x_0)
        elif order == 1:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            val[x > x_end] = y_prime_end
            val[x < x_0] = y_prime_0
        else:
            pass  # because the higher order derivative is 0 anyway.

        return val


    def set_coef(self, coef):
        self.coef = np.ascontiguousarray(coef, dtype=np.float64)
        self.order = self.coef.shape[0] - 1
        for i in range(1, 4):  # 先求1-3阶导数
            self.derivative_coef(order=i)

    def set_valid_x_range(self, x_lower_boundary, x_upper_boundary):
        self.valid_x_range = [x_lower_boundary, x_upper_boundary]

    def derivative_coef(self, order=1):
        # 几阶导数
        if order in self._derivative_coefs:
            return self._derivative_coefs[order]
        else:
            # if order >= len(self.coef):
            dcoef = np.polyder(self.coef, order)
            # if order >= len(self.coef), dcoef is array([]), and polyval([],x) === 0
            self._derivative_coefs[order] = np.ascontiguousarray(dcoef, dtype=np.float64)
            return dcoef


    def calc_point(self, x):
        y_0, y_end = self._get_boundary_value(0)
        y_prime_0, y_prime_end = self._get_boundary_value(1)
        c = np.ascontiguousarray(self.coef, dtype=np.float64)

        if self._isscalar(x):
            return _poly_calc.evaluate_scalar(x, c, self.valid_x_range[0], self.valid_x_range[1], 0,
                                             y_0, y_prime_0, y_end, y_prime_end)

        x = np.ascontiguousarray(x, dtype=np.float64)
        val = _poly_calc.evaluate(x, c, self.valid_x_range[0], self.valid_x_range[1], 0,
                                 y_0, y_prime_0, y_end, y_prime_end)
        val = np.asarray(val)

        # x = np.asarray(x, dtype=float)
        # extra_mask = self._out_of_range_flag(x)
        # val = np.empty_like(x)
        # val[extra_mask] = self.extrapolate(x[extra_mask], order=0) # 数据范围外，按外推规则计算
        # val[~extra_mask] = np.polyval(self.coef, x[~extra_mask]) # 数据范围内，直接计算
        return val

    def calc_first_derivative(self,x):
        y_0, y_end = self._get_boundary_value(0)
        y_prime_0, y_prime_end = self._get_boundary_value(1)
        c = np.ascontiguousarray(self.derivative_coef(order=1), dtype=np.float64)

        if self._isscalar(x):
            return _poly_calc.evaluate_scalar(x, c, self.valid_x_range[0], self.valid_x_range[1], 1, y_0, y_prime_0, y_end,
                                      y_prime_end)

        x = np.ascontiguousarray(x, dtype=np.float64)
        val = _poly_calc.evaluate(x, c, self.valid_x_range[0], self.valid_x_range[1], 1,
                                 y_0, y_prime_0, y_end, y_prime_end)
        val = np.asarray(val)

        # x = np.asarray(x, dtype=float)
        # extra_mask = self._out_of_range_flag(x)
        # val = np.empty_like(x)
        # val[extra_mask] = self.extrapolate(x[extra_mask], order=1)  # 数据范围外，按外推规则计算
        # val[~extra_mask] = np.polyval(self.derivative_coef(order=1), x[~extra_mask]) # 数据范围内，直接计算
        return val

    def calc_second_derivative(self,x):
        y_0, y_end = self._get_boundary_value(0)
        y_prime_0, y_prime_end = self._get_boundary_value(1)
        c = np.ascontiguousarray(self.derivative_coef(order=2), dtype=np.float64)

        if self._isscalar(x):
            return _poly_calc.evaluate_scalar(x, c, self.valid_x_range[0], self.valid_x_range[1], 2, y_0, y_prime_0, y_end,
                                      y_prime_end)

        x = np.ascontiguousarray(x, dtype=np.float64)
        val = _poly_calc.evaluate(x, c, self.valid_x_range[0], self.valid_x_range[1], 2,
                                 y_0, y_prime_0, y_end, y_prime_end)
        val = np.asarray(val)

        # x = np.asarray(x, dtype=float)
        # extra_mask = self._out_of_range_flag(x)
        # val = np.empty_like(x)
        # val[extra_mask] = self.extrapolate(x[extra_mask], order=2)  # 数据范围外，按外推规则计算
        # val[~extra_mask] = np.polyval(self.derivative_coef(order=2), x[~extra_mask])  # 数据范围内，直接计算
        return val

    def calc_third_derivative(self, x):
        y_0, y_end = self._get_boundary_value(0)
        y_prime_0, y_prime_end = self._get_boundary_value(1)
        c = np.ascontiguousarray(self.derivative_coef(order=3), dtype=np.float64)

        if self._isscalar(x):
            return _poly_calc.evaluate_scalar(x, c, self.valid_x_range[0], self.valid_x_range[1], 3, y_0, y_prime_0, y_end,
                                      y_prime_end)

        x = np.ascontiguousarray(x, dtype=np.float64)
        val = _poly_calc.evaluate(x, c, self.valid_x_range[0], self.valid_x_range[1], 3,
                                 y_0, y_prime_0, y_end, y_prime_end)
        val = np.asarray(val)

        # x = np.asarray(x, dtype=float)
        # extra_mask = self._out_of_range_flag(x)
        # val = np.empty_like(x)
        # val[extra_mask] = self.extrapolate(x[extra_mask], order=3)  # 数据范围外，按外推规则计算
        # val[~extra_mask] = np.polyval(self.derivative_coef(order=3), x[~extra_mask])  # 数据范围内，直接计算
        return val

    def fit(self, x_data, y_data, order):
        """
        Fit the polynomial curve to given data points and update coefficients.
        k阶多项式的得到k+1个coef
        """
        coef = np.polyfit(x_data, y_data, order)
        self.set_coef(coef)


class CubicPolynomial(BasePolynomial):
    def __init__(self, coef=None, valid_x_range=None):
        super(CubicPolynomial, self).__init__(coef, valid_x_range)

    def two_point_boundary_value(self, x_0, y_0, y_prime_0, x_end, y_end, y_prime_end):
        '''
        interpolate with two_point_boundary_value constraints.
        And it is recommended to set x_0 to 0 to accelerate the calculation
        todo: 可以提前储存几次方是几，可以加速
        '''
        self.valid_x_range = [x_0, x_end]

        if x_0 != 0:
            # general 的计算方法
            A = np.array([
                [x_0 ** 3, x_0 ** 2, x_0, 1],
                [3 * x_0 ** 2, 2 * x_0, 1, 0],
                [x_end ** 3, x_end ** 2, x_end, 1],
                [3 * x_end ** 2, 2 * x_end, 1, 0],
            ])
            b = np.array([y_0, y_prime_0, y_end, y_prime_end])
            coef = np.linalg.solve(A,b)
            self.set_coef(coef)
        else:
            # x_0 = 0特殊条件下的计算，仅用于加速，数值上等效于general情况下的方法，实验提速1/3
            a0 = y_0
            a1 = y_prime_0

            A = np.array([[x_end**3, x_end**2],
                          [3 * x_end ** 2, 2 * x_end]])
            b = np.array([y_end - a0 - x_end * a1,
                          y_prime_end - a1])
            temp = np.linalg.solve(A, b)

            a3 = temp[0]
            a2 = temp[1]
            self.set_coef([a3, a2, a1, a0])


    @classmethod
    def from_kine_states(cls, x0, y0, yaw0, xe, ye, yawe):
        cp = cls()
        cp.two_point_boundary_value(x0, y0, math.tan(yaw0), xe, ye, math.tan(yawe))
        return cp

class QuarticPolynomial(BasePolynomial):
    def __init__(self, coef=None, valid_x_range=None):
        super(QuarticPolynomial, self).__init__(coef, valid_x_range)
        # self.t_range:list = []

    def two_point_boundary_value(self, x_0, y_0, y_prime_0, y_2prime_0, x_end, y_prime_end, y_2prime_end):
        '''
        interpolate with two_point_boundary_value constraints.
        And it is recommended to set x_0 to 0 to accelerate the calculation
        todo: 可以提前储存几次方是几，可以加速
        '''
        self.valid_x_range = [x_0, x_end]

        if x_0 != 0:
            # general 的计算方法
            A = np.array([
                [x_0 ** 4, x_0 ** 3, x_0 ** 2, x_0, 1],
                [4 * x_0 ** 3, 3 * x_0 ** 2, 2 * x_0, 1, 0],
                [12 * x_0 ** 2, 6 * x_0, 2, 0, 0],
                # [x_end ** 4, x_end ** 3, x_end ** 2, x_end, 1],
                [4 * x_end ** 3, 3 * x_end ** 2, 2 * x_end, 1, 0],
                [12 * x_end ** 2, 6 * x_end, 2, 0, 0]
            ])
            b = np.array([y_0, y_prime_0, y_2prime_0, y_prime_end, y_2prime_end])
            coef = np.linalg.solve(A,b)
            self.set_coef(coef)
        else:
            # x_0 = 0特殊条件下的计算，仅用于加速，数值上等效于general情况下的方法，实验提速1/3
            a0 = y_0
            a1 = y_prime_0
            a2 = y_2prime_0 / 2.0

            A = np.array([[4 * x_end ** 3, 3 * x_end ** 2],
                          [12 * x_end ** 2, 6 * x_end]])
            b = np.array([y_prime_end - a1 - 2 * a2 * x_end,
                          y_2prime_end - 2 * a2])
            temp = np.linalg.solve(A, b)

            a4 = temp[0]
            a3 = temp[1]
            self.set_coef([a4, a3, a2, a1, a0])


    @classmethod
    def from_kine_states(cls, xs, vxs, axs, vxe, axe, te):
        qp = cls()
        qp.two_point_boundary_value(0., xs, vxs, axs, te, vxe, axe)
        return qp

    # def calc_point(self, t):
    #     if t > self.t_range:
    #         xt = self.calc_point(self.t_range) + (t-self.t_range) * self.calc_first_derivative(self.t_range)
    #         return xt
    #
    #     xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
    #          self.a3 * t ** 3 + self.a4 * t ** 4
    #
    #     return xt
    #
    # def calc_first_derivative(self, t):
    #     if t > self.t_range:
    #         xt = self.calc_first_derivative(self.t_range)
    #         return xt
    #
    #     xt = self.a1 + 2 * self.a2 * t + \
    #          3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3
    #
    #     return xt
    #
    # def calc_second_derivative(self, t):
    #     if t > self.t_range:
    #         xt = 0
    #         return xt
    #
    #     xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2
    #
    #     return xt
    #
    # def calc_third_derivative(self, t):
    #     if t > self.t_range:
    #         xt = 0
    #         return xt
    #
    #     xt = 6 * self.a3 + 24 * self.a4 * t
    #
    #     return xt

class QuinticPolynomial(BasePolynomial):

    def __init__(self, coef=None, valid_x_range=None):
        super(QuinticPolynomial, self).__init__(coef, valid_x_range)

    def two_point_boundary_value(self, x_0, y_0, y_prime_0, y_2prime_0, x_end, y_end, y_prime_end, y_2prime_end):
        '''
        interpolate with two_point_boundary_value constraints.
        And it is recommended to set x_0 to 0 to accelerate the calculation
        What's more, a large number (like x>1e5) could cause Ill-Conditioned Matrix!
        todo: 可以提前储存几次方是几，可以加速
        '''
        if x_0 > 1e5: warnings.warn("WARNING! A large number (like x>1e5) could cause ill-Conditioned Matrix!")

        self.valid_x_range = [x_0, x_end]

        if x_0 != 0:
            # general 的计算方法
            A = np.array([
                [x_0 ** 5, x_0 ** 4, x_0 ** 3, x_0 ** 2, x_0, 1],
                [5 * x_0 ** 4, 4 * x_0 ** 3, 3 * x_0 ** 2, 2 * x_0, 1, 0],
                [20 * x_0 ** 3, 12 * x_0 ** 2, 6 * x_0, 2, 0, 0],
                [x_end ** 5, x_end ** 4, x_end ** 3, x_end ** 2, x_end, 1],
                [5 * x_end ** 4, 4 * x_end ** 3, 3 * x_end ** 2, 2 * x_end, 1, 0],
                [20 * x_end ** 3, 12 * x_end ** 2, 6 * x_end, 2, 0, 0]
            ])
            b = np.array([y_0, y_prime_0, y_2prime_0, y_end, y_prime_end, y_2prime_end])
            coef = np.linalg.solve(A, b)
            self.set_coef(coef)
        else:
            # x_0 = 0特殊条件下的计算，仅用于加速，数值上等效于general情况下的方法，实验提速1/3
            a0 = y_0
            a1 = y_prime_0
            a2 = y_2prime_0 / 2.0

            A = np.array([
                [x_end ** 5, x_end ** 4, x_end ** 3],
                [5 * x_end ** 4, 4 * x_end ** 3, 3 * x_end ** 2],
                [20 * x_end ** 3, 12 * x_end ** 2, 6 * x_end]
            ])
            b = np.array([
                y_end - a0 - a1 * x_end - a2 * x_end ** 2,
                y_prime_end - a1 - 2 * a2 * x_end,
                y_2prime_end - 2 * a2
            ])

            temp = np.linalg.solve(A, b)

            a5, a4, a3 = temp
            self.set_coef([a5, a4, a3, a2, a1, a0])

    @classmethod
    def from_kine_states(cls, xs, vxs, axs, xe, vxe, axe, te):
        qp = cls()
        qp.two_point_boundary_value(0., xs, vxs, axs, te, xe, vxe, axe)
        return qp


class PiecewiseQuinticPolynomial(ExplicitCurve):
    def __init__(self, all_points_with_derivatives=None, segment_num=None):
        super(PiecewiseQuinticPolynomial, self).__init__()
        self.segment_num = segment_num
        self.segments:List[QuinticPolynomial] = None if segment_num is None else \
            [QuinticPolynomial() for _ in range(segment_num)]
        self.critical_points = None # 所有临界点的x

        if not (all_points_with_derivatives is None):
            self.calc_coef(all_points_with_derivatives)

    def calc_coef(self, all_points_with_derivatives:np.ndarray):
        '''
        all_points_with_derivatives:
        np.array([
            [x0, y0, y_prime0, y_2prime0],
            [x1, y1, y_prime1, y_2prime1],
            ...
            [xn, yn, y_prime_n, y_2prime_n]
        ])
        '''
        all_points_with_derivatives = np.asarray(all_points_with_derivatives)
        if self.segment_num is None:
            self.segment_num = len(all_points_with_derivatives) - 1
            self.segments: List[QuinticPolynomial] = [QuinticPolynomial() for _ in range(self.segment_num)]
        if all_points_with_derivatives.shape[0] == self.segment_num + 1 \
                and all_points_with_derivatives.shape[1] == 4:
            pass
        else:
            print("Expect %d points with %d features." % (self.segment_num + 1, 4))
            raise ValueError(
                '''
                Wrong size of all_points_with_derivatives. Did you miss any points or miss any information about each point?
                Here is the definition of all_points_with_derivatives:
                np.array([
                    [x0, y0, y_prime0, y_2prime0],
                    [x1, y1, y_prime1, y_2prime1],
                    ...
                    [xn, yn, y_prime_n, y_2prime_n]
                ])
                '''
            )
        for i in range(len(all_points_with_derivatives) - 1):
            # 给每一段曲线插值计算参数
            start = all_points_with_derivatives[i]
            end = all_points_with_derivatives[i+1]
            self.segments[i].two_point_boundary_value(start[0], start[1], start[2], start[3],
                                                      end[0], end[1], end[2], end[3])

        self.critical_points = all_points_with_derivatives[:,0]

    def search_segment_idx(self, x):
        # x需要是scalar
        # n个segment, n+1个临界点, n+2个区间
        idx = bisect.bisect(self.critical_points, x) - 1
        idx = max([0, idx])
        idx = min([idx, self.segment_num-1])
        return idx

    def calc_point(self, x):
        if self._isscalar(x):
            idx = self.search_segment_idx(x)
            val = self.segments[idx](x, order=0)
        else:
            val = np.array([self.evaluate(xi, order=0) for xi in x])
        return val

    def calc_first_derivative(self, x):
        if self._isscalar(x):
            idx = self.search_segment_idx(x)
            val = self.segments[idx](x, order=1)
        else:
            val = np.array([self.evaluate(xi, order=1) for xi in x])
        return val

    def calc_second_derivative(self, x):
        if self._isscalar(x):
            idx = self.search_segment_idx(x)
            val = self.segments[idx](x, order=2)
        else:
            val = np.array([self.evaluate(xi, order=2) for xi in x])
        return val

    def calc_third_derivative(self, x):
        if self._isscalar(x):
            idx = self.search_segment_idx(x)
            val = self.segments[idx](x, order=3)
        else:
            val = np.array([self.evaluate(xi, order=3) for xi in x])
        return val






################ 插值曲线 ################
class InterpolationCurve(ExplicitCurve):
    def __init__(self, x=None, y=None):
        super(InterpolationCurve, self).__init__()
        self.x, self.y = None, None
        self.nx = 0
        self.valid_x_range = [-np.inf,np.inf]

        if not ((x is None) or (y is None)):
            self.set_data(x, y)

    def set_data(self, x, y):
        self.x, self.y = np.array(x), np.array(y)
        self.nx = len(x)
        self.valid_x_range = [x[0], x[-1]]
        self._calc_coef()

    @abstractmethod
    def _calc_coef(self):
        pass


    def _out_of_range_flag(self, x) -> Union[bool, np.ndarray]:
        '''
        if x is a scalar, return bool
        if x is an array, return a boolean array with the same size
        '''
        return (x<self.valid_x_range[0]) | (x>self.valid_x_range[1]) # 不可以用or， 因为要考虑array情况

    @abstractmethod
    def interpolate(self, x, order):
        pass

    @abstractmethod
    def extrapolate(self, x, order):
        pass

    def calc_point(self, x):
        # val = self.interpolate(x, order=0)
        x = np.asarray(x, dtype=float)
        extra_mask = self._out_of_range_flag(x)
        val = np.empty_like(x)
        val[extra_mask] = self.extrapolate(x[extra_mask], order=0)  # 数据范围外，按外推规则计算
        val[~extra_mask] = self.interpolate(x[~extra_mask], order=0)  # 数据范围内，插值计算
        # todo：判断是否在数据范围外的逻辑现在写的太耗时了！降低运算效率70%
        return val

    def calc_first_derivative(self, x):
        x = np.asarray(x, dtype=float)
        extra_mask = self._out_of_range_flag(x)
        val = np.empty_like(x)
        val[extra_mask] = self.extrapolate(x[extra_mask], order=1)  # 数据范围外，按外推规则计算
        val[~extra_mask] = self.interpolate(x[~extra_mask], order=1)  # 数据范围内，插值计算
        return val

    def calc_second_derivative(self, x):
        x = np.asarray(x, dtype=float)
        extra_mask = self._out_of_range_flag(x)
        val = np.empty_like(x)
        val[extra_mask] = self.extrapolate(x[extra_mask], order=2)  # 数据范围外，按外推规则计算
        val[~extra_mask] = self.interpolate(x[~extra_mask], order=2)  # 数据范围内，插值计算
        return val

    def calc_third_derivative(self, x):
        x = np.asarray(x, dtype=float)
        extra_mask = self._out_of_range_flag(x)
        val = np.empty_like(x)
        val[extra_mask] = self.extrapolate(x[extra_mask], order=3)  # 数据范围外，按外推规则计算
        val[~extra_mask] = self.interpolate(x[~extra_mask], order=3)  # 数据范围内，插值计算
        return val


class spCubicSpline(InterpolationCurve):
    """
    CubicSpline class
    三次样条插值
    """
    def __init__(self, x=None, y=None, bc_type='natural'):
        self._sp_csp = None # 屈服了，还是用Scipy的吧
        self.bc_type = bc_type
        super(spCubicSpline, self).__init__(x, y)

    def _calc_coef(self):
        self._sp_csp = scipy.interpolate.CubicSpline(self.x, self.y, bc_type=self.bc_type)#bc_type=((1, 0), (1, 0)))
        #，bc_type=((1, 0), (1, 0)) 指定了第一个和最后一个插值点处的一阶导数为0
        # (1, 0) 表示一阶导数为0，(2, 0) 表示二阶导数为0
        # 不加这个限制的话会出现非常大的振荡,，因为默认Not-a-knot Spline (非结点样条)
        # Natural Spline (自然样条): 在这种情况下，样条的二阶导数在两个端点都被设定为零。这是默认的边界条件。
        # Clamped Spline (夹持样条): 在这种情况下，你可以指定首尾两个点的一阶导数值。这相当于夹持（固定）样条的两个端点。
        # Not-a-knot Spline (非结点样条): 在这种情况下，样条的三阶导数在内部节点上是连续的。这种条件通常在要求样条在内部节点处更平滑的情况下使用。

    def interpolate(self, x, order):
        if np.asarray(x).size == 0:
            return np.empty_like(x)

        val = self._sp_csp(x, order)
        return val

    def extrapolate(self, x, order):
        '''
        在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
        please notice that, if any x in the valid_x_range, the corresponding y maintains 0 value
        '''
        val = np.zeros_like(x)
        if val.size == 0:
            return val

        x_0, x_end = self.valid_x_range
        if order == 0:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            y_0 = self.evaluate(x_0, order=0)
            y_end = self.evaluate(x_end, order=0)
            val[x > x_end] = y_end + y_prime_end * (x[x>x_end] - x_end)
            val[x < x_0] = y_0 + y_prime_0 * (x[x < x_end] - x_0)
        elif order == 1:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            val[x > x_end] = y_prime_end
            val[x < x_0] = y_prime_0
        else:
            pass  # because the higher order derivative is 0 anyway.

        return val

class myCubicSpline(InterpolationCurve):
    """
    Cubic CubicSpline class
    分段三次函数
    """
    def __init__(self, x=None, y=None):
        self.b, self.c, self.d, self.w = [], [], [], []
        # 一定要把需要在calc_coef里面计算用到的属性定义在前面，因为下一句可能会调用calc_coef
        super(myCubicSpline, self).__init__(x, y)


    def _calc_coef(self):
        self.a = list(self.y)
        h = np.diff(self.x)
        # calc coefficient c
        A = self._calc_A(h)
        B = self._calc_B(h)
        self.c = np.linalg.solve(A, B)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                 (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

        self.a, self.b, self.c, self.d, self.w = np.array(self.a), np.array(self.b), np.array(self.c), np.array(self.d), np.array(self.w)


    def _calc_A(self, h):
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

    def _calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                       h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        #  print(B)

        return B

    def _search_index(self, x):
        """
        search data segment index
        """
        if self._isscalar(x):
            pt_idx = bisect.bisect(self.x, x) - 1  # 二分查找插入点的索引
            seg_idx = pt_idx if pt_idx != self.nx - 1 else pt_idx - 1 #如果是最后一个点，那么segment id要往前挪一个
            return seg_idx
        else:
            pt_idxs = [bisect.bisect(self.x, xi) - 1 for xi in x]  # 二分查找插入点的索引
            seg_idxs = np.array(pt_idxs)
            seg_idxs[seg_idxs==self.nx-1] -= 1
            return seg_idxs #如果是最后一个点，那么segment id要往前挪一个

    def interpolate(self, x, order):
        if np.asarray(x).size == 0:
            return np.empty_like(x)

        i = self._search_index(x)
        dx = x - self.x[i]
        if order == 0:
            val = np.polyval([self.d[i], self.c[i],self.b[i], self.a[i]], dx)
        elif order == 1:
            val = np.polyval([3*self.d[i], 2*self.c[i], self.b[i]], dx)
        elif order == 2:
            val = np.polyval([6 * self.d[i], 2 * self.c[i]], dx)
        elif order == 3:
            val = 6.0 * self.d[i]
        else:
            val = 0.
        # if order == 0:
        #     val = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
        # elif order == 1:
        #     val = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        # elif order == 2:
        #     val = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        # elif order == 3:
        #     val = 6.0 * self.d[i]
        # else:
        #     val = 0.
        return val

    def extrapolate(self, x, order):
        '''
        在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
        please notice that, if any x in the valid_x_range, the corresponding y maintains 0 value
        '''
        val = np.zeros_like(x)
        if val.size == 0:
            return val

        x_0, x_end = self.valid_x_range
        if order == 0:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            y_0 = self.evaluate(x_0, order=0)
            y_end = self.evaluate(x_end, order=0)
            val[x > x_end] = y_end + y_prime_end * (x[x>x_end] - x_end)
            val[x < x_0] = y_0 + y_prime_0 * (x[x < x_end] - x_0)
        elif order == 1:
            y_prime_0 = self.evaluate(x_0, order=1)
            y_prime_end = self.evaluate(x_end, order=1)
            val[x > x_end] = y_prime_end
            val[x < x_0] = y_prime_0
        else:
            pass  # because the higher order derivative is 0 anyway.

        return val


class CubicSpline(spCubicSpline):
    pass

class Spline:#Cubic CubicSpline class(abandoned)
    def __init__(self, x, y):
        raise AssertionError("Spline class has been deprecated! Use CubicSpline instead!")

################ 线性插值曲线 ################


################ 三次样条曲线 ################
# class Spline:
#     """
#     Cubic CubicSpline class
#     """
#
#     def __init__(self, x, y):
#         self.b, self.c, self.d, self.w = [], [], [], []
#
#         self.x = x
#         self.y = y
#
#         self.nx = len(x)  # dimension of x
#         h = np.diff(x)
#
#         # calc coefficient c
#         self.a = [iy for iy in y]
#
#         # calc coefficient c
#         A = self._calc_A(h)
#         B = self._calc_B(h)
#         self.c = np.linalg.solve(A, B)
#
#         # calc spline coefficient b and d
#         for i in range(self.nx - 1):
#             self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
#             tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
#                  (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
#             self.b.append(tb)
#
#     def calc(self, t):
#         """
#         Calc position
#
#         if x is outside of the input x, return None
#
#         """
#
#         if t < self.x[0]:
#             return None
#         elif t > self.x[-1]:
#             return None
#         i = self._search_index(t)
#         dx = t - self.x[i]
#         result = self.a[i] + self.b[i] * dx + self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0
#
#         return result
#
#     def calcd(self, t):
#         """
#         Calc first derivative
#
#         if x is outside of the input x, return None
#         """
#
#         if t < self.x[0]:
#             return None
#         elif t > self.x[-1]:
#             return None
#
#         i = self._search_index(t)
#         dx = t - self.x[i]
#         result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
#         return result
#
#     def calcdd(self, t):
#         """
#         Calc second derivative
#         """
#
#         if t < self.x[0]:
#             return None
#         elif t > self.x[-1]:
#             return None
#
#         i = self._search_index(t)
#         dx = t - self.x[i]
#         result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
#         return result
#
#     def _search_index(self, x):
#         """
#         search data segment index
#         """
#         return bisect.bisect(self.x, x) - 1
#
#     def _calc_A(self, h):
#         """
#         calc matrix A for spline coefficient c
#         """
#         A = np.zeros((self.nx, self.nx))
#         A[0, 0] = 1.0
#         for i in range(self.nx - 1):
#             if i != (self.nx - 2):
#                 A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
#             A[i + 1, i] = h[i]
#             A[i, i + 1] = h[i]
#
#         A[0, 1] = 0.0
#         A[self.nx - 1, self.nx - 2] = 0.0
#         A[self.nx - 1, self.nx - 1] = 1.0
#         #  print(A)
#         return A
#
#     def _calc_B(self, h):
#         """
#         calc matrix B for spline coefficient c
#         """
#         B = np.zeros(self.nx)
#         for i in range(self.nx - 2):
#             B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
#                        h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
#         #  print(B)
#
#         return B


class Spline2D:#2D Cubic CubicSpline class(abandoned)
    def __init__(self, x, y):
        raise AssertionError("Spline2D class has been deprecated! Use ParametricCubicSpline instead!")
    #     self.s = self._calc_s(x, y)
    #     self.sx = CubicSpline(self.s, x)
    #     self.sy = CubicSpline(self.s, y)
    #
    # def _calc_s(self, x, y):
    #     dx = np.diff(x)
    #     dy = np.diff(y)
    #     self.ds = [math.sqrt(idx ** 2 + idy ** 2)
    #                for (idx, idy) in zip(dx, dy)]
    #     s = [0]
    #     s.extend(np.cumsum(self.ds))
    #     return s
    #
    # def calc_position(self, s):
    #     """
    #     calc position
    #     """
    #     x = self.sx.calc(s)
    #     y = self.sy.calc(s)
    #
    #     return x, y
    #
    # def calc_curvature(self, s):
    #     """
    #     calc curvature
    #     """
    #     dx = self.sx.calcd(s)
    #     ddx = self.sx.calcdd(s)
    #     dy = self.sy.calcd(s)
    #     ddy = self.sy.calcdd(s)
    #     k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2)
    #     return k
    #
    # def calc_yaw(self, s):
    #     """
    #     calc yaw
    #     """
    #     dx = self.sx.calcd(s)
    #     dy = self.sy.calcd(s)
    #     yaw = math.atan2(dy, dx)
    #     return yaw



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

    def calc_yaw(self, s):
        dx, dy = self.calc_first_derivative(s)
        yaw = np.arctan2(dy, dx)
        return yaw

    def calc_curvature(self, s, absolute:bool=False):
        dx, dy = self.calc_first_derivative(s)
        ddx, ddy = self.calc_second_derivative(s)
        if absolute:
            k = np.abs(ddy*dx - ddx*dy) / (dx**2+dy**2) ** 1.5
        else:
            k = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** 1.5
        return k

    def _isscalar(self, x):
        return np.isscalar(x) or np.ndim(x)==0



################# 参数化3阶样条曲线 #################

class ParametricCubicSpline(ParametricCurve):
    """
    2D Cubic CubicSpline class
    本质是关于s的x,y二维参数化曲线
    """

    def __init__(self, x, y):
        super(ParametricCubicSpline, self).__init__()
        self.s = self._calc_s(x, y)
        self.x = np.array(x)
        self.y = np.array(y)
        self.sx = CubicSpline(self.s, x) # 以s为自变量，以x为因变量的三阶样条曲线
        self.sy = CubicSpline(self.s, y) # 以s为自变量，以y为因变量的三阶样条曲线

    def _calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = [math.sqrt(idx ** 2 + idy ** 2)
                   for (idx, idy) in zip(dx, dy)]
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_point(self, s):
        x = self.sx.calc_point(s)
        y = self.sy.calc_point(s)
        return np.array([x,y]).T#x, y

    def calc_first_derivative(self, s):
        dx = self.sx.calc_first_derivative(s)
        dy = self.sy.calc_first_derivative(s)
        return np.array([dx, dy]).T #dx, dy

    def calc_second_derivative(self, s):
        ddx = self.sx.calc_second_derivative(s)
        ddy = self.sy.calc_second_derivative(s)
        return np.array([ddx, ddy]).T #ddx, ddy

    def calc_third_derivative(self, s):
        dddx = self.sx.calc_third_derivative(s)
        dddy = self.sy.calc_third_derivative(s)
        return np.array([dddx, dddy]).T #dddx, dddy


################# 贝塞尔曲线 #####################
class BezierCurve(ParametricCurve):
    # 预先计算二项式系数
    # 类变量的定义语句在类被定义时就会执行，并且只执行一次, 多次初始化多个类的实例时，类变量的定义语句不会被重新执行
    _binom_coeff_dict = {n: binom(n, np.arange(n+1)) for n in range(1, 9)} # 预先算至多8个控制点的情况
    # 字典中,n:[B_n_0, B_n_1,...B_n_n]储存了n下的所有的二项式系数

    def __init__(self, control_points, pre_calculation=True):
        super(BezierCurve, self).__init__()
        assert len(control_points) >= 2
        self.n = len(control_points) - 1 # 贝塞尔曲线的阶数
        self.control_points = np.asarray(control_points)

        # self._bezier_coeff_dict = {}
        # self._binom_coeff_dict = {}

        self._derivative_bezier_curve = None
        self._arclength = None
        self._s2t = None # displacement to parameter t

        # if pre_calculation:
        #     # 提前计算二项式系数
        #     values = binom(self.n, np.arange(self.n+1))
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
            # self._arclength = _arclength(pts, closed=False)# 用cv2的函数会比自己写的稍快一些
            #self._arclength = cv2.arcLength(pts, closed=False)
            delta_xy = np.diff(pts, axis=0)
            delta = np.linalg.norm(delta_xy,axis=1)
            ss = np.insert(np.cumsum(delta), 0, 0.0)
            self._s2t = scipy.interpolate.interp1d(ss, ts, kind='linear')
            self._arclength = self._s2t.x[-1]# np.sum(delta)
        return self._arclength

    def get_t_for_displacement(self, s):
        if self._s2t is None:
            ts = np.linspace(0, 1, 500)
            pts = self.calc_point_t(ts).astype(dtype=np.float32)
            delta_xy = np.diff(pts, axis=0)
            delta = np.linalg.norm(delta_xy,axis=1)
            ss = np.cumsum(delta)
            ss = np.insert(ss, 0, 0.0)
            self._s2t = scipy.interpolate.interp1d(ss, ts, kind='linear')
        return self._s2t(s)


    def _binom(self, n, i):
        assert i >= 0 and i <= n
        if n in self._binom_coeff_dict:
            return self._binom_coeff_dict[n][i]
        else:
            self._binom_coeff_dict[n] = binom(n, np.arange(n+1))
            return self._binom_coeff_dict[n][i]


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

        t = np.asarray(t)
        # assert np.all(t <= 1) and np.all(t >= 0)

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

    def _out_of_range_flag(self, s) -> Union[bool, np.ndarray]:
        '''
        if x is a scalar, return bool
        if x is an array, return a boolean array with the same size
        '''
        return (s < 0.) | (s > self.arclength)  # 不可以用or， 因为要考虑array情况

    def calc_point(self, s):
        s = np.asarray(s, dtype=float)
        extra_mask = self._out_of_range_flag(s)
        if np.any(extra_mask):
            val = np.empty(shape=(*s.shape, 2))
            val[extra_mask] = self.extrapolate(s[extra_mask], order=0)  # 数据范围外，按外推规则计算
            val[~extra_mask] = self.calc_point_t(self.get_t_for_displacement(s[~extra_mask]))  # 数据范围内，直接计算
        else:
            val = self.calc_point_t(self.get_t_for_displacement(s))

        return val

    def calc_first_derivative(self, s):
        s = np.asarray(s, dtype=float)
        extra_mask = self._out_of_range_flag(s)
        if np.any(extra_mask):
            val = np.empty(shape=(*s.shape, 2))
            val[extra_mask] = self.extrapolate(s[extra_mask], order=1)  # 数据范围外，按外推规则计算
            val[~extra_mask] = self.calc_first_derivative_t(self.get_t_for_displacement(s[~extra_mask]))  # 数据范围内，直接计算
        else:
            val = self.calc_first_derivative_t(self.get_t_for_displacement(s))

        return val

    def calc_second_derivative(self, s):
        s = np.asarray(s, dtype=float)
        extra_mask = self._out_of_range_flag(s)
        if np.any(extra_mask):
            val = np.empty(shape=(*s.shape, 2))
            val[extra_mask] = self.extrapolate(s[extra_mask], order=2)  # 数据范围外，按外推规则计算
            val[~extra_mask] = self.calc_second_derivative_t(self.get_t_for_displacement(s[~extra_mask]))  # 数据范围内，直接计算
        else:
            val = self.calc_second_derivative_t(self.get_t_for_displacement(s))

        return val

    def calc_third_derivative(self, s):
        s = np.asarray(s, dtype=float)
        extra_mask = self._out_of_range_flag(s)
        if np.any(extra_mask):
            val = np.empty(shape=(*s.shape, 2))
            val[extra_mask] = self.extrapolate(s[extra_mask], order=3)  # 数据范围外，按外推规则计算
            val[~extra_mask] = self.calc_third_derivative_t(self.get_t_for_displacement(s[~extra_mask]))  # 数据范围内，直接计算
        else:
            val = self.calc_third_derivative_t(self.get_t_for_displacement(s))

        return val

    # def calc_first_derivative(self, s):
    #     return self.calc_first_derivative_t( np.asarray(s) / self.arclength )
    #
    # def calc_second_derivative(self, s):
    #     return self.calc_second_derivative_t( np.asarray(s) / self.arclength )
    #
    # def calc_third_derivative(self, s):
    #     return self.calc_third_derivative_t( np.asarray(s) / self.arclength )

    def extrapolate(self, s, order):
        '''
        在超出已知数据范围的点上进行插值/外推，默认保持二阶导为0然后外推，即保持起点或终点的斜率
        please notice that, if any x in the valid_x_range, the corresponding y maintains 0 value
        '''
        val = np.zeros(shape=(*s.shape, 2))
        if val.size == 0:
            return val

        s_0, s_end = 0., self.arclength
        if order == 0:
            yaw_0 = self.calc_yaw(s_0)
            yaw_e = self.calc_yaw(s_end)
            # dx_ds_0, dy_ds_0 = self.evaluate(s_0, order=1).T
            # dx_ds_e, dy_ds_e = self.evaluate(s_end, order=1).T
            x_0, y_0 = self.evaluate(s_0, order=0)
            x_e, y_e = self.evaluate(s_end, order=0)

            delta_s_e = s[s > s_end] - s_end
            val[s > s_end] = np.array([
                x_e + np.cos(yaw_e) * delta_s_e,
                y_e + np.sin(yaw_e) * delta_s_e
            ]).T

            delta_s_0 = s[s < s_0] - s_0
            val[s < s_0] = np.array([
                x_0 + np.cos(yaw_0) * delta_s_0,
                y_e + np.sin(yaw_0) * delta_s_0
            ]).T
        elif order == 1:
            dx_ds_0, dy_ds_0 = self.evaluate(s_0, order=1).T
            dx_ds_e, dy_ds_e = self.evaluate(s_end, order=1).T

            val[s > s_end] = np.array([dx_ds_e, dy_ds_e]).T
            val[s < s_0] = np.array([dx_ds_0, dy_ds_0]).T
        else:
            pass  # because the higher order derivative is 0 anyway.

        return val





################## 基础的隐式方程表达的曲线类 #####################
class ImplicitCurve:
    def __init__(self):
        raise NotImplementedError("Wait for completion. Can be used for curves like circle and eclipse")



if __name__ == '__main__':
    # cp = np.array([[0, 0], [2, 3], [4, 4],[8,10]])
    # t = 0.1
    # c = BezierCurve(cp)
    # x = c.calc_point_t(t)
    # dx = c.calc_first_derivative_t(t)
    # ddx = c.calc_second_derivative_t(t)
    # dddx = c.calc_third_derivative_t(t)
    #
    # from time import time
    # t1 = time()
    # for _ in range(10000):
    #     a = c.arclength
    # t2 = time()
    # for _ in range(10000):
    #     # a = {n: binom(n, np.arange(n + 1)) for n in range(1, 9)}
    #     b = c.arclength2
    # t3 = time()
    #
    # print(t2-t1)
    # print(t3-t2)
    # print(a)
    # print(b)


    pass
