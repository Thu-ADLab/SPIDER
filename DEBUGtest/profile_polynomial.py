import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

valid_x_range = [0,20]

class tic:
    t1 = -1
    i = 0
    def toc(self):
        if self.t1 < 0:
            self.t1 = time.time()
        else:
            t2 = time.time()
            print(t2-self.t1,'--'+str(self.i))
            self.t1 = t2


def _out_of_range_flag(x):
    '''
    if x is a scalar, return bool
    if x is an array, return a boolean array with the same size
    '''
    return (x < valid_x_range[0]) | (x > valid_x_range[1])  # 不可以用or， 因为要考虑array情况


coef = [1,2,3,4]
x = np.linspace(0,19,50)
t = tic()

poly_func = np.poly1d(coef)


# from scipy.interpolate import CubicSpline as spcsp
# csp = spcsp(x, np.random.rand(x.size),bc_type='natural')
# yy = csp(np.linspace(0,20,50))


def calculate1(x):
    x = np.asarray(x, dtype=float)
    extra_mask = _out_of_range_flag(x)
    val = np.empty_like(x)
    val[extra_mask] = np.polyval(coef, x[-1]) * np.ones(np.sum(extra_mask))  # 数据范围外，按外推规则计算
    val[~extra_mask] = np.polyval(coef, x[~extra_mask]) # 数据范围内，直接计算
    return val

def calculate2(x):
    x = np.asarray(x, dtype=float)
    extra_mask = _out_of_range_flag(x)
    val = np.empty_like(x)
    val[extra_mask] = poly_func(x[-1]) * np.ones(np.sum(extra_mask))  # 数据范围外，按外推规则计算
    val[~extra_mask] = poly_func(x[~extra_mask])  # 数据范围内，直接计算
    return val

def calculate3(x):
    val = poly_func(x)
    return val

def calculate4(x):
    val = np.polyval(coef, x)
    return val

def calculate5(x):
    # val = np.polyval(coef, x)
    val = np.where((x < valid_x_range[0]) | (x > valid_x_range[1]), poly_func(x[-1]), poly_func(x))
    return val
    # extra_mask = _out_of_range_flag(x)
    # val[extra_mask] = poly_func(x[-1]) * np.ones(np.sum(extra_mask))  # 数据范围外，按外推规则计算

def calculate6(x):
    x = np.asarray(x, dtype=float)
    if (x.min()< valid_x_range[0]) or (x.max() > valid_x_range[1]):
        val = np.where((x < valid_x_range[0]) | (x > valid_x_range[1]), poly_func(x), poly_func(x[-1]))
        # extra_mask = _out_of_range_flag(x)
        # val = np.empty_like(x)
        # val[extra_mask] = np.polyval(coef, x[-1]) * np.ones(np.sum(extra_mask))  # 数据范围外，按外推规则计算
        # val[~extra_mask] = np.polyval(coef, x[~extra_mask])  # 数据范围内，直接计算
    else:
        val = np.polyval(coef, x)
    return val

import pyximport
pyximport.install( language_level=3)#'../elements/poly_calc.pyx',
poly_calc = pyximport.load_module('poly_calc', '../elements/poly_calc.pyx', language_level=3)
# import poly_calc
# # 导入编译后的模块

y_0, y_prime_0, y_end, y_prime_end = 1,-1,10,1

def calculate0(x):
    # x = np.asarray(x)
    if np.isscalar(x) and np.ndim(x) == 0:
        assert 0
    x = np.ascontiguousarray(x, dtype=np.float64)
    c = np.ascontiguousarray(coef, dtype=np.float64)
    val = poly_calc.evaluate(x, c, valid_x_range[0], valid_x_range[1], 0, y_0, y_prime_0, y_end, y_prime_end)
    return np.asarray(val)



# import cProfile
# cProfile.run('for _ in range(10000): calculate1(x)')
all_funcs = [calculate0, calculate1, calculate2, calculate3, calculate4,calculate5,calculate6]

all_val = [func(x) for func in all_funcs]

for func in all_funcs:
    for _ in tqdm(range(100000),desc=str(func.__name__)):
        val = func(x)
    # print(val)


