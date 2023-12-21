import math
from spider.elements.curves import *
import matplotlib.pyplot as plt

######## 显式曲线 #######

x_0, x_end = 0., 30.
y_0, y_end = 0., 3.5
yaw0 = -10*math.pi/180
y_prime_0 = math.tan(yaw0)

yaw_end = 0*math.pi/180
y_prime_end = math.tan(yaw_end)

y_2prime_0, y_2prime_end = 0., 0.

poly3 = CubicPolynomial.from_kine_states(x_0, y_0, yaw0, x_end, y_end, yaw_end)
poly5 = QuinticPolynomial.from_kine_states(y_0, y_prime_0, y_2prime_0, y_end, y_prime_end, y_2prime_end, x_end)
poly4 = QuarticPolynomial.from_kine_states(y_0, y_prime_0, y_2prime_0, y_prime_end, y_2prime_end, x_end)

all_points_with_derivatives = np.array([
    [x_0, y_0, y_prime_0, y_2prime_0],
    [10, 3.5, 0., 0.],
    [20, 1.0, 0., 0.],
    [x_end, y_end, y_prime_end, y_2prime_end]
])
piecewise_poly5 = PiecewiseQuinticPolynomial(all_points_with_derivatives)



xx = np.linspace(x_0-2, x_end+2, 1000)
rows, cols = 2,2
plt.figure(figsize=(12,8))
plt.subplot(rows,cols,1)
# plt.title
plt.plot(xx, poly3(xx, order=0), label='cubic')
plt.plot(xx, poly4(xx, order=0), label='quartic')
plt.plot(xx, poly5(xx, order=0), label='quintic')
plt.plot(xx, piecewise_poly5(xx, order=0), label='piecewise_quintic')
plt.plot(x_0, poly3(x_0, order=0),'xr')
plt.plot(x_0, poly4(x_0, order=0),'xr')
plt.plot(x_0, poly5(x_0, order=0),'xr')
plt.plot(x_0, piecewise_poly5(x_0, order=0), 'xr')
plt.legend()
plt.subplot(rows,cols,2)
plt.plot(xx, poly3(xx, order=1), label='cubic')
plt.plot(xx, poly4(xx, order=1), label='quartic')
plt.plot(xx, poly5(xx, order=1), label='quintic')
plt.plot(xx, piecewise_poly5(xx, order=1), label='piecewise_quintic')
plt.legend()
plt.subplot(rows,cols,3)
plt.plot(xx, poly3(xx, order=2), label='cubic')
plt.plot(xx, poly4(xx, order=2), label='quartic')
plt.plot(xx, poly5(xx, order=2), label='quintic')
plt.plot(xx, piecewise_poly5(xx, order=2), label='piecewise_quintic')
plt.legend()
plt.subplot(rows,cols,4)
plt.plot(xx, poly3(xx, order=3), label='cubic')
plt.plot(xx, poly4(xx, order=3), label='quartic')
plt.plot(xx, poly5(xx, order=3), label='quintic')
plt.plot(xx, piecewise_poly5(xx, order=3), label='piecewise_quintic')
plt.legend()





plt.show()




