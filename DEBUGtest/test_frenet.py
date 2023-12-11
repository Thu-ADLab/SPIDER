import numpy as np
from spider.utils.transform.frenet import FrenetCoordinateTransformer
from spider.elements.trajectory import FrenetTrajectory


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