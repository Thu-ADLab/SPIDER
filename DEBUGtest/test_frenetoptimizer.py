import numpy as np
import matplotlib.pyplot as plt

from spider.elements.trajectory import Path, Trajectory, FrenetTrajectory
from spider.elements import TrackingBoxList
from spider.utils.collision.AABB import AABB_vertices
from spider.elements import TrackingBox
from spider.visualize import draw_polygon
from spider.optimize.TrajectoryOptimizer import FrenetTrajectoryOptimizer




# def draw_polygon(vertices, color='black', lw=1.):
#     vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
#     plt.plot(vertices[:, 0], vertices[:, 1], color=color, linestyle='-', linewidth=lw)

N = steps = 50

dt = 0.1


optim = FrenetTrajectoryOptimizer(50,0.1)

veh_length = 5
veh_width = 2
wheelbase = 3.0

# observation
bboxes = TrackingBoxList()
obs = [
    [15, 3.5 * 1, 3, 0],
    [30, 3.5 * 1, 8, 0],
    [40, -0.7, 0, 0]
]

for x, y, vx, vy in obs:
    vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
    tb = TrackingBox(vertices=vertices, vx=vx, vy=vy)
    bboxes.append(tb)
bboxes.predict([dt * i for i in range(steps)])

s0, s_d0, s_dd0 = 0., 30 / 3.6, 0.
l0, l_d0, l_dd0 = 0., 0., 0.
target_l = 3.5
target_l_bound = [target_l - 3.5 * 0.5, target_l + 3.5 * 0.5]

initial_guess_s = np.array([s0 + s_d0 * i * dt for i in range(steps)])
initial_guess_d = np.array([l0 + 0.12 * i * i * dt * dt for i in range(steps)])
# initial_guess = np.concatenate((initial_guess_s, initial_guess_d))
initial_frenet_trajectory = FrenetTrajectory(50, 0.1)
initial_frenet_trajectory.s, initial_frenet_trajectory.l = initial_guess_s, initial_guess_d
initial_frenet_trajectory.s_dot.append(s_d0)
initial_frenet_trajectory.l_dot.append(l_d0)
initial_frenet_trajectory.s_2dot.append(s_dd0)
initial_frenet_trajectory.l_2dot.append(l_dd0)

optim_traj = optim.optimize_traj(initial_frenet_trajectory,bboxes)
corridors = optim._corridors

for x, y, vx, vy in obs:
    vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
    draw_polygon(vertices, color='black')

for aabb in corridors:
    vertices = AABB_vertices(aabb)
    draw_polygon(vertices, color='green', lw=0.3)

plt.plot([0, 50], [3.5 / 2, 3.5 / 2], 'k--')
plt.plot([0, 50], [3.5 * 1.5, 3.5 * 1.5], 'k-')
plt.plot([0, 50], [-3.5 * 0.5, -3.5 * 0.5], 'k-')
plt.plot(initial_guess_s, initial_guess_d, '.-', label='initial guess')
plt.plot(optim_traj.s, optim_traj.l, '.-', label='optimized trajectory')
plt.show()
