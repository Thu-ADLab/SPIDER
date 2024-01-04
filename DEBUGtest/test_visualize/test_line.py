from spider.visualize.line import *

from copy import deepcopy
plt.figure(figsize=(15,3))

steps, dt = 30, 0.2
traj = Trajectory(steps, dt)
s0, s_d0, s_dd0 = 0., 60 / 3.6, 0.
l0, l_d0, l_dd0 = 0., 0., 0.
traj.x = xs = np.array([s0 + s_d0 * i * dt for i in range(steps)])
traj.y = ys = np.array([l0 + 0.12 * i * i * dt * dt for i in range(steps)])
traj.heading = np.insert(np.arctan2(np.diff(traj.y), np.diff(traj.x)), 0, 0.0)
draw_trajectory(traj, '.-', show_footprint=True, footprint_fill=False)

traj2 = deepcopy(traj)
traj2.y *= -1
traj2.heading *= -1
draw_trajectory(traj2, '.-', show_footprint=True, footprint_fill=True)

traj3 = deepcopy(traj)
traj3.y *= 0
traj3.heading *= 0
# draw_trajectory(traj3, '.-', show_footprint=False)
draw_polyline(np.column_stack((traj3.x, traj3.y)),show_buffer=True)

plt.show()

