import numpy as np
import matplotlib.pyplot as plt

from spider.elements.trajectory import Path, Trajectory, FrenetTrajectory
from spider.elements import TrackingBoxList
from spider.utils.collision.AABB import AABB_vertices
from spider.elements import TrackingBox
import spider.visualize as vis
from spider.optimize.TrajectoryOptimizer import FrenetTrajectoryOptimizer
from spider.utils.transform.frenet import FrenetCoordinateTransformer



N = steps = 50

dt = 0.1

s0, s_d0, s_dd0 = 0., 30 / 3.6, 0.
l0, l_d0, l_dd0 = 0., 0., 0.
target_l = 0
target_l_bound = [target_l - 3.5 * 0.5, target_l + 3.5 * 0.5]

optim = FrenetTrajectoryOptimizer(steps, dt)

veh_length = 5
veh_width = 2
wheelbase = 3.0

# observation
bboxes = TrackingBoxList()
obs = [
    [15, 3.5 * 1, 3, 0],
    [30, 3.5 * 1, 8, 0],
    [40, -0.3, 0, 0]
]

for x, y, vx, vy in obs:
    vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
    tb = TrackingBox.from_vertices(vertices=vertices, vx=vx, vy=vy)
    bboxes.append(tb)
bboxes.predict([dt * i for i in range(steps)])

xs = np.linspace(0,80,80)
ys = np.linspace(0,0,80)
centerline = np.column_stack((xs, ys))
transformer = FrenetCoordinateTransformer(centerline)
bboxes = transformer.cart2frenet4boxes(bboxes, convert_prediction=True)




initial_guess_s = np.array([s0 + s_d0 * i * dt + 0.5*1*(i*dt)**2 for i in range(steps)])
initial_guess_d = np.array([l0 + 0.12 * i * i * dt * dt for i in range(steps)])
# initial_guess = np.concatenate((initial_guess_s, initial_guess_d))
initial_frenet_trajectory = FrenetTrajectory(steps, dt)
initial_frenet_trajectory.s, initial_frenet_trajectory.l = initial_guess_s, initial_guess_d
initial_frenet_trajectory.s_dot.append(s_d0)
initial_frenet_trajectory.l_dot.append(l_d0)
initial_frenet_trajectory.s_2dot.append(s_dd0)
initial_frenet_trajectory.l_2dot.append(l_dd0)

optim_traj = optim.optimize_traj(initial_frenet_trajectory,bboxes,offset_bound=(-3.5*0.5, 3.5*1.5),
                                 target_offset=target_l,target_offset_bound=target_l_bound)
corridors = optim.corridors

for x, y, vx, vy in obs:
    vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
    vis.draw_polygon(vertices, color='black')

for aabb in corridors:
    vertices = AABB_vertices(aabb)
    vis.draw_polygon(vertices, color='green', lw=0.5)

plt.plot([0, 50], [3.5 / 2, 3.5 / 2], 'k--')
plt.plot([0, 50], [3.5 * 1.5, 3.5 * 1.5], 'k-')
plt.plot([0, 50], [-3.5 * 0.5, -3.5 * 0.5], 'k-')
plt.plot(initial_guess_s, initial_guess_d, '.-', label='initial guess')
plt.plot(optim_traj.s, optim_traj.l, '.-', label='optimized trajectory')


plt.figure(figsize=(10, 6))
plt.axes(projection="3d")
plt.gca().view_init(elev=37, azim=-123, roll=0)
t_seq = np.asarray(optim_traj.t)

for i,aabb in enumerate(corridors):
    vertices = AABB_vertices(aabb)
    t = t_seq[i]
    vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    plt.plot(vertices[:, 0], vertices[:, 1], t, color=(0.2,0.8,0.2), linestyle='-')#, linewidth=0.6


plt.plot(initial_frenet_trajectory.s, initial_frenet_trajectory.l, t_seq, '.-', color='black')
plt.plot(optim_traj.s, optim_traj.l, t_seq, '.-', color='blue')


for x, y, vx, vy in obs:
    t_end = t_seq[-1]
    t0 = t_seq[0]
    bottom_vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
    top_vertices = AABB_vertices(
        [x - veh_length / 2 + t_end * vx, y - veh_width / 2 + t_end * vy,
         x + veh_length / 2 + t_end * vx, y + veh_width / 2 + t_end * vy])
    vis.draw_prism(bottom_vertices, t0, top_vertices, t_end)

# 车道线
plt.plot([-5, 80], [3.5 / 2, 3.5 / 2], 'k--')
plt.plot([-5, 80], [3.5 * 1.5, 3.5 * 1.5], 'k-')
plt.plot([-5, 80], [-3.5 * 0.5, -3.5 * 0.5], 'k-')
plt.plot([-5, 80], [3.5 / 2, 3.5 / 2], [t_seq[-1], t_seq[-1]], 'k--')
plt.plot([-5, 80], [3.5 * 1.5, 3.5 * 1.5], [t_seq[-1], t_seq[-1]], 'k-')
plt.plot([-5, 80], [-3.5 * 0.5, -3.5 * 0.5], [t_seq[-1], t_seq[-1]], 'k-')

for t in [t_seq[0], t_seq[-1]]:
    plt.plot([-5, -5], [-3.5 * 0.5, 3.5 * 1.5], [t, t], 'k--', lw=0.8)
    plt.plot([80, 80], [-3.5 * 0.5, 3.5 * 1.5], [t, t], 'k--', lw=0.8)

for m in [-0.5, 0.5, 1.5]:
    plt.plot([-5, -5], [3.5 * m, 3.5 * m], [t_seq[0], t_seq[-1]], 'k--', lw=0.8)
    plt.plot([80, 80], [3.5 * m, 3.5 * m], [t_seq[0], t_seq[-1]], 'k--', lw=0.8)

# 隐藏x、y和z轴刻度
plt.gca().set_facecolor('white')
plt.gca().set_xticks([])
plt.gca().set_yticks([])
plt.gca().set_zticks([])
plt.tight_layout()
plt.gca().set_box_aspect([5, 1, 1.5])
plt.gca().grid(False)
# plt.savefig('./images/figure2.pdf', dpi=300, bbox_inches='tight')
# plt.savefig('./images/figure2.png', dpi=300, bbox_inches='tight')

plt.show()
