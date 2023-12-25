import numpy as np
import matplotlib.pyplot as plt

from spider.elements import FrenetTrajectory, TrackingBoxList, TrackingBox
from spider.elements.Box import obb2vertices
from spider.utils.collision import BoxCollisionChecker

np.random.seed(0)

def check_and_draw(tb):
    tb_list = TrackingBoxList()
    tb_list.append(tb)

    predicted_obstacles = tb_list.predict(traj.dt * np.arange(traj.steps))
    vertices = tb.vertices
    vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    if checker.check_trajectory(traj, predicted_obstacles):
        plt.plot(vertices[:, 0], vertices[:, 1], color='red', linestyle='-', linewidth=1.5)  # 画他车
    else:
        plt.plot(vertices[:, 0], vertices[:, 1], color='green', linestyle='-', linewidth=1.5)  # 画他车

traj = FrenetTrajectory(50)
traj.x = np.linspace(400000., 400050., 50)
traj.y = np.linspace(400000., 400020., 50)
traj.heading = np.ones_like(traj.x) * np.arctan2(20,50)
# 画自车轨迹
plt.plot(traj.x, traj.y)
# 画自车脚印
for x, y, yaw in zip(traj.x, traj.y, traj.heading):
    vertices = obb2vertices((x,y,5.,2.,yaw))
    vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    plt.plot(vertices[:, 0], vertices[:, 1], color='gray', linestyle='-', linewidth=1)


checker = BoxCollisionChecker(5.,2.)

tb = TrackingBox(obb=(400019,400012.5,4,2,np.arctan(3/5)), vx=0,vy=0)
check_and_draw(tb)
# plt.show()



np.random.seed(0)
for i in range(500):
    x = np.random.random()*80 + 400000-20
    y = np.random.random()*50 + 400000-10
    heading = np.random.random()*3.14*2 - 3.14
    tb = TrackingBox(obb=(x,y,4,2,heading), vx=0,vy=0)
    check_and_draw(tb)


plt.show()



