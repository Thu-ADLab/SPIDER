import numpy as np
from spider.utils.transform.frenet import FrenetTransformer
from spider.elements.trajectory import FrenetTrajectory, Trajectory
import matplotlib.pyplot as plt
import torch

data = torch.load('data.pt')

traj_arr = data['traj'].numpy()
cline = data['base_centerline'].numpy()
# indices_to_remove = np.where(np.arange(len(cline)) % 10 == 0)
# cline = np.delete(cline, indices_to_remove, axis=0)

plt.figure()
plt.plot(cline[:,0], cline[:,1], '.-')
plt.plot(traj_arr[:,0], traj_arr[:,1])


transformer = FrenetTransformer()
transformer.set_reference_line(cline)

plt.figure()
ss = np.linspace(0,168,1000)
cline_interp = transformer.refer_line_csp.calc_point(ss)
plt.subplot(131)
plt.plot(ss, cline_interp[:, 0],lw=2)
plt.plot(transformer.refer_line_csp.s, transformer.refer_line_csp.x, '.')
plt.subplot(132)
plt.plot(ss, cline_interp[:, 1],lw=2)
plt.plot(transformer.refer_line_csp.s, transformer.refer_line_csp.y , '.')
plt.subplot(133)
plt.plot(cline_interp[:,0], cline_interp[:,1])

traj = Trajectory.from_trajectory_array(traj_arr, dt=0.1,calc_derivative=False)
frenet_traj = transformer.cart2frenet4traj(traj, order=0)
plt.figure()
plt.plot(frenet_traj.s, frenet_traj.l)


plt.figure()
plt.plot(cline[:50,0], cline[:50,1], '.-')
plt.plot(traj_arr[:,0], traj_arr[:,1])
for i, (x,y) in enumerate(cline):
    plt.text(x,y , str(i))



plt.show()
pass
# import pickle
#
# f = open('data.pt',)
# data = pickle.load('data.pt')
