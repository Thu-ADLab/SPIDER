import matplotlib.pyplot as plt
import numpy as np


def draw_polygon(vertices,color='black',lw=1.):
    vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    plt.plot(vertices[:, 0], vertices[:, 1], color=color, linestyle='-', linewidth=lw)
# def draw_polygon(vertices,**kwargs):
#     vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
#     plt.plot(vertices[:, 0], vertices[:, 1], kwargs)


def draw_rectangle(xmin, ymin, xmax, ymax):
    vertices = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
        [xmin, ymin]
    ])
    plt.plot(vertices[:, 0], vertices[:, 1], color='black', linestyle='-', linewidth=1)


def draw_trajectory(traj,color='red'):
    plt.plot(traj.x, traj.y, '.-', color=color, linewidth=1)