import numpy as np


def linear_predict(vertices:np.ndarray, vx, vy, ts):
    pred_vertices = []
    for t in ts:
        dx, dy = vx * t, vy * t
        pred_vertice = vertices.copy()
        pred_vertice[:,0] += dx
        pred_vertice[:,1] += dy
        pred_vertices.append(pred_vertice)
    return pred_vertices