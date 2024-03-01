import numpy as np
from spider.utils.predict.common import BasePredictor


def linear_predict(vertices:np.ndarray, vx, vy, ts):
    pred_vertices = []
    for t in ts:
        dx, dy = vx * t, vy * t
        pred_vertice = vertices.copy()
        pred_vertice[:,0] += dx
        pred_vertice[:,1] += dy
        pred_vertices.append(pred_vertice)
    return pred_vertices


class LinearPredictor(BasePredictor):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        pass

    def predict(self):
        pass

    def predict_box(self):
        pass

    def predict_vertices(self):
        pass

    def predict_occupancy(self):
        pass
