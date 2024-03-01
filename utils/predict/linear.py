import numpy as np
from spider.utils.predict.common import BasePredictor


def vertices_linear_predict(vertices:np.ndarray, vx, vy, ts):
    pred_vertices = []
    for t in ts:
        dx, dy = vx * t, vy * t
        pred_vertice = vertices.copy()
        pred_vertice[:,0] += dx
        pred_vertice[:,1] += dy
        pred_vertices.append(pred_vertice)
    return pred_vertices


# 目前，将predict结果暂时统一为x,y,theta的序列，以后可能再改
class LinearPredictor(BasePredictor):
    def __init__(self):
        super(LinearPredictor, self).__init__()
        pass

    def predict(self, trackingbox_list, ts):
        if len(trackingbox_list) == 0:
            return np.empty((0,3),dtype=float)
        for tb in trackingbox_list:
            vx, vy, yaw = tb.vx, tb.vy, tb.box_heading
            xs, ys = tb.x + np.asarray(ts) * tb.vx, tb.y + np.asarray(ts) * tb.vy
            yaws = np.ones_like(xs) * yaw
        return np.column_stack((xs, ys, yaws))

    def predict_box(self):
        pass

    def predict_vertices(self):
        pass

    def predict_occupancy(self):
        pass
