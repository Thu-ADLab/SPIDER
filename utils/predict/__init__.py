from spider.utils.predict.linear import linear_predict

class BasePrediction:
    def __init__(self):
        self.pred_traj = None
        self.pred_vertices = None
