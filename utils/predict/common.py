

class BasePrediction:
    def __init__(self):
        self.pred_traj = None
        self.pred_vertices = None


class BasePredictor:
    def __init__(self):
        pass

    def predict(self):
        pass

    def predict_box(self):
        pass

    def predict_vertices(self):
        pass

    def predict_occupancy(self):
        pass
