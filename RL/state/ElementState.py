'''
元素级的状态
'''

class ElementState:
    def __init__(self):
        pass

    def encode_from_obstacles(self, obstacles, ego_veh_state):
        pass


class ElementFrenetState(ElementState):
    def __init__(self):
        super(ElementFrenetState, self).__init__()
        pass

    def encode_from_obstacles(self, obstacles, ego_veh_state, frenet_transformer=None):
        pass
