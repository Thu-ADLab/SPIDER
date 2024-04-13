
# lane_id = function(ego_state, perception, local_map)

class ConstLaneDecision:
    def __init__(self, lane_id=1):
        self.lane_id = lane_id

    def __call__(self, *args, **kwargs):
        return self.decide(*args, **kwargs)

    def decide(self, ego_state, perception, local_map):
        idx = min([self.lane_id, len(local_map.lanes)])
        idx = max([self.lane_id, 0])
        return idx

class NearestLaneDecision:
    def __call__(self, *args, **kwargs):
        return self.decide(*args, **kwargs)

    def decide(self, ego_state, perception, local_map):
        ego_lane_idx = local_map.match_lane(ego_state)
        return ego_lane_idx

class UtilityLaneDecision:
    '''
    todo: 未完成
    '''
    def __call__(self, *args, **kwargs):
        return self.decide(*args, **kwargs)

    def decide(self, ego_state, perception, local_map):
        raise NotImplementedError
