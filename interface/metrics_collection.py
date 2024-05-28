from abc import abstractmethod
import numpy as np

class BaseMetric:

    def reset(self):
        ''' a new rollout '''
        pass

    @abstractmethod
    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        pass

    @abstractmethod
    def get_result(self)->dict:
        pass

class MetricCombiner(BaseMetric):
    def __init__(self, *metrics:BaseMetric):
        self.metrics = metrics

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        for metric in self.metrics:
            metric.evaluate(ego_state, perception, local_map, *args, **kwargs)

    def get_result(self) -> dict:
        result = {}
        for metric in self.metrics:
            result.update(metric.get_result())
        return result

class CompletionMetric(BaseMetric):
    def __init__(self, destination_x_range, destination_y_range, max_duration):
        self.destination_x_range = destination_x_range
        self.destination_y_range = destination_y_range
        self.num_rollout = 0
        self.num_completion = 0
        self.rollout_length = 0
        self.max_duration = max_duration


    def reset(self):
        self.num_rollout += 1
        self.rollout_length = 0

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        self.rollout_length += 1
        ego_x, ego_y = ego_state.x(), ego_state.y()
        if self.destination_x_range[0] <= ego_x <= self.destination_x_range[1]:
            if self.destination_y_range[0] <= ego_y <= self.destination_y_range[1]:
                if self.rollout_length < self.max_duration:
                    self.num_completion += 1

    def get_result(self)->dict:
        return {"route_completion": self.num_completion/self.num_rollout if self.num_rollout>0 else 0.0}

class CollisionRateMetric(BaseMetric):
    def __init__(self, ego_length, ego_width):
        from spider.utils.collision import BoxCollisionChecker

        self.num_rollout = 0
        self.num_collide = 0
        self.collision_flag = False

        self.collision_checker = BoxCollisionChecker(ego_length, ego_width)

    def reset(self):
        self.collision_flag = False
        self.num_rollout += 1

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        if not self.collision_flag:
            # 每个rollout只记录一次碰撞。若已经记录，则不做碰撞检测了
            collision = self.collision_checker.check_state(ego_state, perception)
            if collision:
                self.num_collide += 1
                self.collision_flag = True

    def get_result(self) -> dict:
        return {"collision_rate": self.num_collide / self.num_rollout if self.num_rollout>0 else 0.0}

class TTCMetric(BaseMetric):
    def __init__(self, ego_radius=2.5, low_ttc_thresh=2.7): # 2.7s的ttc
        self.ego_radius = ego_radius
        self.low_ttc_thresh = low_ttc_thresh

        self.ttc_record = []

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        from spider.utils.vector import project

        ego_loc = np.array([ego_state.x(), ego_state.y()])
        ego_v = np.array([ego_state.velocity.x, ego_state.velocity.y])

        ttc_list = [] # 记录对于所有的boundingbox的ttc
        for tbox in perception:
            x, y, length, width, yaw = tbox.obb
            v = np.array([tbox.vx, tbox.vy])
            tbox_radius = max([length, width]) / 2

            # 计算TTC
            rel_v = v - ego_v
            displacement = np.array([x, y]) - ego_loc
            proj, distance = project(-displacement, rel_v, calc_distance=True) # 原点到射线的投影
            if proj < 0: #说明投影点在射线反向延长线，不会碰撞
                ttc = np.inf
            else:
                occupancy_radius = self.ego_radius + tbox_radius
                if abs(distance) > occupancy_radius: # 距离大于占据，不会碰撞
                    ttc = np.inf
                else:
                    delta_proj = np.sqrt(occupancy_radius**2 - distance**2)
                    ttc = max([proj - delta_proj, 0.0]) / np.linalg.norm(rel_v)

            ttc_list.append(ttc)

        if len(ttc_list)>0:
            self.ttc_record.append(min(ttc_list))

    def get_result(self) ->dict:
        if len(self.ttc_record)>0:
            return {
                "min_ttc": min(self.ttc_record) if len(self.ttc_record)>0 else np.inf,
                "low_ttc_rate": (np.array(self.ttc_record) < self.low_ttc_thresh).sum() / len(self.ttc_record)
            }
        else:
            return {
                "min_ttc": 0.0,
                "low_ttc_rate": 0.0
            }

class StuckMetric(BaseMetric):
    def __init__(self, stuck_v_thresh=0.1):
        self.stuck_v_thresh = stuck_v_thresh
        self.stuck_frames = 0
        self.stuck_flag = False
        self.max_stuck_frames = 0

    def reset(self):
        self.stuck_flag = False

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        if ego_state.v() < self.stuck_v_thresh:
            self.stuck_frames += 1
            self.stuck_flag = True
            if self.stuck_frames > self.max_stuck_frames:
                self.max_stuck_frames = self.stuck_frames
        else:
            if self.stuck_flag:
                self.stuck_flag = False
                self.stuck_frames = 0

    def get_result(self) -> dict:
        return {"max_stuck_frames": self.max_stuck_frames}

class SpeedMetric(BaseMetric):
    def __init__(self):
        self.total_speed = 0.0
        self.num_frames = 0

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        self.total_speed += ego_state.v()
        self.num_frames += 1

    def get_result(self) -> dict:
        return {"avg_speed": self.total_speed / self.num_frames if self.num_frames > 0 else 0.0}

class JerkMetric(BaseMetric):
    def __init__(self, delta_t):
        self.dt = delta_t
        self.total_jerk = 0.0
        self.num_frames = 0

        self.last_a = None

    def reset(self):
        self.last_a = None

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        if self.last_a is not None:
            jerk = (ego_state.a() - self.last_a) / self.dt
            self.total_jerk += abs(jerk)
            self.num_frames += 1
        self.last_a = ego_state.a()

    def get_result(self) -> dict:
        return {"avg_jerk": self.total_jerk / self.num_frames if self.num_frames > 0 else 0.0}

class LateralOffsetMetric(BaseMetric):
    '''未完成'''
    def __init__(self):
        self.total_offset = 0.0
        self.num_frames = 0

    def evaluate(self, ego_state, perception, local_map, *args, **kwargs):
        self.total_offset += abs(ego_state.y())
        self.num_frames += 1

    def get_result(self) -> dict:
        return {"avg_lateral_offset": self.total_offset / self.num_frames if self.num_frames > 0 else 0.0}

