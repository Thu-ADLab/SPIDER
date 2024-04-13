from abc import abstractmethod, ABC
from spider.utils.collision.SAT import SAT_check
from spider.utils.collision.disks import disk_check_for_box
from spider.utils.collision.AABB import AABB_check

from spider.elements import Trajectory, VehicleState
from spider.elements.box import TrackingBoxList, obb2vertices, vertices2obb, dilate
import spider
# from spider.param import *


class BaseCollisionChecker(ABC):
    def __init__(self, method):
        self.method = method
        pass

    @abstractmethod
    def check(self, ego_state, observation) -> bool:
        pass

    @abstractmethod
    def check_trajectory(self, trajectory, observation) -> bool:
        pass



class BoxCollisionChecker(BaseCollisionChecker):
    def __init__(self, ego_veh_length=5., ego_veh_width=2.,
                 method=spider.COLLISION_CHECKER_SAT, safe_dist=(0.0, 0.0)):
        super(BoxCollisionChecker, self).__init__(method)
        # self.method = method_flag
        self.ego_box_vertices = None

        self.bboxes_vertices = None
        # self.ogm = None

        self.ego_length = ego_veh_length #0.
        self.ego_width = ego_veh_width

        self.safe_dist = safe_dist # 分别为纵向、横向的安全距离阈值

        # todo: 这里以后考虑一下如何和vertices统一
        self.ego_box_obb = None
        self.bboxes_obb = None

    def set_ego_veh_size(self,length,width):
        self.ego_length = length
        self.ego_width = width

    # setEgoVehicleBox
    def set_ego_box(self, ego_box_vertices=None):
        self.ego_box_vertices = ego_box_vertices
        if self.method == spider.COLLISION_CHECKER_DISK: # todo:以后去掉
            self.ego_box_obb = vertices2obb(ego_box_vertices)

    def set_obstacles(self, bboxes_vertices=None):
        self.bboxes_vertices = bboxes_vertices
        if self.method == spider.COLLISION_CHECKER_DISK:  # todo:以后去掉
            self.bboxes_obb = [vertices2obb(vs) for vs in bboxes_vertices]

    def check(self, ego_box_vertices=None, bboxes_vertices=None, safe_dilate=False):
        if not (ego_box_vertices is None):
            self.set_ego_box(ego_box_vertices)
        if not (bboxes_vertices is None):
            self.set_obstacles(bboxes_vertices)

        if safe_dilate:
            self.set_ego_box(dilate(ego_box_vertices, 2 * self.safe_dist[0], 2 * self.safe_dist[1]))

        collision = False
        if self.method == spider.COLLISION_CHECKER_SAT:
            for bbox_vertices in self.bboxes_vertices:
                if SAT_check(self.ego_box_vertices, bbox_vertices):
                    collision = True
                    break
        elif self.method == spider.COLLISION_CHECKER_DISK:
            for obb in self.bboxes_obb:
                # todo: 这里的逻辑要改的很多，因为现在输入和预测只接收了顶点，但disk check需要Obb。来回转换很蠢
                if disk_check_for_box(self.ego_box_obb, obb):
                    collision = True
                    break
        elif self.method == spider.COLLISION_CHECKER_AABB:
            for bbox_vertices in self.bboxes_vertices:
                if AABB_check(self.ego_box_vertices, bbox_vertices):
                    collision = True
                    break
        else:
            raise ValueError("INVALID method for box collision checker")

        return collision  # True就是撞了，False就是没撞

    def check_trajectory(self, traj:Trajectory, predicted_obstacles:TrackingBoxList, ego_length=0.0, ego_width=0.0):
        if ego_length and ego_width:
            self.set_ego_veh_size(ego_length,ego_width)

        for i in range(traj.steps):
            x, y, heading = traj.x[i], traj.y[i], traj.heading[i]
            ego_box_vertices = obb2vertices(
                [x, y, self.ego_length+2*self.safe_dist[0], self.ego_width+2*self.safe_dist[1], heading])
            collision = self.check(ego_box_vertices, predicted_obstacles.get_vertices_at(i))
            if collision:
                return True
        return False


    def check_state(self, ego_veh_state:VehicleState, obstacles:TrackingBoxList):
        ego_box_vertices = obb2vertices(ego_veh_state.obb)
        return self.check(ego_box_vertices, obstacles.get_vertices_at(0))


class GridCollisionChecker(BaseCollisionChecker):
    pass

