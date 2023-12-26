from abc import abstractmethod, ABC
from spider.utils.collision.SAT import SAT_check
from spider.utils.collision.disks import disk_check_for_box
from spider.utils.collision.AABB import AABB_check
from spider.elements.trajectory import Trajectory
from spider.elements.Box import TrackingBoxList, obb2vertices, vertices2obb
import spider
# from spider.param import *


class CollisionChecker(ABC):
    def __init__(self, methodflag):
        self.method = methodflag
        pass

    @abstractmethod
    def check(self, ego_state, observation) -> bool:
        pass

    @abstractmethod
    def check_trajectory(self, trajectory, observation) -> bool:
        pass



class BoxCollisionChecker(CollisionChecker):
    def __init__(self, ego_veh_length, ego_veh_width, method_flag=spider.COLLISION_CHECKER_SAT):
        super(BoxCollisionChecker, self).__init__(method_flag)
        # self.method = method_flag
        self.ego_box_vertices = None

        self.bboxes_vertices = None
        # self.ogm = None

        self.ego_length = ego_veh_length #0.
        self.ego_width = ego_veh_width


        # todo: 这里以后考虑一下如何和vertices统一
        self.ego_box_obb = None
        self.bboxes_obb = None

    def set_ego_veh_size(self,length,width):
        self.ego_length = length
        self.ego_width = width

    def setEgoVehicleBox(self,ego_box_vertices=None):
        self.ego_box_vertices = ego_box_vertices
        if self.method == spider.COLLISION_CHECKER_DISK: # todo:以后去掉
            self.ego_box_obb = vertices2obb(ego_box_vertices)

    def setObstacles(self, bboxes_vertices=None):
        self.bboxes_vertices = bboxes_vertices
        if self.method == spider.COLLISION_CHECKER_DISK:  # todo:以后去掉
            self.bboxes_obb = [vertices2obb(vs) for vs in bboxes_vertices]


    def check_trajectory(self, traj:Trajectory, predicted_obstacles:TrackingBoxList, ego_length=0.0, ego_width=0.0):
        if ego_length and ego_width:
            self.set_ego_veh_size(ego_length,ego_width)

        for i in range(traj.steps):
            x, y, heading = traj.x[i], traj.y[i], traj.heading[i]
            ego_box_vertices = obb2vertices([x,y,self.ego_length,self.ego_width,heading])
            collision = self.check(ego_box_vertices, predicted_obstacles.getBoxVertices(i))
            if collision:
                return True
        return False


    def check(self, ego_box_vertices=None, bboxes_vertices=None):
        if not (ego_box_vertices is None):
            self.setEgoVehicleBox(ego_box_vertices)
        if not (bboxes_vertices is None):
            self.setObstacles(bboxes_vertices)


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

        return collision # True就是撞了，False就是没撞


class GridCollisionChecker(CollisionChecker):
    pass

