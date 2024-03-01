import numpy as np
from typing import List, Sequence
import warnings
import spider
from spider.elements.vector import rotate
from spider.utils.predict.linear import vertices_linear_predict
from spider.elements.vehicle import FrenetKinematicState



def AABB_vertices(AABB):
    '''

    :param AABB: xmin, ymin, xmax, ymax
    :return:
    '''
    xmin, ymin, xmax, ymax = AABB
    vertices = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin]
    ])
    return vertices

def obb2vertices(obb):
    xc, yc, length, width, heading = obb
    vertices = np.array([
        [xc + length / 2, yc + width / 2],
        [xc + length / 2, yc - width / 2],
        [xc - length / 2, yc - width / 2],
        [xc - length / 2, yc + width / 2],
    ])
    vertices = rotate(vertices,[xc,yc], heading)
    return vertices


def vertices2obb(vertices):
    vertices = np.asarray(vertices)
    edges = [
        vertices[0] - vertices[1],
        vertices[1] - vertices[2],
    ]
    edges_length = [np.linalg.norm(edge) for edge in edges]
    xc, yc = (vertices[0] + vertices[2]) / 2
    if edges_length[0] > edges_length[1]:
        length, width = edges_length[0], edges_length[1]
        heading = np.arctan2(edges[0][1], edges[0][0])
    else:
        length, width = edges_length[1], edges_length[0]
        heading = np.arctan2(edges[1][1], edges[1][0])
    return [xc,yc,length,width,heading]

def dilate(vertices, length_add, width_add):
    xc, yc, length, width, heading = vertices2obb(vertices)
    return obb2vertices((xc, yc, length+length_add, width+width_add, heading))


class BoundingBox:

    def __init__(self, obb=None, *, vertices=None):
        '''
        # 现在obb和vertices一起输入造成混淆，已经分出from_vertices方法
        :param vertices: a list(4) of four vertices of bounding box
        :param obb: a list(5) of [xc, yc, length, width, heading]
        '''
        self.obb = None
        self.vertices = None

        if vertices is not None: # 未来把这个if下的删掉，并且将BBOX和trackingbox输入的vertices删去，即可
            warnings.warn("The initialization of BBOX will support vertices input NO MORE! Please use from_vertices instead!",
                          DeprecationWarning)
            self.set_vertices(vertices)

        if obb is not None:
            self.set_obb(obb)

        # else:
        #     raise ValueError("Invalid Input of Bounding Box")

    @classmethod
    def from_vertices(cls, vertices):
        bbox = cls()
        bbox.set_vertices(vertices)
        return bbox

    def set_obb(self, obb):
        self.obb = obb#np.asarray(obb)
        self.vertices = obb2vertices(obb)

    def set_vertices(self, vertices):
        self.vertices = np.asarray(vertices)
        self.obb = vertices2obb(vertices)

    # def dilate(self,radius):
    #     obb = self.obb.copy()
    #     obb[2] += radius
    #     obb[3] += radius
    #     self.set_obb(obb)
    def dilate(self,length_add, width_add):
        obb = self.obb.copy()
        obb[2] += length_add
        obb[3] += width_add
        self.set_obb(obb)

    @property
    def x(self): return self.obb[0]

    @property
    def y(self): return self.obb[1]

    @property
    def box_heading(self): return self.obb[4]

    @property
    def length(self): return self.obb[2]

    @property
    def width(self): return self.obb[3]

    def __str__(self):
        return "BoundingBox: OBB:%s" % str(self.obb)



class TrackingBox(BoundingBox):
    def __init__(self, obb=None, vx=0., vy=0., id=0, *, vertices=None):
        super(TrackingBox, self).__init__(obb, vertices=vertices)
        self.id = id
        self.vx = vx
        self.vy = vy
        self.pred_vertices = []

        # todo: 以规范形式定义prediction和history，很重要
        self.prediction = None # sequence of x, y, theta?
        self.history = None

        self.frenet_state = FrenetKinematicState()
        self.frenet_prediction = None
        self.frenet_history = None

    # @property
    # def frenet_prediction(self):
    #     if self._frenet_prediction is None:
    #         assert not (self.prediction is None), "TrackingBox cartesian prediction is not done!"
    #         self._frenet_prediction = self.frenet_state.predict(self.prediction)
    #     return self._frenet_prediction

    @classmethod
    def from_vertices(cls, vertices, vx=0., vy=0., id=0):
        tbox = cls()
        tbox.set_vertices(vertices)
        tbox.set_velocity(vx, vy)
        return tbox

    def set_velocity(self, vx, vy):
        self.vx, self.vy = vx,vy

    @property
    def speed(self):
        return np.sqrt(self.vx**2 + self.vy**2)

    def dilate(self,length_add, width_add):
        super(TrackingBox, self).dilate(length_add, width_add)
        if len(self.pred_vertices) > 0:
            warnings.warn("TrackingBox dilation after prediction Detected! This might cause incorrect prediction!")

    def predict(self, ts, methodflag=spider.PREDICTION_LINEAR):
        # 不应该有这个方法，应该在外部的类中完成
        assert self.vertices is not None
        if methodflag == spider.PREDICTION_LINEAR:
            vx, vy, yaw = self.vx, self.vy, self.box_heading
            xs, ys = self.x + np.asarray(ts) * self.vx, self.y + np.asarray(ts) * self.vy
            yaws = np.ones_like(xs) * yaw
            self.prediction = np.column_stack((xs, ys, yaws))
            self.pred_vertices = self._pathseqence2verticeseqence(self.prediction)
        else:
            raise ValueError("Invalid method flag")

    def _pathseqence2verticeseqence(self, path_sequence):
        # prediction&history目前定义为x,y,theta序列，转换为vertices序列
        vertice_seq = []
        for x, y, yaw in path_sequence:
            vertices = obb2vertices((x, y, self.length, self.width, yaw))
            vertice_seq.append(vertices)
        return vertice_seq

    def __str__(self):
        return "TrackingBox: id:%d, OBB:%s, velocity:(%.1f, %.1f)" % (self.id, str(self.obb), self.vx, self.vy)


class TrackingBoxList(list): # List[TrackingBox]
    def __init__(self, seq:Sequence[TrackingBox]=()):
        super(TrackingBoxList, self).__init__(seq)

    def predict(self, ts, methodflag=spider.PREDICTION_LINEAR):
        # todo:预测要单独写一个类
        for tb in self:
            tb.predict(ts, methodflag)
        return self

    def dilate(self,length_add, width_add):
        for tb in self:
            tb.dilate(length_add, width_add)
        return self

    def get_vertices_at(self, step=0):
        '''
        获取的是第step预测的，所有障碍物bbox的顶点集合
        有预测就预测 没预测就直接用当前的顶点
        :param step: the ith step of prediction
        :return:
        '''
        bboxes_vertices = []
        for tb in self:
            if step == 0 or len(tb.pred_vertices) <= step:
                bboxes_vertices.append(tb.vertices)
            else:
                bboxes_vertices.append(tb.pred_vertices[step])
        return bboxes_vertices

    def get_frenet_vertices_at(self, step=0):
        '''
        这个函数命名非常奇怪说实话
        获取的是第step预测的，所有障碍物bbox的顶点集合(frenet下）
        有预测就预测 没预测就直接用当前的顶点
        :param step: the ith step of prediction
        :return:
        '''
        frenet_vertices = []
        for tb in self:
            s, l, frenet_yaw = tb.prediction[step]
            frenet_vertices.append(obb2vertices([s, l, tb.length, tb.width, frenet_yaw]))
        return frenet_vertices

    @classmethod
    def from_obbs(cls, obb_set_with_vel:Sequence, obbs_history=None, obbs_prediction=None, ids=None):
        '''
        输入的是 带有速度信息的obb的集合，基本格式为：
        [
            [x, y, len, wid, yaw, vx, vy],
            ......
        ]
        '''
        tbox_list = cls()
        if ids is None:
            ids = range(len(obb_set_with_vel))
        for obb_info, id in zip(obb_set_with_vel, ids):
            tbox_list.append(TrackingBox(obb=obb_info[:5], vx=obb_info[5], vy=obb_info[6], id=id))
        return tbox_list


# todo: 应该允许更多几何形状的物体，比如圆形物体的输入，所以应该改为TrackingObjectList
#  ，trackingobject可以包含box，circle等等几何形状，这样子来适配disk check等等




