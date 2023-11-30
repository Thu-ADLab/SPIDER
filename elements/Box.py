import numpy as np
from typing import List

from spider.elements.vector import rotate
from spider.utils.predict.linear import linear_predict
import warnings

flagLinear = 0

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
    vertices = np.array(vertices)
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


class BoundingBox:
    def __init__(self, *, vertices=None, obb=None):
        '''

        :param vertices: a list(4) of four vertices of bounding box
        :param obb: a list(5) of [xc, yc, length, width, heading]
        '''
        if vertices is not None:
            self.setVertices(vertices) # 同时输入时，以顶点为准
        elif obb is not None:
            self.setObb(obb)
        else:
            self.vertices = None
            self.obb = None
        # else:
        #     raise ValueError("Invalid Input of Bounding Box")

    def setObb(self,obb):
        self.obb = obb
        self.vertices = obb2vertices(obb)

    def setVertices(self,vertices):
        self.vertices = np.array(vertices)
        self.obb = vertices2obb(vertices)

    # def dilate(self,radius):
    #     obb = self.obb.copy()
    #     obb[2] += radius
    #     obb[3] += radius
    #     self.setObb(obb)
    def dilate(self,length_add, width_add):
        obb = self.obb.copy()
        obb[2] += length_add
        obb[3] += width_add
        self.setObb(obb)


class TrackingBox(BoundingBox):
    def __init__(self, *, vertices=None, obb=None, vx=0, vy=0):
        super(TrackingBox, self).__init__(vertices=vertices, obb=obb)
        self.vx = vx
        self.vy = vy
        self.pred_vertices = []

    def setVelocity(self, vx, vy):
        self.vx, self.vy = vx,vy

    def dilate(self,length_add, width_add):
        super(TrackingBox, self).dilate(length_add, width_add)
        if len(self.pred_vertices) > 0:
            warnings.warn("TrackingBox dilation after prediction Detected! This might cause incorrect prediction!")

    def predict(self, ts, methodflag=flagLinear):
        assert self.vertices is not None
        if methodflag == flagLinear:
            self.pred_vertices = linear_predict(self.vertices, self.vx, self.vy, ts)
        else:
            raise ValueError("Invalid method flag")


class TrackingBoxList(list):
    def __init__(self, seq=()):
        super(TrackingBoxList, self).__init__(seq)

    def predict(self, ts, methodflag=flagLinear):
        # todo:预测要单独写一个类
        for tb in self:
            tb.predict(ts, methodflag)
        return self

    def dilate(self,length_add, width_add):
        for tb in self:
            tb.dilate(length_add, width_add)
        return self

    def getBoxVertices(self, step=0):
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



# import matplotlib.pyplot as plt
# def draw_polygon(vertices):
#     vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
#     plt.plot(vertices[:, 0], vertices[:, 1], color='blue', linestyle='-', linewidth=1)



# if __name__ == '__main__':
#     pass
    # obs = TrackingBoxList()
    # vertices1 = [
    #     [0,0],
    #     [0,2],
    #     [5,2],
    #     [5,0]
    # ]
    # obs.append(TrackingBox(vertices=vertices1, vx=1, vy=0))
    #
    # vertices2 = [
    #     [0, 0],
    #     [0, 5],
    #     [2, 5],
    #     [2, 0]
    # ]
    # obs.append(TrackingBox(vertices=vertices2, vx=1, vy=1))
    #
    #
    # ts = list(range(10))
    # obs.predict(ts)
    #
    # for i in range(10):
    #     bboxes_vertices = obs.getBoxVertices(i)
    #     for vertice in bboxes_vertices:
    #         draw_polygon(vertice)
    #         plt.pause(0.1)
    # plt.show()

