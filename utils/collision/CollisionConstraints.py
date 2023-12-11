# 碰撞躲避约束的建模方法
import spider.utils.collision.CollisionChecker
from spider.elements.trajectory import Trajectory
from spider.utils.collision.CollisionChecker import BoxCollisionChecker
# from utils.collision.SAT import SAT_check
import numpy as np
from spider.elements.Box import obb2vertices,AABB_vertices,TrackingBox,TrackingBoxList





def triangle_area():
    # 三角面积法
    # TODO:补充这个方法
    pass


def generate_corridor(initial_guess:Trajectory, observation, delta=0.2, max_expand=5.0):
    '''

    :param initial_guess:
    :param observation:
    :param delta:
    :param max_expand:
    :return: corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    '''
    if isinstance(observation,TrackingBoxList):
        corridor = generate_corridor_bboxes(initial_guess, observation, delta,max_expand)
    # elif isinstance(observation, ): # TODO:补充OGM
    else:
        raise ValueError("Invalid input")

    return corridor


def generate_corridor_bboxes(initial_guess:Trajectory, bboxes:TrackingBoxList,
                             delta=0.2, max_expand=5.0):
    # lon_offset为负表示在几何中心后方

    # bboxes.dilate(radius)
    # bboxes.predict(initial_guess.x) # TODO:QZL:是不是要把预测放到外面
    collision_checker = BoxCollisionChecker(utils.collision.BoxCollisionChecker.flagSAT)

    corridor = []
    for i in range(len(initial_guess.t)):
        x, y, heading, t = initial_guess.x[i], initial_guess.y[i], initial_guess.heading[i], initial_guess.t[i]

        if t == 0:
            continue

        # collision_checker.setEgoVehicleBox(obb2vertices((x,y,ego_veh_size[0],ego_veh_size[1],heading)))
        collision_checker.setObstacles(bboxes_vertices=bboxes.getBoxVertices(step=i))

        seed = np.float32([x-0.1, y-0.1, x+0.1, y+0.1])  # 坍缩为一个小区域,四个方向发散以扩展
        sign = [-1, -1, 1, 1]
        space = seed.copy()
        StopIterFlag = [False, False, False, False]

        while not np.all(StopIterFlag):  # 全部方向都停止迭代才break
            for j in range(4):  # 每个方向尝试拓展
                if StopIterFlag[j]:
                    continue

                temp_space = space.copy()
                temp_space[j] += sign[j] * delta

                collision_checker.setEgoVehicleBox(AABB_vertices(temp_space))

                if np.abs(temp_space[j] - seed[j]) > max_expand or collision_checker.check():
                    # 超界 或者 碰撞
                    # TODO:记得加上道路边界的碰撞
                    StopIterFlag[j] = True
                    continue
                space = temp_space
        corridor.append(space)
    return np.array(corridor)

def generate_corridor_ogm(initial_guess:Trajectory, ogm,
                             delta=0.2, max_expand=5.0):
    # TODO:补充OGM
    return []





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from utils.Visualize import *

    traj = Trajectory()
    traj.x = np.arange(11)
    traj.y = 0.05 *(traj.x-5) ** 3
    traj.t = np.array(list(range(11)))*0.1
    traj.heading = 45 * np.pi / 180 * np.ones_like(traj.t)

    bboxes = TrackingBoxList()
    vertices1 = np.array([
        [4,-4],
        [6,-4],
        [6,-1.5],
        [4,-1.5]
    ])
    bboxes.append(TrackingBox(vertices=vertices1, vx=0, vy=0))

    vertices2 = np.array([
        [-2, -7.5],
        [-2, -5],
        [-1, -5],
        [-1, -7.5]
    ])
    bboxes.append(TrackingBox(vertices=vertices2, vx=4, vy=14))

    corridor = generate_corridor(traj, bboxes)
    for i, rect in enumerate(corridor):
        x1, y1, x2, y2 = rect
        draw_rectangle(x1,y1,x2,y2)
        bboxes_vertices = bboxes.getBoxVertices(i)
        for vertice in bboxes_vertices:
            draw_polygon(vertice,color='red')
            plt.pause(0.01)

    draw_trajectory(traj)
    plt.show()
