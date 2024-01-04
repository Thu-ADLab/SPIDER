from typing import Union
import numpy as np
from cvxopt import matrix, solvers


from spider.optimize.BaseOptimizer import BaseOptimizer
from spider.optimize.common import FrenetTrajOptimParam
from spider.elements.trajectory import Path, Trajectory, FrenetTrajectory
from spider.elements import TrackingBoxList, OccupancyGrid2D
from spider.utils.collision.CollisionChecker import BoxCollisionChecker
from spider.utils.collision.AABB import AABB_vertices
from spider.vehicle_model import Bicycle


def generate_corridor_bboxes(initial_guess:np.ndarray, bboxes:TrackingBoxList,
                             x_bound=None, y_bound=None,
                             delta=0.1, max_expand=50.0,
                             ):
    '''
    # todo: 没加上障碍物膨胀的功能，应该先根据自车搞一些近似的disks出来（嫌麻烦就设定1个圆盘），然后根据圆盘半径来设定膨胀的半径
    initial_guess: xy pair sequence or sl pair sequence, [ [x1, y1], [x2, y2]... ]
    return: corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    '''

    # lon_offset为负表示在几何中心后方

    # bboxes.dilate(radius)
    # bboxes.predict(initial_guess.t) # TODO:QZL:是不是要把预测放到外面
    x_bound = (-max_expand, max_expand) if x_bound is None else x_bound
    y_bound = (-max_expand, max_expand) if y_bound is None else y_bound

    collision_checker = BoxCollisionChecker()

    corridor = []
    for i in range(len(initial_guess)):
        x, y = initial_guess[i,0], initial_guess[i,1]

        # if t == 0:
        #     continue

        collision_checker.setObstacles(bboxes_vertices=bboxes.getBoxVertices(step=i))

        seed = np.float64([x-0.01, y-0.01, x+0.01, y+0.01])  # 坍缩为一个小区域,四个方向发散以扩展
        sign = [-1, -1, 1, 1]
        road_bound = [x_bound[0], y_bound[0], x_bound[1], y_bound[1]]#[-1, -3.5*0.5, 80, 3.5*1.5]
        space = seed.copy()
        StopIterFlag = [False, False, False, False]

        while not np.all(StopIterFlag):  # 全部方向都停止迭代才break
            for j in range(4):  # 每个方向尝试拓展
                if StopIterFlag[j]:
                    continue

                temp_space = space.copy()
                temp_space[j] += sign[j] * delta

                collision_checker.setEgoVehicleBox(AABB_vertices(temp_space))

                if np.abs(temp_space[j] - seed[j]) > max_expand or collision_checker.check() or \
                        (road_bound[j]-temp_space[j])*sign[j]<0:
                    StopIterFlag[j] = True
                    continue
                space = temp_space
        space = [round(x,1) for x in space]
        corridor.append(space)
    return np.array(corridor)

def getConsMat(s0,s_d0,s_dd0,l0,l_d0,l_dd0, target_l_bound,
               initial_guess:np.ndarray, bboxes, l_bound=None,
               s_dot_bound=(0.,60/3.6), l_dot_bound=(-5.,5.), s_2dot_bound=(-8.,8.), l_2dot_bound=(-3.,3.),
               *, param: FrenetTrajOptimParam):
    '''
    todo: qzl: 这里输入太多了，要好好整理打包一下再传进来，否则可读性非常差
    包含构建三部分约束所需的参数:
    两点边值约束（起终点条件）
    外部环境约束（碰撞约束、道路边界约束l_bound）
    内部系统运动学约束（速度极限、加速度极限等）
    '''
    p = param
    N, dt = p.N, p.dt

    Aineq_list = []
    bineq_list = []
    Aeq_list = []
    beq_list = []

    ones_minus = p.ones_N_minus_1
    # ones = np.ones(N)

    # 速度
    sdlb, sdub = s_dot_bound
    ldlb, ldub = l_dot_bound
    Aineq_list += [p.Diff_s, -p.Diff_s, p.Diff_l, -p.Diff_l]
    bineq_list += [sdub*ones_minus, -sdlb*ones_minus, ldub*ones_minus, -ldlb*ones_minus]

    # 加速度
    sddlb, sddub = s_2dot_bound
    lddlb, lddub = l_2dot_bound
    Aineq_list += [p.G2Ms, -p.G2Ms, p.G2Ml, -p.G2Ml]
    bineq_list += [sddub - p.H2_1*s_d0,
                   -sddlb + p.H2_1*s_d0,
                   lddub - p.H2_1*l_d0,
                   -lddlb + p.H2_1*l_d0]

    # 碰撞约束
    # traj = Trajectory(steps=N,dt=dt)
    # traj.t = [dt*i for i in range(N)]
    # ss,ls = p.Ms@initial_guess, p.Ml@initial_guess
    # traj.x = ss
    # traj.y = ls
    # traj.heading = traj.heading = np.insert(np.arctan2(np.diff(traj.y), np.diff(traj.x)), 0, np.arctan2(l_d0,s_d0))
    # veh_model = Bicycle(s0,l0, s_d0, s_dd0,heading=0., dt=dt, wheelbase=wheelbase)
    # traj.derivative(veh_model,xs=ss,ys=ls)
    # bboxes.predict(traj.t) # 放到外面去了
    corridors = generate_corridor_bboxes(initial_guess, bboxes,x_bound=(-1,80), y_bound=l_bound)
    #:return: corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    slb,llb, sub,lub = corridors.T#[:,0],corridors[:,1],corridors[:,2],corridors[:,3]
    Aineq_list += [p.Ms,-p.Ms,p.Ml,-p.Ml]
    bineq_list += [sub,-slb,lub,-llb]



    # 两点边值约束
    if not (target_l_bound is None):
        target_l_lb, target_l_ub = target_l_bound
        Aineq_list += [p.Final_l, -p.Final_l] # 末端横向不超过要求的范围
        bineq_list += [[target_l_ub], [-target_l_lb]]
    Aeq_list += [p.Final_l_dot, p.Final_l_2dot_coef, p.Final_s_2dot_coef, p.First_l, p.First_s]
    beq_list += [0., -l_d0*p.Final_l_2dot_bias, -s_d0*p.Final_s_2dot_bias, l0, s0]
    # 分别为：末端横向速度、横向加速度、纵向加速度为0；初始横纵向位置固定

    Aeq = np.vstack(Aeq_list)
    Aineq = np.vstack(Aineq_list)

    # beq = np.concatenate(beq_list)
    bineq = np.concatenate(bineq_list)
    beq = np.array(beq_list)
    return Aineq, bineq, Aeq, beq, corridors


def getCostFunMat(s_d0,s_dd0,l_d0,l_dd0, target_l, *, param:FrenetTrajOptimParam):
    p = param
    N = p.N
    w1,w2,w3 = np.array([0.2, 2, 1]) # *1e2
    # 舒适
    k1 = 5.

    hs = p.H3_1 * s_d0 + p.H3_2 * s_dd0
    hl = p.H3_1 * l_d0 + p.H3_2 * l_dd0
    # G3Ms, G3Ml = G3@Ms, G3@Ml
    Q_comf = p.G3Ms.T @ p.G3Ms + k1 * p.G3Ml.T @ p.G3Ml
    f_comf = 2 * (hs.T @ p.G3Ms + k1 * hl.T @ p.G3Ml)

    # 效率： 最后一个点的s，和第一个点的s的差
    Q_eff = p.zeros_2N_2N
    f_eff = - p.s_displacement

    # 安全, l距离target_l的偏移量
    k2 = 1000.
    W = np.eye(N)
    W[-1,-1] = k2 # 认为最后目标点的偏移量最重要
    Q_safe = p.Ml.T @ W @ p.Ml
    f_safe = -2 * target_l * np.ones(N) @ W @ p.Ml
    # Q_safe = Ml.T @ Ml + k2 * (Final@Ml).T @ (Final@Ml)
    # f_safe = -2*target_l*np.ones(N)@Ml - k2*2*target_l*Final@Ml

    # 加权和
    Q = 2 * (w1*Q_comf + w2*Q_eff + w3*Q_safe) # 乘2是因为认为接受的目标函数为1/2 * Q.T @ X @ Q + f @ X
    f = w1*f_comf + w2*f_eff + w3*f_safe
    return Q, f

class TrajectoryOptimizer(BaseOptimizer):
    '''
    对于等时间间隔的
    '''
    def __init__(self, steps, dt):
        super(TrajectoryOptimizer, self).__init__()
        pass

    def optimize_traj(self, trajectory,):
        pass

    def optimize_traj_arr(self):
        pass


class FrenetTrajectoryOptimizer(BaseOptimizer):
    '''
    对于等时间间隔的(s,l)对，在给定观测情况下，做优化
    '''
    def __init__(self, steps, dt):
        super(FrenetTrajectoryOptimizer, self).__init__()
        self.steps = steps
        self.dt = dt
        self.param = FrenetTrajOptimParam(steps, dt)
        self._corridors = None
        pass

    def optimize_traj(self,
                      initial_frenet_trajectory: FrenetTrajectory, # Union[FrenetTrajectory, np.ndarray],
                      perception:Union[TrackingBoxList, OccupancyGrid2D],
                      offset_bound=None,  # 过程中所有处的横向l的范围，这一项现在直接被corridor纳入考虑，不显式建模为l的上下界
                      target_offset=0.0,
                      target_offset_bound=None, # 终点处的横向l的范围
                      # todo: offset_bound约束需要改，目前是定值，后面改成可以跟l是同大小的数组，以应对车道收窄或交叉口接虚拟车道的情况
                      ):
        if not isinstance(perception, TrackingBoxList):
            raise NotImplementedError("Optimization under occupancy has not been implemented. It is recommended to Use cartesian Optimizer")
        # if not isinstance(initial_frenet_trajectory, FrenetTrajectory):
            # raise NotImplementedError("FrenetTrajectory not supported now. Please convert it to ndarray")

        traj = initial_frenet_trajectory
        traj_sl_array = np.column_stack((traj.s, traj.l))
        s_0, l_0 = traj.s[0], traj.l[0]
        s_dot0, s_2dot0, l_dot0, l_2dot0 = traj.s_dot[0], traj.s_2dot[0], traj.l_dot[0], traj.l_2dot[0]

        Q, f = getCostFunMat(s_dot0, s_2dot0, l_dot0, l_2dot0, target_offset, param=self.param)
        Aineq, bineq, Aeq, beq, corridors = getConsMat(s_0, s_dot0, s_2dot0, l_0, l_dot0, l_2dot0, target_offset_bound,
                                                       traj_sl_array, perception, offset_bound, param=self.param)


        Q, f, Aineq, bineq, Aeq, beq = [matrix(i.astype(np.float64)) for i in [Q, f, Aineq, bineq, Aeq, beq]]
        sol = solvers.qp(Q, f, Aineq, bineq, Aeq, beq)  # kktsolver='ldl', options={'kktreg':1e-9}

        optim_traj_arr = np.array(sol['x']) # size of [2N, 1] squeeze的过程在下面赋值的时候加了0的列索引
        optim_traj = FrenetTrajectory(traj.steps, traj.dt)

        optim_traj.s = s = optim_traj_arr[:self.steps, 0] # size of [N,]
        optim_traj.l = l = optim_traj_arr[self.steps:, 0] # size of [N,]

        self._corridors = corridors
        return optim_traj


    def optimize_traj_arr(self,
                      initial_sl_pair_array: np.ndarray,
                      perception:Union[TrackingBoxList, OccupancyGrid2D],
                      target_offset=0.0,
                      target_offset_bound=None, # 终点处的横向l的范围
                      offset_bound = None # 过程中所有处的横向l的范围
                      ):
        # todo: 输入sl_pair_array & 初始横纵向状态
        pass

    def get_corridors(self):
        return self._corridors



if __name__ == '__main__':
    from spider.elements import TrackingBox
    from spider.visualize import draw_polygon
    import matplotlib.pyplot as plt


    # def draw_polygon(vertices, color='black', lw=1.):
    #     vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    #     plt.plot(vertices[:, 0], vertices[:, 1], color=color, linestyle='-', linewidth=lw)
    steps = 50
    dt = 0.1

    optim = FrenetTrajectoryOptimizer(steps,dt)

    veh_length = 5
    veh_width = 2
    wheelbase = 3.0

    # observation
    bboxes = TrackingBoxList()
    obs = [
        [15, 3.5 * 1, 3, 0],
        [30, 3.5 * 1, 8, 0],
        [40, -0.7, 0, 0]
    ]

    for x, y, vx, vy in obs:
        vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
        tb = TrackingBox(vertices=vertices, vx=vx, vy=vy)
        bboxes.append(tb)
    bboxes.predict([dt * i for i in range(steps)])

    s0, s_d0, s_dd0 = 0., 30 / 3.6, 0.
    l0, l_d0, l_dd0 = 0., 0., 0.
    target_l = 3.5
    target_l_bound = [target_l - 3.5 * 0.5, target_l + 3.5 * 0.5]

    initial_guess_s = np.array([s0 + s_d0 * i * dt for i in range(steps)])
    initial_guess_d = np.array([l0 + 0.12 * i * i * dt * dt for i in range(steps)])
    # initial_guess = np.concatenate((initial_guess_s, initial_guess_d))
    initial_frenet_trajectory = FrenetTrajectory(50, 0.1)
    initial_frenet_trajectory.s, initial_frenet_trajectory.l = initial_guess_s, initial_guess_d
    initial_frenet_trajectory.s_dot.append(s_d0)
    initial_frenet_trajectory.l_dot.append(l_d0)
    initial_frenet_trajectory.s_2dot.append(s_dd0)
    initial_frenet_trajectory.l_2dot.append(l_dd0)

    optim_traj = optim.optimize_traj(initial_frenet_trajectory, bboxes, offset_bound=(-3.5*0.5, 3.5*1.5))
    corridors = optim.get_corridors()

    for x, y, vx, vy in obs:
        vertices = AABB_vertices([x - veh_length / 2, y - veh_width / 2, x + veh_length / 2, y + veh_width / 2])
        draw_polygon(vertices, color='black')

    for aabb in corridors:
        vertices = AABB_vertices(aabb)
        draw_polygon(vertices, color='green', lw=0.3)

    plt.plot([0, 50], [3.5 / 2, 3.5 / 2], 'k--')
    plt.plot([0, 50], [3.5 * 1.5, 3.5 * 1.5], 'k-')
    plt.plot([0, 50], [-3.5 * 0.5, -3.5 * 0.5], 'k-')
    plt.plot(initial_guess_s, initial_guess_d, '.-', label='initial guess')
    plt.plot(optim_traj.s, optim_traj.l, '.-', label='optimized trajectory')
    plt.show()
