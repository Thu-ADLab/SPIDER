from typing import Union
import numpy as np
from cvxopt import matrix, solvers


from spider.optimize.BaseOptimizer import BaseOptimizer
from spider.elements.trajectory import Path, Trajectory, FrenetTrajectory
from spider.elements import TrackingBoxList, OccupancyGrid2D
from spider.utils.collision.CollisionChecker import BoxCollisionChecker
from spider.utils.collision.AABB import AABB_vertices
from spider.vehicle_model import Bicycle

# todo: 要改要改！
N = steps = 50

dt = 0.1

Ms = np.zeros((N,2*N))
Ms[:,:N] = np.eye(N)

Ml = np.zeros((N,2*N))
Ml[:,N:] = np.eye(N)

Minus = np.zeros(N)
Minus[0] = -1
Minus[-1] = 1

Diff = np.zeros((N-1,N))
for i in range(N-1):
    Diff[i,i] = -1
    Diff[i, i+1] = 1
Diff /= dt

Appe = np.zeros((N,N-1))
Appe[1:,:] = np.eye(N-1)

First = np.zeros(N)
First[0] = 1


# Final = np.zeros((N,1))
Final = np.zeros(N)
Final[-1] = 1
# Final = np.reshape(Final,(N))

G1 = Appe@Diff # 1阶导公式中x的参数矩阵
H1_1 = First # 1阶导公式中1阶导初始值的参数矩阵

G2 = G1@G1 # 2阶导公式中x的参数矩阵
H2_1 = G1@H1_1 # 2阶导公式中1阶导初始值的参数矩阵
H2_2 = First # 2阶导公式中2阶导初始值的参数矩阵

G3 = Diff@G2 # 2阶导公式中x的参数矩阵
H3_1 = Diff@H2_1 # 3阶导公式中1阶导初始值的参数矩阵
H3_2 = Diff@H2_2 # 3阶导公式中2阶导初始值的参数矩阵

weight_comf = 0.1
weight_jerk_s = 1
weight_jerk_l = 10

weight_eff = 2
weight_eff_s = 1
weight_eff_s_d = 0.5

weight_safe = 2
weight_safe_lateral_offset = 1



def generate_corridor_bboxes(initial_guess, bboxes:TrackingBoxList,
                             delta=0.1, max_expand=100.0):
    '''
    # todo: 没加上障碍物膨胀的功能，应该先根据自车搞一些近似的disks出来（嫌麻烦就设定1个圆盘），然后根据圆盘半径来设定膨胀的半径
    initial_guess: xy pair sequence or sl pair sequence, [ [x1, y1], [x2, y2]... ]
    return: corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    '''

    # lon_offset为负表示在几何中心后方

    # bboxes.dilate(radius)
    # bboxes.predict(initial_guess.t) # TODO:QZL:是不是要把预测放到外面
    collision_checker = BoxCollisionChecker()

    corridor = []
    for i in range(len(initial_guess.t)):
        x, y, heading, t = initial_guess.x[i], initial_guess.y[i], initial_guess.heading[i], initial_guess.t[i]

        # if t == 0:
        #     continue

        # collision_checker.setEgoVehicleBox(obb2vertices((x,y,ego_veh_size[0],ego_veh_size[1],heading)))
        collision_checker.setObstacles(bboxes_vertices=bboxes.getBoxVertices(step=i))

        seed = np.float64([x-0.01, y-0.01, x+0.01, y+0.01])  # 坍缩为一个小区域,四个方向发散以扩展
        sign = [-1, -1, 1, 1]
        road_bound = [-1, -3.5*0.5, 80, 3.5*1.5]#steps*dt*60/3.6
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
                    # 超界 或者 碰撞
                    # TODO:记得加上道路边界的碰撞
                    StopIterFlag[j] = True
                    continue
                space = temp_space
        space = [round(x,1) for x in space]
        corridor.append(space)
    return np.array(corridor)

def getConsMat(s0,s_d0,s_dd0,l0,l_d0,l_dd0, target_l_bound,
               initial_guess:np.ndarray, bboxes, l_bound=None,
               s_dot_bound=(0.,60/3.6), l_dot_bound=(-5.,5.), s_2dot_bound=(-8.,8.), l_2dot_bound=(-3.,3.)):
    '''
    todo: qzl: 这里输入太多了，要好好整理打包一下再传进来，否则可读性非常差
    包含构建三部分约束所需的参数:
    两点边值约束（起终点条件）
    外部环境约束（碰撞约束、道路边界约束l_bound）
    内部系统运动学约束（速度极限、加速度极限等）
    '''
    Aineq_list = []
    bineq_list = []
    Aeq_list = []
    beq_list = []

    ones_lack = np.ones(N-1)
    ones = np.ones(N)

    # 速度
    sdlb, sdub = 0, 60/3.6
    ldlb, ldub = -5, 5
    Aineq_list += [Diff@Ms, -Diff@Ms, Diff@Ml, -Diff@Ml]
    bineq_list += [sdub*ones_lack, -sdlb*ones_lack, ldub*ones_lack, -ldlb*ones_lack]

    # 加速度
    sddlb, sddub = -5, 5
    lddlb, lddub = -3, 3
    Aineq_list += [G2 @ Ms, -G2 @ Ms, G2 @ Ml, -G2 @ Ml]
    bineq_list += [sddub - H2_1*s_d0,
                   -sddlb + H2_1*s_d0,
                   lddub - H2_1*l_d0,
                   -lddlb + H2_1*l_d0]

    # 碰撞约束
    traj = Trajectory(steps=50,dt=dt)
    traj.t = [dt*i for i in range(N)]
    ss,ls = Ms@initial_guess, Ml@initial_guess
    veh_model = Bicycle(s0,l0, s_d0, s_dd0,heading=0., dt=dt, wheelbase=wheelbase)
    traj.derivative(veh_model,xs=ss,ys=ls) # 这里是不对的
    bboxes.predict(traj.t)
    corridors = generate_corridor_bboxes(traj, bboxes)#:return: corridor: [ [x1_min, y1_min, x1_max, y1_max],... ]
    slb,llb, sub,lub = corridors.T#[:,0],corridors[:,1],corridors[:,2],corridors[:,3]
    Aineq_list += [Ms,-Ms,Ml,-Ml]
    bineq_list += [sub,-slb,lub,-llb]



    # 两点边值约束
    if not (target_l_bound is None):
        target_l_lb, target_l_ub = target_l_bound
        Aineq_list += [Final@Ml, -Final@Ml] # 末端横向不超过要求的范围
        bineq_list += [[target_l_ub], [-target_l_lb]]
    Aeq_list += [Final@G1@Ml, Final@G2@Ml, Final@G2@Ms, First@Ml, First@Ms]
    beq_list += [0., -Final@H2_1*l_d0, -Final@H2_1*s_d0, l0, s0]
    # 分别为：末端横向速度、横向加速度、纵向加速度为0；初始横纵向位置固定

    Aeq = np.vstack(Aeq_list)
    Aineq = np.vstack(Aineq_list)

    # beq = np.concatenate(beq_list)
    bineq = np.concatenate(bineq_list)
    beq = np.array(beq_list)
    return Aineq, bineq, Aeq, beq, corridors


def getCostFunMat(s_d0,s_dd0,l_d0,l_dd0, target_l):
    w1,w2,w3 = np.array([0.3,2.,1.])*1e2
    # 舒适
    k1 = 5

    hs = H3_1 * s_d0 + H3_2 * s_dd0
    hl = H3_1 * l_d0 + H3_2 * l_dd0
    G3Ms, G3Ml = G3@Ms, G3@Ml
    Q_comf = G3Ms.T @ G3Ms + k1 * G3Ml.T @ G3Ml
    f_comf = 2 * (hs.T @ G3Ms + k1 * hl.T @ G3Ml)

    # 效率
    Q_eff = np.zeros((2*N, 2*N))
    f_eff = - Minus @ Ms

    # 安全
    k2 = 1000
    W = np.eye(N)
    W[-1,-1] = k2
    Q_safe = Ml.T @ W @ Ml
    f_safe = -2 * target_l * np.ones(N) @ W @ Ml
    # Q_safe = Ml.T @ Ml + k2 * (Final@Ml).T @ (Final@Ml)
    # f_safe = -2*target_l*np.ones(N)@Ml - k2*2*target_l*Final@Ml

    # 加权和
    Q = 2 * (w1*Q_comf + w2*Q_eff + w3*Q_safe)
    f = w1*f_comf + w2*f_eff + w3*f_safe
    return Q,f

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
        self._corridors = None
        pass

    def optimize_traj(self,
                      initial_frenet_trajectory: FrenetTrajectory, # Union[FrenetTrajectory, np.ndarray],
                      perception:Union[TrackingBoxList, OccupancyGrid2D],
                      target_offset=0.0,
                      target_offset_bound=None, # 终点处的横向l的范围
                      offset_bound = None # 过程中所有处的横向l的范围 # todo: 这个约束要加进去，输入可以是与l同大小的数组也可以是一个值
                      ):
        if not isinstance(perception, TrackingBoxList):
            raise NotImplementedError("Optimization under occupancy has not been implemented. It is recommended to Use cartesian Optimizer")
        # if not isinstance(initial_frenet_trajectory, FrenetTrajectory):
            # raise NotImplementedError("FrenetTrajectory not supported now. Please convert it to ndarray")

        traj = initial_frenet_trajectory
        traj_sl_array = np.concatenate((traj.s, traj.l))
        s_0, l_0 = traj.s[0], traj.l[0]
        s_dot0, s_2dot0, l_dot0, l_2dot0 = traj.s_dot[0], traj.s_2dot[0], traj.l_dot[0], traj.l_2dot[0]

        Q, f = getCostFunMat(s_dot0, s_2dot0, l_dot0, l_2dot0, target_offset)
        Aineq, bineq, Aeq, beq, corridors = getConsMat(s_0, s_dot0, s_2dot0, l_0, l_dot0, l_2dot0, target_offset_bound,
                                                       traj_sl_array, perception)

        Q, f, Aineq, bineq, Aeq, beq = [matrix(i.astype(np.float64)) for i in [Q, f, Aineq, bineq, Aeq, beq]]
        sol = solvers.qp(Q, f, Aineq, bineq, Aeq, beq)  # kktsolver='ldl', options={'kktreg':1e-9}

        optim_traj_arr = np.array(sol['x'])
        optim_traj = FrenetTrajectory(traj.steps, traj.dt)

        optim_traj.s = s = optim_traj_arr[:steps]#[:,0]
        optim_traj.l = l = optim_traj_arr[steps:]#[:, 1]

        self._corridors = corridors
        return optim_traj

        # optim_traj.s_dot =

    def optimize_traj_arr(self,
                      initial_sl_pair_array: np.ndarray,
                      perception:Union[TrackingBoxList, OccupancyGrid2D],
                      target_offset=0.0,
                      target_offset_bound=None, # 终点处的横向l的范围
                      offset_bound = None # 过程中所有处的横向l的范围
                      ):
        # todo: 输入sl_pair_array & 初始横纵向状态
        pass



if __name__ == '__main__':
    from spider.elements import TrackingBox
    import matplotlib.pyplot as plt


    def draw_polygon(vertices, color='black', lw=1.):
        vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
        plt.plot(vertices[:, 0], vertices[:, 1], color=color, linestyle='-', linewidth=lw)


    optim = FrenetTrajectoryOptimizer(50,0.1)

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

    optim_traj = optim.optimize_traj(initial_frenet_trajectory,bboxes)
    corridors = optim._corridors

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
