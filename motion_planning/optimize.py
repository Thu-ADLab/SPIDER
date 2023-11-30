import numpy as np
import scipy.optimize as opt
import cyipopt
from utils.collision.CollisionConstraints import generate_corridor
from elements.trajectory import Trajectory
from utils.collision.Disks import disks_approximate
from copy import deepcopy




N = steps = 14
dt = 0.1

M_A = np.zeros((N, 2*N))
M_PHI = np.zeros((N, 2*N))
for i in range(N):
    M_A[i,2*i] = 1
    M_PHI[i, 2*i+1] = 1

M_j = np.zeros((N-1,N))
for i in range(N-1):
    M_j[i,i] = -1
    M_j[i, i+1] = 1

M_V = np.zeros((N+1, N)) #0~N
M_V[1:,:] = np.tril(np.ones((N,N)))

# M_X = np.zeros(N,N) #1~N
M_X = np.tril(np.ones((N,N))) #1~N

w1,w2,w3,w4 = 1,1,1,5

def cost(U:np.ndarray):
    #U=[acceleration0, steering0,acceleration1, steering1,...]
    A = M_A @ U # 0~N-1
    PHI = M_PHI @ U # 0~N-1

    V = M_V @ A * dt + v0 #0~N

    ## jerk cost
    jerk = M_j @ A /dt
    jerk0 = (A[0] - a_pre)/dt
    jerk_cost = jerk.T @ jerk + jerk0 ** 2

    #steer velocity
    steer_velocity = M_j @ PHI / dt * 180 /np.pi
    steer_velocity0 = (PHI[0] - 0.0) / dt * 180 /np.pi# FIXME:qzl:0.0不对，应该减去其观测到的初值
    steer_velocity_cost = steer_velocity.T @ steer_velocity + steer_velocity0 ** 2

    ## centripetal acceleration
    cen_acc = V[:-1] * V[:-1] * PHI / (veh_length/2)
    cen_acc_cost = np.sum(cen_acc**2)

    ## efficiency
    efficiency_cost = np.sum(V*dt)

    cost = jerk_cost*w1 + cen_acc_cost*w2 + efficiency_cost * w3 + steer_velocity_cost * w4
    return cost

def constraints(acc_bound, steer_bound, speed_bound,
                initial_guess:Trajectory, observation, delta=0.2, max_expand=5.0, center_offset=tuple([0.])):
    # eq表示 函数结果等于0 ； ineq 表示 表达式大于等于0
    cons = []
    # cons += getAccBoundCons(acc_bound)
    # cons += getSteerBoundCons(steer_bound)
    cons += getVelocityCons(speed_bound)
    cons += getCorridorCons(initial_guess, observation, delta, max_expand, center_offset)
    cons += getGoalCons(initial_guess)
    return tuple(cons)

def getGoalCons(initial_guess, goal_scale=(0.2, 0.2)):
    xscale,yscale = goal_scale
    cons = [
        {'type': 'ineq', 'fun': lambda U: veh_model.accsteer2state(M_A@U, M_PHI@U)[0][-1] - initial_guess.x[-1] + xscale},
        {'type': 'ineq', 'fun': lambda U: initial_guess.x[-1] - veh_model.accsteer2state(M_A@U, M_PHI@U)[0][-1] + xscale},
        {'type': 'ineq', 'fun': lambda U: veh_model.accsteer2state(M_A@U, M_PHI@U)[1][-1] - initial_guess.y[-1] + yscale},
        {'type': 'ineq', 'fun': lambda U: initial_guess.y[-1] - veh_model.accsteer2state(M_A@U, M_PHI@U)[1][-1] + yscale},

        {'type': 'ineq',
         'fun': lambda U: veh_model.accsteer2state(M_A @ U, M_PHI @ U)[3][-1] - initial_guess.heading[-1] + 0.05},
        {'type': 'ineq',
         'fun': lambda U: initial_guess.heading[-1] - veh_model.accsteer2state(M_A @ U, M_PHI @ U)[3][-1] + 0.05},
    ]
    return cons

def getAccBoundCons(acc_bound):
    lb, ub = acc_bound # 上下界
    cons = []
    # for i in range(steps):
    #     cons.append({'type': 'ineq', 'fun': lambda U: U[2*i] - lb})
    #     cons.append({'type': 'ineq', 'fun': lambda U: ub - U[2 * i]})
    cons = [
        {'type': 'ineq', 'fun': lambda U: M_A @ U - lb},
        {'type': 'ineq', 'fun': lambda U: ub - M_A @ U},
    ]
    return cons

def getSteerBoundCons(steer_bound):
    lb, ub = steer_bound  # 上下界
    cons = [
        {'type': 'ineq', 'fun': lambda U: M_PHI @ U - lb},
        {'type': 'ineq', 'fun': lambda U: ub - M_PHI @ U},
    ]
    return cons

def getVelocityCons(speed_bound):
    # V = M_V[1:,:] @ M_A @ U * dt + v0 # 1~N, M_V要截取是因为只检查未来时刻的速度
    temp_M = M_V[1:,:] @ M_A * dt
    lb, ub = speed_bound  # 上下界
    cons = [
        {'type': 'ineq', 'fun': lambda U: temp_M @ U + v0 - lb},
        {'type': 'ineq', 'fun': lambda U: ub - temp_M @ U - v0},
    ]
    return cons


def getCorridorCons(initial_guess:Trajectory, observation, delta=0.2, max_expand=5.0,center_offset=tuple([0.])):
    # lon_offset为负表示在几何中心后方
    cons = []
    for lon_offset in center_offset:
        # 计算一个offset对应的轨迹
        traj = deepcopy(initial_guess)
        traj.offsetAdjust(lon_offset) # 调整offset
        corridor = generate_corridor(initial_guess, observation, delta, max_expand)
        xlb, xub = corridor[:,0], corridor[:,2]
        ylb, yub = corridor[:,1], corridor[:,3]
        temp_cons = [
            {'type': 'ineq', 'fun': lambda U: veh_model.accsteer2state(M_A@U, M_PHI@U)[0][1:] - xlb}, # [1:]是只检查1~N
            {'type': 'ineq', 'fun': lambda U: xub - veh_model.accsteer2state(M_A@U, M_PHI@U)[0][1:]}, # [1:]是只检查1~N
            {'type': 'ineq', 'fun': lambda U: veh_model.accsteer2state(M_A@U, M_PHI@U)[1][1:] - ylb}, # [1:]是只检查1~N
            {'type': 'ineq', 'fun': lambda U: yub - veh_model.accsteer2state(M_A@U, M_PHI@U)[1][1:]}, # [1:]是只检查1~N
        ]
        # U = U0
        # assert np.all(veh_model.accsteer2state(M_A@U0, M_PHI@U0)[0][1:] - xlb>0)
        # assert np.all(xub - veh_model.accsteer2state(M_A@U, M_PHI@U)[0][1:]>0)
        # assert np.all(veh_model.accsteer2state(M_A@U, M_PHI@U)[1][1:] - ylb>0)
        # assert np.all(yub - veh_model.accsteer2state(M_A@U, M_PHI@U)[1][1:]>0)

        # temp_cons = [
        #     {'type': 'ineq', 'fun': lambda U: accsteer2x(U) - xlb},
        #     {'type': 'ineq', 'fun': lambda U: xub - accsteer2x(U)},
        #     {'type': 'ineq', 'fun': lambda U: accsteer2y(U) - ylb},
        #     {'type': 'ineq', 'fun': lambda U: yub - accsteer2y(U)},
        # ]
        cons += temp_cons

    return cons




if __name__ == '__main__':
    import pickle
    from utils.Visualize import *
    from elements.Box import TrackingBoxList,AABB_vertices,TrackingBox
    from elements.trajectory import Trajectory
    from vehicle_model.bicycle import Bicycle

    file = open('D:/Workspace/TrajDETR/dataset/groundtruth/gt.pickle', 'rb')
    groundtruth = pickle.load(file)
    file.close()
    file = open('D:/Workspace/TrajDETR/dataset/observation/obs.pickle', 'rb')
    observations = pickle.load(file)
    file.close()
    idx = 0

    available_traj = groundtruth[idx]
    obs = observations[idx]

    veh_length = 5
    veh_width = 2
    wheelbase = 3.0

    # observation
    bboxes = TrackingBoxList()
    for x,y,vx,vy in obs:
        vertices = AABB_vertices([x-veh_length/2, y-veh_width/2,x+veh_length/2, y+veh_width/2])
        tb = TrackingBox(vertices=vertices, vx=vx,vy=vy)
        bboxes.append(tb)

    # initial guess
    center_offset, radius = disks_approximate(veh_length, veh_width, 3)
    bboxes.dilate(radius)

    for traj in available_traj[:]:

        veh_model = Bicycle(traj['x'][0],traj['y'][0],traj['v0'],traj['a0'],
                            traj['yaw'][0],0,0,dt=dt,wheelbase=wheelbase)
        x0,y0,v0,a_pre,heading0 = veh_model.x,veh_model.y,veh_model.velocity,veh_model.acceleration,veh_model.heading # TODO:这几个量要写到函数里面去
        initial_guess = Trajectory(dt=dt)
        initial_guess.derivative(deepcopy(veh_model),traj['x'],traj['y'])
        initial_guess.t = traj['t']

        bboxes.predict(initial_guess.t)

        U0 = []
        for i in range(len(initial_guess.t)):
            if i == 0:
                continue
            U0 += [initial_guess.a[i], initial_guess.steer[i]]
        U0 = np.array(U0)

        acc_bd = (-6.,6.)
        steer_bd = (-40.*np.pi/180, 40.*np.pi/180)
        speed_bd = (0., 100./3.6)
        bd = []
        for i in range(steps):
            bd.append(acc_bd)
            bd.append(steer_bd)
        cons = constraints(acc_bd, steer_bd, speed_bd,
                           initial_guess=initial_guess, observation=bboxes)

        # res = cyipopt.minimize_ipopt(cost, U0, bounds=tuple(bd), constraints=cons)
        res = opt.minimize(cost, U0, bounds=tuple(bd),constraints=cons)
        # res = opt.minimize(cost, U0, constraints=cons)

        # veh_model = Bicycle(traj['x'][0], traj['y'][0], traj['v0'], traj['a0'],
        #                     traj['yaw'][0], 0, 0, dt=dt, wheelbase=wheelbase)
        optim_traj = Trajectory(dt=dt)
        optim_traj.step(deepcopy(veh_model), M_A@res.x, M_PHI@res.x)
        # optim_traj.step(veh_model, M_A @ U0, M_PHI @ U0)
        optim_traj.t = traj['t']

        plt.figure(1)
        for x,y,vx,vy in obs:
            vertices = AABB_vertices([x-veh_length/2, y-veh_width/2,x+veh_length/2, y+veh_width/2])
            draw_polygon(vertices,color='blue')

        egox, egoy = traj['x'][0], traj['y'][0]
        vertices = AABB_vertices([egox-veh_length/2, egoy-veh_width/2,egox+veh_length/2, egoy+veh_width/2])
        draw_polygon(vertices,color='black')

        draw_trajectory(initial_guess, 'blue')
        draw_trajectory(optim_traj,'red')

        # for lon_offset in center_offset:
        # 计算一个offset对应的轨迹
        traject = deepcopy(initial_guess)
        traject.offsetAdjust(lon_offset=0.)  # 调整offset
        corridor = generate_corridor(initial_guess, bboxes)
        for aabb in corridor:
            vertices = AABB_vertices(aabb)
            draw_polygon(vertices, color='green',lw=0.3)



        plt.figure(2)
        plt.subplot(211)
        plt.plot(np.arange(steps),M_A @ U0)
        plt.plot(np.arange(steps), M_A @ res.x, '--')
        plt.subplot(212)
        plt.plot(np.arange(steps), M_PHI @ U0)
        plt.plot(np.arange(steps), M_PHI @ res.x, '--')
        print('-----')
        print(cost(U0))
        print(cost(res.x))



    plt.figure(1)
    plt.gca().set_aspect(1)
    plt.xlim([175,235])
    plt.show()
    pass





