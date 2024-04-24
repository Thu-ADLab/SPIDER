import numpy as np
import matplotlib.pyplot as plt


'''
要求势场处处可导！
'''

# 定义风险势能场的图像表示，假设风险势能场是一个二维数组
# risk_potential_field = np.array([[0.1, 0.2, 0.3],
#                                  [0.2, 0.5, 0.4],
#                                  [0.3, 0.4, 0.6]])
x, y = np.arange(600), np.arange(200)
xx, yy = np.meshgrid(x,y)
dist = np.sqrt((xx-380) ** 2 + (yy-100) ** 2)
dist[dist<0.001] = 0.001
risk_potential_field = 10/dist # 这里是反比，而不是平方反比
risk_potential_field[risk_potential_field>1] = 1.0

risk_potential_field -= risk_potential_field.min()
risk_potential_field /= risk_potential_field.max()


# 定义初始轨迹，假设是一个简单的二维数组，表示轨迹的坐标点
initial_trajectory0 = initial_trajectory = np.column_stack([
    np.linspace(200.,500, 50),
    np.linspace(50., 150, 50)
])

# 梯度下降的学习率/step size
learning_rate = 5.

# 最大迭代次数
max_iterations = 1000

# 停止迭代的条件
stop_dcost_thresh = 0.01
stop_dtraj_thresh = 0.01
stop_dcost_endure = 30
dcost_endure_count = 0
dtraj_endure_count = 0

cost_record = []
for i in range(max_iterations):
    # 计算当前轨迹的总风险势能
    total_potential = risk_potential_field[
        initial_trajectory[:, 1].astype(int),
        initial_trajectory[:, 0].astype(int)
    ].sum()

    if len(cost_record) > 0 and abs(total_potential - cost_record[-1]) < stop_dcost_thresh:
        dcost_endure_count += 1
        if dcost_endure_count > stop_dcost_endure:
            print("Converged at iteration %d" % i)
            break
    endure_count = 0
    cost_record.append(total_potential)

    # grad descend to optimize the traj
    gradient_x = np.gradient(risk_potential_field, axis=1)
    gradient_y = np.gradient(risk_potential_field, axis=0)
    gradient_x = gradient_x[
        initial_trajectory[:, 1].astype(int),
        initial_trajectory[:, 0].astype(int)
    ]
    gradient_y = gradient_y[
        initial_trajectory[:, 1].astype(int),
        initial_trajectory[:, 0].astype(int)
    ]
    updated_trajectory = initial_trajectory - learning_rate * np.column_stack((gradient_x, gradient_y))

    if np.linalg.norm(updated_trajectory - initial_trajectory) < stop_dtraj_thresh:
        dtraj_endure_count += 1
        if dtraj_endure_count > stop_dcost_endure:
            print("Converged at iteration %d" % i)
            break

    initial_trajectory = updated_trajectory

    plt.cla()
    plt.plot(cost_record)
    plt.pause(0.01)

plt.figure()
plt.plot(np.diff(cost_record))

plt.figure()
plt.imshow(risk_potential_field, cmap='Reds')
plt.plot(initial_trajectory0[:,0], initial_trajectory0[:,1],'-k')
plt.plot(initial_trajectory[:,0], initial_trajectory[:,1],'-r')
plt.show()


# for i in range(max_iterations):
#     total_potential = 0
#     for point in initial_trajectory:
#         x, y = point
#         total_potential += risk_potential_field[int(y), int(x)]
#
#     updated_trajectory = np.copy(initial_trajectory)
#     for j, point in enumerate(initial_trajectory):
#         x, y = point
#         gradient_x = (risk_potential_field[int(y), int(min(x + 1, risk_potential_field.shape[0] - 1))] -
#                       risk_potential_field[int(y), int(max(x - 1, 0))])
#         gradient_y = (risk_potential_field[int(min(y + 1, risk_potential_field.shape[1] - 1)), int(x)] -
#                       risk_potential_field[int(max(y - 1, 0)), int(x)])
#
#         # 更新轨迹点的坐标
#         updated_trajectory[j, 0] += learning_rate * gradient_x
#         updated_trajectory[j, 1] += learning_rate * gradient_y
#
#     initial_trajectory = updated_trajectory