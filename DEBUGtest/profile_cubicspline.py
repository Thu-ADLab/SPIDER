import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from spider.elements.curves import myCubicSpline as mycsp
from spider.elements.curves import spCubicSpline as my_spcsp
from scipy.interpolate import CubicSpline as spcsp


def test_spcsp():
    x = np.arange(10)
    y = np.sin(x)
    cs = spcsp(x, y)
    xs = np.arange(-0.5, 9.6, 0.1)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(x, y, 'o', label='data')
    ax.plot(xs, np.sin(xs), label='true')
    ax.plot(xs, cs(xs), label="S")
    ax.plot(xs, cs(xs, 1), label="S'")
    ax.plot(xs, cs(xs, 2), label="S''")
    ax.plot(xs, cs(xs, 3), label="S'''")
    ax.set_xlim(-0.5, 9.5)
    ax.legend(loc='lower left', ncol=2)
    plt.show()


if __name__ == '__main__':
    # test_spcsp()
    # assert  0

    # 生成一些二维点序列，这里以随机点为例
    np.random.seed(6)
    x = np.arange(10) + np.random.normal(0, 0.3, 10)
    y = np.sin(x) + np.random.normal(0, 0.1, len(x))

    x = np.append(x, 20)
    y = np.append(y, y[-1])


    # 使用三次样条插值
    csp1 = mycsp(x, y)
    csp2 = my_spcsp(x, y)
    csp3 = spcsp(x, y, bc_type='natural')

    x_interp = np.linspace(min(x), max(x)+5, 100)

    ############## 计时 ####################
    for csp in [csp1, csp2, csp3]:
        for _ in tqdm(range(10000),desc=str(csp.__class__)):
            y_interp = csp(x_interp)
    ##########################################
    # <class 'spider.elements.curves.CubicSpline'>: 100%|██████████| 100000/100000 [00:04<00:00, 21689.01it/s]
    # <class 'scipy.interpolate._cubic.CubicSpline'>: 100%|██████████| 100000/100000 [00:00<00:00, 282696.19it/s]
    # 离谱，scipy自带的比我设计的要快10倍以上


    # 生成插值点
    plt.figure(figsize=(12,8))
    for order in range(4):
        plt.subplot(2,2,order+1)
        for csp in [csp1, csp2, csp3]:

            y_interp = csp(x_interp, order)
            # 绘制原始点和插值曲线
            plt.plot(x_interp, y_interp, label=str(csp.__class__).split('.')[-1])

        if order == 0: plt.scatter(x, y, label='Original Points')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Cubic Spline for order '+str(order))
    plt.show()




