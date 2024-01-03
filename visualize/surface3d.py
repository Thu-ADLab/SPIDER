import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import mpl_toolkits.mplot3d.art3d as art3d

from spider.visualize.surface import draw_polygon


def draw_prism(bottom_vertices, bottom_z, top_vertices, top_z, color=(0.8, 0.4, 0.6), alpha=0.4):
    '''
    画棱柱
    '''

    ax = plt.gca()

    # 定义顶部和底部顶点坐标
    top_vertices = np.array(top_vertices)
    bottom_vertices = np.array(bottom_vertices)

    # 绘制底部的面
    bottom = draw_polygon(bottom_vertices, fill=True, facecolor=color, alpha=alpha)
    ax.add_patch(bottom)
    art3d.pathpatch_2d_to_3d(bottom, z=bottom_z)


    vertices = np.vstack((bottom_vertices, bottom_vertices[0]))  # recurrent to close polyline
    plt.plot(vertices[:, 0], vertices[:, 1], bottom_z, color=color, linestyle='-')

    # 绘制顶部的面
    # x_top = top_vertices[:, 0]
    # y_top = top_vertices[:, 1]
    top = draw_polygon(top_vertices, fill=True, facecolor=color, alpha=alpha)
    ax.add_patch(top)
    art3d.pathpatch_2d_to_3d(top, z=top_z)
    #
    vertices = np.vstack((top_vertices, top_vertices[0]))  # recurrent to close polyline
    plt.plot(vertices[:, 0], vertices[:, 1], top_z, color=color, linestyle='-')

    # 绘制四棱柱的边
    for i in range(len(top_vertices)):
        x = [bottom_vertices[i][0], top_vertices[i][0]]
        y = [bottom_vertices[i][1], top_vertices[i][1]]
        z = [bottom_z,top_z]
        ax.plot(x, y, z, 'r-', alpha=0.6)


if __name__ == '__main__':

    # 创建一个三维坐标轴
    fig = plt.figure()
    axes = plt.axes(projection="3d")
    top_vertices = np.array([[2, 0], [2, 1],  [3, 1],[3, 0]])
    bottom_vertices = np.array([[0, 0], [0, 1],  [1, 1], [1, 0]])

    draw_prism(bottom_vertices, 0, top_vertices, 8)
    plt.show()
