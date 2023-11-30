import matplotlib.pyplot as plt
import numpy as np

def disks_approximate(length,width,N=3):
    '''

    :param length:
    :param width:
    :param N:
    :return: 返回几个圆盘的圆心在heading方向上与车辆质心的距离
    '''
    radius = np.sqrt((width/2) ** 2 + (length/(2*N)) ** 2)
    center_offset = np.array([i * length/N for i in range(N)]) + length/(2*N) - length/2
    return center_offset,radius






if __name__ == '__main__':
    from utils.Visualize import *
    l,w = 5., 2.
    xc, yc = l/2, w/2
    center_pos, radius = disks_approximate(l,w,2)
    draw_rectangle(0,0,l,w)

    center_x = center_pos+xc
    center_y = yc * np.ones_like(center_x)
    for xx,yy in zip(center_x,center_y):
        circle = plt.Circle((xx,yy), radius, fill=False,color='red')
        plt.gca().add_artist(circle)
    plt.gca().set_aspect(1)
    plt.xlim([-2, l+2])
    plt.ylim([-2,w+2])
    plt.show()