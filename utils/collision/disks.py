import numpy as np
from typing import Type, Union, Sequence
import spider.elements as elm

_target_theta = 45 / 180 * np.pi
_tan_target_theta = np.tan(_target_theta)

#bounding_box: Union[Type[elm.BoundingBox], Sequence]
def disks_approximate(bounding_box: Union[Type[elm.BoundingBox], Sequence],disk_num:int=-1):
    '''

    :param length:
    :param width:
    :param N:
    :return: 返回几个圆盘的圆心在heading方向上与车辆质心的距离
    '''
    if isinstance(bounding_box, elm.BoundingBox):
        x,y, length, width, heading = bounding_box.obb
    else:
        x, y, length, width, heading = bounding_box

    if disk_num < 0: # 自动以45度theta角为目标计算disk_num
        disk_num = _tan_target_theta * length / width
        disk_num = int(np.ceil(disk_num))

    radius = np.sqrt((width/2) ** 2 + (length/(2*disk_num)) ** 2)
    delta_length = length / disk_num

    center_offset = delta_length*np.arange(disk_num) + (delta_length/2. - length/2.) # 把yaw对齐x轴时，圆心的x
    centers = np.column_stack((center_offset, np.zeros_like(center_offset))) # 把yaw对齐x轴时，圆心的x,y
    centers = elm.vector.rotate(centers, (0.,0.), heading)

    centers[:, 0] += x
    centers[:, 1] += y

    return centers, radius

def disk_check_for_box(obb1, obb2):
    '''
    :param obb1: 多边形1
    :param obb2: 多边形2
    :return: 碰撞与否
    '''
    centers1, radius1 = disks_approximate(obb1)
    centers2, radius2 = disks_approximate(obb2)

    # 使用广播计算所有点对之间的距离的平方
    all_dist_2 = np.sum((centers1[:, np.newaxis, :] - centers2)**2, axis=2)

    threshold = (radius1 + radius2) ** 2
    if np.any(all_dist_2 <= threshold):
        return True
    else:
        return False


def disk_check(center1, radius1, center2, radius2):
    '''
    检查两个disk之间是否碰撞
    return True: 碰撞
    '''
    distance = np.linalg.norm(center2-center1)
    if distance > radius1+radius2:
        return False
    else:
        return True


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    obb1 = [100, 200, 8, 2, 30/180*3.14]
    cs, r = disks_approximate(obb1)

    vs = elm.Box.obb2vertices(obb1)
    vs = np.vstack((vs,vs[0,:]))
    plt.plot(vs[:,0],vs[:,1])

    for pt in cs:
        plt.plot(pt[0], pt[1], 'r.')
        temp = plt.Circle(pt, r, fill=False, color='r')
        plt.gca().add_artist(temp)

    obb2 = [105, 199, 5, 2, 150 / 180 * 3.14]
    cs, r = disks_approximate(obb2)

    vs = elm.Box.obb2vertices(obb2)
    vs = np.vstack((vs, vs[0, :]))
    plt.plot(vs[:, 0], vs[:, 1])

    for pt in cs:
        plt.plot(pt[0], pt[1], 'r.')
        temp = plt.Circle(pt, r, fill=False, color='r')
        plt.gca().add_artist(temp)


    if disk_check_for_box(obb1, obb2):
        print("COLLIDE!")
    else:
        print("SAFE!")

    plt.gca().set_aspect("equal")
    plt.show()

