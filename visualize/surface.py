import matplotlib.pyplot as plt
import numpy as np
from spider.elements.Box import obb2vertices


def draw_polygon(vertices, *args, fill=False, **kwargs):
    # vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    # plt.plot(vertices[:, 0], vertices[:, 1], *args, **kwargs)

    polygon = plt.Polygon(vertices, *args, fill=fill, **kwargs)
    #closed=True, edgecolor='black', color='green'
    plt.gca().add_patch(polygon)
    return polygon


def draw_obb(obb, *args, fill=False, **kwargs):
    vertices = obb2vertices(obb)
    return draw_polygon(vertices, *args, fill=fill, **kwargs)


def draw_aabb(aabb_rect, *args, fill=False, **kwargs):
    xmin, ymin, xmax, ymax = aabb_rect
    vertices = np.array([
        [xmin, ymin],
        [xmin, ymax],
        [xmax, ymax],
        [xmax, ymin],
        # [xmin, ymin]
    ])
    # return plt.plot(vertices[:, 0], vertices[:, 1], *args, **kwargs)
    return draw_polygon(vertices, *args, fill=fill, **kwargs)

def draw_circle(center, radius, *args, mark_center=False, fill=False, alpha=1.0, **kwargs):
    # fill=False, color='r'
    circle = plt.Circle(center, radius, *args, fill=fill,alpha=alpha, **kwargs)
    plt.gca().add_artist(circle)

    if mark_center:
        marker_color = 'black' if fill else circle.get_edgecolor()
        plt.plot(center[0], center[1],'.', color=marker_color)




if __name__ == '__main__':
    from spider.elements import TrackingBox, TrackingBoxList
    tb = TrackingBox(obb=[1,6,5,2,3.14/6])
    tb_list = TrackingBoxList([tb])
    draw_polygon(tb.vertices, color='red')#,

    draw_circle([2, 5], 1,mark_center=True, fill=True, alpha=0.6)
    draw_circle([3, 7], 1, mark_center=True, color='blue')
    plt.show()
