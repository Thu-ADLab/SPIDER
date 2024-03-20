import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from spider.elements.Box import obb2vertices


def draw_polygon(vertices, *args, fill=False, **kwargs):
    # vertices = np.vstack((vertices, vertices[0]))  # recurrent to close polyline
    # plt.plot(vertices[:, 0], vertices[:, 1], *args, **kwargs)

    if fill:
        # 这里设计的key不太好，想想重新设计一下
        face_alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
        edge_alpha = kwargs['edge_alpha'] if 'edge_alpha' in kwargs else 0.8
        if 'alpha' in kwargs: kwargs.pop('alpha')
        if 'edge_alpha' in kwargs: kwargs.pop('edge_alpha')
    # if fill:
    #     if 'color' in kwargs and 'alpha' in kwargs:
    #         kwargs['edgecolor'] = to_rgba(kwargs['color'], 0.8)
    #         kwargs['facecolor'] = to_rgba(kwargs['color'], kwargs['alpha'])
    #         del kwargs['color'], kwargs['alpha']


    polygon = plt.Polygon(vertices, *args, fill=fill, **kwargs)

    if fill:
        polygon.set_facecolor(to_rgba(polygon.get_facecolor(), face_alpha))
        if 'color' in kwargs or 'edgecolor' in kwargs: # 如果边框已经被设置过颜色，就只改透明度即可
            polygon.set_edgecolor(to_rgba(polygon.get_edgecolor(), edge_alpha))
        else: # 如果边框没有被设置过颜色，就把颜色改成facecolor
            polygon.set_edgecolor(to_rgba(polygon.get_facecolor(), edge_alpha))

    plt.gca().add_patch(polygon)
    #closed=True, edgecolor='black', color='green'

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
    pass
