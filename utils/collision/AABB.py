import numpy as np
from spider.elements.Box import aabb2vertices, vertices2aabb

def AABB_check(vertices1:np.ndarray ,vertices2:np.ndarray):
    xmin1, ymin1, xmax1, ymax1 = vertices2aabb(vertices1)
    xmin2, ymin2, xmax2, ymax2 = vertices2aabb(vertices2)
    length2, width2 = xmax2-xmin2, ymax2-ymin2
    expanded_aabb1 = [xmin1, ymin1, xmax1+length2, ymax1+width2]
    if expanded_aabb1[0]<xmax2<expanded_aabb1[2] and expanded_aabb1[1]<ymax2<expanded_aabb1[3]:
        collision = True
    else:
        collision = False
    return collision
