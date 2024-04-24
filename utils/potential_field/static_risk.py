
import numpy as np


G, K_risk = 9.0, 0.3
Ms = {i:1 for i in range(50)} # qzl: a ratio determined by the class of boundary points
max_risk = 1.0
_epsilon = 0.001

'''
Euclidean Signed Distance Field (ESDF)
'''
def static_risk(target_xs, target_ys, obs_x, obs_y, obs_class=0, ego_radius=1.0):
    '''
    计算静态风险
    '''
    dx = target_xs - obs_x
    dy = target_ys - obs_y
    dist = np.sqrt(dx ** 2 + dy ** 2)

    rr = (dist - ego_radius)**2
    rr[rr < _epsilon] = _epsilon  # avoid division by zero

    M = Ms.get(int(obs_class), 1.0)
    risk = G * M / rr

    risk[dist <= ego_radius] = max_risk
    risk[risk > max_risk] = max_risk
    return risk
